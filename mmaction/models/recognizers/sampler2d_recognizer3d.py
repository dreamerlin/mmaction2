import torch
from torch.distributions import Bernoulli
import torch.nn.functional as F

from .. import builder
from ..registry import RECOGNIZERS
from .base import BaseRecognizer


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


@RECOGNIZERS.register_module()
class Sampler2DRecognizer3D(BaseRecognizer):

    def __init__(self,
                 sampler,
                 backbone,
                 cls_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 bp_mode='gradient_policy',
                 num_segments=16):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg)
        self.num_segments = num_segments
        self.bp_mode = bp_mode
        assert bp_mode in ['gradient_policy', 'gumbel_softmax']

    def init_weights(self):
        """Initialize the model network weights."""
        self.sampler.init_weights()
        self.backbone.init_weights()
        self.cls_head.init_weights()
        if hasattr(self, 'neck'):
            self.neck.init_weights()

    def gumbel_softmax(self, logits, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        temperature = self.train_cfg.get('temperature', 1)
        y = gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard, y

    def sample(self, imgs, probs, test_mode=False):
        if test_mode:
            sample_index = probs.topk(self.num_segments, dim=1)[1]
            sorted_sample_index, _ = sample_index.sort(dim=1, descending=False)
            distribution = None
        else:
            if self.bp_mode == 'gradient_policy':
                distribution = Bernoulli(probs)
                num_sample_times = self.num_segments * self.train_cfg.get(
                    'num_sample_times', 1)
                sample_result = torch.zeros_like(probs)
                for _ in range(num_sample_times):
                    sample_result += distribution.sample()
                sample_index = sample_result.topk(self.num_segments, dim=1)[1]
                sorted_sample_index, _ = sample_index.sort(dim=1, descending=False)
            else:
                hard = self.train_cfg.get('hard_gumbel_softmax', True)
                sample_index, distribution = self.gumbel_softmax(probs, hard)
                sorted_sample_index, _ = sample_index.sort(dim=1, descending=False)

        # num_batches, num_segments
        sorted_sample_index = sorted_sample_index.squeeze(-1)
        num_batches = sorted_sample_index.shape[0]
        batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sorted_sample_index)
        selected_imgs = imgs[batch_inds, sorted_sample_index]
        return selected_imgs, distribution

    def forward_sampler(self, imgs, num_batches, test_mode=False, **kwargs):
        probs = self.sampler(imgs, num_batches)
        imgs = imgs.reshape((num_batches, -1) + (imgs.shape[-3:]))

        if test_mode:
            selected_imgs, distribution = self.sample(imgs, probs, True)
        else:
            alpha = self.train_cfg.get('alpha', 0.8)
            probs = probs * alpha + (1 - probs) * (1 - alpha)
            selected_imgs, distribution = self.sample(imgs, probs, False)

        return selected_imgs, distribution

    def get_reward(self, cls_score, gt_labels):
        reward, pred = torch.max(cls_score, dim=1, keepdim=True)
        match = (pred == gt_labels).data
        reward[~match] = self.train_cfg.get('penalty', -1)

        return reward, match.float()

    def forward_train(self, imgs, labels, **kwargs):
        num_batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
        imgs, distribution = self.forward_sampler(imgs, num_batches, test_mode=False, **kwargs)
        num_clips = 1

        imgs = imgs.reshape((num_batches, num_clips, 3, self.num_segments) +
                            imgs.shape[-2:])
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        losses.update(loss_cls)

        if self.train_cfg.get('use_reward', False):
            reward, match = self.get_reward(cls_score, labels)
            losses.update(dict(reward=reward))

        return losses

    def _do_test(self, imgs):
        num_batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
        imgs, _ = self.forward_sampler(imgs, num_batches, test_mode=True)
        num_clips_crops = imgs.shape[0] // num_batches
        imgs = imgs.reshape((-1, 3, self.num_segments) +
                            imgs.shape[-2:])

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, _ = self.neck(x)
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score, num_clips_crops)
        return cls_score

    def forward_test(self, imgs):
        return self._do_test(imgs).cpu().numpy()

    def forward_gradcam(self, imgs):
        pass
