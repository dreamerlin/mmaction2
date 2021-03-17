import torch
from torch.distributions import Bernoulli

from .. import builder
from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Sampler2DRecognizer3D(BaseRecognizer):

    def __init__(self,
                 sampler,
                 backbone,
                 cls_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_segments=16):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg)
        self.num_segments = num_segments

    def init_weights(self):
        """Initialize the model network weights."""
        self.sampler.init_weights()
        self.backbone.init_weights()
        self.cls_head.init_weights()
        if hasattr(self, 'neck'):
            self.neck.init_weights()

    def sample(self, imgs, probs, test_mode=False):
        if test_mode:
            sample_index = probs.topk(self.num_segments, dim=1)[1]
            sorted_sample_index, _ = sample_index.sort(dim=1, descending=False)
            num_batchs = sorted_sample_index.shape[0]
            batch_inds = torch.arange(num_batchs).unsqueeze(-1).expand_as(sorted_sample_index)
            selected_imgs = imgs[batch_inds, sorted_sample_index]
        else:
            distribution = Bernoulli(probs)
            num_sample_times = self.num_segments * self.train_cfg.get(
                'num_sample_times', 1)
            sample_result = torch.zeros_like(probs)
            for _ in range(num_sample_times):
                sample_result += distribution.sample()
            sample_index = sample_result.topk(self.num_segments, dim=1)[1]
            sorted_sample_index, _ = sample_index.sort(dim=1, descending=False)
            num_batchs = sorted_sample_index.shape[0]
            batch_inds = torch.arange(num_batchs).unsqueeze(-1).expand_as(sorted_sample_index)
            selected_imgs = imgs[batch_inds, sorted_sample_index]

        return selected_imgs

    def forward_sampler(self, imgs, num_batchs, test_mode=False, **kwargs):
        if test_mode:
            probs = self.sampler(imgs, num_batchs)
            imgs = imgs.reshape((num_batchs, -1) + (imgs.shape[-3:]))
            selected_imgs = self.sample(imgs, probs, True)
        else:
            probs = self.sampler(imgs, num_batchs)
            imgs = imgs.reshape((num_batchs, -1) + (imgs.shape[-3:]))
            alpha = self.train_cfg.get('alpha', 0.8)
            probs = probs * alpha + (1 - probs) * (1 - alpha)

            selected_imgs = self.sample(imgs, probs, False)
        return selected_imgs

    def get_reward(self, cls_score, gt_labels):
        reward, pred = torch.max(cls_score, dim=1, keepdim=True)
        match = (pred == gt_labels).data

        # reward = torch.empty_like(pred)
        # for i, score in enumerate(pred):
        #     if match[i]:
        #         reward[i] = score
        #     else:
        #         reward[i] = self.train_cfg.get('penalty', -1)
        reward[1 - match] = self.train_cfg.get('penalty', -1)

        return reward, match.float()

    def forward_train(self, imgs, labels, **kwargs):
        num_batchs = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
        imgs = self.forward_sampler(imgs, num_batchs, test_mode=False, **kwargs)
        num_clips = 1

        imgs = imgs.reshape((num_batchs, num_clips, 3, self.num_segments) +
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

        return losses

    def _do_test(self, imgs):
        num_batchs = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
        imgs = self.forward_sampler(imgs, num_batchs, test_mode=True)
        num_clips_crops = imgs.shape[0] // num_batchs
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
