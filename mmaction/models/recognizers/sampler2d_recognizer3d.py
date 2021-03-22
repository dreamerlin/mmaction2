import torch
from torch.distributions import Bernoulli
import torch.nn.functional as F

import numpy as np

from .. import builder
from ..registry import RECOGNIZERS
from .base import BaseRecognizer


def sample_gumbel(logits, eps=1e-20):
    shape = logits.shape
    U = torch.rand(shape, device=logits.device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits)
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
                 num_segments=16,
                 use_sampler=False):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        self.num_segments = num_segments
        self.bp_mode = bp_mode
        assert bp_mode in ['gradient_policy', 'gumbel_softmax']

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
        _, ind = y.topk(self.num_segments, dim=1)
        ind_float = ind.float()
        y_hard = torch.ones_like(y).view(-1, shape[-1]) * -1.
        y_hard.scatter_(1, ind, ind_float)
        y_hard = y_hard.view(*shape)

        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard, y

    def sample(self, imgs, probs, test_mode=False):
        if self.bp_mode == 'gradient_policy':

            if test_mode:
                sample_index = probs.topk(self.num_segments, dim=1)[1]
                sample_index, _ = sample_index.sort(dim=1, descending=False)
                distribution = None
                policy = None
            else:
                num_batches, original_segments = probs.shape
                cumsum_probs = probs.reshape(-1).cumsum(0)
                boundary_sum = cumsum_probs[original_segments-1]
                boundary = torch.arange(num_batches) * boundary_sum
                policy = torch.zeros_like(cumsum_probs).int()
                sample_index = []
                for _ in range(self.num_segments):
                    rand_number_list = torch.rand(num_batches) * boundary_sum + boundary
                    sub_sample_index = []
                    for rand_number in rand_number_list:
                        judge = (cumsum_probs < rand_number).sum()
                        policy[judge] += 1
                        sub_sample_index.append(judge % original_segments)
                    sample_index.append(torch.tensor(sub_sample_index, device=probs.device).int())
                sample_index = torch.stack(sample_index, dim=1).long().sort()[0]
                policy = policy.reshape(num_batches, -1)
                distribution = probs
            # num_batches, num_segments
            num_batches = sample_index.shape[0]
            batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
            selected_imgs = imgs[batch_inds, sample_index]
            return selected_imgs, distribution, policy

        else:
            if test_mode:
                sample_index = probs.topk(self.num_segments, dim=1)[1]
                sorted_sample_index, _ = sample_index.sort(dim=1, descending=False)
                distribution = None
                policy = None
            else:
                hard = self.train_cfg.get('hard_gumbel_softmax', True)
                probs = probs.squeeze()
                sample_map, distribution = self.gumbel_softmax(probs, hard)
                num_batches = imgs.shape[0]
                sorted_sample_index = sample_map[sample_map >= 0].reshape(num_batches, -1).long()
                policy = torch.zeros_like(probs).int()
                policy.scatter_(1, sorted_sample_index, 1)

            # num_batches, num_segments
            num_batches = sorted_sample_index.shape[0]
            batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sorted_sample_index)
            selected_imgs = imgs[batch_inds, sorted_sample_index]
            return selected_imgs, distribution, policy

    def forward_sampler(self, imgs, num_batches, test_mode=False, **kwargs):
        probs = self.sampler(imgs, num_batches)
        imgs = imgs.reshape((num_batches, -1) + (imgs.shape[-3:]))

        if test_mode:
            selected_imgs, distribution, policy = self.sample(imgs, probs, True)
        else:
            # alpha = self.train_cfg.get('alpha', 0.8)
            # probs = probs * alpha + (1 - probs) * (1 - alpha)
            # probs = F.softmax(probs, dim=1)
            selected_imgs, distribution, policy = self.sample(imgs, probs, False)

        return selected_imgs, distribution, policy

    def get_reward(self, cls_score, gt_labels):
        reward, pred = torch.max(cls_score, dim=1, keepdim=True)
        match = (pred == gt_labels).data
        reward[~match] = self.train_cfg.get('penalty', -1)

        return reward, match.float()

    def forward_train(self, imgs, labels, **kwargs):
        num_batches = imgs.shape[0]
        if hasattr(self, 'sampler'):
            imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
            imgs, distribution, policy = self.forward_sampler(imgs, num_batches, test_mode=False, **kwargs)
            imgs = imgs.transpose(1, 2).contiguous()
        else:
            imgs = imgs.transpose(1, 2).contiguous()

        losses = dict()
        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        if gt_labels.shape == torch.Size([]):
            gt_labels = gt_labels.unsqueeze(0)
        cls_loss_list = []
        for i in range(cls_score.shape[0]):
            cls_loss_item = F.cross_entropy(cls_score[i].unsqueeze(0), gt_labels[i].unsqueeze(0))
            cls_loss_item = torch.exp(-cls_loss_item)
            cls_loss_list.append(cls_loss_item)
        loss_cls_ = torch.tensor(cls_loss_list, device=imgs.device, requires_grad=True)
        eps = 1e-10
        policy_cross_entropy = -torch.sum(policy * torch.log(distribution + eps), dim=1)
        loss_cls_ = (loss_cls_ * policy_cross_entropy).mean()
        loss_cls['loss_cls'] = loss_cls_

        losses.update(loss_cls)

        # if self.train_cfg.get('use_reward', False):
        #     reward, match = self.get_reward(cls_score, labels)
        #     losses.update(dict(reward=reward))

        return losses

    def _do_test(self, imgs):
        num_batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
        imgs, _, _ = self.forward_sampler(imgs, num_batches, test_mode=True)
        num_clips_crops = imgs.shape[0] // num_batches

        imgs = imgs.transpose(1, 2).contiguous()

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
