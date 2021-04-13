import torch
from torch.distributions import Bernoulli
import torch.nn.functional as F
import random

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
                 num_segments=4,
                 use_sampler=False,
                 combine_predict=False):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        self.num_segments = num_segments
        self.bp_mode = bp_mode
        self.combine_predict = combine_predict
        assert bp_mode in ['gradient_policy', 'gumbel_softmax']

    def sample(self, imgs, probs, test_mode=False):

        if test_mode:
            sample_index = probs.topk(self.num_segments, dim=1)[1]
            sample_index, _ = sample_index.sort(dim=1, descending=False)
            distribution = probs
            policy = None
            sample_probs = None
        else:
            num_batches, original_segments = probs.shape
            probs_flat = probs.reshape(-1)

            probs_clone = probs.clone().detach()
            # probs_sum_clone = probs_clone.sum(1)

            probs_flat_clone = probs_clone.reshape(-1)
            # cumsum_probs = probs_flat_clone.cumsum(0)
            # cumsum_probs_unflat = cumsum_probs.reshape(-1, original_segments)
            # upper_cumsum_probs = cumsum_probs_unflat[:, -1]

            # lower_cumsum_probs = F.pad(upper_cumsum_probs[:-1], (1, 0))

            # boundary_sum = cumsum_probs[original_segments-1]
            # boundary = torch.arange(num_batches) * boundary_sum
            policy = torch.zeros_like(probs_flat_clone).int()
            sample_index = []
            sample_probs = []
            if self.combine_predict:
                for _ in range(self.num_segments):
                    sub_sample_index = []
                    sub_sample_probs = []

                    # may cause sum not 1, so need to update
                    cumsum_probs = probs_flat_clone.cumsum(0)
                    cumsum_probs_unflat = cumsum_probs.reshape(-1, original_segments)
                    upper_cumsum_probs = cumsum_probs_unflat[:, -1]
                    lower_cumsum_probs = F.pad(upper_cumsum_probs[:-1], (1, 0))
                    probs_sum_clone = probs_flat_clone.reshape(-1, original_segments).sum(1)

                    for i, base_number, i_range in zip(range(len(lower_cumsum_probs)), lower_cumsum_probs, probs_sum_clone):
                        duplicate_time = 0
                        lower_bound = i * original_segments
                        higger_bound = (i + 1) * original_segments
                        while True:
                            rand_number = random.random() * i_range + base_number
                            assert base_number <= rand_number <= upper_cumsum_probs[i], f'Got {base_number} <= {rand_number} <= {upper_cumsum_probs[i]}'

                            judge = (cumsum_probs < rand_number).sum(0, keepdim=True)
                            if judge == higger_bound:
                                # avoid overflow
                                judge = judge - 1
                            if judge == lower_bound - 1:
                                judge = judge + 1
                                
                            assert lower_bound <= judge < higger_bound, f'Got {lower_bound} <= {judge} < {higger_bound}, ' \
                                                                        f'cumsum_probs: {cumsum_probs}, ' \
                                                                        f'rand_number: {rand_number}, ' \
                                                                        f'upper_cumsum_probs: {upper_cumsum_probs}, ' \
                                                                        f'lower_cumsum_probs: {lower_cumsum_probs}'

                            if policy[judge] == 1:
                                while duplicate_time <= self.num_segments:
                                    duplicate_time += 1
                                    rand_number = random.random() * i_range + base_number
                                    assert base_number <= rand_number <= upper_cumsum_probs[i], f'Got {base_number} <= {rand_number} <= {upper_cumsum_probs[i]}'

                                    judge = (cumsum_probs < rand_number).sum(0, keepdim=True)
                                    if judge == higger_bound:
                                        judge = judge - 1
                                    if judge == lower_bound - 1:
                                        judge = judge + 1
                                    assert lower_bound <= judge < higger_bound, f'Got {lower_bound} <= {judge} < {higger_bound}, ' \
                                                                                f'cumsum_probs: {cumsum_probs}, ' \
                                                                                f'rand_number: {rand_number}, ' \
                                                                                f'upper_cumsum_probs: {upper_cumsum_probs}, ' \
                                                                                f'lower_cumsum_probs: {lower_cumsum_probs}'

                                    if policy[judge] == 0:
                                        break

                                while policy[judge] == 1:
                                    judge = (judge + 1) % original_segments + i * original_segments
                                    assert lower_bound <= judge < higger_bound, f'Got {lower_bound} <= {judge} < {higger_bound}'

                            policy[judge] = 1
                            sub_sample_index.append(judge % original_segments)
                            sub_sample_probs.append(probs_flat[judge])
                            probs_flat_clone[i * original_segments:(i+1) * original_segments] /= (1-probs_flat_clone[judge]) # may cause sum not 1

                            probs_flat_clone[judge] = 0
                            break
                    # probs_flat_clone = torch.clamp(probs_flat_clone, 0.00001, 0.99999)
                    # cumsum_probs = probs_flat_clone.cumsum(0)
                    sample_index.append(torch.cat(sub_sample_index))
                    sample_probs.append(torch.cat(sub_sample_probs))
            else:
                for _ in range(self.num_segments):
                    rand_number_list = torch.rand(num_batches) * boundary_sum + boundary
                    sub_sample_index = []
                    for rand_number in rand_number_list:
                        judge = (cumsum_probs < rand_number).sum(0, keepdim=True)

                        if judge == len(policy):
                            # avoid overflow
                            judge = judge - 1

                        policy[judge] += 1
                        sub_sample_index.append(judge % original_segments)
                    sample_index.append(torch.cat(sub_sample_index).long())
            sample_index = torch.stack(sample_index, dim=1).long().sort()[0]
            sample_probs = torch.stack(sample_probs, dim=1)

            policy = policy.reshape(num_batches, -1)
            distribution = probs
        # num_batches, num_segments
        num_batches = sample_index.shape[0]
        batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
        selected_imgs = imgs[batch_inds, sample_index]
        return selected_imgs, distribution, policy, sample_index, sample_probs

    def forward_sampler(self, imgs, num_batches, test_mode=False, **kwargs):
        probs = self.sampler(imgs, num_batches)
        imgs = imgs.reshape((num_batches, -1) + (imgs.shape[-3:]))

        selected_imgs, distribution, policy, sample_index, sample_probs = self.sample(imgs, probs, test_mode)

        return selected_imgs, distribution, policy, sample_index, sample_probs

    def forward_train(self, imgs, labels, **kwargs):
        num_batches = imgs.shape[0]

        watch_map = kwargs.pop('watch_map', None)
        frame_name_list = kwargs.pop('frame_name_list', None)

        if hasattr(self, 'sampler'):
            imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
            imgs, distribution, policy, sample_index, sample_probs = self.forward_sampler(imgs, num_batches, **kwargs)

        imgs = imgs.transpose(1, 2).contiguous()

        selected_frame_names = []
        if watch_map is not None and frame_name_list is not None:
            for i, need_watch in enumerate(watch_map):
                if need_watch:
                    frame_names = np.array(frame_name_list[i])
                    sample_ind = sample_index[i].cpu().numpy().tolist()
                    selected_frame_names.append(frame_names[sample_ind])
                else:
                    selected_frame_names.append(None)
        else:
            selected_frame_names = [None] * num_batches

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
        # cls_loss_list = []
        # for i in range(cls_score.shape[0]):
        #     cls_loss_item = F.cross_entropy(cls_score[i].unsqueeze(0), gt_labels[i].unsqueeze(0))
        #     cls_loss_item = torch.exp(-cls_loss_item)
        #     cls_loss_list.append(cls_loss_item)
        # loss_cls_ = torch.tensor(cls_loss_list, device=imgs.device, requires_grad=True)

        reward_list = []
        for i in range(gt_labels.shape[0]):
            gt_label = gt_labels[i]
            gt_score = F.softmax(cls_score[i])[gt_label].unsqueeze(0)
            max_score, max_index = torch.max(F.softmax(cls_score[i]), 0)
            if max_index == gt_label:
                reward = gt_score
            else:
                reward = gt_score - max_score
                # reward = torch.clamp()
            reward_list.append(reward)
        reward = torch.cat(reward_list).clone().detach()
        # reward = torch.cat(loss_list)

        loss_cls['reward'] = reward.mean()

        eps = 1e-9
        # policy_cross_entropy = -torch.sum(torch.log(sample_probs + eps), dim=1)

        ones = torch.ones(sample_probs.shape[0], sample_probs.shape[1], device=imgs.device)
        # divisor = F.pad(sample_probs, (1, 0)) / (ones - sample_probs.cumsum(dim=1))[:, :-1]
        divisor = ones - F.pad(sample_probs.cumsum(dim=1)[:, :-1], (1,0))
        # divisor = torch.clamp(divisor, 1e-7, 1)

        sample_probs = sample_probs / divisor
        # sample_probs = torch.clamp(sample_probs, 1e-7, 0.9999999)

        entropy = torch.sum(-distribution * torch.log(distribution), dim=1)

        policy_cross_entropy = torch.sum(torch.log(sample_probs + eps), dim=1)
        # policy_cross_entropy = torch.log(sample_probs + eps)

        # loss_cls_ = -(reward * policy_cross_entropy - 0.1 * entropy).mean()
        if self.cls_head.final_loss:
            loss_cls_ = -(reward * policy_cross_entropy).mean()
            loss_cls['entropy_fc'] = loss_cls['loss_cls'].clone().detach()
            loss_cls['loss_cls'] = loss_cls_ + loss_cls['loss_cls']
        else:
            # loss_cls_ = -(reward * policy_cross_entropy - 0.1 * entropy).mean()
            # loss_cls_ = -(reward * policy_cross_entropy).mean()
            loss_cls_ = -(reward * policy_cross_entropy + 0.1 * entropy).mean()
            loss_cls['loss_cls'] = loss_cls_
            loss_cls['entropy'] = entropy.clone().detach() * 0.1
            # loss_cls['entropy'] = entropy.clone().detach() * 1

        losses.update(loss_cls)

        return losses, selected_frame_names

    def _do_test(self, imgs, **kwargs):
        num_batches = imgs.shape[0]

        watch_map = kwargs.get('watch_map')
        frame_name_list = kwargs.get('frame_name_list')

        imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
        imgs, distribution, _, sample_index, _ = self.forward_sampler(imgs, num_batches, test_mode=True)
        num_clips_crops = imgs.shape[0] // num_batches

        imgs = imgs.transpose(1, 2).contiguous()

        selected_frame_names = []
        if watch_map is not None and frame_name_list is not None:
            for i, need_watch in enumerate(watch_map):
                if need_watch:
                    frame_names = np.array(frame_name_list[i])
                    sample_ind = sample_index[i].cpu().numpy().tolist()
                    selected_frame_names.append(frame_names[sample_ind])
                else:
                    selected_frame_names.append(None)
        else:
            selected_frame_names = [None] * num_batches

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, _ = self.neck(x)
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score, num_clips_crops)
        return cls_score.cpu().numpy(), selected_frame_names, distribution.cpu().numpy()

    def forward_test(self, imgs, **kwargs):
        img_metas = kwargs['img_metas']
        watch_map = []
        frame_name_list = []
        for item in img_metas:
            watch_map.append(item['need_watch'])
            frame_name_list.append(item['frame_name_list'])
        kwargs['frame_name_list'] = frame_name_list
        kwargs['watch_map'] = watch_map
        # return self._do_test(imgs, **kwargs).cpu().numpy()
        return self._do_test(imgs, **kwargs)

    def forward_gradcam(self, imgs):
        pass
