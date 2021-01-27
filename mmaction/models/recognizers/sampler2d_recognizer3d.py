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
            selected_imgs = torch.empty((sorted_sample_index.shape) + (imgs.shape[-3:]),
                                        device=sorted_sample_index.device)
            num_batchs = sorted_sample_index.shape[0]
            for i in range(num_batchs):
                sample_index = sorted_sample_index[i]
                selected_imgs[i] = imgs[i, sample_index]
        else:
            distribution = Bernoulli(probs)
            num_sample_times = self.num_segments * self.train_cfg.get(
                'num_sample_times', 1)
            sample_result = torch.zeros_like(probs)
            for _ in range(num_sample_times):
                sample_result += distribution.sample()
            sample_index = sample_result.topk(self.num_segments, dim=1)[1]
            sorted_sample_index, _ = sample_index.sort(dim=1, descending=False)
            # print(sorted_sample_index.shape)
            # print(imgs.shape)
            # selected_imgs = torch.gather(
            #     imgs, dim=1, index=sorted_sample_index)
            selected_imgs = torch.empty((sorted_sample_index.shape) + (imgs.shape[-3:]),
                                        device=sorted_sample_index.device)
            num_batchs = sorted_sample_index.shape[0]
            for i in range(num_batchs):
                sample_index = sorted_sample_index[i]
                selected_imgs[i] = imgs[i, sample_index]

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

    # def forward_train(self, imgs, labels, **kwargs):
    # losses = dict()
    # cls_score_list = []
    #
    # for i in range(sample_map.shape[0]):
    #     sample = sample_map[i]
    #
    #     img = imgs[sample]
    #     img = img.reshape((1, ) + img.shape)
    #     img = img.reshape((-1, ) + img.shape[2:])
    #
    #     x = self.extract_feat(img)
    #
    #     cls_score_item = self.cls_head(x)
    #     cls_score_list.append(cls_score_item)
    #
    # cls_score = torch.cat(cls_score_list)
    # gt_labels = labels.squeeze()
    # loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
    # losses.update(loss_cls)
    # return losses

    # def _do_test(self, imgs):
    #     sample_map = self.sampler(imgs, test_mode=True)
    #     cls_score_list = []
    #
    #     for i in range(sample_map.shape[0]):
    #         sample = sample_map[i]
    #
    #         img = imgs[sample]
    #         img = img.reshape((1, ) + img.shape)
    #         img = img.reshape((-1, ) + img.shape[2:])
    #
    #         x = self.extract_feat(img)
    #
    #         cls_score_item = self.cls_head(x)
    #         cls_score_list.append(cls_score_item)
    #
    #     cls_score = torch.cat(cls_score_list)
    #     cls_score = self.average_clip(cls_score)
    #
    #     return cls_score

    def forward_test(self, imgs):
        return self._do_test(imgs).cpu().numpy()

    def forward_gradcam(self, imgs):
        pass
