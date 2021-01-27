import torch
import torch.nn as nn

from .. import builder
from ..registry import SAMPLER


@SAMPLER.register_module()
class StepSampler(nn.Module):

    def __init__(self, backbone, head):
        super().__init__()

        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)

        self.init_weights()

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        self.cls_head.init_weights()

    def forward_train(self, imgs):
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.backbone(imgs)
        score = self.head(x, num_segs)

        rand_score = torch.rand(score.shape, device=score.device)

        sample_map = score >= rand_score

        return sample_map

    def forward_test(self, imgs):
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.backbone(imgs)
        score = self.head(x, num_segs)

        sample_map = score >= 0.5

        return sample_map

    def forward(self, imgs, test_mode=False):
        if test_mode:
            return self.forward_train(imgs)
        return self.forward_test(imgs)
