import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES
from .mobilenet_v2 import MobileNetV2


@BACKBONES.register_module()
class MobileNet3DV2(MobileNetV2):

    def __init__(self,
                 *args,
                 pretrained2d=False,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 **kwargs):
        super(MobileNetV2, self).__init__(
            *args, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs)
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pretrained2d = pretrained2d

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)

            else:
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
