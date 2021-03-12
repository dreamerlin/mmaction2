import torch.nn as nn

from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class GSTHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        pass

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1, ) + x.size()[2:])
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
