import math

import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ...utils import get_root_logger
from ..registry import HEADS
from .base import BaseHead


@HEADS.register_module()
class VAPHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.001,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)
        self.logger = get_root_logger()
        self.num_segments = num_segments
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.spatial_type = spatial_type
        self.in_channels = in_channels
        self.init_std = init_std

        self.vap_level = int(math.log(self.num_segments, 2))

        self.logger.info(f'Using {self.vap_level}-level VAP')

        total_timescale = 0
        for i in range(self.vap_level):
            timescale = 2**i
            total_timescale += timescale
            vap = nn.MaxPool3d((self.num_segments // timescale, 1, 1), 1, 0,
                               (timescale, 1, 1))
            setattr(self, f'vap_{timescale}', vap)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.tes = nn.Sequential(
            nn.Linear(total_timescale, total_timescale * 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_timescale * 4, total_timescale, bias=False))
        self.softmax = nn.Softmax(dim=1)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=self.init_std)

    def forward(self, x, num_segs=None):
        if self.avg_pool is not None:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        _, d = x.size()
        x = x.view(-1, self.num_segments, d, 1, 1).permute(0, 2, 1, 3, 4)
        x = torch.cat(
            tuple([
                getattr(self, f'vap_{2**i}')(x) for i in range(self.vap_level)
            ]), 2)
        x = x.squeeze(3).squeeze(3).permute(0, 2, 1)
        w = self.gap(x).squeeze(2)
        w = self.softmax(self.tes(w))
        x = x * w.unsqueeze(2)
        x = x.sum(dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x.view(-1, d))
        return x
