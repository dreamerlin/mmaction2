import torch
from torch import nn as nn

from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module()
class MOCHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 k=7,
                 hm_bias=-2.19,
                 mid_wh_channel=64,
                 mid_conv_channel=256,
                 hm_loss=dict(type='FocalLoss', loss_factor=1.0),
                 mv_loss=dict(type='RegL1Loss', loss_factor=1.0),
                 wh_loss=dict(type='RegL1Loss', loss_factor=0.1)):
        super().__init__()

        hm_channels = num_classes
        mv_channels = 2 * k
        wh_channels = 2 * k

        self.hm = nn.Sequential(
            nn.Conv2d(
                k * in_channels, mid_conv_channel, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_conv_channel,
                hm_channels,
                1,
                stride=1,
                padding=0,
                bias=True))

        self.mv = nn.Sequential(
            nn.Conv2d(
                k * in_channels, mid_conv_channel, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_conv_channel,
                mv_channels,
                1,
                stride=1,
                padding=0,
                bias=True))

        self.wh = nn.Sequential(
            nn.Conv2d(in_channels, mid_wh_channel, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_wh_channel,
                wh_channels // k,
                1,
                stride=1,
                padding=0,
                bias=True))

        self.hm_loss = build_loss(hm_loss)
        self.mv_loss = build_loss(mv_loss)
        self.wh_loss = build_loss(wh_loss)

        self.hm_bias = hm_bias

    @staticmethod
    def fill_fc_weights(layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights(self):
        self.hm[-1].bias.data.fill_(self.hm_bias)
        self.fill_fc_weights(self.mv)
        self.fill_fc_weights(self.wh)

    def forward(self, x):
        output = {}
        output_wh = []
        for feature in x:
            output_wh.append(self.wh(feature))
        input_chunk = torch.cat(x, dim=1)
        output_wh = torch.cat(output_wh, dim=1)
        output['hm'] = self.hm(input_chunk)
        output['mov'] = self.mov(input_chunk)
        output['wh'] = output_wh
        return output

    def loss(self, output, **batch):
        loss = dict()

        output['hm'] = torch.clamp(
            output['hm'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = self.hm_loss(output['hm'], batch['hm'])
        mv_loss = self.mv_loss(output['mv'], batch['mask'], batch['index'],
                               batch['mv'])
        wh_loss = self.wh_loss(output['wh'], batch['mask'], batch['index'],
                               batch['wh'], batch['index_all'])
        losses = hm_loss + mv_loss + wh_loss

        loss['hm_loss'] = hm_loss
        loss['mv_loss'] = mv_loss
        loss['wh_loss'] = wh_loss
        loss['losses'] = losses

        return losses
