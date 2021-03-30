import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import BACKBONES
from .resnet import ResNet


class PA(nn.Module):

    modality_length = {'RGB': 1, 'PA': 4, 'PALite': 4, 'Flow': 5}

    def __init__(self, modality='RGB', init_std=0.001):
        super().__init__()

        assert modality in ['RGB', 'Flow', 'PA', 'PALite']
        self.modality = modality
        self.init_std = init_std

        self.num_length = self.modality_length[self.modality]
        self.shallow_conv = nn.Conv2d(3, 8, 7, 1, 3)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=self.init_std)

    def forward(self, x):
        h, w = x.size(-2), x.size(-1)
        x = x.view((-1, 3) + x.size()[-2:])
        x = self.shallow_conv(x)
        x = x.view(-1, self.num_length, x.size(-3), x.size(-2) * x.size(-1))
        for i in range(self.num_length - 1):
            distance_i = nn.PairwiseDistance(p=2)(x[:, i, :, :],
                                                  x[:,
                                                    i + 1, :, :]).unsqueeze(1)
            if i == 0:
                distance = distance_i
            else:
                distance = torch.cat((distance, distance_i), 1)
        pa = distance.view(-1, 1 * (self.num_length - 1), h, w)
        return pa


@BACKBONES.register_module()
class ResNetPAN(ResNet):

    def __init__(self, *args, depth, modality='RGB', **kwargs):
        super().__init__(*args, depth=depth, **kwargs)

        self.modality = modality
        self.pa = PA(modality=self.modality)

    def init_weights(self):
        super().init_weights()
        if self.modality == 'PALite':
            # modify parameters, assume the first blob
            # contains the convolution kernels
            first_conv = self.conv1.conv
            weight_param = first_conv.weight
            bias_param = first_conv.bias

            kernel_size = weight_param.size()
            new_kernel_size = kernel_size[:1] + (6, ) + kernel_size[2:]
            new_kernels = weight_param.data.mean(
                dim=1, keepdim=True).expand(new_kernel_size).contiguous()

            new_conv = nn.Conv2d(
                6,
                first_conv.out_channels,
                first_conv.kernel_size,
                first_conv.stride,
                first_conv.padding,
                bias=True if len(weight_param) == 2 else False)
            new_conv.weight.data = new_kernels
            # import pdb; pdb.set_trace()

            if bias_param is not None:
                new_conv.bias.data = first_conv.bias.data

            self.conv1.conv = new_conv

    def forward(self, x):
        if self.modality == 'PA':
            x = self.pa(x)
        if self.modality == 'PALite':
            pa = self.pa(x)
            rgb = x.view((-1, self.pa.num_length) + x.size()[-3:])[:,
                                                                   0, :, :, :]
            x = torch.cat((rgb, pa), 1)
        out = super().forward(x)
        return out
