import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import _load_checkpoint
from torch.utils import checkpoint as cp

from ..registry import BACKBONES
from .resnet3d import Bottleneck3d, ResNet3d


class GSTBottleneck3d(Bottleneck3d):

    def __init__(self,
                 inplanes,
                 planes,
                 alpha,
                 beta,
                 *args,
                 spatial_stride=1,
                 **kwargs):
        super().__init__(inplanes, planes, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.conv2 = nn.Conv3d(
            planes // beta,
            planes // alpha * (alpha - 1),
            kernel_size=(1, 3, 3),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, 1, 1),
            bias=False)
        self.Tconv = nn.Conv3d(
            planes // beta,
            planes // alpha,
            kernel_size=3,
            bias=False,
            stride=(1, spatial_stride, spatial_stride),
            padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):

        def _inner_forward(x):

            identity = x
            out = self.conv1(x)

            if self.beta == 2:
                num_channels = out.size()[1] // self.beta
                left = out[:, :num_channels]
                right = out[:, num_channels:]

                out1 = self.conv2(left)
                out2 = self.Tconv(right)
            else:
                out1 = self.conv2(out)
                out2 = self.Tconv(out)

            out = torch.cat((out1, out2), dim=1)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


@BACKBONES.register_module()
class ResNetGST(ResNet3d):

    def __init__(self,
                 *args,
                 alpha,
                 beta,
                 conv1_kernel=(1, 7, 7),
                 conv1_stride_t=1,
                 pool1_stride_t=1,
                 with_pools=False,
                 inflate=(0, 0, 0, 0),
                 **kwargs):
        self.arch_settings = {
            # 18: (BasicBlock3d, (2, 2, 2, 2)),
            # 34: (BasicBlock3d, (3, 4, 6, 3)),
            50: (GSTBottleneck3d, (3, 4, 6, 3)),
            101: (GSTBottleneck3d, (3, 4, 23, 3)),
            152: (GSTBottleneck3d, (3, 8, 36, 3))
        }
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            *args,
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            with_pools=with_pools,
            inflate=inflate,
            alpha=alpha,
            beta=beta**kwargs)

    def _inflate_conv_params(self,
                             conv3d,
                             state_dict_2d,
                             module_name_2d,
                             inflated_param_names,
                             is_conv2=False):
        weight_2d_name = module_name_2d + '.weight'

        conv2d_weight = state_dict_2d[weight_2d_name]

        new_weight = conv2d_weight.data.unsqueeze(2)
        if is_conv2:
            num_out, num_in = conv2d_weight.size()[:2]
            new_weight = new_weight[:num_out // self.alpha *
                                    (self.alpha - 1), :num_in // self.beta,
                                    ...]
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_weights(self, logger):
        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    self._inflate_conv_params(module.conv, state_dict_r2d,
                                              original_conv_name,
                                              inflated_param_names)
                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)

            elif 'conv2' in name and isinstance(module, nn.Conv3d):
                original_conv_name = name
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    self._inflate_conv_params(module.conv, state_dict_r2d,
                                              original_conv_name,
                                              inflated_param_names, True)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')
