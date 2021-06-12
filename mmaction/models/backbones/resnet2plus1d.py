import torch.nn as nn
from mmcv.cnn import ConvModule, NonLocal3d
from torch.nn.modules.utils import _triple

from ..builder import BACKBONES
from .resnet3d import (BasicBlock3d, Bottleneck3d, ResNet3d,
                       build_activation_layer)


class BasicBlock2plus1d(BasicBlock3d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Bottleneck2plus1d(Bottleneck3d):
    """Bottleneck 3d block for ResNet2plus1D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        non_local (bool): Determine whether to apply non-local module in this
            block. Default: False.
        non_local_cfg (dict): Config for non-local module. Default: ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 inflate=True,
                 inflate_style='3x3x3',
                 non_local=False,
                 non_local_cfg=dict(),
                 conv_cfg=dict(type='Conv2plus1d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 all_r2puls1d=False):
        super(Bottleneck3d, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        if self.style == 'pytorch':
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            bias=False,
            conv_cfg=conv_cfg if all_r2puls1d else dict(type='Conv3d'),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=dict(type='Conv2plus1d'),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            bias=False,
            conv_cfg=conv_cfg if all_r2puls1d else dict(type='Conv3d'),
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(self.conv3.norm.num_features,
                                              **self.non_local_cfg)


@BACKBONES.register_module()
class ResNet2Plus1d(ResNet3d):
    """ResNet (2+1)d backbone.

    This model is proposed in `A Closer Look at Spatiotemporal Convolutions for
    Action Recognition <https://arxiv.org/abs/1711.11248>`_
    """

    arch_settings = {
        18: (BasicBlock2plus1d, (2, 2, 2, 2)),
        34: (BasicBlock2plus1d, (3, 4, 6, 3)),
        50: (Bottleneck2plus1d, (3, 4, 6, 3)),
        101: (Bottleneck2plus1d, (3, 4, 23, 3)),
        152: (Bottleneck2plus1d, (3, 8, 36, 3))
    }

    def __init__(self,
                 *args,
                 with_pool1=False,
                 with_pool2=False,
                 all_2plus1d=False,
                 **kwargs):
        self.all_2plus1d = all_2plus1d
        super().__init__(
            *args,
            with_pool1=with_pool1,
            with_pool2=with_pool2,
            all_2plus1d=all_2plus1d,
            **kwargs)

        assert self.pretrained2d is False
        assert self.conv_cfg['type'] == 'Conv2plus1d'

    def _make_stem_layer(self):
        super()._make_stem_layer()
        mid_channels = None if self.all_2plus1d else 45
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False,
            conv_cfg=dict(type='Conv2plus1d', mid_channels=mid_channels),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    @staticmethod
    def make_res_layer(block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       style='pytorch',
                       inflate=1,
                       inflate_style='3x3x3',
                       non_local=0,
                       non_local_cfg=dict(),
                       norm_cfg=None,
                       act_cfg=None,
                       conv_cfg=dict(type='Conv2plus1d'),
                       with_cp=False,
                       all_2plus1d=True,
                       **kwargs):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``,
                the stride-two layer is the 3x3 conv layer, otherwise
                the stride-two layer is the first 1x1 conv layer.
                Default: ``pytorch``.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: '3x3x3'.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Default: 0.
            non_local_cfg (dict): Config for non-local module.
                Default: ``dict()``.
            conv_cfg (dict | None): Config for norm layers.
                Default: dict(type='Conv2plus1d').
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool | None): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Default: False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(
            non_local, int) else (non_local, ) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg if all_2plus1d else dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=None)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    **kwargs))

        return nn.Sequential(*layers)
