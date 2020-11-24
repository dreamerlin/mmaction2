import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


def _gather_feature(feature, index, index_all=None):
    # dim = channel = 2*K
    # feature b, h*w , c
    # index  b, N --> b, N, c
    if index_all is not None:
        index0 = index_all
    else:
        dim = feature.size(2)
        index0 = index.unsqueeze(2).expand(index.size(0), index.size(1), dim)
    feature = feature.gather(1, index0)
    # feature --> b, N, 2*K
    return feature


def _tranpose_and_gather_feature(feature, index, index_all=None):
    # b,c,h,w --> b,h,w,c
    feature = feature.permute(0, 2, 3, 1).contiguous()
    # b,h,w,c --> b,h*w,c
    feature = feature.view(feature.size(0), -1, feature.size(3))
    feature = _gather_feature(feature, index, index_all=index_all)
    # feature --> b, N, 2*K
    return feature


@LOSSES.register_module()
class RegL1Loss(BaseWeightedLoss):

    def _forward(self, output, mask, index, target, index_all=None):
        pred = _tranpose_and_gather_feature(output, index, index_all=index_all)
        # pred --> b, N, 2*K
        # mask --> b, N ---> b, N, 2*K
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss
