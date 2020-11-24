import torch

from ..registry import LOSSES
from .base import BaseWeightedLoss


def _neg_loss(pred, target):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                               2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


@LOSSES.register_module()
class FocalLoss(BaseWeightedLoss):

    def _forward(self, pred, target):
        return _neg_loss(pred, target)
