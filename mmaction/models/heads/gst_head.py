from ..registry import HEADS
from .tsn_head import TSNHead


@HEADS.register_module()
class GSTHead(TSNHead):

    def __init__(self,
                 *args,
                 num_classes,
                 in_channels,
                 dropout_ratio=0.3,
                 init_std=0.001,
                 **kwargs):
        super().__init__(
            num_classes,
            in_channels,
            *args,
            dropout_ratio=dropout_ratio,
            init_std=init_std,
            **kwargs)

    def forward(self, x):
        num_sges = x.shape[2]
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1, ) + x.size()[2:])
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc_cls(x)
        x = x.view((-1, num_sges) + x.size()[1:])
        x = self.consensus(x)
        x = x.squeeze(1)

        return x
