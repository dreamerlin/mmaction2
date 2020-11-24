from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module()
class MOC(BaseDetector):

    def __init__(self, backbone, head, train_cfg=None, test_cfg=None):
        super().__init__()

        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.k = self.head.k

    def init_weights(self):
        self.backbone.init_weights()
        if hasattr(self, 'head'):
            self.head.init_weights()

    def forward_train(self, imgs, **kwargs):
        losses = dict()

        chunk = [self.backbone(imgs[i]) for i in range(self.k)]
        output = self.head(chunk)

        losses.update(self.head.loss(output, **kwargs))

        return losses

    def forward_test(self, imgs, **kwargs):
        assert self.k == len(imgs) // 2
        chunk1 = [self.backbone(imgs[i]) for i in range(self.k)]
        chunk2 = [self.backbone(imgs[i + self.k] for i in range(self.k))]

        return [self.head(chunk1), self.head(chunk2)]

    def forward(self, imgs, flip=False, return_loss=True, **kwargs):
        if not return_loss and flip:
            return self.forward_test(imgs, **kwargs)
        return self.forward_train(imgs, **kwargs)
