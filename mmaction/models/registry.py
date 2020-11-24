from mmcv.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
RECOGNIZERS = Registry('recognizer')
DETECTORS = Registry('detector')
LOSSES = Registry('loss')
LOCALIZERS = Registry('localizer')
