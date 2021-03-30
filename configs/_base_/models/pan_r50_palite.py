# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetPAN',
        pretrained='torchvision://resnet50',
        depth=50,
        modality='PALite',
        norm_eval=False),
    cls_head=dict(
        type='PANHead',
        num_classes=400,
        in_channels=2048,
        num_segments=8,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.001),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
