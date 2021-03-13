# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNetGST',
        depth=50,
        alpha=4,
        beta=2,
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        zero_init_residual=False),
    cls_head=dict(
        type='GSTHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.3,
        init_std=0.001),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
