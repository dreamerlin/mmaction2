# model settings
model = dict(
    type='Sampler2DRecognizer3D',
    num_segments=8,
    # bp_mode='gumbel_softmax',
    bp_mode='gradient_policy',
    sampler=dict(
        type='MobileNetV2',
        pretrained='mmcls://mobilenet_v2',
        is_sampler=True,
        num_segments=16),
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        lateral=False,
        frozen_stages=4,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=2048,
        num_classes=200,
        final_loss=False,
        frozen=True,
        spatial_type='avg',
        dropout_ratio=0.5))

# model training and testing settings
train_cfg = dict(use_sampler=True)
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'RawframeDataset'
# data_root = 'data/ActivityNet/rgb_340x256'
# data_root_val = 'data/ActivityNet/rgb_340x256'
# ann_file_train = 'data/ActivityNet/anet_train_video.txt'
# ann_file_val = 'data/ActivityNet/anet_val_video.txt'
# ann_file_test = 'data/ActivityNet/anet_val_video.txt'
data_root = 'data/ActivityNet/rawframes_train'
data_root_val = 'data/ActivityNet/rawframes_val'
ann_file_train = 'data/ActivityNet/new_anet_train_video.txt'
ann_file_val = 'data/ActivityNet/new_anet_val_video.txt'
ann_file_test = 'data/ActivityNet/new_anet_val_video.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
mc_cfg = dict(
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=16, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=16, num_clips=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=16, num_clips=1, test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=8, workers_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=6, workers_per_gpu=4),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        # with_offset=True,
        filename_tmpl='image_{:05d}.jpg'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        # with_offset=True,
        filename_tmpl='image_{:05d}.jpg'))
# optimizer
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 256
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/simple_sampler/'  # noqa: E501
load_from = 'modelzoo/slowonly_pretrained_uniform_r50_1x1x16_40e_anet_video_rgb_uniformsample.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
