auto_scale_lr = dict(base_batch_size=64, enable=False)
backend_args = None
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        0,
        0,
        0,
    ],
    pad_size_divisor=32,
    std=[
        255.0,
        255.0,
        255.0,
    ],
    type='DetDataPreprocessor')
data_root = '/root/autodl-tmp/mmdetection-main/data/COCO/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook',save_best='auto'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/root/autodl-tmp/mmdetection-main/pretrain/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=53,
        init_cfg=dict(checkpoint='open-mmlab://darknet53', type='Pretrained'),
        out_indices=(
            3,
            4,
            5,
        ),
        type='Darknet'),
    bbox_head=dict(
        anchor_generator=dict(
            base_sizes=[
                [
                    (
                        116,
                        90,
                    ),
                    (
                        156,
                        198,
                    ),
                    (
                        373,
                        326,
                    ),
                ],
                [
                    (
                        30,
                        61,
                    ),
                    (
                        62,
                        45,
                    ),
                    (
                        59,
                        119,
                    ),
                ],
                [
                    (
                        10,
                        13,
                    ),
                    (
                        16,
                        30,
                    ),
                    (
                        33,
                        23,
                    ),
                ],
            ],
            strides=[
                32,
                16,
                8,
            ],
            type='YOLOAnchorGenerator'),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[
            32,
            16,
            8,
        ],
        in_channels=[
            512,
            256,
            128,
        ],
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_conf=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_wh=dict(loss_weight=2.0, reduction='sum', type='MSELoss'),
        loss_xy=dict(
            loss_weight=2.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=13,
        out_channels=[
            1024,
            512,
            256,
        ],
        type='YOLOV3Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0,
            0,
            0,
        ],
        pad_size_divisor=32,
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            1024,
            512,
            256,
        ],
        num_scales=3,
        out_channels=[
            512,
            256,
            128,
        ],
        type='YOLOV3Neck'),
    test_cfg=dict(
        conf_thr=0.005,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.45, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        assigner=dict(
            min_pos_iou=0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='GridAssigner')),
    type='YOLOV3')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=2000, start_factor=0.1, type='LinearLR'),
    dict(
        by_epoch=True, gamma=0.1, milestones=[
            175,
            200,
        ], type='MultiStepLR'),
]
resume = False

test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/root/autodl-tmp/mmdetection-main/data/COCO/annotations/instances_test2017.json',
        backend_args=None,
        data_prefix=dict(img='test2017/'),
        data_root='/root/autodl-tmp/mmdetection-main/data/COCO/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/root/autodl-tmp/mmdetection-main/data/COCO/annotations/instances_test2017.json',
    backend_args=None,
    metric='bbox',
    outfile_prefix='./work_dirs/coco_detection/test',
    classwise=True,
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=200, type='EpochBasedTrainLoop', val_interval=10)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='/root/autodl-tmp/mmdetection-main/data/COCO/annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='/root/autodl-tmp/mmdetection-main/data/COCO/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                mean=[
                    0,
                    0,
                    0,
                ],
                ratio_range=(
                    1,
                    2,
                ),
                to_rgb=True,
                type='Expand'),
            dict(
                min_crop_size=0.3,
                min_ious=(
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ),
                type='MinIoURandomCrop'),
            dict(
                keep_ratio=True,
                scale=[
                    (
                        320,
                        320,
                    ),
                    (
                        608,
                        608,
                    ),
                ],
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(mean=[
        0,
        0,
        0,
    ], ratio_range=(
        1,
        2,
    ), to_rgb=True, type='Expand'),
    dict(
        min_crop_size=0.3,
        min_ious=(
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),
        type='MinIoURandomCrop'),
    dict(
        keep_ratio=True,
        scale=[
            (
                320,
                320,
            ),
            (
                608,
                608,
            ),
        ],
        type='RandomResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='/root/autodl-tmp/mmdetection-main/data/COCO/annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='/root/autodl-tmp/mmdetection-main/data/COCO/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                608,
                608,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/root/autodl-tmp/mmdetection-main/data/COCO/annotations/instances_val2017.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'autodl-tmp/mmdetection-main/yolov3-results'
