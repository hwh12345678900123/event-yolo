_file_client_args = dict(backend='disk')
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.5
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
anchors = [
    [
        (
            68,
            69,
        ),
        (
            154,
            91,
        ),
        (
            143,
            162,
        ),
    ],
    [
        (
            242,
            160,
        ),
        (
            189,
            287,
        ),
        (
            391,
            207,
        ),
    ],
    [
        (
            353,
            337,
        ),
        (
            539,
            341,
        ),
        (
            443,
            432,
        ),
    ],
]
base_lr = 0.01
batch_shapes_cfg = dict(
    batch_size=1,
    extra_pad_ratio=0.5,
    img_size=640,
    size_divisor=32,
    type='BatchShapePolicy')
class_name = ('car', )
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]
data_root = '/home/openinnolab/work/current/mmyolo/data/event/dataset/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.33
default_hooks = dict(
    checkpoint=dict(
        interval=10, max_keep_ckpts=2, save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=120,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook',
        warmup_mim_iter=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        test_out_dir='show_results',
        type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
launcher = 'none'
load_from = 'work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/best_coco_bbox_mAP_epoch_30.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 0.05
loss_cls_weight = 0.5
loss_obj_weight = 1.0
lr_factor = 0.01
max_epochs = 120
max_keep_ckpts = 3
metainfo = dict(
    classes=('car', ), palette=[
        (
            20,
            220,
            60,
        ),
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        frozen_stages=4,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv5CSPDarknet',
        widen_factor=0.5),
    bbox_head=dict(
        head_module=dict(
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            num_base_priors=3,
            num_classes=1,
            type='YOLOv5HeadModule',
            widen_factor=0.5),
        loss_bbox=dict(
            bbox_format='xywh',
            eps=1e-07,
            iou_mode='ciou',
            loss_weight=0.05,
            reduction='mean',
            return_iou=True,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        obj_level_weights=[
            4.0,
            1.0,
            0.4,
        ],
        prior_generator=dict(
            base_sizes=[
                [
                    (
                        68,
                        69,
                    ),
                    (
                        154,
                        91,
                    ),
                    (
                        143,
                        162,
                    ),
                ],
                [
                    (
                        242,
                        160,
                    ),
                    (
                        189,
                        287,
                    ),
                    (
                        391,
                        207,
                    ),
                ],
                [
                    (
                        353,
                        337,
                    ),
                    (
                        539,
                        341,
                    ),
                    (
                        443,
                        432,
                    ),
                ],
            ],
            strides=[
                8,
                16,
                32,
            ],
            type='mmdet.YOLOAnchorGenerator'),
        prior_match_thr=4.0,
        type='YOLOv5Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv5PAFPN',
        widen_factor=0.5),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.65, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 1
num_det_layers = 3
obj_level_weights = [
    4.0,
    1.0,
    0.4,
]
optim_wrapper = dict(
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=12,
        lr=0.01,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = None
persistent_workers = True
pre_transform = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
prior_match_thr = 4.0
resume = False
save_checkpoint_intervals = 10
strides = [
    8,
    16,
    32,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test.json',
        batch_shapes_cfg=dict(
            batch_size=1,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='images/'),
        data_root='/home/openinnolab/work/current/mmyolo/data/event/dataset/',
        metainfo=dict(classes=('car', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/openinnolab/work/current/mmyolo/data/event/dataset/annotations/test.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 12
train_cfg = dict(max_epochs=120, type='EpochBasedTrainLoop', val_interval=10)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=12,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/'),
        data_root='/home/openinnolab/work/current/mmyolo/data/event/dataset/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=('car', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(
                        file_client_args=dict(backend='disk'),
                        type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='Mosaic'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 4
train_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test.json',
        batch_shapes_cfg=dict(
            batch_size=1,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='images/'),
        data_root='/home/openinnolab/work/current/mmyolo/data/event/dataset/',
        metainfo=dict(classes=('car', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/openinnolab/work/current/mmyolo/data/event/dataset/annotations/test.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.0005
widen_factor = 0.5
work_dir = './work_dirs/yolov5_s-v61_fast_1xb12-40e_cat'
