_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco.py'
]
checkpoint_config = dict(interval=20)
evaluation = dict(interval=20, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=0.001,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[90, 120])
total_epochs = 140
channel_cfg = dict(
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

data_cfg = dict(
    image_size=512,
    base_size=256,
    base_sigma=2,
    heatmap_size=[128],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='DisentangledKeypointRegressor',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
    ),
    keypoint_head=dict(
        type='DEKRHead',
        in_channels=(32, 64, 128, 256),
        in_index=(0, 1, 2, 3),
        num_heatmap_filters=32,
        num_joints=channel_cfg['dataset_joints'],
        input_transform='resize_concat',
        heatmap_loss=dict(
            type='JointsMSELoss',
            use_target_weight=True,
            loss_weight=1.0,
        ),
        offset_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            loss_weight=0.002,
            beta=1 / 9.0,
        )),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        project2image=False,
        align_corners=False,
        max_pool_kernel=5,
        use_nms=True,
        nms_dist_thr=0.05,
        nms_joints_thr=8,
        keypoint_threshold=0.01,
        rescore_cfg=dict(
            in_channels=74,
            norm_indexes=(5, 6),
            pretrained='https://download.openmmlab.com/mmpose/'
            'pretrain_models/kpt_rescore_coco-33d58c5c.pth'),
        flip_test=True))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='GetKeypointCenterArea'),
    dict(
        type='BottomUpGenerateHeatmapTarget',
        sigma=(2, 4),
        gen_center_heatmap=True,
        bg_weight=0.1,
    ),
    dict(
        type='BottomUpGenerateOffsetTarget',
        radius=4,
    ),
    dict(
        type='Collect',
        keys=['img', 'heatmaps', 'masks', 'offsets', 'offset_weights'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index', 'num_joints', 'skeleton',
            'image_size', 'heatmap_size'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/coco'
data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=10),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
