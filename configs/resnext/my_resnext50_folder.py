# D:\ECE 549 Final\mmpretrain\configs\resnext\my_resnext50_folder.py

default_scope = 'mmpretrain'

# ===== 模型 =====
num_classes = 120
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        groups=32,
        width_per_group=4,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)

# ===== 数据预处理 =====
data_root = r'D:\ECE 549 Final\Archive'
data_preprocessor = dict(
    num_classes=num_classes,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# ===== 训练策略 =====
# 说明：原 base_batch_size=256；本配置单卡 batch_size=32 => LR 缩放到 0.1 * (32/256) = 0.0125
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=1e-4),
    loss_scale='dynamic'
)

# 先线性warmup 1个epoch，再余弦退火到极低
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1),  # 1 epoch warm-up 0.0125 * 0.1 开始，epoch1后结束
    dict(type='CosineAnnealingLR', T_max=39, by_epoch=True, begin=1, end=40, eta_min=1e-5)
]
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)

val_cfg = dict()
test_cfg = dict()

# 仅做记录，不启用自动缩放
auto_scale_lr = dict(enable=False, base_batch_size=256)

# ===== 数据管线 =====
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

# 使用文件夹结构的数据读取（按子目录名建类表）
dataset_type = 'CustomDataset'

# ===== DataLoader（GPU 友好参数）=====
# Windows 下可用 num_workers=4；若内存足够可再升到 6/8
# persistent_workers 只有在 num_workers>0 时可设为 True
_common_loader_kwargs = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
)

train_dataloader = dict(
    **_common_loader_kwargs,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train',  # 读取 data_root/train/* 作为类别
        with_label=True,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    **_common_loader_kwargs,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='val',
        with_label=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    **_common_loader_kwargs,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='test',  # 如无 test，可改为 'val'
        with_label=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = val_evaluator

# ===== 运行时配置 =====
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best = 'accuracy/top1'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

# Windows 单机训练 backend 用 gloo（NCCL 在 Windows 不可用）
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

log_level = 'INFO'

# =====（可选）加载 ImageNet 预训练权重以加速收敛 =====
# 若不想加载预训练，设为 None
load_from = 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'

resume = False

# 可被命令行 --work-dir 覆盖
work_dir = r'.\work_dirs\resnext50_mydata'
