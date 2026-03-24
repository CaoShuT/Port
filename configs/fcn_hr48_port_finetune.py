_base_ = './fcn_hr48_loveda.py'

# 港口数据集类别数
PORT_NUM_CLASSES = 7

# 从 LoveDA 预训练权重加载
load_from = 'work_dirs/fcn_hr48_loveda/iter_80000.pth'

# 港口数据集配置
# 默认类别与 LoveDA 一致：background, building, road, water, barren, forest, agriculture
# 如需自定义港口类别（例如 water, road, building, container, crane, ship），
# 请修改下方 metainfo 中的 classes 和 palette 字段。
PORT_CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
                'agriculture')
PORT_PALETTE = [[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                [159, 129, 183], [0, 255, 0], [255, 195, 128]]

# 自定义港口类别示例（取消注释并修改即可）：
# PORT_CLASSES = ('water', 'road', 'building', 'container', 'crane', 'ship')
# PORT_PALETTE = [[0,0,255],[255,255,0],[255,0,0],[0,255,255],[255,0,255],[128,128,0]]

# 港口训练 pipeline（增加垂直翻转增强）
port_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),  # 港口数据标注不需要 reduce_zero_label
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

port_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='CustomDataset',
        data_root='data/port',
        metainfo=dict(classes=PORT_CLASSES, palette=PORT_PALETTE),
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=port_train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomDataset',
        data_root='data/port',
        metainfo=dict(classes=PORT_CLASSES, palette=PORT_PALETTE),
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=port_test_pipeline))

test_dataloader = val_dataloader

# 评估指标：mIoU 和 mFscore
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

# 微调学习率策略
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-5,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False)
]

# 微调训练轮次
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
