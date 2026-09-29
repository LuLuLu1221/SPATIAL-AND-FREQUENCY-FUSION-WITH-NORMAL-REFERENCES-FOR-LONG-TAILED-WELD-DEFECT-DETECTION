"""ROG transfer baseline on the in-house periodic weld-strip detection task.

This config preserves the official ROG classification head and GAP ranking
loss, while replacing Mask R-CNN with Faster R-CNN because the weld corpus
contains bounding boxes but no instance masks.  It is therefore a controlled
transfer baseline, not an official LVIS reproduction.
"""

_base_ = [
    r'D:\1\项目论文\zhwk_project\third_party\ROG_official_snapshot\ROG-main\configs\_base_\models\faster_rcnn_r50_fpn.py',
    r'D:\1\项目论文\zhwk_project\third_party\ROG_official_snapshot\ROG-main\configs\_base_\default_runtime.py',
]

# Keep native category IDs 0--22 for an exact post-hoc source-level audit.
# The audit subsequently removes non-study labels and merges the two agreed
# long-tail families; no label remapping is performed during ROG training.
classes = (
    '\u5c0f\u710a\u70df', '\u710a\u70b8', '\u710a\u70df\u56e2', '\u710a\u70df',
    '\u710a\u6e23', '\u957f\u710a\u9ad8\u88c2', '\u70b9\u710a\u9ad8', '\u710a\u6d1e',
    '\u710a\u5751', '\u710a\u6d1e\u957f', '\u957f\u710a\u9ad8', '\u94a2\u5e3d',
    '\u65ad\u710a', '\u710a\u9ad8\u7eb9', '\u957f\u710a\u9ad8\u584c',
    '\u7f3a\u710a\u84dd\u9ed1', '\u710a\u9ad8\u70df', '\u7f3a\u710a\u88c2',
    '\u710a\u7f1d\u84dd\u9ed1', '\u710a\u76d8\u79bb', '\u710a\u7070\u8272',
    '\u710a\u9ad8\u7f1d', '\u8131\u710a',
)

model = dict(
    # A local cache avoids a network-dependent ImageNet initialization download.
    backbone=dict(init_cfg=dict(
        type='Pretrained',
        checkpoint=r'C:\Users\28137\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth')),
    roi_head=dict(
        bbox_head=dict(
            type='ROGShared2FCBBoxHead',
            num_classes=23,
            gamma=1.0,
            lam=0.1,
            loss_cls=dict(
                type='SigmoidCrossEntropyLoss', num_classes=23,
                loss_weight=1.0),
        )),
    # Keep the low native ROG score floor so the fixed t=0.25 audit can be
    # performed offline without losing candidates before evaluation.
    test_cfg=dict(rcnn=dict(score_thr=0.0001, max_per_img=300)),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Geometry-preserving input: a periodic seam tile stays 1024 x 160, followed
# only by the standard divisor-32 padding (which is a no-op for this size).
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 160), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 160),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data_root = r'D:\1\项目论文\zhwk_unwrapped_150px_cyclic_tiles_padded160_v1'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type='CocoDataset',
            classes=classes,
            ann_file=data_root + r'\annotations\instances_train_diverse_neg0p5_ascii.json',
            img_prefix=data_root + r'\images\train',
            pipeline=train_pipeline)),
    val=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=data_root + r'\annotations\instances_val.json',
        img_prefix=data_root + r'\images\val',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=data_root + r'\annotations\instances_val.json',
        img_prefix=data_root + r'\images\val',
        pipeline=test_pipeline),
)

evaluation = dict(interval=4, metric='bbox', classwise=True)
checkpoint_config = dict(interval=4, max_keep_ckpts=3)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# ROG's official 1x schedule uses SGD lr=0.02 for global batch 16.  This
# single-GPU batch-4 transfer uses the linear-scale lr=0.005 and 32 epochs to
# match the fixed epoch budget of the in-house detector comparison.
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=500,
    warmup_ratio=0.0001, step=[21, 29])
runner = dict(type='EpochBasedRunner', max_epochs=32)

seed = 0
deterministic = False
workflow = [('train', 1)]
work_dir = r'D:\1\项目论文\zhwk_runs\rog_faster_rcnn_zhwk_150px_32ep_20260924_seed0'
