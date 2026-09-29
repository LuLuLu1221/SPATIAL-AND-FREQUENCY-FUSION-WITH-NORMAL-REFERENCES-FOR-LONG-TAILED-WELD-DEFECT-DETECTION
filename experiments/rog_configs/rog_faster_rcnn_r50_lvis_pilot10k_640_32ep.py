"""ROG transfer baseline on the frozen LVIS-v1 pilot10k protocol.

This is a controlled *detection* transfer of ROG: its official
ROGShared2FCBBoxHead and GAP ranking loss are retained, while Faster R-CNN is
used rather than the paper's Mask R-CNN because this study reports bbox-only
metrics.  It is not an official full-LVIS reproduction.

Protocol alignment:
  * same frozen pilot annotations as the D-FINE-S LVIS study (10k/2k images);
  * geometry-preserving, aspect-ratio-preserving 640-pixel resize;
  * batch 4, seed 0, and a 32-epoch transfer budget;
  * the native ROG low score floor is preserved; thresholded candidate audits
    are performed separately by audit_lvis_detection_predictions.py.
"""

_base_ = [
    r'D:\1\项目论文\zhwk_project\third_party\ROG_official_snapshot\ROG-main\configs\_base_\models\faster_rcnn_r50_fpn.py',
    r'D:\1\项目论文\zhwk_project\third_party\ROG_official_snapshot\ROG-main\configs\_base_\default_runtime.py',
]

model = dict(
    # Local ImageNet weights avoid a network-dependent initialization download.
    backbone=dict(init_cfg=dict(
        type='Pretrained',
        checkpoint=r'C:\Users\28137\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth')),
    roi_head=dict(
        bbox_head=dict(
            type='ROGShared2FCBBoxHead',
            num_classes=1203,
            gamma=1.0,
            lam=0.1,
            loss_cls=dict(
                type='SigmoidCrossEntropyLoss', num_classes=1203,
                loss_weight=1.0))),
    # Keep candidates before the explicitly reported audit threshold is applied.
    test_cfg=dict(rcnn=dict(score_thr=0.0001, max_per_img=300)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ResizeMaxPad in the D-FINE protocol is represented here by an equivalent
# keep-ratio resize followed by divisor-32 padding: no anisotropic stretching.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
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
        img_scale=(640, 640),
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

data_root = r'D:\1\项目论文\public_datasets\LVIS'
pilot_root = data_root + r'\dfine_annotations_pilot10k'
image_root = data_root + r'\images'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type='LVISV1Dataset',
            ann_file=pilot_root + r'\lvis_v1_train_dfine_smoke.json',
            img_prefix=image_root,
            pipeline=train_pipeline)),
    val=dict(
        type='LVISV1Dataset',
        ann_file=pilot_root + r'\lvis_v1_val_dfine_smoke.json',
        img_prefix=image_root,
        # Retain all 2,000 pilot validation images, including empty images.
        filter_empty_gt=False,
        pipeline=test_pipeline),
    test=dict(
        type='LVISV1Dataset',
        ann_file=pilot_root + r'\lvis_v1_val_dfine_smoke.json',
        img_prefix=image_root,
        # Retain all 2,000 pilot validation images, including empty images.
        filter_empty_gt=False,
        pipeline=test_pipeline))

# ROG's official 1x schedule is lr=0.02 at global batch 16.  The transfer
# uses the linearly scaled lr=0.005 at batch 4 under a fixed 32-epoch budget.
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=500,
                 warmup_ratio=0.0001, step=[21, 29])
runner = dict(type='EpochBasedRunner', max_epochs=32)
evaluation = dict(interval=4, metric='bbox', classwise=False)
checkpoint_config = dict(interval=4, max_keep_ckpts=3)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
seed = 0
deterministic = False
workflow = [('train', 1)]
work_dir = r'D:\1\项目论文\zhwk_runs\rog_faster_rcnn_lvis_pilot10k_640_32ep_20260925_seed0'
