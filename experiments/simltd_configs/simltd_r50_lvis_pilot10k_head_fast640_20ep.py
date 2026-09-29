_base_ = r"D:/1/项目论文/zhwk_project/simltd_pilot_configs/simltd_r50_lvis_pilot10k_head.py"

# Pilot-fast protocol: fixed 640-pixel letterbox, box-only annotations, and
# 20 data-pass equivalents. The detector consumes boxes only; CopyPaste is
# removed because it requires dense instance masks and caused host-memory OOM
# on the pilot's high-instance images.
bbox_only_load_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", prob=0.5),
]
fast_train_pipeline = [
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type="RandAugment", aug_space=_base_.color_space, aug_num=1),
    dict(type="PackDetInputs"),
]
fast_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(640, 640), keep_ratio=True),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(pipeline=fast_train_pipeline,
                 dataset=dict(dataset=dict(pipeline=bbox_only_load_pipeline))))
val_dataloader = dict(batch_size=1, num_workers=2,
                      dataset=dict(pipeline=fast_test_pipeline))
test_dataloader = val_dataloader

# 10k pilot images / physical batch 2 * 20 passes = 100k iterations.
# Gradient accumulation preserves the effective batch of four.
train_cfg = dict(max_iters=100000, val_interval=10000)
optim_wrapper = dict(accumulative_counts=2)
param_scheduler = [dict(type="MultiStepLR", begin=0, end=100000,
                        by_epoch=False, milestones=[88889], gamma=0.1)]
default_hooks = dict(checkpoint=dict(interval=10000, max_keep_ckpts=2,
                                    by_epoch=False, save_optimizer=False,
                                    save_param_scheduler=False))
randomness = dict(seed=0)
work_dir = r"D:/1/项目论文/zhwk_runs/simltd_official_lvis_pilot_20260924/fast640_32ep/stage1_head866_20ep"

