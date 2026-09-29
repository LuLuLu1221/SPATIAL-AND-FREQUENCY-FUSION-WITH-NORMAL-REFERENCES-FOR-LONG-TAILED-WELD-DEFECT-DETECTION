_base_ = r"D:/1/项目论文/zhwk_project/simltd_pilot_configs/simltd_r50_lvis_pilot10k_head.py"

# The released SimLTD LVIS recipe requests segmentation masks even though this
# Deformable-DETR stage supervises boxes only.  The pilot annotations contain
# very dense images, for which materializing every mask exhausts host RAM.
# Preserve every detection annotation and augmentation; remove only the unused
# mask decoding from the inner labelled dataset pipeline.
bbox_only_load_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", prob=0.5),
]

# Match the inherited MultiImageMixDataset -> ClassBalancedDataset ->
# LVISV1Dataset nesting. Only this inner pipeline is replaced.
train_dataloader = dict(
    dataset=dict(dataset=dict(dataset=dict(pipeline=bbox_only_load_pipeline))))

