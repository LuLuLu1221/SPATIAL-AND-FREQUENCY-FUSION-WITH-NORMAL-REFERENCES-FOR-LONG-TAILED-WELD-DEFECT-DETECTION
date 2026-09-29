_base_ = r"D:/1/项目论文/zhwk_project/third_party/simltd_official_snapshot/simltd-main/configs/simltd/deformable-detr-resnet/deformable-detr-refine-twostage_r50_lvis_v1_head866.py"

# Pilot-specific paths.  The architecture and stage-1 schedule remain from
# SimLTD's released R50 Deformable-DETR configuration.
data_root = r"D:/1/项目论文/public_datasets/LVIS/images/"
train_ann = r"D:/1/项目论文/public_datasets/LVIS/dfine_annotations_pilot10k/lvis_v1_train_dfine_smoke.json"
val_ann = r"D:/1/项目论文/public_datasets/LVIS/dfine_annotations_pilot10k/lvis_v1_val_dfine_smoke.json"
classes_file = r"D:/1/项目论文/zhwk_runs/simltd_official_lvis_pilot_20260924/data/lvis_v1_head_classes.txt"
METAINFO = dict(classes=classes_file)

labeled_dataset = _base_.labeled_dataset
labeled_dataset.dataset.dataset.data_root = data_root
labeled_dataset.dataset.dataset.ann_file = train_ann
labeled_dataset.dataset.dataset.metainfo = METAINFO

train_dataloader = dict(batch_size=4, num_workers=2, dataset=labeled_dataset)
val_dataloader = dict(batch_size=2, num_workers=2,
                      dataset=dict(data_root=data_root, ann_file=val_ann, metainfo=METAINFO))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=val_ann, metric="bbox")
test_evaluator = val_evaluator
work_dir = r"D:/1/项目论文/zhwk_runs/simltd_official_lvis_pilot_20260924/stage1_head866"
