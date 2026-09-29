# ROG on the frozen LVIS-v1 pilot10k protocol

## Scope

`rog_faster_rcnn_r50_lvis_pilot10k_640_32ep.py` is a controlled ROG
**detection transfer baseline**, not a reproduction of ROG's full Mask R-CNN
LVIS experiment. It keeps the official ROG classification head and GAP loss
(`gamma=1.0`, `lambda=0.1`) while using Faster R-CNN and bbox-only reporting.

The frozen pilot split is:

| Split | Images | Annotations | Classes |
| --- | ---: | ---: | ---: |
| train | 10,000 | 117,605 | 1,203 |
| val | 2,000 | 20,891 | 1,203 |

All validation images are retained, including the 19 images without a positive
annotation. Input uses an aspect-ratio-preserving 640-pixel resize and
divisor-32 padding; it does not perform anisotropic stretching. The transfer
run uses batch 4, seed 0, 32 epochs, and linearly scaled SGD learning rate
0.005 (ROG's official 0.02 at batch 16).

## Train

Run from the official ROG snapshot with the isolated ROG environment:

```powershell
$py = 'D:\miniconda3\envs\rog_weld_pip_py38\python.exe'
$repo = 'D:\1\项目论文\zhwk_project\third_party\ROG_official_snapshot\ROG-main'
$cfg = 'D:\1\项目论文\zhwk_project\rog_configs\rog_faster_rcnn_r50_lvis_pilot10k_640_32ep.py'
& $py "$repo\tools\train.py" $cfg --work-dir D:\1\项目论文\zhwk_runs\rog_faster_rcnn_lvis_pilot10k_640_32ep_YYYYMMDD_seed0 --seed 0
```

## Evaluate and export predictions

Use `tools/test.py` once for standard LVIS bbox AP and once to export a
standard LVIS-format prediction JSON. The latter is required for candidate
audit, but is **not** itself an AP evaluation.

```powershell
$out = 'D:\1\项目论文\zhwk_runs\rog_faster_rcnn_lvis_pilot10k_640_32ep_YYYYMMDD_seed0'
$ckpt = "$out\epoch_32.pth"
& $py "$repo\tools\test.py" $cfg $ckpt --work-dir $out --out "$out\rog_lvis_outputs.pkl" --eval bbox
& $py "$repo\tools\test.py" $cfg $ckpt --out "$out\rog_lvis_outputs.pkl" --format-only --eval-options "jsonfile_prefix=$out\rog_lvis_predictions"
```

The expected formatted detection file is
`rog_lvis_predictions.bbox.json`.

## Thresholded candidate audit

The audit is intentionally separate from official AP. It reports a disclosed,
one-to-one, IoU-0.50 candidate outcome on the 20,891 annotated validation
instances:

- **correct**: same-category match;
- **wrong_class**: no correct match, but an overlapping different-category
  prediction;
- **miss**: neither match exists.

Same-category matches are reserved first, preventing a wrong label from
consuming an instance for which a correct detection exists. The script also
reports LVIS `r` (rare), `c` (common), and `f` (frequent) class groups.

```powershell
& $py D:\1\项目论文\zhwk_project\scripts\audit_lvis_detection_predictions.py `
  --annotations D:\1\项目论文\public_datasets\LVIS\dfine_annotations_pilot10k\lvis_v1_val_dfine_smoke.json `
  --predictions "$out\rog_lvis_predictions.bbox.json" `
  --output-dir "$out\audit_t025_iou50" `
  --method-name 'ROG transfer' --score-threshold 0.25 --match-iou 0.50
```

This audit produces `audit_summary.json`, `per_category.csv`, and
`per_instance_outcomes.json`. Its C/W/M values must not be reported as LVIS AP.
