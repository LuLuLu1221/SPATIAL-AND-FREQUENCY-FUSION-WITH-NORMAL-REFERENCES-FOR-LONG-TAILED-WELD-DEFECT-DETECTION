# Long-tailed weld-defect detection experiments

This directory contains the experiment code and configuration files for a two-stage long-tailed weld-defect detection study.  The first stage uses D-FINE to produce high-recall candidates from unwrapped circular weld beads.  The second stage constructs instance-based objects (IBOs), aggregates cyclic-window candidates, and uses normal-reference and frequency-domain evidence to calibrate reliability and correct confusable classes.

The repository deliberately contains **code, configuration, and documentation only**.  The industrial images, annotations, trained checkpoints, intermediate features, and run outputs are not distributed.

## Experiment setting

- Input representation: annular weld bead unwrapped into a strip.
- Tiling: cyclic windows of 1024 x 150 pixels, stride 768 pixels (256-pixel overlap).
- Detector input: tiles are padded to 1024 x 160 rather than radially resized.
- Detector: D-FINE with the HGNetv2-B0 backbone.
- Objective: retain high-recall defect candidates, then reduce long-tail and visually confusable-class errors through IBO evidence fusion and second-stage correction.

The detailed method, evaluation protocol, current results, and limitations are in [docs/EXPERIMENT_OVERVIEW.md](docs/EXPERIMENT_OVERVIEW.md).  The public-dataset comparison plan is in [docs/公开数据集_LoHi-WELD_对比实验方案_v1_20260912.md](docs/公开数据集_LoHi-WELD_对比实验方案_v1_20260912.md).

## Repository layout

```text
.
├── dfine_configs/       # D-FINE training and dataset configurations
├── docs/                # Experiment reports and public-dataset plan
├── scripts/             # Data preparation, IBO, evaluation, and ablations
├── requirements.txt     # Python dependencies used by the auxiliary scripts
└── 文件清单与作用.md       # Chinese file-by-file index
```

## Setup

1. Create a Python environment (the experiments were developed with Python 3.11).
2. Install the auxiliary-script dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Clone and install [D-FINE](https://github.com/Peterande/D-FINE), then update the `__include__` paths in the training YAML files if your checkout layout differs from this project.
4. Prepare your data in COCO format.  The scripts expect separate train/validation annotations and image roots; no proprietary data paths are included here.

## Main workflow

1. Use `scripts/build_unwrapped_tiled_coco_dataset.py` to turn unwrapped weld strips into 150-pixel cyclic tiles and COCO annotations.
2. Train D-FINE with `dfine_configs/dfine_hgnetv2_s_zhwk_unwrapped_150px_cyclic_tiles.yml`.
3. Export detector candidates and create IBO records with `scripts/build_ibo_from_dfine_candidates.py`.
4. Merge duplicated candidates across overlapping cyclic windows with `scripts/evaluate_ibo_cyclic_candidate_merge.py`.
5. Train normal-reference/frequency evidence fusion with `scripts/train_ibo_group_evidence_fusion_v2.py`, then evaluate with `scripts/evaluate_ibo_reliability_postprocess.py`.
6. Run the second-stage and ablation scripts as needed for targeted confusable-class analysis.

Run every script with `--help` before use to see its required input paths.  Some scripts preserve the original experiment defaults, which point to the author's local run directory; supply explicit command-line paths when reproducing an experiment elsewhere.

## Reproducibility notes

- The reported 150-pixel results are from an internal industrial dataset and should not be interpreted as a public benchmark.
- The original study uses a fixed split and seed for the reported experiment.  Use repeated runs or cross-validation when comparing methods on a small dataset.
- The LoHi-WELD plan is a proposed public-data evaluation protocol; it is not presented as completed benchmark results.

## Citation

If this code supports your work, please cite this repository.  A paper citation will be added when available.
