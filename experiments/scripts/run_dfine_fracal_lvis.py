"""FRACAL sigmoid-head calibration on a frozen D-FINE-S LVIS pilot model.

The runner keeps the 640-pixel no-stretch D-FINE-S checkpoint fixed.  It
recomputes the full query-by-class scores using the public FRACAL equation and
evaluates raw and calibrated rankings with the same LVIS API.  To avoid a
candidate-volume confound, the calibrated output retains exactly the number of
candidates selected by raw D-FINE-S (score >= raw-threshold) for each image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_ibo_lvis_nostretch_from_dfine_candidates import build_model_no_stretch
from run_dfine_fracal_weld import compute_fracal_adjustments


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dfine-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--train-annotations", type=Path, required=True)
    p.add_argument("--val-annotations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--raw-threshold", type=float, default=0.25)
    p.add_argument("--top-k", type=int, default=300)
    p.add_argument("--fractal-exponent", type=float, default=2.0)
    p.add_argument("--grid-max", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def xyxy_to_xywh(box: torch.Tensor) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    return [round(x1, 4), round(y1, 4), round(max(0.0, x2 - x1), 4), round(max(0.0, y2 - y1), 4)]
def topk_predictions_no_stretch(logits: torch.Tensor, boxes: torch.Tensor, sizes: torch.Tensor, top_k: int, adjustment: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select query/class scores and decode through ResizeMaxPad geometry."""
    if adjustment is None:
        scores_per_class = torch.sigmoid(logits)
    else:
        calibration = adjustment.to(logits.device).view(1, 1, -1)
        scores_per_class = torch.sigmoid(logits) * torch.sigmoid(logits + calibration)
    xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    widths, heights = sizes[:, 0], sizes[:, 1]
    long_side = torch.maximum(widths, heights)
    xyxy = xyxy * long_side[:, None, None]
    xyxy[..., 0::2] = xyxy[..., 0::2].clamp(min=0.0)
    xyxy[..., 1::2] = xyxy[..., 1::2].clamp(min=0.0)
    xyxy[..., 0::2] = torch.minimum(xyxy[..., 0::2], widths[:, None, None])
    xyxy[..., 1::2] = torch.minimum(xyxy[..., 1::2], heights[:, None, None])
    scores, flat_index = torch.topk(scores_per_class.flatten(1), min(top_k, scores_per_class.shape[1] * scores_per_class.shape[2]), dim=-1)
    class_count = scores_per_class.shape[-1]
    labels = flat_index.remainder(class_count)
    query_index = flat_index.div(class_count, rounding_mode="floor")
    selected_boxes = xyxy.gather(1, query_index.unsqueeze(-1).repeat(1, 1, 4))
    return labels, selected_boxes, scores


def to_original_xyxy(boxes: torch.Tensor, sizes: torch.Tensor, eval_size: int) -> torch.Tensor:
    widths, heights = sizes[:, 0], sizes[:, 1]
    scales = float(eval_size) / torch.maximum(widths, heights)
    valid_w, valid_h = widths * scales, heights * scales
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=0.0)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=0.0)
    boxes[..., 0::2] = torch.minimum(boxes[..., 0::2], valid_w[:, None, None])
    boxes[..., 1::2] = torch.minimum(boxes[..., 1::2], valid_h[:, None, None])
    boxes /= scales[:, None, None]
    boxes[..., 0::2] = torch.minimum(boxes[..., 0::2], widths[:, None, None])
    boxes[..., 1::2] = torch.minimum(boxes[..., 1::2], heights[:, None, None])
    return boxes


def eval_lvis(annotations: Path, predictions: Path) -> dict[str, float]:
    # LVIS 0.5.3 still refers to the removed NumPy alias on recent NumPy.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    from lvis import LVIS, LVISEval, LVISResults

    gt = LVIS(str(annotations))
    results = LVISResults(gt, str(predictions))
    evaluator = LVISEval(gt, results, "bbox")
    evaluator.run()
    keys = ("AP", "AP50", "AP75", "APr", "APc", "APf", "AR@300")
    return {key: float(evaluator.results[key]) for key in keys}


def main() -> None:
    args = args_parser()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    train, val = load_json(args.train_annotations), load_json(args.val_annotations)
    classes = [int(row["id"]) for row in train["categories"]]
    adjustment, stats = compute_fracal_adjustments(train, classes, args.fractal_exponent, args.grid_max)
    spec = SimpleNamespace(config=args.config, checkpoint=args.checkpoint, name="FRACAL-LVIS")
    wrapped, transform, spatial = build_model_no_stretch(spec, args.dfine_root, SCRIPT_DIR, device)
    eval_size = int(spatial[0])
    raw_predictions, calibrated_predictions = [], []
    records = sorted(val["images"], key=lambda row: int(row["id"]))
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            images, sizes = [], []
            for row in batch:
                path = args.image_root / str(row["file_name"])
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    sizes.append([float(image.width), float(image.height)])
                    images.append(transform(image))
            size_tensor = torch.tensor(sizes, dtype=torch.float32, device=device)
            outputs = wrapped.model(torch.stack(images).to(device))
            raw_labels, raw_boxes, raw_scores = topk_predictions_no_stretch(outputs["pred_logits"], outputs["pred_boxes"], size_tensor, args.top_k, None)
            cal_labels, cal_boxes, cal_scores = topk_predictions_no_stretch(outputs["pred_logits"], outputs["pred_boxes"], size_tensor, args.top_k, adjustment)
            # topk_predictions already scales normalized boxes to original image coordinates.
            raw_boxes = raw_boxes.cpu()
            cal_boxes = cal_boxes.cpu()
            for index, row in enumerate(batch):
                # Keep GPU indices for logits/scores and CPU indices for projected boxes.
                keep_gpu = raw_scores[index] >= args.raw_threshold
                keep_cpu = keep_gpu.cpu()
                count = int(keep_gpu.sum().item())
                for label, box, score in zip(raw_labels[index][keep_gpu].tolist(), raw_boxes[index][keep_cpu], raw_scores[index][keep_gpu].tolist()):
                    raw_predictions.append({"image_id": int(row["id"]), "category_id": int(label), "bbox": xyxy_to_xywh(box), "score": round(float(score), 7)})
                # Exact per-image candidate budget matching: only the top count
                # calibrated query/class predictions are retained.
                for label, box, score in zip(cal_labels[index][:count].tolist(), cal_boxes[index][:count], cal_scores[index][:count].tolist()):
                    calibrated_predictions.append({"image_id": int(row["id"]), "category_id": int(label), "bbox": xyxy_to_xywh(box), "score": round(float(score), 7)})
            if (start + len(batch)) % 100 == 0 or start + len(batch) == len(records):
                print(f"inferred {start + len(batch)}/{len(records)} LVIS validation images", flush=True)
    raw_path = args.output_dir / "dfine_raw_budgetmatched_predictions.json"
    cal_path = args.output_dir / "fracal_budgetmatched_predictions.json"
    raw_path.write_text(json.dumps(raw_predictions), encoding="utf-8")
    cal_path.write_text(json.dumps(calibrated_predictions), encoding="utf-8")
    (args.output_dir / "fracal_train_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "D-FINE-S raw": {"predictions": len(raw_predictions), "metrics": eval_lvis(args.val_annotations, raw_path)},
        "D-FINE-S + FRACAL (budget-matched)": {"predictions": len(calibrated_predictions), "metrics": eval_lvis(args.val_annotations, cal_path)},
    }
    metadata = vars(args) | {"eval_spatial_size": spatial, "formula": "sigmoid(z) * sigmoid(z + frequency_adjustment + fractal_adjustment)", "selection": "FRACAL retains raw D-FINE-S candidate count per image; no detector weights are updated."}
    (args.output_dir / "official_lvis_ap_summary.json").write_text(json.dumps({"settings": metrics, "metadata": {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
