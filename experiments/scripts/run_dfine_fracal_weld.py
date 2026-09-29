#!/usr/bin/env python
"""Evaluate exact FRACAL-style logit calibration on a frozen D-FINE weld detector.

This runner never updates detector weights.  It recomputes the raw D-FINE
top-k selection and the FRACAL selection from the same query logits, then
audits both results using the source-instance rules used by the existing
strict D-FINE evaluator.  The implementation follows the public FRACAL
sigmoid-head equation: sigmoid(z) * sigmoid(z + w_freq + w_fractal).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_dfine_strict_detection_errors import (  # noqa: E402
    analyse_native_input,
    load_json,
    source_annotation_id,
    source_image_id,
    source_outcome,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dfine-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--train-annotations", type=Path, required=True)
    p.add_argument("--native-val-annotations", type=Path, required=True)
    p.add_argument("--native-extra-annotations", type=Path)
    p.add_argument("--native-extra-image-root", type=Path)
    p.add_argument("--source-annotations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--fracal-thresholds", default="")
    p.add_argument("--grouped-protocol", action="store_true", help="Use the manuscript 13-group protocol.")
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--partial-coverage", type=float, default=0.20)
    p.add_argument("--fragmented-coverage", type=float, default=0.50)
    p.add_argument("--top-k", type=int, default=300)
    p.add_argument("--fractal-exponent", type=float, default=2.0)
    p.add_argument("--grid-max", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_raw_model(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(args.dfine_root))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config), resume=str(args.checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    spatial = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    if not isinstance(spatial, (list, tuple)) or len(spatial) != 2:
        raise ValueError(f"Unexpected eval_spatial_size: {spatial!r}")
    transform = T.Compose([T.Resize(tuple(int(v) for v in spatial)), T.ToTensor()])
    return cfg.model.deploy().to(device).eval(), transform, tuple(int(v) for v in spatial)


def compute_fracal_adjustments(train: dict[str, Any], classes: list[int], exponent: float, grid_max: int):
    images = {int(x["id"]): x for x in train["images"]}
    n_images = len(images)
    centres: dict[int, list[tuple[float, float]]] = {c: [] for c in classes}
    instance_count = Counter()
    image_classes: dict[int, set[int]] = defaultdict(set)
    for ann in train["annotations"]:
        c = int(ann["category_id"])
        if c not in centres:
            continue
        image = images[int(ann["image_id"])]
        x, y, w, h = (float(v) for v in ann["bbox"])
        centres[c].append(((x + w * 0.5) / float(image["width"]), (y + h * 0.5) / float(image["height"])))
        instance_count[c] += 1
        image_classes[int(ann["image_id"])].add(c)
    doc_count = Counter(c for values in image_classes.values() for c in values)
    dimensions = np.arange(1, grid_max + 1, dtype=np.float64)
    fractal: dict[int, float] = {}
    for c in classes:
        points = np.asarray(centres[c], dtype=np.float64)
        valid_dims, occupied = [], []
        if len(points) >= 4:
            for d in dimensions:
                if instance_count[c] < d * d:
                    continue
                index = np.floor(np.clip(points, 0.0, 1.0 - 1e-9) * d).astype(np.int64)
                occupied_boxes = len({(int(x), int(y)) for x, y in index})
                if occupied_boxes > 0:
                    valid_dims.append(float(d))
                    occupied.append(float(occupied_boxes))
        if len(valid_dims) >= 2:
            slope = float(np.polyfit(np.log(valid_dims), np.log(occupied), 1)[0])
            fractal[c] = max(1.0, slope)
        else:
            fractal[c] = 1.0
    c_count = len(classes)
    fractal_vec = np.asarray([fractal[c] ** exponent for c in classes], dtype=np.float64)
    fractal_prob = -np.log10(fractal_vec / fractal_vec.sum()) + math.log10(1.0 / c_count)
    frequency = np.asarray([max(1, doc_count[c]) for c in classes], dtype=np.float64)
    frequency_weight = -np.log10(frequency / float(n_images)) - math.log10(float(c_count))
    adjustment = frequency_weight + fractal_prob
    rows = [{
        "class_id": c,
        "class_name": next(x["name"] for x in train["categories"] if int(x["id"]) == c),
        "train_instances": int(instance_count[c]),
        "train_images": int(doc_count[c]),
        "fractal_dimension": round(fractal[c], 8),
        "frequency_adjustment": round(float(frequency_weight[i]), 8),
        "fractal_adjustment": round(float(fractal_prob[i]), 8),
        "total_logit_adjustment": round(float(adjustment[i]), 8),
    } for i, c in enumerate(classes)]
    return torch.tensor(adjustment, dtype=torch.float32), rows


def topk_predictions(logits: torch.Tensor, boxes: torch.Tensor, sizes: torch.Tensor, top_k: int, adjustment: torch.Tensor | None):
    if adjustment is None:
        scores_per_class = torch.sigmoid(logits)
    else:
        calibration = adjustment.to(logits.device).view(1, 1, -1)
        scores_per_class = torch.sigmoid(logits) * torch.sigmoid(logits + calibration)
    xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    xyxy = xyxy * sizes.repeat(1, 2).unsqueeze(1)
    scores, index = torch.topk(scores_per_class.flatten(1), min(top_k, scores_per_class.shape[1] * scores_per_class.shape[2]), dim=-1)
    class_count = scores_per_class.shape[-1]
    labels = index.remainder(class_count)
    query_index = index.div(class_count, rounding_mode="floor")
    selected_boxes = xyxy.gather(1, query_index.unsqueeze(-1).repeat(1, 1, 4))
    return labels, selected_boxes, scores


GROUP_MAP = {0: 0, 2: 0, 3: 0, 12: 12, 14: 12, 15: 12, 18: 12}
EXCLUDED_GROUP_PROTOCOL = {11, 17, 19, 21, 22}
GROUP_NAMES = {0: "烟类合并", 12: "缺焊长塌合并"}


def canonical_label(label: int, grouped: bool) -> int:
    if not grouped:
        return int(label)
    label = int(label)
    if label in EXCLUDED_GROUP_PROTOCOL:
        return -1
    return GROUP_MAP.get(label, label)


def canonicalize_dataset(data: dict[str, Any], grouped: bool) -> dict[str, Any]:
    """Project labels into the final 13-category manuscript protocol."""
    if not grouped:
        return data
    out = dict(data)
    category_names = {int(row["id"]): str(row["name"]) for row in data["categories"]}
    active_ids = sorted({canonical_label(int(row["category_id"]), True) for row in data["annotations"]
                         if canonical_label(int(row["category_id"]), True) >= 0})
    out["categories"] = [
        {"id": cid, "name": GROUP_NAMES.get(cid, category_names.get(cid, f"class_{cid}"))}
        for cid in active_ids
    ]
    out["annotations"] = [
        dict(row, category_id=canonical_label(int(row["category_id"]), True))
        for row in data["annotations"] if canonical_label(int(row["category_id"]), True) >= 0
    ]
    return out


def combine_native(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    if extra is None:
        return base
    base_ids = {int(row["id"]) for row in base["images"]}
    if base_ids.intersection(int(row["id"]) for row in extra["images"]):
        raise ValueError("Native validation and holdout tile image IDs overlap")
    return {
        "info": base.get("info", {}),
        "licenses": base.get("licenses", []),
        "categories": base["categories"],
        "images": list(base["images"]) + list(extra["images"]),
        "annotations": list(base["annotations"]) + list(extra["annotations"]),
    }

def empty_source_details(source: dict[str, Any]):
    categories = {int(x["id"]): str(x["name"]) for x in source["categories"]}
    details = {}
    for ann in source["annotations"]:
        aid = source_annotation_id(ann)
        details[aid] = {
            "source_annotation_id": aid,
            "source_image_id": source_image_id(next(x for x in source["images"] if int(x["id"]) == int(ann["image_id"]))),
            "class_id": int(ann["category_id"]),
            "class_name": categories[int(ann["category_id"])],
            "represented": False,
            "native_occurrences": 0,
            "matched": False,
            "max_same_class_coverage": 0.0,
            "max_same_class_prediction_count": 0,
            "wrong_class_categories": set(),
        }
    return details, categories


def audit_method(name: str, batches, native: dict[str, Any], source: dict[str, Any], threshold: float, match_iou: float, partial: float, fragmented: float):
    source_details, categories = empty_source_details(source)
    native_categories = {int(x["id"]): str(x["name"]) for x in native["categories"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in native["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
        aid = source_annotation_id(ann)
        if aid in source_details:
            source_details[aid]["represented"] = True
            source_details[aid]["native_occurrences"] += 1
    per_prediction, prediction_counts = [], Counter()
    for record, labels, boxes, scores in batches:
        image_id = int(record["id"])
        source_id = source_image_id(record)
        width, height = float(record["width"]), float(record["height"])
        predictions = []
        for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
            if int(label) < 0 or float(score) < threshold:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            x0, x1 = sorted((max(0.0, x0), min(width - 1.0, x1)))
            y0, y1 = sorted((max(0.0, y0), min(height - 1.0, y1)))
            if x1 > x0 and y1 > y0:
                predictions.append({"category_id": int(label), "score": float(score), "box": np.asarray([x0, y0, x1, y1], dtype=np.float32)})
        ground_truth = anns_by_image.get(image_id, [])
        local_gt, prediction_labels = analyse_native_input(ground_truth, predictions, match_iou)
        for detail in local_gt.values():
            origin = source_details.get(int(detail["source_annotation_id"]))
            if origin is not None:
                origin["matched"] = origin["matched"] or bool(detail["matched"])
                origin["max_same_class_coverage"] = max(origin["max_same_class_coverage"], float(detail["same_class_union_coverage"]))
                origin["max_same_class_prediction_count"] = max(origin["max_same_class_prediction_count"], int(detail["same_class_prediction_count"]))
                origin["wrong_class_categories"].update(detail["wrong_class_prediction_categories"])
        for pred, pred_label in zip(predictions, prediction_labels):
            outcome = str(pred_label["outcome"])
            prediction_counts[(int(pred["category_id"]), outcome)] += 1
            per_prediction.append({"method": name, "native_image_id": image_id, "source_image_id": source_id,
                                   "class_id": int(pred["category_id"]), "class_name": native_categories.get(int(pred["category_id"]), categories.get(int(pred["category_id"]), "class_{}".format(int(pred["category_id"])))),
                                   "score": round(float(pred["score"]), 7), "outcome": outcome})
    instance_rows, counts = [], Counter()
    for aid in sorted(source_details):
        d = source_details[aid]
        strict = source_outcome(d, partial, fragmented)
        if strict == "correct":
            cwm = "correct"
        elif strict == "miss" or strict == "not_represented":
            cwm = "miss"
        else:
            cwm = "wrong_class_or_partial"
        counts[cwm] += 1
        instance_rows.append({"method": name, "source_annotation_id": aid, "source_image_id": d["source_image_id"],
                              "class_id": d["class_id"], "class_name": d["class_name"], "strict_outcome": strict,
                              "cwm_outcome": cwm, "represented": d["represented"],
                              "max_same_class_coverage": round(float(d["max_same_class_coverage"]), 7),
                              "wrong_class_categories": ";".join(categories[c] for c in sorted(d["wrong_class_categories"]) if c in categories)})
    rows = []
    for c in sorted(categories):
        selected = [r for r in instance_rows if int(r["class_id"]) == c]
        item = Counter(r["cwm_outcome"] for r in selected)
        rows.append({"method": name, "class_id": c, "class_name": categories[c], "instances": len(selected),
                     "correct": item["correct"], "wrong": item["wrong_class_or_partial"], "miss": item["miss"],
                     "correct_rate": round(item["correct"] / len(selected), 6) if selected else None})
    summary = {"method": name, "source_instances": len(instance_rows), "correct": counts["correct"],
               "wrong": counts["wrong_class_or_partial"], "miss": counts["miss"],
               "correct_rate": round(counts["correct"] / len(instance_rows), 6), "threshold": threshold,
               "match_iou": match_iou, "coverage_threshold": partial,
               "native_predictions_kept": int(sum(prediction_counts.values()))}
    return summary, rows, instance_rows, per_prediction


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    required = [args.config, args.checkpoint, args.image_root, args.train_annotations,
                args.native_val_annotations, args.source_annotations]
    if args.native_extra_annotations:
        required.extend([args.native_extra_annotations, args.native_extra_image_root])
    for path in required:
        if path is None or not path.exists():
            raise FileNotFoundError(path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    train = load_json(args.train_annotations)
    base_native = load_json(args.native_val_annotations)
    extra_native = load_json(args.native_extra_annotations) if args.native_extra_annotations else None
    native = canonicalize_dataset(combine_native(base_native, extra_native), args.grouped_protocol)
    source = canonicalize_dataset(load_json(args.source_annotations), args.grouped_protocol)
    classes = [int(x["id"]) for x in train["categories"]]
    adjustment, stat_rows = compute_fracal_adjustments(train, classes, args.fractal_exponent, args.grid_max)
    model, transform, spatial = build_raw_model(args, device)
    args.output_dir.mkdir(parents=True)
    records = sorted(native["images"], key=lambda x: int(x["id"]))
    roots = {int(row["id"]): args.image_root for row in base_native["images"]}
    if extra_native:
        roots.update({int(row["id"]): args.native_extra_image_root for row in extra_native["images"]})
    raw_batches, fracal_batches = [], []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            part = records[start:start + args.batch_size]
            tensors, sizes = [], []
            for record in part:
                image_path = roots[int(record["id"])] / str(record["file_name"])
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    tensors.append(transform(image))
                    sizes.append([float(image.width), float(image.height)])
            size_tensor = torch.tensor(sizes, dtype=torch.float32, device=device)
            outputs = model(torch.stack(tensors).to(device))
            raw = topk_predictions(outputs["pred_logits"], outputs["pred_boxes"], size_tensor, args.top_k, None)
            calibrated = topk_predictions(outputs["pred_logits"], outputs["pred_boxes"], size_tensor, args.top_k, adjustment)
            for index, record in enumerate(part):
                raw_labels = torch.tensor([canonical_label(int(v), args.grouped_protocol) for v in raw[0][index].tolist()])
                fracal_labels = torch.tensor([canonical_label(int(v), args.grouped_protocol) for v in calibrated[0][index].tolist()])
                raw_batches.append((record, raw_labels, raw[1][index].cpu(), raw[2][index].cpu()))
                fracal_batches.append((record, fracal_labels, calibrated[1][index].cpu(), calibrated[2][index].cpu()))
            completed = min(start + len(part), len(records))
            if completed % 100 == 0 or completed == len(records):
                print(f"inferred {completed}/{len(records)} native validation tiles", flush=True)

    threshold_grid = [float(args.threshold)]
    if args.fracal_thresholds.strip():
        threshold_grid = [float(x.strip()) for x in args.fracal_thresholds.split(",") if x.strip()]
    all_summary, all_class_rows, all_instance_rows, all_prediction_rows = [], [], [], []
    raw_result = audit_method("D-FINE-S raw", raw_batches, native, source, args.threshold,
                              args.match_iou, args.partial_coverage, args.fragmented_coverage)
    all_summary.append(raw_result[0]); all_class_rows.extend(raw_result[1]); all_instance_rows.extend(raw_result[2]); all_prediction_rows.extend(raw_result[3])
    fracal_trials = []
    for threshold in threshold_grid:
        result = audit_method(f"D-FINE-S + FRACAL (t={threshold:.3f})", fracal_batches, native, source, threshold,
                              args.match_iou, args.partial_coverage, args.fragmented_coverage)
        fracal_trials.append(result)
        all_summary.append(result[0]); all_class_rows.extend(result[1]); all_instance_rows.extend(result[2]); all_prediction_rows.extend(result[3])
    # C/N is the protocol's primary candidate-retention score; break ties by fewer wrong labels.
    best = max(fracal_trials, key=lambda result: (result[0]["correct_rate"], -result[0]["wrong"], -result[0]["miss"]))
    write_csv(args.output_dir / "fracal_train_statistics.csv", stat_rows)
    write_csv(args.output_dir / "threshold_grid_summary.csv", all_summary)
    write_csv(args.output_dir / "per_class_cwm.csv", all_class_rows)
    write_csv(args.output_dir / "per_source_instance_cwm.csv", all_instance_rows)
    write_csv(args.output_dir / "per_native_prediction_outcomes.csv", all_prediction_rows)
    write_csv(args.output_dir / "summary.csv", [raw_result[0], best[0]])
    metadata = vars(args) | {
        "eval_spatial_size_h_w": list(spatial),
        "fracal_equation": "sigmoid(z) * sigmoid(z + frequency_adjustment + fractal_adjustment)",
        "source": "FRACAL public sigmoid-head implementation",
        "selected_fracal_threshold": best[0]["threshold"],
        "protocol": "13 grouped categories; 1122 original validation instances plus 4 long-weld-high diagnostic holdout instances" if args.grouped_protocol else "raw categories",
    }
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# FRACAL on frozen D-FINE-S\\n\\nInference-only post-calibration. The raw D-FINE-S row remains at its fixed threshold; FRACAL is scanned separately because its multiplicative sigmoid score has a different scale. `wrong` combines wrong-class, partial-coverage and fragmented-coverage outcomes.\\n", encoding="utf-8")
    print(json.dumps({"raw": raw_result[0], "best_fracal": best[0], "all_fracal_thresholds": [x[0] for x in fracal_trials]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()