#!/usr/bin/env python
"""Build IBO candidate crops from a fixed D-FINE detector.

The script is intentionally candidate-centric:
1. Run a trained D-FINE checkpoint on unwrapped weld-strip tiles.
2. Save each prediction above a recall-oriented threshold as an IBO crop.
3. Keep traceability back to source image / tile / annotation.
4. Summarize whether each GT instance is covered by low-threshold and
   high-confidence candidates.

This does not train a new model.  It prepares the candidate-region evidence
bank used by later normal-reference, frequency-evidence, reliability-fusion
and calibration experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-root", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--candidate-threshold", type=float, default=0.25)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.40)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--context-scale", type=float, default=2.0)
    parser.add_argument("--min-crop-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude-class-id", type=int, action="append", default=[])
    parser.add_argument(
        "--match-family",
        action="append",
        default=[],
        help="Comma-separated class ids that should be treated as one class for candidate/GT matching, e.g. 12,14,15.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Debug only: limit number of tile images.")
    return parser.parse_args()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: Path
    checkpoint: Path
    image_root: Path
    annotations: Path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_annotation_id(annotation: dict[str, Any]) -> int:
    return int(annotation.get("source_annotation_id", annotation["id"]))


def source_image_id(image: dict[str, Any]) -> int:
    return int(image.get("source_image_id", image["id"]))


def xyxy_from_xywh(box: list[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def gt_coverage(gt_box: np.ndarray, pred_boxes: list[np.ndarray]) -> float:
    if not pred_boxes:
        return 0.0
    # One-dimensional strip defects can be split into several candidate boxes.
    # We therefore compute union coverage on the GT area rather than only max IoU.
    events: list[tuple[float, int, float, float]] = []
    for box in pred_boxes:
        x0 = max(float(gt_box[0]), float(box[0]))
        y0 = max(float(gt_box[1]), float(box[1]))
        x1 = min(float(gt_box[2]), float(box[2]))
        y1 = min(float(gt_box[3]), float(box[3]))
        if x1 > x0 and y1 > y0:
            events.append((x0, 1, y0, y1))
            events.append((x1, -1, y0, y1))
    if not events:
        return 0.0
    events.sort()
    active: list[tuple[float, float]] = []
    previous_x: float | None = None
    area = 0.0
    for x, flag, y0, y1 in events:
        if previous_x is not None and x > previous_x and active:
            intervals = sorted(active)
            merged = 0.0
            current_start, current_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged += current_end - current_start
                    current_start, current_end = start, end
            merged += current_end - current_start
            area += (x - previous_x) * merged
        if flag > 0:
            active.append((y0, y1))
        else:
            try:
                active.remove((y0, y1))
            except ValueError:
                pass
        previous_x = x
    return min(1.0, area / max(1e-6, box_area(gt_box)))


def expand_box(box: np.ndarray, width: int, height: int, scale: float, min_size: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bw = max(float(min_size), (x1 - x0) * scale)
    bh = max(float(min_size), (y1 - y0) * scale)
    nx0 = max(0.0, cx - bw / 2.0)
    ny0 = max(0.0, cy - bh / 2.0)
    nx1 = min(float(width), cx + bw / 2.0)
    ny1 = min(float(height), cy + bh / 2.0)
    return np.asarray([nx0, ny0, nx1, ny1], dtype=np.float32)


def clamp_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = sorted((max(0.0, x0), min(float(width), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(height), y1)))
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_") or "item"


def build_model(spec: ModelSpec, dfine_root: Path, scripts_root: Path, device: torch.device):
    sys.path.insert(0, str(scripts_root))
    from evaluate_dfine_strict_detection_errors import build_model as _build_model

    return _build_model(spec, dfine_root, device)


def parse_match_families(values: list[str]) -> list[set[int]]:
    families: list[set[int]] = []
    for value in values:
        family = {int(item.strip()) for item in str(value).split(",") if item.strip()}
        if len(family) >= 2:
            families.append(family)
    return families


def class_matches(gt_class: int, pred_class: int, match_families: list[set[int]]) -> bool:
    if int(gt_class) == int(pred_class):
        return True
    for family in match_families:
        if int(gt_class) in family and int(pred_class) in family:
            return True
    return False


def candidate_label(
    prediction: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    match_iou: float,
    coverage_threshold: float,
    match_families: list[set[int]],
) -> dict[str, Any]:
    if not ground_truth:
        return {
            "outcome": "background",
            "matched_native_gt_id": "",
            "matched_source_annotation_id": "",
            "matched_gt_class_id": "",
            "matched_gt_class_name": "",
            "best_iou_any": 0.0,
            "best_iou_same": 0.0,
            "best_coverage_any": 0.0,
            "best_coverage_same": 0.0,
        }
    pred_box = prediction["box"]
    pred_class = int(prediction["category_id"])
    best_any = None
    best_same = None
    for gt in ground_truth:
        gt_box = xyxy_from_xywh(gt["bbox"])
        item_iou = iou(pred_box, gt_box)
        item_cov = intersection_area(gt_box, pred_box) / max(1e-6, box_area(gt_box))
        packed = (item_iou, item_cov, gt)
        if best_any is None or (item_iou, item_cov) > (best_any[0], best_any[1]):
            best_any = packed
        if class_matches(int(gt["category_id"]), pred_class, match_families) and (best_same is None or (item_iou, item_cov) > (best_same[0], best_same[1])):
            best_same = packed

    if best_same is not None and best_same[0] >= match_iou:
        best = best_same
        outcome = "correct"
    elif best_any is not None and best_any[0] >= match_iou:
        best = best_any
        outcome = "wrong_class"
    elif best_same is not None and best_same[1] >= coverage_threshold:
        best = best_same
        outcome = "partial_same_class"
    elif best_any is not None and best_any[1] >= coverage_threshold:
        best = best_any
        outcome = "overlaps_other_class"
    else:
        best = best_any
        outcome = "background"

    matched_gt = best[2] if best is not None else {}
    return {
        "outcome": outcome,
        "matched_native_gt_id": int(matched_gt["id"]) if matched_gt else "",
        "matched_source_annotation_id": source_annotation_id(matched_gt) if matched_gt else "",
        "matched_gt_class_id": int(matched_gt["category_id"]) if matched_gt else "",
        "best_iou_any": round(float(best_any[0]), 6) if best_any else 0.0,
        "best_iou_same": round(float(best_same[0]), 6) if best_same else 0.0,
        "best_coverage_any": round(float(best_any[1]), 6) if best_any else 0.0,
        "best_coverage_same": round(float(best_same[1]), 6) if best_same else 0.0,
    }


def main() -> None:
    args = parse_args()
    exclude_class_ids = set(int(v) for v in args.exclude_class_id)
    match_families = parse_match_families(args.match_family)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    data = load_json(args.annotations)
    source_data = load_json(args.source_annotations)
    categories = {int(item["id"]): str(item["name"]) for item in data.get("categories", source_data["categories"])}
    category_slugs = {
        int(item["id"]): str(item.get("slug") or safe_name(str(item["name"])))
        for item in data.get("categories", source_data["categories"])
    }
    records = sorted(data["images"], key=lambda item: int(item["id"]))
    if args.max_images > 0:
        records = records[: args.max_images]

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        if int(annotation["category_id"]) in exclude_class_ids:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    source_images = {int(item["id"]): item for item in source_data["images"]}
    source_details: dict[int, dict[str, Any]] = {}
    for annotation in source_data["annotations"]:
        if int(annotation["category_id"]) in exclude_class_ids:
            continue
        sid = int(annotation["id"])
        source_details[sid] = {
            "source_annotation_id": sid,
            "source_image_id": int(annotation["image_id"]),
            "source_file": source_images.get(int(annotation["image_id"]), {}).get("file_name", ""),
            "class_id": int(annotation["category_id"]),
            "class_name": categories.get(int(annotation["category_id"]), f"class_{int(annotation['category_id'])}"),
            "represented_in_tiles": False,
            "native_occurrences": 0,
            "max_iou_any_low": 0.0,
            "max_iou_same_low": 0.0,
            "max_coverage_any_low": 0.0,
            "max_coverage_same_low": 0.0,
            "max_iou_any_high": 0.0,
            "max_iou_same_high": 0.0,
            "max_coverage_any_high": 0.0,
            "max_coverage_same_high": 0.0,
        }
    for annotation in data["annotations"]:
        if int(annotation["category_id"]) in exclude_class_ids:
            continue
        sid = source_annotation_id(annotation)
        if sid not in source_details:
            source_details[sid] = {
                "source_annotation_id": sid,
                "source_image_id": "",
                "source_file": "",
                "class_id": int(annotation["category_id"]),
                "class_name": categories.get(int(annotation["category_id"]), f"class_{int(annotation['category_id'])}"),
                "represented_in_tiles": False,
                "native_occurrences": 0,
                "max_iou_any_low": 0.0,
                "max_iou_same_low": 0.0,
                "max_coverage_any_low": 0.0,
                "max_coverage_same_low": 0.0,
                "max_iou_any_high": 0.0,
                "max_iou_same_high": 0.0,
                "max_coverage_any_high": 0.0,
                "max_coverage_same_high": 0.0,
            }
        source_details[sid]["represented_in_tiles"] = True
        source_details[sid]["native_occurrences"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_root = args.output_dir / "crops" / args.split
    manifest_path = args.output_dir / f"ibo_manifest_{args.split}.jsonl"
    summary_path = args.output_dir / f"ibo_summary_{args.split}.md"
    config_path = args.output_dir / f"ibo_build_config_{args.split}.json"

    spec = ModelSpec(
        name="dfine_s_unwrapped_150px_cyclic_tiles",
        config=args.config,
        checkpoint=args.checkpoint,
        image_root=args.image_root,
        annotations=args.annotations,
    )
    model, transform, _ = build_model(spec, args.dfine_root, args.scripts_root, device)

    candidate_counts = Counter()
    gt_rows: list[dict[str, Any]] = []
    crop_count = 0
    print(f"[IBO] split={args.split} images={len(records)} device={device} output={args.output_dir}", flush=True)

    with manifest_path.open("w", encoding="utf-8") as manifest, torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            tensors: list[torch.Tensor] = []
            sizes: list[list[float]] = []
            pil_images: list[Image.Image] = []
            for record in batch_records:
                image_path = args.image_root / str(record["file_name"])
                image = Image.open(image_path).convert("RGB")
                pil_images.append(image)
                tensors.append(transform(image))
                sizes.append([float(image.width), float(image.height)])
            labels, boxes, scores = model(
                torch.stack(tensors).to(device),
                torch.tensor(sizes, dtype=torch.float32, device=device),
            )
            for batch_index, record in enumerate(batch_records):
                tile_image_id = int(record["id"])
                width = int(sizes[batch_index][0])
                height = int(sizes[batch_index][1])
                ground_truth = annotations_by_image.get(tile_image_id, [])
                predictions: list[dict[str, Any]] = []
                high_predictions: list[dict[str, Any]] = []
                for category_id, box, score in zip(
                    labels[batch_index].tolist(),
                    boxes[batch_index].tolist(),
                    scores[batch_index].tolist(),
                ):
                    category_id = int(category_id)
                    score = float(score)
                    if category_id in exclude_class_ids or score < args.candidate_threshold:
                        continue
                    pred_box = clamp_box(np.asarray(box, dtype=np.float32), width, height)
                    if pred_box[2] <= pred_box[0] or pred_box[3] <= pred_box[1]:
                        continue
                    prediction = {"category_id": category_id, "score": score, "box": pred_box}
                    predictions.append(prediction)
                    if score >= args.high_confidence_threshold:
                        high_predictions.append(prediction)

                # Source-level GT coverage aggregation, including fragmented boxes.
                for gt in ground_truth:
                    gt_box = xyxy_from_xywh(gt["bbox"])
                    same_low = [p["box"] for p in predictions if class_matches(int(gt["category_id"]), int(p["category_id"]), match_families)]
                    any_low = [p["box"] for p in predictions]
                    same_high = [p["box"] for p in high_predictions if class_matches(int(gt["category_id"]), int(p["category_id"]), match_families)]
                    any_high = [p["box"] for p in high_predictions]
                    sid = source_annotation_id(gt)
                    detail = source_details[sid]
                    for suffix, same_boxes, any_boxes in [
                        ("low", same_low, any_low),
                        ("high", same_high, any_high),
                    ]:
                        if any_boxes:
                            detail[f"max_iou_any_{suffix}"] = max(detail[f"max_iou_any_{suffix}"], max(iou(gt_box, b) for b in any_boxes))
                            detail[f"max_coverage_any_{suffix}"] = max(detail[f"max_coverage_any_{suffix}"], gt_coverage(gt_box, any_boxes))
                        if same_boxes:
                            detail[f"max_iou_same_{suffix}"] = max(detail[f"max_iou_same_{suffix}"], max(iou(gt_box, b) for b in same_boxes))
                            detail[f"max_coverage_same_{suffix}"] = max(detail[f"max_coverage_same_{suffix}"], gt_coverage(gt_box, same_boxes))

                image = pil_images[batch_index]
                for prediction_index, prediction in enumerate(predictions):
                    label = candidate_label(prediction, ground_truth, args.match_iou, args.coverage_threshold, match_families)
                    class_id = int(prediction["category_id"])
                    class_name = categories.get(class_id, f"class_{class_id}")
                    confidence_group = "high_conf" if float(prediction["score"]) >= args.high_confidence_threshold else "recall_protect"
                    outcome = str(label["outcome"])
                    candidate_counts[(class_id, outcome, confidence_group)] += 1
                    expanded = expand_box(prediction["box"], width, height, args.context_scale, args.min_crop_size)
                    crop = image.crop(tuple(int(round(v)) for v in expanded))
                    crop_count += 1
                    crop_rel = Path("crops") / args.split / safe_name(category_slugs.get(class_id, class_name)) / outcome / f"ibo_{args.split}_{crop_count:07d}.jpg"
                    crop_path = args.output_dir / crop_rel
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(crop_path, quality=92)
                    row = {
                        "ibo_id": f"{args.split}_{crop_count:07d}",
                        "split": args.split,
                        "tile_image_id": tile_image_id,
                        "tile_file": record["file_name"],
                        "tile_width": width,
                        "tile_height": height,
                        "source_image_id": source_image_id(record),
                        "source_file": record.get("source_file", ""),
                        "tile_start": record.get("tile_start", ""),
                        "phase_offset": record.get("phase_offset", ""),
                        "pred_class_id": class_id,
                        "pred_class_name": class_name,
                        "pred_score": round(float(prediction["score"]), 6),
                        "confidence_group": confidence_group,
                        "candidate_outcome": outcome,
                        "pred_xyxy": [round(float(v), 3) for v in prediction["box"].tolist()],
                        "crop_xyxy": [round(float(v), 3) for v in expanded.tolist()],
                        "crop_path": str(crop_rel).replace("\\", "/"),
                        **label,
                    }
                    if row["matched_gt_class_id"] != "":
                        row["matched_gt_class_name"] = categories.get(int(row["matched_gt_class_id"]), f"class_{row['matched_gt_class_id']}")
                    else:
                        row["matched_gt_class_name"] = ""
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            for image in pil_images:
                image.close()
            done = start + len(batch_records)
            if done == len(records) or done % max(args.batch_size * 25, 100) == 0:
                print(f"[IBO] {args.split}: {done}/{len(records)} images, crops={crop_count}", flush=True)

    for detail in source_details.values():
        row = {
            "source_annotation_id": detail["source_annotation_id"],
            "source_image_id": detail["source_image_id"],
            "source_file": detail["source_file"],
            "class_id": detail["class_id"],
            "class_name": detail["class_name"],
            "represented_in_tiles": bool(detail["represented_in_tiles"]),
            "native_occurrences": detail["native_occurrences"],
        }
        for suffix in ["low", "high"]:
            row[f"max_iou_any_{suffix}"] = round(float(detail[f"max_iou_any_{suffix}"]), 6)
            row[f"max_iou_same_{suffix}"] = round(float(detail[f"max_iou_same_{suffix}"]), 6)
            row[f"max_coverage_any_{suffix}"] = round(float(detail[f"max_coverage_any_{suffix}"]), 6)
            row[f"max_coverage_same_{suffix}"] = round(float(detail[f"max_coverage_same_{suffix}"]), 6)
            row[f"covered_any_iou50_{suffix}"] = detail[f"max_iou_any_{suffix}"] >= args.match_iou
            row[f"covered_same_iou50_{suffix}"] = detail[f"max_iou_same_{suffix}"] >= args.match_iou
            row[f"covered_any_cov20_{suffix}"] = detail[f"max_coverage_any_{suffix}"] >= args.coverage_threshold
            row[f"covered_same_cov20_{suffix}"] = detail[f"max_coverage_same_{suffix}"] >= args.coverage_threshold
        gt_rows.append(row)

    write_csv(args.output_dir / f"gt_candidate_coverage_{args.split}.csv", gt_rows)

    by_class_rows: list[dict[str, Any]] = []
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        by_class[int(row["class_id"])].append(row)
    for class_id in sorted(by_class):
        rows = by_class[class_id]
        gt_count = len(rows)
        represented = sum(1 for row in rows if row["represented_in_tiles"])
        out = {
            "class_id": class_id,
            "class_name": categories.get(class_id, f"class_{class_id}"),
            "gt_instances": gt_count,
            "represented_in_tiles": represented,
        }
        for suffix in ["low", "high"]:
            out[f"same_class_cov20_recall_{suffix}"] = round(sum(1 for row in rows if row[f"covered_same_cov20_{suffix}"]) / gt_count, 6) if gt_count else 0.0
            out[f"any_class_cov20_recall_{suffix}"] = round(sum(1 for row in rows if row[f"covered_any_cov20_{suffix}"]) / gt_count, 6) if gt_count else 0.0
            out[f"same_class_iou50_recall_{suffix}"] = round(sum(1 for row in rows if row[f"covered_same_iou50_{suffix}"]) / gt_count, 6) if gt_count else 0.0
            out[f"any_class_iou50_recall_{suffix}"] = round(sum(1 for row in rows if row[f"covered_any_iou50_{suffix}"]) / gt_count, 6) if gt_count else 0.0
        for suffix in ["recall_protect", "high_conf"]:
            out[f"candidate_count_{suffix}"] = sum(count for (cid, _, group), count in candidate_counts.items() if cid == class_id and group == suffix)
        by_class_rows.append(out)
    write_csv(args.output_dir / f"candidate_recall_by_class_{args.split}.csv", by_class_rows)

    candidate_rows = [
        {
            "class_id": cid,
            "class_name": categories.get(cid, f"class_{cid}"),
            "candidate_outcome": outcome,
            "confidence_group": group,
            "count": count,
        }
        for (cid, outcome, group), count in sorted(candidate_counts.items())
    ]
    write_csv(args.output_dir / f"candidate_counts_by_outcome_{args.split}.csv", candidate_rows)

    config = {
        "split": args.split,
        "dfine_root": str(args.dfine_root),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "image_root": str(args.image_root),
        "annotations": str(args.annotations),
        "source_annotations": str(args.source_annotations),
        "candidate_threshold": args.candidate_threshold,
        "high_confidence_threshold": args.high_confidence_threshold,
        "match_iou": args.match_iou,
        "coverage_threshold": args.coverage_threshold,
        "context_scale": args.context_scale,
        "min_crop_size": args.min_crop_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "exclude_class_id": sorted(exclude_class_ids),
        "match_families": [sorted(family) for family in match_families],
        "image_count": len(records),
        "crop_count": crop_count,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    total_gt = len(gt_rows)
    low_same_cov = sum(1 for row in gt_rows if row["covered_same_cov20_low"])
    high_same_cov = sum(1 for row in gt_rows if row["covered_same_cov20_high"])
    lines = [
        f"# IBO 候选区域构建摘要（{args.split}）",
        "",
        f"- 输入 tile 数：{len(records)}",
        f"- 导出 IBO crop 数：{crop_count}",
        f"- 候选阈值：{args.candidate_threshold}",
        f"- 高置信阈值：{args.high_confidence_threshold}",
        f"- GT 实例数（已排除类别 {sorted(exclude_class_ids)}）：{total_gt}",
        f"- 合并匹配类别族：{[sorted(family) for family in match_families]}",
        f"- 低阈值同类/同族候选覆盖率（GT 覆盖 >= {args.coverage_threshold:.2f}）：{low_same_cov}/{total_gt} = {low_same_cov / total_gt:.2%}" if total_gt else "- 低阈值同类/同族候选覆盖率：NA",
        f"- 高置信同类/同族候选覆盖率（GT 覆盖 >= {args.coverage_threshold:.2f}）：{high_same_cov}/{total_gt} = {high_same_cov / total_gt:.2%}" if total_gt else "- 高置信同类/同族候选覆盖率：NA",
        "",
        "## 主要产物",
        "",
        f"- `ibo_manifest_{args.split}.jsonl`：每个 IBO crop 的来源、预测框、类别、置信度、匹配状态。",
        f"- `gt_candidate_coverage_{args.split}.csv`：每个人工标注实例是否被候选框覆盖。",
        f"- `candidate_recall_by_class_{args.split}.csv`：每类缺陷的候选覆盖率。",
        f"- `candidate_counts_by_outcome_{args.split}.csv`：候选框按类别/结果/置信组统计。",
        f"- `crops/{args.split}/`：候选框上下文裁剪图。",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[IBO] done split={args.split} crops={crop_count} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
