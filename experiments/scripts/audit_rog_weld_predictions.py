#!/usr/bin/env python
"""Audit MMDetection/ROG weld predictions with the established source C/W/M protocol.

The script consumes the native 150-pixel-tile COCO annotations and the ordered
``tools/test.py --out`` pickle.  It deliberately reuses the strict matching,
coverage and source-instance aggregation rules used by
``evaluate_dfine_strict_detection_errors.py``; only prediction acquisition is
different.  Thus this is an audit adapter, not a changed evaluation metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_dfine_strict_detection_errors import (  # noqa: E402
    analyse_native_input,
    source_annotation_id,
    source_image_id,
    source_outcome,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--native-annotations", type=Path, required=True)
    parser.add_argument("--predictions-pkl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="ROG")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--partial-coverage", type=float, default=0.10)
    parser.add_argument("--fragmented-coverage", type=float, default=0.50)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> list:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def mmdet_boxes(result: object, native_category_ids: list[int], threshold: float) -> list[dict]:
    """Convert a single MMDetection bbox result into strict-audit boxes."""
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, (list, tuple)) or len(result) != len(native_category_ids):
        raise ValueError("Unexpected MMDetection bbox result format or class count")
    boxes: list[dict] = []
    for class_index, class_boxes in enumerate(result):
        arr = np.asarray(class_boxes, dtype=np.float32)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 5)
        for x0, y0, x1, y1, score in arr:
            if float(score) < threshold:
                continue
            if float(x1) <= float(x0) or float(y1) <= float(y0):
                continue
            boxes.append({
                "category_id": int(native_category_ids[class_index]),
                "score": float(score),
                "box": np.asarray([x0, y0, x1, y1], dtype=np.float32),
            })
    return boxes


def main() -> None:
    args = parse_args()
    if not 0 <= args.score_threshold <= 1 or not 0 < args.match_iou <= 1:
        raise ValueError("Invalid score threshold or IoU")
    if not 0 <= args.partial_coverage < args.fragmented_coverage <= 1:
        raise ValueError("Require 0 <= partial < fragmented <= 1")

    source = load_json(args.source_annotations)
    native = load_json(args.native_annotations)
    predictions = load_predictions(args.predictions_pkl)
    records = sorted(native["images"], key=lambda item: int(item["id"]))
    if len(records) != len(predictions):
        raise ValueError(f"Prediction/image count mismatch: {len(predictions)} vs {len(records)}")

    categories = {int(item["id"]): str(item["name"]) for item in source["categories"]}
    native_category_ids = [int(item["id"]) for item in native["categories"]]
    if set(native_category_ids) != set(categories):
        raise ValueError("Source/native category ids differ")
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in native["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    source_details: dict[int, dict] = {}
    for annotation in source["annotations"]:
        source_id = int(annotation["id"])
        source_details[source_id] = {
            "source_annotation_id": source_id,
            "source_image_id": int(annotation["image_id"]),
            "class_id": int(annotation["category_id"]),
            "class_name": categories[int(annotation["category_id"])],
            "represented": False,
            "native_occurrences": 0,
            "matched": False,
            "true_positive_native_matches": 0,
            "max_same_class_iou": 0.0,
            "max_same_class_coverage": 0.0,
            "max_same_class_prediction_count": 0,
            "wrong_class_categories": set(),
        }
    for annotation in native["annotations"]:
        source_id = source_annotation_id(annotation)
        if source_id not in source_details:
            raise KeyError(f"Native annotation maps to unavailable source id {source_id}")
        source_details[source_id]["represented"] = True
        source_details[source_id]["native_occurrences"] += 1

    prediction_counts: Counter = Counter()
    source_prediction_error_counts: dict[int, Counter] = defaultdict(Counter)
    native_prediction_rows: list[dict] = []
    predictions_by_source_image: Counter = Counter()
    for record, result in zip(records, predictions):
        image_id = int(record["id"])
        predictions_for_image = mmdet_boxes(result, native_category_ids, args.score_threshold)
        source_image = source_image_id(record)
        predictions_by_source_image[source_image] += len(predictions_for_image)
        ground_truth = annotations_by_image.get(image_id, [])
        local_gt, prediction_labels = analyse_native_input(ground_truth, predictions_for_image, args.match_iou)
        for detail in local_gt.values():
            origin = source_details[detail["source_annotation_id"]]
            origin["matched"] = origin["matched"] or detail["matched"]
            origin["true_positive_native_matches"] += int(detail["matched"])
            origin["max_same_class_iou"] = max(origin["max_same_class_iou"], detail["max_same_class_iou"])
            origin["max_same_class_coverage"] = max(origin["max_same_class_coverage"], detail["same_class_union_coverage"])
            origin["max_same_class_prediction_count"] = max(origin["max_same_class_prediction_count"], detail["same_class_prediction_count"])
            origin["wrong_class_categories"].update(detail["wrong_class_prediction_categories"])
        for prediction_index, (prediction, label) in enumerate(zip(predictions_for_image, prediction_labels)):
            outcome = str(label["outcome"])
            category_id = int(prediction["category_id"])
            prediction_counts[(category_id, outcome)] += 1
            anchor_source = ""
            if label["matched_gt_index"] is not None:
                anchor_source = source_annotation_id(ground_truth[int(label["matched_gt_index"])])
                if outcome != "true_positive":
                    source_prediction_error_counts[int(anchor_source)][outcome] += 1
            native_prediction_rows.append({
                "native_image_id": image_id,
                "source_image_id": source_image,
                "native_file": str(record["file_name"]),
                "prediction_index": prediction_index,
                "predicted_class_id": category_id,
                "predicted_class_name": categories[category_id],
                "confidence": round(float(prediction["score"]), 6),
                "prediction_outcome": outcome,
                "anchor_source_annotation_id": anchor_source,
            })

    instance_rows: list[dict] = []
    for source_id in sorted(source_details):
        detail = source_details[source_id]
        outcome = source_outcome(detail, args.partial_coverage, args.fragmented_coverage)
        instance_rows.append({
            "source_annotation_id": source_id,
            "source_image_id": detail["source_image_id"],
            "class_id": detail["class_id"],
            "class_name": detail["class_name"],
            "represented_in_native_dataset": detail["represented"],
            "native_occurrences": detail["native_occurrences"],
            "true_positive_native_matches": detail["true_positive_native_matches"],
            "max_same_class_iou": round(float(detail["max_same_class_iou"]), 6),
            "max_same_class_union_coverage": round(float(detail["max_same_class_coverage"]), 6),
            "wrong_class_predictions": ";".join(categories[item] for item in sorted(detail["wrong_class_categories"])),
            "same_tile_duplicate_prediction_boxes": source_prediction_error_counts[source_id]["same_tile_duplicate"],
            "wrong_class_prediction_boxes": source_prediction_error_counts[source_id]["wrong_class"],
            "localization_or_fragment_prediction_boxes": source_prediction_error_counts[source_id]["localization_or_fragment"],
            "strict_gt_outcome": outcome,
        })
    gt_names = ("correct", "fragmented_coverage", "partial_coverage", "wrong_class", "miss", "not_represented")
    prediction_names = ("true_positive", "same_tile_duplicate", "wrong_class", "localization_or_fragment", "background")
    per_class_rows: list[dict] = []
    for category_id in sorted(categories):
        rows = [row for row in instance_rows if int(row["class_id"]) == category_id]
        counts = Counter(str(row["strict_gt_outcome"]) for row in rows)
        per_class_rows.append({
            "class_id": category_id, "class_name": categories[category_id],
            "ground_truth_instances": len(rows),
            "evaluable_instances": len(rows) - counts["not_represented"],
            **{name: counts[name] for name in gt_names},
            "mean_same_class_coverage": round(float(np.mean([float(row["max_same_class_union_coverage"]) for row in rows])) if rows else 0.0, 6),
        })
    per_prediction_class_rows: list[dict] = []
    for category_id in sorted(categories):
        counts = {name: prediction_counts[(category_id, name)] for name in prediction_names}
        per_prediction_class_rows.append({
            "predicted_class_id": category_id, "predicted_class_name": categories[category_id],
            "prediction_boxes": sum(counts.values()), **counts,
        })

    instance_by_source = {int(row["source_annotation_id"]): row for row in instance_rows}
    image_to_instances: dict[int, list[int]] = defaultdict(list)
    for row in instance_rows:
        image_to_instances[int(row["source_image_id"])].append(int(row["source_annotation_id"]))
    errors_by_source_image: Counter = Counter()
    for row in native_prediction_rows:
        if row["prediction_outcome"] != "true_positive":
            errors_by_source_image[int(row["source_image_id"])] += 1
    image_rows: list[dict] = []
    image_outcomes: Counter = Counter()
    for image in sorted(source["images"], key=lambda item: int(item["id"])):
        source_image = int(image["id"])
        gt_rows = [instance_by_source[x] for x in image_to_instances.get(source_image, [])]
        non_tp = sum(row["strict_gt_outcome"] != "correct" for row in gt_rows)
        fp_count = errors_by_source_image[source_image]
        if not gt_rows:
            outcome = "fully_correct" if fp_count == 0 else "prediction_error_only"
        elif non_tp:
            outcome = "miss_or_partial"
        elif fp_count:
            outcome = "detected_with_extra_error"
        else:
            outcome = "fully_correct"
        image_outcomes[outcome] += 1
        image_rows.append({
            "source_image_id": source_image, "source_file": str(image["file_name"]),
            "ground_truth_instances": len(gt_rows), "non_true_positive_ground_truth_instances": non_tp,
            "prediction_error_boxes": fp_count,
            "cross_tile_repeat_true_positive_boxes": sum(max(0, int(row["true_positive_native_matches"]) - 1) for row in gt_rows),
            "native_predictions_at_threshold": predictions_by_source_image[source_image],
            "strict_image_outcome": outcome,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_ground_truth_class_strict_stats.csv", per_class_rows)
    write_csv(args.output_dir / "per_prediction_class_error_stats.csv", per_prediction_class_rows)
    write_csv(args.output_dir / "per_source_defect_instance_strict_outcomes.csv", instance_rows)
    write_csv(args.output_dir / "per_source_image_strict_outcomes.csv", image_rows)
    write_csv(args.output_dir / "per_native_prediction_outcomes.csv", native_prediction_rows)
    gt_counts = Counter(str(row["strict_gt_outcome"]) for row in instance_rows)
    result = {
        "model": args.method_name,
        "predictions_pkl": str(args.predictions_pkl),
        "native_validation_images": len(records),
        "native_validation_annotations": len(native["annotations"]),
        "source_ground_truth_instances": len(instance_rows),
        "confidence_threshold": args.score_threshold,
        "match_iou": args.match_iou,
        "partial_coverage_threshold": args.partial_coverage,
        "fragmented_coverage_threshold": args.fragmented_coverage,
        **{f"gt_{name}": gt_counts[name] for name in gt_names},
        **{f"pred_{name}": sum(prediction_counts[(category, name)] for category in categories) for name in prediction_names},
        "cross_tile_repeat_true_positive_boxes": sum(max(0, int(row["true_positive_native_matches"]) - 1) for row in instance_rows),
        "image_fully_correct": image_outcomes["fully_correct"],
        "image_detected_with_extra_error": image_outcomes["detected_with_extra_error"],
        "image_miss_or_partial": image_outcomes["miss_or_partial"],
        "image_prediction_error_only": image_outcomes["prediction_error_only"],
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "strict_model_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader(); writer.writerow(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
