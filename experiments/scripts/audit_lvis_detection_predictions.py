"""Thresholded C/W/M audit for LVIS-format detection predictions.

This script is deliberately separate from LVIS AP.  It audits a finite set of
predictions at a disclosed score threshold, assigning every ground-truth
instance exactly one outcome:

* correct: one-to-one IoU match (>= threshold) with the same category;
* wrong_class: no same-class match, but a one-to-one overlapping prediction
  of another category;
* miss: neither condition holds.

Same-class matches are resolved first, so an overlapping wrong-class candidate
cannot consume a ground-truth instance that has a valid correct candidate.
Predictions must be a standard LVIS/COCO result JSON list with ``image_id``,
``category_id``, ``bbox`` (xywh), and ``score``.  Category IDs are read from
the supplied annotation file; no category remapping is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="LVIS pilot validation annotation JSON.")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Standard LVIS/COCO detection result JSON list.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="detector")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def xywh_iou(left: Iterable[float], right: Iterable[float]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2 = min(lx + max(0.0, lw), rx + max(0.0, rw))
    iy2 = min(ly + max(0.0, lh), ry + max(0.0, rh))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, lw) * max(0.0, lh) + max(0.0, rw) * max(0.0, rh) - inter
    return inter / union if union > 0.0 else 0.0


def empty_counts() -> Counter:
    return Counter(correct=0, wrong_class=0, miss=0)


def as_summary(counts: Counter) -> dict[str, Any]:
    total = int(counts["correct"] + counts["wrong_class"] + counts["miss"])
    return {
        "correct": int(counts["correct"]),
        "wrong_class": int(counts["wrong_class"]),
        "miss": int(counts["miss"]),
        "instances": total,
        "correct_rate": (float(counts["correct"]) / total if total else None),
        "triple": f"{counts['correct']} / {counts['wrong_class']} / {counts['miss']}",
    }


def choose_best(candidates: list[tuple[float, int]], used: set[int]) -> int | None:
    for _, index in sorted(candidates, reverse=True):
        if index not in used:
            return index
    return None


def audit_image(
    ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], match_iou: float
) -> dict[int, str]:
    """Return one outcome per GT index with class-aware matching first."""
    outcomes: dict[int, str] = {index: "miss" for index in range(len(ground_truth))}
    used_gt: set[int] = set()
    ranked = sorted(predictions, key=lambda row: float(row["score"]), reverse=True)

    # Pass 1: reserve all valid same-class detections for their GT instances.
    for prediction in ranked:
        compatible = [
            (xywh_iou(prediction["bbox"], truth["bbox"]), index)
            for index, truth in enumerate(ground_truth)
            if int(prediction["category_id"]) == int(truth["category_id"])
            and xywh_iou(prediction["bbox"], truth["bbox"]) >= match_iou
        ]
        target = choose_best(compatible, used_gt)
        if target is not None:
            used_gt.add(target)
            outcomes[target] = "correct"

    # Pass 2: only unmatched GT can become wrong-class.  A candidate of the
    # right label is not allowed to turn a localization miss into wrong-class.
    for prediction in ranked:
        compatible = [
            (xywh_iou(prediction["bbox"], truth["bbox"]), index)
            for index, truth in enumerate(ground_truth)
            if index not in used_gt
            and int(prediction["category_id"]) != int(truth["category_id"])
            and xywh_iou(prediction["bbox"], truth["bbox"]) >= match_iou
        ]
        target = choose_best(compatible, used_gt)
        if target is not None:
            used_gt.add(target)
            outcomes[target] = "wrong_class"
    return outcomes


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    source = read_json(args.annotations)
    predictions = read_json(args.predictions)
    if not isinstance(predictions, list):
        raise TypeError("--predictions must contain a JSON list of LVIS/COCO detections")
    categories = {int(row["id"]): row for row in source["categories"]}
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in source["annotations"]:
        if int(annotation["category_id"]) not in categories:
            raise ValueError(f"Unknown GT category ID: {annotation['category_id']}")
        gt_by_image[int(annotation["image_id"])].append(annotation)
    selected_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        required = {"image_id", "category_id", "bbox", "score"}
        if not required.issubset(prediction):
            raise ValueError(f"Prediction lacks required keys: {prediction}")
        if int(prediction["category_id"]) not in categories:
            raise ValueError(f"Unknown prediction category ID: {prediction['category_id']}")
        if float(prediction["score"]) >= args.score_threshold:
            selected_by_image[int(prediction["image_id"])].append(prediction)

    overall, rare, common, frequent = empty_counts(), empty_counts(), empty_counts(), empty_counts()
    by_category: dict[int, Counter] = {category_id: empty_counts() for category_id in categories}
    outcome_rows: list[dict[str, Any]] = []
    for image_id, ground_truth in gt_by_image.items():
        outcomes = audit_image(ground_truth, selected_by_image.get(image_id, []), args.match_iou)
        for index, annotation in enumerate(ground_truth):
            category_id = int(annotation["category_id"])
            frequency = str(categories[category_id].get("frequency", "unknown"))
            outcome = outcomes[index]
            overall[outcome] += 1
            by_category[category_id][outcome] += 1
            if frequency == "r":
                rare[outcome] += 1
            elif frequency == "c":
                common[outcome] += 1
            elif frequency == "f":
                frequent[outcome] += 1
            outcome_rows.append({
                "image_id": image_id,
                "annotation_id": int(annotation["id"]),
                "category_id": category_id,
                "category_name": str(categories[category_id]["name"]),
                "frequency": frequency,
                "outcome": outcome,
            })

    category_rows = []
    for category_id, counts in sorted(by_category.items()):
        if not sum(counts.values()):
            continue
        category_rows.append({
            "category_id": category_id,
            "category_name": str(categories[category_id]["name"]),
            "frequency": str(categories[category_id].get("frequency", "unknown")),
            **as_summary(counts),
        })
    report = {
        "protocol": {
            "method_name": args.method_name,
            "annotations": str(args.annotations.resolve()),
            "predictions": str(args.predictions.resolve()),
            "score_threshold": args.score_threshold,
            "match_iou": args.match_iou,
            "definition": {
                "correct": "one-to-one same-category match at IoU >= match_iou",
                "wrong_class": "unmatched GT with one-to-one overlapping different-category prediction",
                "miss": "remaining unmatched GT",
                "note": "This thresholded C/W/M audit is not LVIS AP.",
            },
        },
        "selected_predictions": sum(len(rows) for rows in selected_by_image.values()),
        "all": as_summary(overall),
        "rare": as_summary(rare),
        "common": as_summary(common),
        "frequent": as_summary(frequent),
        "per_category": category_rows,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_category.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(category_rows[0]) if category_rows else ["category_id"])
        writer.writeheader()
        writer.writerows(category_rows)
    with (args.output_dir / "per_instance_outcomes.json").open("w", encoding="utf-8") as handle:
        json.dump(outcome_rows, handle, ensure_ascii=False)
    print(json.dumps({"method": args.method_name, "all": report["all"], "rare": report["rare"],
                      "selected_predictions": report["selected_predictions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
