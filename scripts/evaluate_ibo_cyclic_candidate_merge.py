#!/usr/bin/env python
"""Cyclic/global-coordinate merge experiment for IBO candidates.

The previous reliability post-processing kept tile-level candidates.  This
script maps them back to the unwrapped weld-strip coordinate system and merges
duplicate detections from overlapping cyclic tiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibo-manifest", type=Path, required=True)
    parser.add_argument("--reliability-scores", type=Path, required=True)
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--strip-boxes", type=Path, required=True)
    parser.add_argument("--tiles-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reliability-threshold", type=float, default=0.45)
    parser.add_argument("--protect-threshold", type=float, default=0.25)
    parser.add_argument("--merge-iou", type=float, action="append", default=[])
    parser.add_argument("--same-class-only", action="store_true")
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--exclude-class-id", type=int, action="append", default=[])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def xyxy_from_xywh(values: list[float]) -> np.ndarray:
    x, y, w, h = (float(v) for v in values)
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


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


def y_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    y0 = max(float(a[1]), float(b[1]))
    y1 = min(float(a[3]), float(b[3]))
    overlap = max(0.0, y1 - y0)
    return overlap / max(1e-6, min(float(a[3] - a[1]), float(b[3] - b[1])))


def best_shifted_box(box: np.ndarray, target: np.ndarray, strip_width: float) -> np.ndarray:
    candidates = [box + np.asarray([shift, 0, shift, 0], dtype=np.float32) for shift in (-strip_width, 0.0, strip_width)]
    target_center = (target[0] + target[2]) / 2.0
    return min(candidates, key=lambda item: abs(float((item[0] + item[2]) / 2.0 - target_center)))


def pair_should_merge(a: dict[str, Any], b: dict[str, Any], merge_iou: float, same_class_only: bool) -> bool:
    if int(a["source_image_id"]) != int(b["source_image_id"]):
        return False
    if same_class_only and int(a["class_id"]) != int(b["class_id"]):
        return False
    width = float(a["strip_width"])
    box_a = np.asarray(a["global_box"], dtype=np.float32)
    box_b = best_shifted_box(np.asarray(b["global_box"], dtype=np.float32), box_a, width)
    value_iou = iou(box_a, box_b)
    if value_iou >= merge_iou:
        return True
    # Long defects may be fragmented.  If two boxes overlap along x and have
    # similar y support, allow a weaker spatial merge.  This is conservative:
    # it still requires the boxes to touch/overlap in global strip space.
    x_gap = max(0.0, max(float(box_a[0]), float(box_b[0])) - min(float(box_a[2]), float(box_b[2])))
    x_overlap = min(float(box_a[2]), float(box_b[2])) - max(float(box_a[0]), float(box_b[0]))
    if x_overlap > 0.0 and y_overlap_ratio(box_a, box_b) >= 0.35 and value_iou >= max(0.05, merge_iou * 0.35):
        return True
    if x_gap <= 16.0 and y_overlap_ratio(box_a, box_b) >= 0.60 and (not same_class_only or int(a["class_id"]) == int(b["class_id"])):
        return True
    return False


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def rows_with_scores(manifest_path: Path, scores_path: Path, tiles_path: Path, exclude_class_ids: set[int]) -> list[dict[str, Any]]:
    scores = {row["ibo_id"]: float(row["reliability_score"]) for row in read_csv(scores_path)}
    tiles = {int(row["tile_image_id"]): row for row in load_json(tiles_path)}
    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(manifest_path):
        if raw["ibo_id"] not in scores:
            continue
        class_id = int(raw["pred_class_id"])
        if class_id in exclude_class_ids:
            continue
        tile_id = int(raw["tile_image_id"])
        tile = tiles.get(tile_id)
        if not tile:
            continue
        pred = np.asarray([float(v) for v in raw["pred_xyxy"]], dtype=np.float32)
        strip_width = float(tile["strip_width"])
        global_box = np.asarray(
            [
                float(tile["tile_start"]) + float(pred[0]),
                float(pred[1]) - float(tile.get("y_pad", 0)),
                float(tile["tile_start"]) + float(pred[2]),
                float(pred[3]) - float(tile.get("y_pad", 0)),
            ],
            dtype=np.float32,
        )
        row = {
            "ibo_id": raw["ibo_id"],
            "source_image_id": int(raw["source_image_id"]),
            "source_file": raw.get("source_file", ""),
            "tile_image_id": tile_id,
            "tile_file": raw["tile_file"],
            "class_id": class_id,
            "class_name": raw["pred_class_name"],
            "pred_score": float(raw["pred_score"]),
            "reliability_score": float(scores[raw["ibo_id"]]),
            "strip_width": strip_width,
            "strip_height": float(tile.get("strip_height", 150)),
            "global_box": global_box,
        }
        rows.append(row)
    return rows


def merge_candidates(candidates: list[dict[str, Any]], merge_iou: float, same_class_only: bool) -> list[dict[str, Any]]:
    grouped_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped_by_source[int(item["source_image_id"])].append(item)

    merged: list[dict[str, Any]] = []
    next_id = 1
    for source_id, items in grouped_by_source.items():
        uf = UnionFind(len(items))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if pair_should_merge(items[i], items[j], merge_iou, same_class_only):
                    uf.union(i, j)
        components: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for i, item in enumerate(items):
            components[uf.find(i)].append(item)
        for comp in components.values():
            width = float(comp[0]["strip_width"])
            anchor = np.asarray(comp[0]["global_box"], dtype=np.float32)
            aligned_boxes: list[np.ndarray] = []
            weights: list[float] = []
            class_votes: Counter[int] = Counter()
            class_names: dict[int, str] = {}
            for item in comp:
                box = best_shifted_box(np.asarray(item["global_box"], dtype=np.float32), anchor, width)
                weight = max(1e-6, float(item["reliability_score"]) * float(item["pred_score"]))
                aligned_boxes.append(box)
                weights.append(weight)
                cid = int(item["class_id"])
                class_votes[cid] += weight
                class_names[cid] = str(item["class_name"])
            weights_np = np.asarray(weights, dtype=np.float32)
            boxes_np = np.stack(aligned_boxes)
            weighted_box = np.average(boxes_np, axis=0, weights=weights_np)
            union_box = np.asarray([boxes_np[:, 0].min(), boxes_np[:, 1].min(), boxes_np[:, 2].max(), boxes_np[:, 3].max()], dtype=np.float32)
            # Weighted box is more stable for compact duplicate detections; union
            # box is safer for long fragmented defects.  Use a hybrid that keeps
            # the weighted center but at least the weighted median-like extent.
            final_box = union_box if len(comp) >= 3 else weighted_box.astype(np.float32)
            box_width = float(final_box[2] - final_box[0])
            center = float((final_box[0] + final_box[2]) / 2.0) % width
            final_box = np.asarray([center - box_width / 2.0, final_box[1], center + box_width / 2.0, final_box[3]], dtype=np.float32)
            cid = int(max(class_votes.items(), key=lambda kv: kv[1])[0])
            merged.append(
                {
                    "merged_id": f"merge_{next_id:07d}",
                    "source_image_id": source_id,
                    "source_file": comp[0].get("source_file", ""),
                    "class_id": cid,
                    "class_name": class_names.get(cid, f"class_{cid}"),
                    "strip_width": width,
                    "global_box": final_box,
                    "member_count": len(comp),
                    "member_ids": ",".join(str(item["ibo_id"]) for item in comp),
                    "max_reliability": max(float(item["reliability_score"]) for item in comp),
                    "mean_reliability": float(np.mean([float(item["reliability_score"]) for item in comp])),
                    "max_pred_score": max(float(item["pred_score"]) for item in comp),
                    "class_vote_margin": float(
                        (sorted(class_votes.values(), reverse=True)[0] - (sorted(class_votes.values(), reverse=True)[1] if len(class_votes) > 1 else 0.0))
                        / max(1e-6, sum(class_votes.values()))
                    ),
                }
            )
            next_id += 1
    return merged


def coverage_union_against_gt(gt_box: np.ndarray, boxes: list[np.ndarray], strip_width: float) -> float:
    shifted = [best_shifted_box(box, gt_box, strip_width) for box in boxes]
    events: list[tuple[float, int, float, float]] = []
    for box in shifted:
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
    prev_x: float | None = None
    area = 0.0
    for x, flag, y0, y1 in events:
        if prev_x is not None and x > prev_x and active:
            intervals = sorted(active)
            start, end = intervals[0]
            total = 0.0
            for ys, ye in intervals[1:]:
                if ys <= end:
                    end = max(end, ye)
                else:
                    total += end - start
                    start, end = ys, ye
            total += end - start
            area += (x - prev_x) * total
        if flag > 0:
            active.append((y0, y1))
        else:
            try:
                active.remove((y0, y1))
            except ValueError:
                pass
        prev_x = x
    return min(1.0, area / max(1e-6, box_area(gt_box)))


def evaluate_global(
    candidates: list[dict[str, Any]],
    source_data: dict[str, Any],
    strip_boxes: list[dict[str, Any]],
    exclude_class_ids: set[int],
    coverage_threshold: float,
    match_iou: float,
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_items = {int(item["source_image_id"]): item for item in strip_boxes}
    source_images = {int(item["id"]): item for item in source_data.get("images", [])}
    category_names = {int(item["id"]): str(item["name"]) for item in source_data.get("categories", [])}
    strip_gt_by_source_annotation: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for item in strip_boxes:
        for box in item["boxes"]:
            strip_gt_by_source_annotation[int(box["source_annotation_id"])] = (item, box)
    gt_details: list[dict[str, Any]] = []
    categories: dict[int, str] = dict(category_names)
    for annotation in source_data.get("annotations", []):
        cid = int(annotation["category_id"])
        if cid in exclude_class_ids:
            continue
        source_annotation_id = int(annotation["id"])
        source_image_id = int(annotation["image_id"])
        source_file = source_images.get(source_image_id, {}).get("file_name", "")
        represented = source_annotation_id in strip_gt_by_source_annotation
        if represented:
            item, box = strip_gt_by_source_annotation[source_annotation_id]
            gt_box = xyxy_from_xywh([box["x"], box["y"], box["width"], box["height"]])
            strip_width = float(item["strip_width"])
            source_file = item.get("source_file", source_file)
        else:
            gt_box = None
            strip_width = float("nan")
        gt_details.append(
            {
                "source_image_id": source_image_id,
                "source_file": source_file,
                "strip_width": strip_width,
                "source_annotation_id": source_annotation_id,
                "class_id": cid,
                "gt_box": gt_box,
                "represented_in_strip": represented,
            }
        )
    # Pull readable class names from predictions where available.
    for cand in candidates:
        categories[int(cand["class_id"])] = str(cand["class_name"])

    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        by_source[int(cand["source_image_id"])].append(cand)

    instance_rows: list[dict[str, Any]] = []
    class_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for gt in gt_details:
        if gt["gt_box"] is None:
            outcome = "miss"
            same_cov = same_iou = other_cov = other_iou = 0.0
            class_counts[int(gt["class_id"])][outcome] += 1
            instance_rows.append(
                {
                    "method": method,
                    "source_annotation_id": gt["source_annotation_id"],
                    "source_image_id": gt["source_image_id"],
                    "source_file": gt["source_file"],
                    "class_id": gt["class_id"],
                    "class_name": categories.get(int(gt["class_id"]), f"class_{gt['class_id']}"),
                    "outcome": outcome,
                    "max_same_coverage": round(float(same_cov), 6),
                    "max_same_iou": round(float(same_iou), 6),
                    "max_other_coverage": round(float(other_cov), 6),
                    "max_other_iou": round(float(other_iou), 6),
                    "represented_in_strip": False,
                }
            )
            continue
        cands = by_source.get(int(gt["source_image_id"]), [])
        same_boxes = [np.asarray(c["global_box"], dtype=np.float32) for c in cands if int(c["class_id"]) == int(gt["class_id"])]
        other_boxes = [np.asarray(c["global_box"], dtype=np.float32) for c in cands if int(c["class_id"]) != int(gt["class_id"])]
        same_cov = coverage_union_against_gt(gt["gt_box"], same_boxes, float(gt["strip_width"]))
        same_iou = max((iou(gt["gt_box"], best_shifted_box(box, gt["gt_box"], float(gt["strip_width"]))) for box in same_boxes), default=0.0)
        other_cov = coverage_union_against_gt(gt["gt_box"], other_boxes, float(gt["strip_width"]))
        other_iou = max((iou(gt["gt_box"], best_shifted_box(box, gt["gt_box"], float(gt["strip_width"]))) for box in other_boxes), default=0.0)
        if same_cov >= coverage_threshold:
            outcome = "correct"
        elif other_cov >= coverage_threshold or other_iou >= match_iou:
            outcome = "wrong"
        else:
            outcome = "miss"
        class_counts[int(gt["class_id"])][outcome] += 1
        instance_rows.append(
            {
                "method": method,
                "source_annotation_id": gt["source_annotation_id"],
                "source_image_id": gt["source_image_id"],
                "source_file": gt["source_file"],
                "class_id": gt["class_id"],
                "class_name": categories.get(int(gt["class_id"]), f"class_{gt['class_id']}"),
                "outcome": outcome,
                "max_same_coverage": round(float(same_cov), 6),
                "max_same_iou": round(float(same_iou), 6),
                "max_other_coverage": round(float(other_cov), 6),
                "max_other_iou": round(float(other_iou), 6),
                "represented_in_strip": True,
            }
        )

    # Candidate-level false positives after merge.
    fp_by_source: Counter[int] = Counter()
    for cand in candidates:
        source = source_items.get(int(cand["source_image_id"]))
        if not source:
            continue
        gt_boxes = [
            xyxy_from_xywh([box["x"], box["y"], box["width"], box["height"]])
            for box in source["boxes"]
            if int(box["category_id"]) not in exclude_class_ids
        ]
        width = float(source["strip_width"])
        cand_box = np.asarray(cand["global_box"], dtype=np.float32)
        overlaps = [
            iou(gt_box, best_shifted_box(cand_box, gt_box, width)) > 0.0
            or intersection_area(gt_box, best_shifted_box(cand_box, gt_box, width)) / max(1e-6, box_area(gt_box)) >= coverage_threshold
            for gt_box in gt_boxes
        ]
        if not any(overlaps):
            fp_by_source[int(cand["source_image_id"])] += 1

    details_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        details_by_image[int(row["source_image_id"])].append(row)
    image_rows: list[dict[str, Any]] = []
    image_counts = Counter()
    for source_image_id, details in details_by_image.items():
        if any(row["outcome"] == "miss" for row in details):
            outcome = "miss"
        elif any(row["outcome"] == "wrong" for row in details) or fp_by_source[source_image_id] > 0:
            outcome = "wrong"
        else:
            outcome = "correct"
        image_counts[outcome] += 1
        image_rows.append(
            {
                "method": method,
                "source_image_id": source_image_id,
                "source_file": details[0]["source_file"],
                "gt_instances": len(details),
                "false_positive_merged_candidates": fp_by_source[source_image_id],
                "outcome": outcome,
            }
        )

    class_rows: list[dict[str, Any]] = []
    for cid in sorted(class_counts):
        counts = class_counts[cid]
        total = counts["correct"] + counts["wrong"] + counts["miss"]
        class_rows.append(
            {
                "method": method,
                "class_id": cid,
                "class_name": categories.get(cid, f"class_{cid}"),
                "gt_instances": total,
                "correct": counts["correct"],
                "wrong": counts["wrong"],
                "miss": counts["miss"],
                "accuracy": round(counts["correct"] / total, 6) if total else 0.0,
                "triple": f"{counts['correct']} / {counts['wrong']} / {counts['miss']}",
            }
        )
    total = sum(row["gt_instances"] for row in class_rows)
    correct = sum(row["correct"] for row in class_rows)
    wrong = sum(row["wrong"] for row in class_rows)
    miss = sum(row["miss"] for row in class_rows)
    image_total = sum(image_counts.values())
    summary = {
        "method": method,
        "merged_candidates": len(candidates),
        "instances": total,
        "correct": correct,
        "wrong": wrong,
        "miss": miss,
        "instance_accuracy": round(correct / total, 6) if total else 0.0,
        "source_images": image_total,
        "image_correct": image_counts["correct"],
        "image_wrong": image_counts["wrong"],
        "image_miss": image_counts["miss"],
        "image_accuracy": round(image_counts["correct"] / image_total, 6) if image_total else 0.0,
    }
    return summary, class_rows, image_rows + instance_rows


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclude = set(int(v) for v in args.exclude_class_id)
    merge_ious = args.merge_iou or [0.30, 0.40, 0.50]
    all_rows = rows_with_scores(args.ibo_manifest, args.reliability_scores, args.tiles_metadata, exclude)
    source_data = load_json(args.source_annotations)
    strip_boxes = load_json(args.strip_boxes)

    selected_main = [row for row in all_rows if float(row["reliability_score"]) >= args.reliability_threshold]
    selected_protect = [row for row in all_rows if float(row["reliability_score"]) >= args.protect_threshold]
    print(f"[CYCLIC-MERGE] candidates all={len(all_rows)} main={len(selected_main)} protect={len(selected_protect)}", flush=True)

    summaries: list[dict[str, Any]] = []
    class_rows_all: list[dict[str, Any]] = []
    detail_rows_all: list[dict[str, Any]] = []
    merged_rows_all: list[dict[str, Any]] = []

    base_summary, base_classes, base_details = evaluate_global(
        selected_main, source_data, strip_boxes, exclude, args.coverage_threshold, args.match_iou, f"pre_merge_rel_ge_{args.reliability_threshold:.2f}"
    )
    summaries.append(base_summary)
    class_rows_all.extend(base_classes)
    detail_rows_all.extend(base_details)

    best_summary = base_summary
    best_classes = base_classes
    best_method = base_summary["method"]
    for merge_iou in merge_ious:
        for source_name, selected in [("main", selected_main), ("protect", selected_protect)]:
            for same_class_only in [True, False]:
                mode = "sameclass" if same_class_only else "classvote"
                method = f"merge_{source_name}_rel{args.reliability_threshold:.2f}_p{args.protect_threshold:.2f}_iou{merge_iou:.2f}_{mode}"
                merged = merge_candidates(selected, merge_iou, same_class_only)
                summary, class_rows, detail_rows = evaluate_global(
                    merged, source_data, strip_boxes, exclude, args.coverage_threshold, args.match_iou, method
                )
                summaries.append(summary)
                class_rows_all.extend(class_rows)
                detail_rows_all.extend(detail_rows)
                for row in merged:
                    packed = dict(row)
                    packed["method"] = method
                    packed["global_box"] = [round(float(v), 4) for v in np.asarray(row["global_box"]).tolist()]
                    merged_rows_all.append(packed)
                print(
                    f"[CYCLIC-MERGE] {method}: candidates={summary['merged_candidates']} "
                    f"inst={summary['correct']}/{summary['wrong']}/{summary['miss']} "
                    f"img={summary['image_correct']}/{summary['image_wrong']}/{summary['image_miss']}",
                    flush=True,
                )
                if (
                    summary["correct"] >= 1000
                    and summary["miss"] <= 40
                    and summary["merged_candidates"] < best_summary["merged_candidates"]
                ) or (
                    summary["correct"] > best_summary["correct"]
                    and summary["miss"] <= 45
                    and summary["merged_candidates"] <= 4500
                ):
                    best_summary = summary
                    best_classes = class_rows
                    best_method = method

    write_csv(args.output_dir / "cyclic_merge_summary.csv", summaries)
    write_csv(args.output_dir / "cyclic_merge_by_class.csv", class_rows_all)
    write_csv(args.output_dir / "cyclic_merge_details.csv", detail_rows_all)
    write_csv(args.output_dir / "merged_candidates_all_methods.csv", merged_rows_all)

    # Compact curve: candidate count vs accuracy for merge methods.
    try:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
        xs = [int(row["merged_candidates"]) for row in summaries]
        ys = [float(row["instance_accuracy"]) for row in summaries]
        colors = ["#d95f02" if row["method"] == best_method else "#1b9e77" for row in summaries]
        ax.scatter(xs, ys, c=colors)
        for row in summaries:
            label = row["method"].replace("merge_", "").replace("pre_merge_", "pre_")
            ax.annotate(label, (int(row["merged_candidates"]), float(row["instance_accuracy"])), fontsize=6, alpha=0.75)
        ax.set_xlabel("Candidate / merged group count")
        ax.set_ylabel("Instance accuracy")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "cyclic_merge_tradeoff.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[CYCLIC-MERGE] plot skipped: {exc}", flush=True)

    baseline = summaries[0]
    lines = [
        "# IBO 跨滑窗候选归并实验报告 v1",
        "",
        "## Material Passport",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        "| 实验 | IBO reliability 候选框映射到展开焊道全局坐标后的跨滑窗归并 |",
        f"| 主阈值 | reliability ≥ {args.reliability_threshold:.2f} |",
        f"| 召回保护阈值 | reliability ≥ {args.protect_threshold:.2f} |",
        "| 数据 | 验证集，排除钢帽 |",
        f"| 输出目录 | `{args.output_dir}` |",
        "",
        "## 总体结果",
        "",
        "| 方法 | 候选/归并框数 | 正确 / 错误 / 漏检 | 实例准确率 | 图片完全正确 / 错误 / 漏检 | 图片准确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        star = " **推荐**" if row["method"] == best_method else ""
        lines.append(
            f"| {row['method']}{star} | {row['merged_candidates']} | {row['correct']} / {row['wrong']} / {row['miss']} | "
            f"{fmt_pct(float(row['instance_accuracy']))} | {row['image_correct']} / {row['image_wrong']} / {row['image_miss']} | {fmt_pct(float(row['image_accuracy']))} |"
        )
    base_by_class = {int(row["class_id"]): row for row in base_classes}
    best_by_class = {int(row["class_id"]): row for row in best_classes}
    lines.extend(
        [
            "",
            "## 推荐参数类别级对比",
            "",
            "| 类别 | pre-merge 0.45 | 归并后推荐 | 正确实例变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for cid in sorted(best_by_class):
        b = base_by_class.get(cid)
        r = best_by_class[cid]
        if not b:
            continue
        lines.append(
            f"| {r['class_name']} | {b['triple']}<br>{fmt_pct(float(b['accuracy']))} | "
            f"{r['triple']}<br>{fmt_pct(float(r['accuracy']))} | {int(r['correct']) - int(b['correct']):+d} |"
        )
    lines.extend(
        [
            "",
            "## 阶段判断",
            "",
            "- 这一阶段只做坐标回投和候选归并，不重新训练 D-FINE，也不引入验证集人工框作为候选来源。",
            "- 如果推荐参数能把候选数压低且漏检基本不反弹，说明跨滑窗重复是主要冗余来源。",
            "- 如果候选数下降但正确实例明显下降，说明当前归并太激进，需要降低 merge 强度或引入正常参照/频率证据后再筛。",
            "",
            "## 输出文件",
            "",
            "- `cyclic_merge_summary.csv`：所有归并参数总体指标。",
            "- `cyclic_merge_by_class.csv`：每类实例级统计。",
            "- `cyclic_merge_details.csv`：图片级和实例级明细。",
            "- `merged_candidates_all_methods.csv`：各参数下的归并候选框。",
            "- `cyclic_merge_tradeoff.png`：候选数与实例准确率权衡图。",
        ]
    )
    (args.output_dir / "IBO跨滑窗候选归并实验报告_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[CYCLIC-MERGE] done best={best_method} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
