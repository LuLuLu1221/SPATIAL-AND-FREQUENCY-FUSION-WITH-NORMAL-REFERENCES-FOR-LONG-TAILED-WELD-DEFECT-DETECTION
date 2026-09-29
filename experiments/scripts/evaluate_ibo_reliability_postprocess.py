#!/usr/bin/env python
"""Evaluate reliability-score post-processing for IBO candidates on validation.

This script projects candidate-level IBO reliability scores back to source GT
instances and reports the same "correct / wrong / missed" instance-level
statistics used in the D-FINE audits.
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
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--tile-annotations", type=Path, required=True)
    parser.add_argument("--ibo-manifest", type=Path, required=True)
    parser.add_argument("--reliability-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-class-id", type=int, action="append", default=[])
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument(
        "--match-family",
        action="append",
        default=[],
        help="Comma-separated class ids treated as one class for matching, e.g. 12,14,15. May be repeated.",
    )
    parser.add_argument("--dfine-baseline-threshold", type=float, default=0.40)
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


def parse_match_families(values: list[str]) -> list[set[int]]:
    families: list[set[int]] = []
    for value in values:
        family = {int(part.strip()) for part in value.split(",") if part.strip()}
        if len(family) >= 2:
            families.append(family)
    return families


def class_matches(gt_class: int, pred_class: int, families: list[set[int]]) -> bool:
    if gt_class == pred_class:
        return True
    return any(gt_class in family and pred_class in family for family in families)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_annotation_id(annotation: dict[str, Any]) -> int:
    return int(annotation.get("source_annotation_id", annotation["id"]))


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


def union_coverage(gt_box: np.ndarray, pred_boxes: list[np.ndarray]) -> float:
    if not pred_boxes:
        return 0.0
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
            start, end = intervals[0]
            for y_start, y_end in intervals[1:]:
                if y_start <= end:
                    end = max(end, y_end)
                else:
                    merged += end - start
                    start, end = y_start, y_end
            merged += end - start
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


def parse_box(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray([float(v) for v in value], dtype=np.float32)


def prepare_rows(manifest_path: Path, score_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    score_rows = read_csv(score_path)
    if not score_rows:
        raise ValueError(f"No reliability scores found in {score_path}")
    score_column = "reliability_score"
    if score_column not in score_rows[0]:
        score_column = "reliability_fusion_score"
    if score_column not in score_rows[0]:
        raise KeyError(
            "Expected reliability_score or reliability_fusion_score in "
            f"{score_path}; found {sorted(score_rows[0])}"
        )
    score_by_id = {row["ibo_id"]: float(row[score_column]) for row in score_rows}
    merged: list[dict[str, Any]] = []
    for row in rows:
        if row["ibo_id"] not in score_by_id:
            continue
        item = dict(row)
        item["reliability_score"] = score_by_id[row["ibo_id"]]
        item["pred_score"] = float(item["pred_score"])
        item["pred_class_id"] = int(item["pred_class_id"])
        item["tile_image_id"] = int(item["tile_image_id"])
        item["source_image_id"] = int(item["source_image_id"])
        item["pred_box"] = parse_box(item["pred_xyxy"])
        merged.append(item)
    return merged


def evaluate_selection(
    selected: list[dict[str, Any]],
    source_data: dict[str, Any],
    tile_data: dict[str, Any],
    exclude_class_ids: set[int],
    coverage_threshold: float,
    match_iou: float,
    match_families: list[set[int]],
    method: str,
    threshold: float | str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    categories = {int(item["id"]): item for item in source_data["categories"]}
    source_images = {int(item["id"]): item for item in source_data["images"]}

    source_details: dict[int, dict[str, Any]] = {}
    for annotation in source_data["annotations"]:
        class_id = int(annotation["category_id"])
        if class_id in exclude_class_ids:
            continue
        sid = int(annotation["id"])
        source_details[sid] = {
            "source_annotation_id": sid,
            "source_image_id": int(annotation["image_id"]),
            "source_file": source_images.get(int(annotation["image_id"]), {}).get("file_name", ""),
            "class_id": class_id,
            "class_name": str(categories[class_id]["name"]),
            "class_status": str(categories[class_id].get("status", "")),
            "represented_in_tiles": False,
            "native_occurrences": 0,
            "max_same_coverage": 0.0,
            "max_same_iou": 0.0,
            "wrong_class_ids": set(),
        }

    tile_gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in tile_data["annotations"]:
        class_id = int(annotation["category_id"])
        if class_id in exclude_class_ids:
            continue
        tile_gt_by_image[int(annotation["image_id"])].append(annotation)
        sid = source_annotation_id(annotation)
        if sid in source_details:
            source_details[sid]["represented_in_tiles"] = True
            source_details[sid]["native_occurrences"] += 1

    selected_by_tile: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        if int(row["pred_class_id"]) in exclude_class_ids:
            continue
        selected_by_tile[int(row["tile_image_id"])].append(row)

    false_positive_by_source_image: Counter[int] = Counter()
    selected_candidate_count = 0
    for tile_image_id, candidates in selected_by_tile.items():
        ground_truth = tile_gt_by_image.get(tile_image_id, [])
        selected_candidate_count += len(candidates)
        for candidate in candidates:
            pred_box = candidate["pred_box"]
            pred_class = int(candidate["pred_class_id"])
            matched_any = False
            for gt in ground_truth:
                gt_box = xyxy_from_xywh(gt["bbox"])
                sid = source_annotation_id(gt)
                if sid not in source_details:
                    continue
                cov = intersection_area(gt_box, pred_box) / max(1e-6, box_area(gt_box))
                item_iou = iou(gt_box, pred_box)
                if cov > 0.0 or item_iou > 0.0:
                    matched_any = True
                if class_matches(int(gt["category_id"]), pred_class, match_families):
                    source_details[sid]["max_same_coverage"] = max(source_details[sid]["max_same_coverage"], cov)
                    source_details[sid]["max_same_iou"] = max(source_details[sid]["max_same_iou"], item_iou)
                elif cov >= coverage_threshold or item_iou >= match_iou:
                    source_details[sid]["wrong_class_ids"].add(pred_class)
            if not matched_any:
                false_positive_by_source_image[int(candidate["source_image_id"])] += 1

    instance_rows: list[dict[str, Any]] = []
    class_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for detail in source_details.values():
        if detail["max_same_coverage"] >= coverage_threshold:
            outcome = "correct"
        elif detail["wrong_class_ids"]:
            outcome = "wrong"
        else:
            outcome = "miss"
        detail["outcome"] = outcome
        class_counts[int(detail["class_id"])][outcome] += 1
        instance_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "source_annotation_id": detail["source_annotation_id"],
                "source_image_id": detail["source_image_id"],
                "source_file": detail["source_file"],
                "class_id": detail["class_id"],
                "class_name": detail["class_name"],
                "class_status": detail["class_status"],
                "outcome": outcome,
                "max_same_coverage": round(float(detail["max_same_coverage"]), 6),
                "max_same_iou": round(float(detail["max_same_iou"]), 6),
                "wrong_class_ids": ",".join(str(v) for v in sorted(detail["wrong_class_ids"])),
                "represented_in_tiles": bool(detail["represented_in_tiles"]),
                "native_occurrences": detail["native_occurrences"],
            }
        )

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for detail in source_details.values():
        by_image[int(detail["source_image_id"])].append(detail)
    image_counts = Counter()
    image_rows: list[dict[str, Any]] = []
    for source_image_id, details in by_image.items():
        if any(detail["outcome"] == "miss" for detail in details):
            outcome = "miss"
        elif any(detail["outcome"] == "wrong" for detail in details) or false_positive_by_source_image[source_image_id] > 0:
            outcome = "wrong"
        else:
            outcome = "correct"
        image_counts[outcome] += 1
        image_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "source_image_id": source_image_id,
                "source_file": source_images.get(source_image_id, {}).get("file_name", ""),
                "gt_instances": len(details),
                "false_positive_candidates": false_positive_by_source_image[source_image_id],
                "outcome": outcome,
            }
        )

    class_rows: list[dict[str, Any]] = []
    non_tail = Counter()
    tail = Counter()
    for class_id in sorted(class_counts):
        counts = class_counts[class_id]
        total = counts["correct"] + counts["wrong"] + counts["miss"]
        status = str(categories[class_id].get("status", ""))
        group_counter = tail if status in {"tail_candidate", "insufficient"} else non_tail
        group_counter.update(counts)
        class_rows.append(
            {
                "method": method,
                "threshold": threshold,
                "class_id": class_id,
                "class_name": str(categories[class_id]["name"]),
                "class_status": status,
                "gt_instances": total,
                "correct": counts["correct"],
                "wrong": counts["wrong"],
                "miss": counts["miss"],
                "accuracy": round(counts["correct"] / total, 6) if total else 0.0,
                "triple": f"{counts['correct']} / {counts['wrong']} / {counts['miss']}",
            }
        )

    total_instances = sum(row["gt_instances"] for row in class_rows)
    total_correct = sum(row["correct"] for row in class_rows)
    total_wrong = sum(row["wrong"] for row in class_rows)
    total_miss = sum(row["miss"] for row in class_rows)
    non_tail_total = non_tail["correct"] + non_tail["wrong"] + non_tail["miss"]
    tail_total = tail["correct"] + tail["wrong"] + tail["miss"]
    total_images = sum(image_counts.values())
    summary = {
        "method": method,
        "threshold": threshold,
        "selected_candidates": selected_candidate_count,
        "source_images": total_images,
        "image_correct": image_counts["correct"],
        "image_wrong": image_counts["wrong"],
        "image_miss": image_counts["miss"],
        "image_accuracy": round(image_counts["correct"] / total_images, 6) if total_images else 0.0,
        "instances": total_instances,
        "instance_correct": total_correct,
        "instance_wrong": total_wrong,
        "instance_miss": total_miss,
        "instance_accuracy": round(total_correct / total_instances, 6) if total_instances else 0.0,
        "non_tail_instances": non_tail_total,
        "non_tail_correct": non_tail["correct"],
        "non_tail_wrong": non_tail["wrong"],
        "non_tail_miss": non_tail["miss"],
        "non_tail_accuracy": round(non_tail["correct"] / non_tail_total, 6) if non_tail_total else 0.0,
        "tail_instances": tail_total,
        "tail_correct": tail["correct"],
        "tail_wrong": tail["wrong"],
        "tail_miss": tail["miss"],
        "tail_accuracy": round(tail["correct"] / tail_total, 6) if tail_total else 0.0,
    }
    return summary, class_rows, image_rows + instance_rows


def make_thresholds(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclude_class_ids = set(int(v) for v in args.exclude_class_id)
    match_families = parse_match_families(args.match_family)

    source_data = load_json(args.source_annotations)
    tile_data = load_json(args.tile_annotations)
    rows = prepare_rows(args.ibo_manifest, args.reliability_scores)
    print(f"[IBO-EVAL] loaded candidates={len(rows)}", flush=True)

    all_summaries: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    selected_detail_rows: list[dict[str, Any]] = []

    baseline_selected = [row for row in rows if row["pred_score"] >= args.dfine_baseline_threshold]
    summary, class_rows, detail_rows = evaluate_selection(
        baseline_selected,
        source_data,
        tile_data,
        exclude_class_ids,
        args.coverage_threshold,
        args.match_iou,
        match_families,
        "dfine_score_ge_0.40",
        args.dfine_baseline_threshold,
    )
    all_summaries.append(summary)
    all_class_rows.extend(class_rows)
    selected_detail_rows.extend(detail_rows)

    thresholds = make_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)
    sweep_records: dict[float, tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for threshold in thresholds:
        selected = [row for row in rows if row["reliability_score"] >= threshold]
        result = evaluate_selection(
            selected,
            source_data,
            tile_data,
            exclude_class_ids,
            args.coverage_threshold,
            args.match_iou,
            match_families,
            "ibo_reliability",
            threshold,
        )
        sweep_records[threshold] = result
        all_summaries.append(result[0])

    reliability_summaries = [row for row in all_summaries if row["method"] == "ibo_reliability"]
    best_by_instance = max(reliability_summaries, key=lambda row: (row["instance_accuracy"], -row["instance_wrong"], -row["selected_candidates"]))
    best_by_image = max(reliability_summaries, key=lambda row: (row["image_accuracy"], row["instance_accuracy"], -row["selected_candidates"]))
    default_threshold = 0.50
    key_thresholds = sorted({float(best_by_instance["threshold"]), float(best_by_image["threshold"]), default_threshold})
    for threshold in key_thresholds:
        result = sweep_records.get(threshold)
        if result is None:
            continue
        all_class_rows.extend(result[1])
        selected_detail_rows.extend(result[2])

    write_csv(args.output_dir / "threshold_sweep_summary.csv", all_summaries)
    write_csv(args.output_dir / "instance_by_class_key_methods.csv", all_class_rows)
    write_csv(args.output_dir / "image_and_instance_details_key_methods.csv", selected_detail_rows)

    try:
        xs = [float(row["threshold"]) for row in reliability_summaries]
        acc = [float(row["instance_accuracy"]) for row in reliability_summaries]
        wrong = [int(row["instance_wrong"]) for row in reliability_summaries]
        miss = [int(row["instance_miss"]) for row in reliability_summaries]
        cand = [int(row["selected_candidates"]) for row in reliability_summaries]
        fig, ax1 = plt.subplots(figsize=(8, 4.6), dpi=160)
        ax1.plot(xs, acc, marker="o", label="instance accuracy")
        ax1.set_xlabel("Reliability threshold")
        ax1.set_ylabel("Instance accuracy")
        ax1.set_ylim(0, 1)
        ax2 = ax1.twinx()
        ax2.plot(xs, wrong, color="#d95f02", marker="s", label="wrong")
        ax2.plot(xs, miss, color="#7570b3", marker="^", label="miss")
        ax2.plot(xs, [v / max(cand) * max(max(wrong), max(miss), 1) for v in cand], color="#1b9e77", linestyle="--", label="candidates scaled")
        ax2.set_ylabel("Count")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
        ax1.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "ibo_reliability_threshold_sweep.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[IBO-EVAL] plot skipped: {exc}", flush=True)

    baseline = all_summaries[0]
    best_threshold = float(best_by_instance["threshold"])
    best_class_rows = sweep_records[best_threshold][1]
    baseline_by_class = {row["class_id"]: row for row in class_rows}
    best_by_class = {row["class_id"]: row for row in best_class_rows}
    comparison_lines = [
        "# IBO 可靠性回投验证集评估报告",
        "",
        "## Material Passport",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        "| 实验 | D-FINE 候选框 + IBO 可靠性分数后处理 |",
        "| 数据 | 展开焊道验证集，排除钢帽类别 |",
        f"| 评价口径 | 同类覆盖 ≥ {args.coverage_threshold:.2f} 算正确；无正确但有异类覆盖算错误；否则漏检 |",
        f"| 输出目录 | `{args.output_dir}` |",
        "",
        "## 总体结果",
        "",
        "| 方法 | 阈值 | 候选数 | 正确 / 错误 / 漏检 | 实例准确率 | 图片完全正确 / 错误 / 漏检 | 图片准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| D-FINE baseline | {baseline['threshold']} | {baseline['selected_candidates']} | {baseline['instance_correct']} / {baseline['instance_wrong']} / {baseline['instance_miss']} | {format_percent(baseline['instance_accuracy'])} | {baseline['image_correct']} / {baseline['image_wrong']} / {baseline['image_miss']} | {format_percent(baseline['image_accuracy'])} |",
        f"| IBO reliability 最佳实例阈值 | {best_by_instance['threshold']} | {best_by_instance['selected_candidates']} | {best_by_instance['instance_correct']} / {best_by_instance['instance_wrong']} / {best_by_instance['instance_miss']} | {format_percent(best_by_instance['instance_accuracy'])} | {best_by_instance['image_correct']} / {best_by_instance['image_wrong']} / {best_by_instance['image_miss']} | {format_percent(best_by_instance['image_accuracy'])} |",
        f"| IBO reliability 最佳图片阈值 | {best_by_image['threshold']} | {best_by_image['selected_candidates']} | {best_by_image['instance_correct']} / {best_by_image['instance_wrong']} / {best_by_image['instance_miss']} | {format_percent(best_by_image['instance_accuracy'])} | {best_by_image['image_correct']} / {best_by_image['image_wrong']} / {best_by_image['image_miss']} | {format_percent(best_by_image['image_accuracy'])} |",
        "",
        "## 每类实例级对比（D-FINE baseline vs IBO reliability 最佳实例阈值）",
        "",
        "| 类别 | 标注实例 | D-FINE baseline | IBO reliability | 变化 |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_id in sorted(best_by_class):
        b = baseline_by_class[class_id]
        r = best_by_class[class_id]
        delta = int(r["correct"]) - int(b["correct"])
        comparison_lines.append(
            f"| {r['class_name']} | {r['gt_instances']} | {b['triple']}<br>{format_percent(b['accuracy'])} | {r['triple']}<br>{format_percent(r['accuracy'])} | {delta:+d} |"
        )
    comparison_lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 这个评估只改变“保留哪些 D-FINE 候选框”，不改变框的位置，也不生成新框。",
            "- 如果阈值过低，会保留更多低置信候选，漏检少，但图片级误检可能增加。",
            "- 如果阈值过高，会压掉误检，但也会把低置信真缺陷压掉，漏检增加。",
            "- 因此这一步主要用于确定后续 IBO 后处理的工作区间，而不是最终结论。",
            "",
            "## 输出文件",
            "",
            "- `threshold_sweep_summary.csv`：可靠性阈值扫描总表。",
            "- `instance_by_class_key_methods.csv`：关键阈值的每类实例级统计。",
            "- `image_and_instance_details_key_methods.csv`：关键阈值下每张图/每个实例明细。",
            "- `ibo_reliability_threshold_sweep.png`：阈值扫描曲线。",
        ]
    )
    (args.output_dir / "IBO可靠性回投验证集评估报告.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    print(
        "[IBO-EVAL] done "
        f"baseline={baseline['instance_correct']}/{baseline['instance_wrong']}/{baseline['instance_miss']} "
        f"best_instance_t={best_by_instance['threshold']} "
        f"best={best_by_instance['instance_correct']}/{best_by_instance['instance_wrong']}/{best_by_instance['instance_miss']} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

