#!/usr/bin/env python
"""Calibrate tail-aware thresholds for IBO reliability/group evidence scores.

This script does not train D-FINE. It reuses the group-level IBO evidence CSVs
and searches class-aware thresholds on the training split, then evaluates them
on the validation split with the same instance-level coverage protocol used by
train_ibo_group_evidence_fusion_v2.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--val-evidence", type=Path, required=True)
    parser.add_argument("--train-strip-boxes", type=Path, required=True)
    parser.add_argument("--val-strip-boxes", type=Path, required=True)
    parser.add_argument("--source-train-annotations", type=Path, required=True)
    parser.add_argument("--source-val-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-column", default="reliability_v2_score")
    parser.add_argument("--base-threshold", type=float, default=0.05)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--exclude-class-id", type=int, action="append", default=[])
    parser.add_argument("--pure-tail-class-id", type=int, action="append", default=[0, 8, 17, 18, 20, 21])
    parser.add_argument("--match-family", action="append", default=[])
    parser.add_argument("--max-total-accuracy-drop", type=float, default=0.005)
    return parser.parse_args()


def load_parent_functions(scripts_root: Path):
    sys.path.insert(0, str(scripts_root))
    import train_ibo_group_evidence_fusion_v2 as parent

    return parent


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def thresholds(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    x = start
    while x <= stop + 1e-9:
        values.append(round(x, 4))
        x += step
    return values


def class_name_map(*rows_list: list[dict[str, str]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for rows in rows_list:
        for row in rows:
            out[int(row["class_id"])] = row.get("class_name", f"class_{row['class_id']}")
    return out


def selected_by_thresholds(rows: list[dict[str, str]], score_column: str, class_thresholds: dict[int, float]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        cid = int(row["class_id"])
        threshold = class_thresholds.get(cid, class_thresholds.get(-1, 0.05))
        if float(row[score_column]) >= threshold:
            selected.append(row)
    return selected


def candidate_fbeta_thresholds(
    train_rows: list[dict[str, str]],
    score_column: str,
    grid: list[float],
    pure_tail: set[int],
    default: float,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    by_class: dict[int, list[dict[str, str]]] = {}
    for row in train_rows:
        by_class.setdefault(int(row["class_id"]), []).append(row)
    thresholds_by_class: dict[int, float] = {-1: default}
    audit: list[dict[str, Any]] = []
    for cid, rows in sorted(by_class.items()):
        beta = 2.0 if cid in pure_tail else 1.0
        best: tuple[float, float, float, float, float] | None = None
        for th in grid:
            tp = sum(1 for row in rows if float(row[score_column]) >= th and int(row["label"]) == 1)
            fp = sum(1 for row in rows if float(row[score_column]) >= th and int(row["label"]) == 0)
            fn = sum(1 for row in rows if float(row[score_column]) < th and int(row["label"]) == 1)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            beta2 = beta * beta
            fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0
            kept = tp + fp
            # Tie-break: tail classes prefer lower thresholds to keep recall; non-tail prefer fewer false positives.
            tie = -th if cid in pure_tail else -fp
            key = (fbeta, recall, precision, tie, -kept)
            if best is None or key > best:
                best = key
                thresholds_by_class[cid] = th
                best_stats = (tp, fp, fn, precision, recall, fbeta)
        tp, fp, fn, precision, recall, fbeta = best_stats
        audit.append(
            {
                "class_id": cid,
                "threshold": thresholds_by_class[cid],
                "train_tp": tp,
                "train_fp": fp,
                "train_fn": fn,
                "train_precision": round(precision, 6),
                "train_recall": round(recall, 6),
                "train_fbeta": round(fbeta, 6),
                "tail_policy": "recall_weighted_f2" if cid in pure_tail else "balanced_f1",
            }
        )
    return thresholds_by_class, audit


def evaluate(parent, rows, source_data, strip_boxes, exclude, match_families, coverage_threshold, match_iou, method):
    return parent.evaluate_instance_from_selected(
        rows,
        source_data,
        strip_boxes,
        coverage_threshold,
        match_iou,
        exclude,
        match_families,
        method,
    )


def aggregate(classes: list[dict[str, Any]], ids: set[int], name: str) -> dict[str, Any]:
    subset = [row for row in classes if int(row["class_id"]) in ids]
    total = sum(int(row["gt_instances"]) for row in subset)
    correct = sum(int(row["correct"]) for row in subset)
    wrong = sum(int(row["wrong"]) for row in subset)
    miss = sum(int(row["miss"]) for row in subset)
    return {
        "group": name,
        "classes": ",".join(str(int(row["class_id"])) for row in subset),
        "gt_instances": total,
        "correct": correct,
        "wrong": wrong,
        "miss": miss,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "triple": f"{correct} / {wrong} / {miss}",
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent = load_parent_functions(args.scripts_root)
    match_families = parent.parse_match_families(args.match_family)
    exclude = {int(x) for x in args.exclude_class_id}
    pure_tail = {int(x) for x in args.pure_tail_class_id}
    grid = thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)

    train_rows = read_csv_rows(args.train_evidence)
    val_rows = read_csv_rows(args.val_evidence)
    train_strip_boxes = load_json(args.train_strip_boxes)
    val_strip_boxes = load_json(args.val_strip_boxes)
    train_source = load_json(args.source_train_annotations)
    val_source = load_json(args.source_val_annotations)
    names = class_name_map(train_rows, val_rows)

    base_thresholds = {-1: args.base_threshold}
    base_train_selected = selected_by_thresholds(train_rows, args.score_column, base_thresholds)
    base_val_selected = selected_by_thresholds(val_rows, args.score_column, base_thresholds)
    base_train_summary, base_train_classes = evaluate(parent, base_train_selected, train_source, train_strip_boxes, exclude, match_families, args.coverage_threshold, args.match_iou, "base_all_ge_0.05_train")
    base_val_summary, base_val_classes = evaluate(parent, base_val_selected, val_source, val_strip_boxes, exclude, match_families, args.coverage_threshold, args.match_iou, "base_all_ge_0.05_val")

    fbeta_thresholds, fbeta_audit = candidate_fbeta_thresholds(train_rows, args.score_column, grid, pure_tail, args.base_threshold)
    fbeta_train_selected = selected_by_thresholds(train_rows, args.score_column, fbeta_thresholds)
    fbeta_val_selected = selected_by_thresholds(val_rows, args.score_column, fbeta_thresholds)
    fbeta_train_summary, fbeta_train_classes = evaluate(parent, fbeta_train_selected, train_source, train_strip_boxes, exclude, match_families, args.coverage_threshold, args.match_iou, "class_fbeta_train")
    fbeta_val_summary, fbeta_val_classes = evaluate(parent, fbeta_val_selected, val_source, val_strip_boxes, exclude, match_families, args.coverage_threshold, args.match_iou, "class_fbeta_val")

    # Tail-priority policy: keep tail at base threshold, use learned thresholds only for non-tail classes
    # if doing so does not drop total validation accuracy too much. This is descriptive on validation,
    # not a frozen-test claim.
    tail_priority_thresholds = dict(fbeta_thresholds)
    for cid in pure_tail:
        tail_priority_thresholds[cid] = args.base_threshold
    tail_train_selected = selected_by_thresholds(train_rows, args.score_column, tail_priority_thresholds)
    tail_val_selected = selected_by_thresholds(val_rows, args.score_column, tail_priority_thresholds)
    tail_train_summary, tail_train_classes = evaluate(parent, tail_train_selected, train_source, train_strip_boxes, exclude, match_families, args.coverage_threshold, args.match_iou, "tail_priority_train")
    tail_val_summary, tail_val_classes = evaluate(parent, tail_val_selected, val_source, val_strip_boxes, exclude, match_families, args.coverage_threshold, args.match_iou, "tail_priority_val")

    all_methods = [
        ("base_all_ge_0.05", base_val_summary, base_val_classes, base_thresholds),
        ("class_fbeta_thresholds", fbeta_val_summary, fbeta_val_classes, fbeta_thresholds),
        ("tail_priority_thresholds", tail_val_summary, tail_val_classes, tail_priority_thresholds),
    ]
    method_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for method, summary, classes, ths in all_methods:
        tail_group = aggregate(classes, pure_tail, "pure_tail")
        non_tail_ids = {int(row["class_id"]) for row in classes} - pure_tail
        non_tail_group = aggregate(classes, non_tail_ids, "non_tail")
        method_rows.append(
            {
                "method": method,
                "selected_groups": summary["selected_groups"],
                "gt_instances": summary["instances"],
                "correct": summary["correct"],
                "wrong": summary["wrong"],
                "miss": summary["miss"],
                "accuracy": summary["instance_accuracy"],
                "triple": f"{summary['correct']} / {summary['wrong']} / {summary['miss']}",
                "pure_tail_triple": tail_group["triple"],
                "pure_tail_accuracy": tail_group["accuracy"],
                "non_tail_triple": non_tail_group["triple"],
                "non_tail_accuracy": non_tail_group["accuracy"],
            }
        )
        for group in (tail_group, non_tail_group):
            group["method"] = method
            group_rows.append(group)
        for row in classes:
            out = dict(row)
            out["method"] = method
            out["threshold"] = ths.get(int(row["class_id"]), ths.get(-1, args.base_threshold))
            class_rows.append(out)

    threshold_rows = []
    for cid, th in sorted(fbeta_thresholds.items()):
        if cid == -1:
            continue
        threshold_rows.append({"class_id": cid, "class_name": names.get(cid, f"class_{cid}"), "threshold": th, "is_pure_tail": cid in pure_tail})

    write_csv(args.output_dir / "tail_calibration_method_summary.csv", method_rows)
    write_csv(args.output_dir / "tail_calibration_group_summary.csv", group_rows)
    write_csv(args.output_dir / "tail_calibration_by_class.csv", class_rows)
    write_csv(args.output_dir / "tail_calibration_learned_thresholds.csv", threshold_rows)
    write_csv(args.output_dir / "tail_calibration_candidate_fbeta_train_audit.csv", fbeta_audit)

    payload = {
        "score_column": args.score_column,
        "base_threshold": args.base_threshold,
        "pure_tail_class_ids": sorted(pure_tail),
        "match_families": args.match_family,
        "exclude_class_ids": sorted(exclude),
        "method_summary": method_rows,
    }
    (args.output_dir / "tail_calibration_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 尾类专用 IBO 阈值校准实验报告",
        "",
        "## Material Passport",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 分数列 | `{args.score_column}` |",
        f"| 基础阈值 | {args.base_threshold} |",
        f"| 纯尾类 | {', '.join(str(x) for x in sorted(pure_tail))} |",
        f"| 类别族匹配 | {'; '.join(args.match_family)} |",
        "",
        "## 验证集方法对比",
        "",
        "| 方法 | group数 | 总体 正确/错类/漏检 | 总准确率 | 纯尾类 正确/错类/漏检 | 纯尾类准确率 | 非尾类准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    best_tail = max(method_rows, key=lambda row: (float(row["pure_tail_accuracy"]), float(row["accuracy"])))
    for row in method_rows:
        mark = " **推荐观察**" if row["method"] == best_tail["method"] else ""
        lines.append(
            f"| {row['method']}{mark} | {row['selected_groups']} | {row['triple']} | {fmt_pct(float(row['accuracy']))} | "
            f"{row['pure_tail_triple']} | {fmt_pct(float(row['pure_tail_accuracy']))} | {fmt_pct(float(row['non_tail_accuracy']))} |"
        )
    lines.extend(
        [
            "",
            "## 推荐观察方法的类别级结果",
            "",
            "| 类别 | 标注实例 | 正确 / 错类 / 漏检 | 准确率 | 使用阈值 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        if row["method"] != best_tail["method"]:
            continue
        cid = int(row["class_id"])
        prefix = "尾类-" if cid in pure_tail else ""
        lines.append(
            f"| {prefix}{row['class_name']} | {row['gt_instances']} | {row['triple']} | {fmt_pct(float(row['accuracy']))} | {float(row['threshold']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 这一步没有重新训练检测器，而是在 IBO 空间/频率/正常参照证据分数上做类别敏感阈值校准。",
            "- 如果纯尾类准确率提升而总体准确率基本不掉，说明长尾改善主要来自“尾类候选保留/校准”；如果总体下降明显，则该阈值方案不能作为正式主结果。",
            "- 本报告基于验证集分析，后续若用于论文主结论，需要在独立测试集或交叉验证上复核。",
        ]
    )
    (args.output_dir / "尾类专用IBO阈值校准实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
