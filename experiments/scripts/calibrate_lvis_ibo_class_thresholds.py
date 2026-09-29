#!/usr/bin/env python
"""Class-conditional calibration for LVIS/SimLTD IBO reliability scores.

The previous stage learns a candidate-level reliability score.  This script
uses the training split to learn class-conditional score thresholds, applies
them to validation candidates, and evaluates instance-level correct/wrong/miss
with the same back-projection logic used by evaluate_ibo_reliability_postprocess.py.

This stage is a calibration / confusion-suppression pilot.  It does not move
boxes and does not create new candidates.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--tile-annotations", type=Path, required=True)
    parser.add_argument("--val-ibo-manifest", type=Path, required=True)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--val-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--global-threshold", type=float, default=0.15)
    parser.add_argument("--min-class-candidates", type=int, default=30)
    parser.add_argument("--min-class-positives", type=int, default=5)
    parser.add_argument("--target-precision", type=float, default=0.50)
    return parser.parse_args()


def load_eval_module(path: Path):
    spec = importlib.util.spec_from_file_location("ibo_eval_backproject", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import eval script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def thresholds(start: float, stop: float, step: float) -> list[float]:
    vals: list[float] = []
    current = start
    while current <= stop + 1e-9:
        vals.append(round(current, 4))
        current += step
    return vals


def candidate_stats(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    positives = sum(int(row["label"]) for row in rows)
    selected = [row for row in rows if float(row["score"]) >= threshold]
    tp = sum(int(row["label"]) for row in selected)
    fp = len(selected) - tp
    fn = positives - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    return {
        "threshold": threshold,
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / max(1e-12, b2 * precision + recall)


def best_threshold(rows: list[dict[str, Any]], grid: list[float], beta: float, target_precision: float | None = None) -> tuple[float, dict[str, float], str]:
    scored: list[dict[str, float]] = []
    for threshold in grid:
        stats = candidate_stats(rows, threshold)
        stats["fbeta"] = fbeta(stats["precision"], stats["recall"], beta)
        scored.append(stats)
    eligible = scored
    reason = f"max_f{beta:g}"
    if target_precision is not None:
        guarded = [row for row in scored if row["precision"] >= target_precision and row["selected"] > 0]
        if guarded:
            eligible = guarded
            reason = f"precision_ge_{target_precision:.2f}_then_max_recall_f{beta:g}"
    best = max(
        eligible,
        key=lambda row: (row["fbeta"], row["recall"], row["precision"], -row["selected"]),
    )
    return float(best["threshold"]), best, reason


def normalize_evidence_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "ibo_id": row["ibo_id"],
        "class_id": int(row["pred_class_id"]),
        "class_name": row.get("pred_class_name", f"class_{row['pred_class_id']}"),
        "score": float(row["reliability_fusion_score"]),
        "label": int(row["label"]),
        "candidate_outcome": row.get("candidate_outcome", ""),
    }


def learn_policy_thresholds(
    train_rows: list[dict[str, Any]],
    grid: list[float],
    args: argparse.Namespace,
    policy: str,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    class_names: dict[int, str] = {}
    for row in train_rows:
        by_class[int(row["class_id"])].append(row)
        class_names[int(row["class_id"])] = str(row["class_name"])

    if policy == "global_f1":
        global_threshold, global_stats, global_reason = best_threshold(train_rows, grid, beta=1.0)
    elif policy == "global_f05":
        global_threshold, global_stats, global_reason = best_threshold(train_rows, grid, beta=0.5)
    elif policy == "global_precision_guard":
        global_threshold, global_stats, global_reason = best_threshold(train_rows, grid, beta=0.5, target_precision=args.target_precision)
    else:
        global_threshold = float(args.global_threshold)
        global_stats = candidate_stats(train_rows, global_threshold)
        global_reason = "user_global_default"

    thresholds_by_class: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for class_id in sorted(by_class):
        items = by_class[class_id]
        positives = sum(int(row["label"]) for row in items)
        use_global = len(items) < args.min_class_candidates or positives < args.min_class_positives
        reason = global_reason if use_global else ""
        if use_global:
            threshold = global_threshold
            stats = candidate_stats(items, threshold)
        else:
            if policy == "class_f1":
                threshold, stats, reason = best_threshold(items, grid, beta=1.0)
            elif policy == "class_f05":
                threshold, stats, reason = best_threshold(items, grid, beta=0.5)
            elif policy == "class_precision_guard":
                threshold, stats, reason = best_threshold(items, grid, beta=0.5, target_precision=args.target_precision)
            else:
                threshold = global_threshold
                stats = candidate_stats(items, threshold)
                reason = global_reason
        thresholds_by_class[class_id] = threshold
        rows.append(
            {
                "policy": policy,
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"class_{class_id}"),
                "train_candidates": len(items),
                "train_positive": positives,
                "threshold": threshold,
                "selected": int(stats["selected"]),
                "tp": int(stats["tp"]),
                "fp": int(stats["fp"]),
                "fn": int(stats["fn"]),
                "precision": round(float(stats["precision"]), 6),
                "recall": round(float(stats["recall"]), 6),
                "fbeta": round(float(stats.get("fbeta", fbeta(stats["precision"], stats["recall"], 1.0))), 6),
                "reason": reason,
            }
        )
    rows.insert(
        0,
        {
            "policy": policy,
            "class_id": "GLOBAL",
            "class_name": "GLOBAL",
            "train_candidates": len(train_rows),
            "train_positive": sum(int(row["label"]) for row in train_rows),
            "threshold": global_threshold,
            "selected": int(global_stats["selected"]),
            "tp": int(global_stats["tp"]),
            "fp": int(global_stats["fp"]),
            "fn": int(global_stats["fn"]),
            "precision": round(float(global_stats["precision"]), 6),
            "recall": round(float(global_stats["recall"]), 6),
            "fbeta": round(float(global_stats.get("fbeta", fbeta(global_stats["precision"], global_stats["recall"], 1.0))), 6),
            "reason": global_reason,
        },
    )
    return thresholds_by_class, rows


def prepare_eval_rows(eval_module, manifest_path: Path, evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_rows = eval_module.read_jsonl(manifest_path)
    evidence_by_id = {row["ibo_id"]: row for row in evidence_rows}
    merged: list[dict[str, Any]] = []
    for row in manifest_rows:
        evidence = evidence_by_id.get(row["ibo_id"])
        if evidence is None:
            continue
        item = dict(row)
        item["reliability_score"] = float(evidence["score"])
        item["pred_score"] = float(item["pred_score"])
        item["pred_class_id"] = int(item["pred_class_id"])
        item["tile_image_id"] = int(item["tile_image_id"])
        item["source_image_id"] = int(item["source_image_id"])
        item["pred_box"] = eval_module.parse_box(item["pred_xyxy"])
        merged.append(item)
    return merged


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_module = load_eval_module(args.eval_script)
    grid = thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)

    train_rows = [normalize_evidence_row(row) for row in read_csv(args.train_evidence)]
    val_evidence_rows = [normalize_evidence_row(row) for row in read_csv(args.val_evidence)]
    val_eval_rows = prepare_eval_rows(eval_module, args.val_ibo_manifest, val_evidence_rows)
    source_data = eval_module.load_json(args.source_annotations)
    tile_data = eval_module.load_json(args.tile_annotations)

    policies = [
        "global_default",
        "global_f1",
        "global_f05",
        "global_precision_guard",
        "class_f1",
        "class_f05",
        "class_precision_guard",
    ]
    all_summary_rows: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, Any]] = []
    class_rows_for_best: list[dict[str, Any]] = []
    detail_rows_for_best: list[dict[str, Any]] = []

    for policy in policies:
        threshold_by_class, threshold_rows = learn_policy_thresholds(train_rows, grid, args, policy)
        all_threshold_rows.extend(threshold_rows)
        selected = [
            row for row in val_eval_rows
            if float(row["reliability_score"]) >= threshold_by_class.get(int(row["pred_class_id"]), args.global_threshold)
        ]
        summary, class_rows, detail_rows = eval_module.evaluate_selection(
            selected,
            source_data,
            tile_data,
            set(),
            args.coverage_threshold,
            args.match_iou,
            [],
            policy,
            "class_conditional",
        )
        all_summary_rows.append(summary)
        if not class_rows_for_best or (
            summary["instance_accuracy"],
            -summary["instance_wrong"],
            -summary["selected_candidates"],
        ) > (
            all_summary_rows[-2]["instance_accuracy"] if len(all_summary_rows) >= 2 else -1,
            -all_summary_rows[-2]["instance_wrong"] if len(all_summary_rows) >= 2 else 0,
            -all_summary_rows[-2]["selected_candidates"] if len(all_summary_rows) >= 2 else 0,
        ):
            class_rows_for_best = class_rows
            detail_rows_for_best = detail_rows
        print(
            f"[CLASS-CAL] {policy}: selected={summary['selected_candidates']} "
            f"correct/wrong/miss={summary['instance_correct']}/{summary['instance_wrong']}/{summary['instance_miss']} "
            f"acc={summary['instance_accuracy']}",
            flush=True,
        )

    best = max(
        all_summary_rows,
        key=lambda row: (row["instance_accuracy"], -row["instance_wrong"], -row["selected_candidates"]),
    )
    # Recompute rows for the true best policy for detail export.
    best_thresholds, _ = learn_policy_thresholds(train_rows, grid, args, str(best["method"]))
    best_selected = [
        row for row in val_eval_rows
        if float(row["reliability_score"]) >= best_thresholds.get(int(row["pred_class_id"]), args.global_threshold)
    ]
    _, class_rows_for_best, detail_rows_for_best = eval_module.evaluate_selection(
        best_selected,
        source_data,
        tile_data,
        set(),
        args.coverage_threshold,
        args.match_iou,
        [],
        str(best["method"]),
        "class_conditional",
    )

    write_csv(args.output_dir / "class_threshold_policy_summary.csv", all_summary_rows)
    write_csv(args.output_dir / "learned_class_thresholds.csv", all_threshold_rows)
    write_csv(args.output_dir / "best_policy_instance_by_class.csv", class_rows_for_best)
    write_csv(args.output_dir / "best_policy_details.csv", detail_rows_for_best)

    lines = [
        "# LVIS/SimLTD IBO 类别条件阈值校准结果",
        "",
        "## Material Passport",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        "| 实验 | IBO reliability 类别条件阈值校准 |",
        "| 阈值学习 | train split 候选级 label |",
        "| 应用与评估 | val split 实例级回投 correct / wrong / miss |",
        f"| 输出目录 | `{args.output_dir}` |",
        "",
        "## 策略汇总",
        "",
        "| 策略 | 候选数 | correct / wrong / miss | 实例准确率 | image correct / wrong / miss | 图片准确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in all_summary_rows:
        lines.append(
            f"| {row['method']} | {row['selected_candidates']} | "
            f"{row['instance_correct']} / {row['instance_wrong']} / {row['instance_miss']} | "
            f"{float(row['instance_accuracy']) * 100:.2f}% | "
            f"{row['image_correct']} / {row['image_wrong']} / {row['image_miss']} | "
            f"{float(row['image_accuracy']) * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 最佳策略",
            "",
            json.dumps(best, ensure_ascii=False, indent=2),
            "",
            "## 说明",
            "",
            "- 该实验只做类别条件阈值校准，不改变候选框位置，也不新增候选。",
            "- 阈值由 train split 学习，再应用于 val split，避免直接用 val 选择每类阈值。",
            "- 如果 wrong 明显下降但 miss 上升，说明类别校准更保守；如果 correct 增加同时 wrong 增加，则仍需要二阶段重分类纠错。",
        ]
    )
    (args.output_dir / "LVIS_IBO类别条件阈值校准报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[CLASS-CAL] done best={best['method']} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
