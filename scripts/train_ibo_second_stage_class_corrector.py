#!/usr/bin/env python
"""Train a lightweight second-stage class corrector for IBO candidates.

The detector proposes boxes first. This script extracts non-leaky IBO image
features for those boxes, assigns training labels from source-level GT overlap,
trains a small multiclass classifier, and evaluates whether re-classifying
candidate boxes improves tail-class instance accuracy.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BACKGROUND = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--strip-root", type=Path, required=True)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--val-evidence", type=Path, required=True)
    parser.add_argument("--train-strip-boxes", type=Path, required=True)
    parser.add_argument("--val-strip-boxes", type=Path, required=True)
    parser.add_argument("--source-val-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-reliability", type=float, default=0.05)
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--exclude-class-id", type=int, action="append", default=[])
    parser.add_argument("--pure-tail-class-id", type=int, action="append", default=[0, 8, 17, 18, 20, 21])
    parser.add_argument("--tail-or-family-class-id", type=int, action="append", default=[0, 8, 12, 14, 15, 17, 18, 20, 21])
    parser.add_argument("--match-family", action="append", default=[])
    parser.add_argument("--override-prob-threshold", type=float, default=0.45)
    parser.add_argument("--disable-original-class-feature", action="store_true")
    parser.add_argument("--tail-sample-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def load_parent(scripts_root: Path):
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


def parse_box(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    return np.asarray(ast.literal_eval(str(value)), dtype=np.float32)


def class_name_map(source_data: dict[str, Any], evidence_rows: list[dict[str, str]]) -> dict[int, str]:
    names = {int(item["id"]): str(item.get("name") or item.get("slug") or f"class_{item['id']}") for item in source_data.get("categories", [])}
    for row in evidence_rows:
        names.setdefault(int(row["class_id"]), row.get("class_name", f"class_{row['class_id']}"))
    names[BACKGROUND] = "background"
    return names


def build_family_canonical(train_rows: list[dict[str, str]], families: list[set[int]]) -> dict[int, int]:
    counts = Counter(int(row["class_id"]) for row in train_rows)
    mapping: dict[int, int] = {}
    for family in families:
        canonical = max(sorted(family), key=lambda cid: (counts[cid], -cid))
        for cid in family:
            mapping[cid] = canonical
    return mapping


def canonicalize(cid: int, mapping: dict[int, int]) -> int:
    return mapping.get(cid, cid)


def assign_target_class(
    row: dict[str, str],
    gt_by_source: dict[int, list[dict[str, Any]]],
    family_canonical: dict[int, int],
    coverage_threshold: float,
    match_iou: float,
) -> tuple[int, float, float, int | None]:
    sid = int(row["source_image_id"])
    box = parse_box(row["global_box"])
    strip_width = float(row["strip_width"])
    best: tuple[float, float, int, int] | None = None
    for gt in gt_by_source.get(sid, []):
        shifted = parent_best_shifted_box(box, gt["box"], strip_width)
        cov = parent_intersection_area(shifted, gt["box"]) / max(1e-6, parent_box_area(gt["box"]))
        ov = parent_iou(shifted, gt["box"])
        score = max(cov, ov)
        item = (score, cov, int(gt["class_id"]), int(gt["source_annotation_id"]))
        if best is None or item > best:
            best = item
    if best is None:
        return BACKGROUND, 0.0, 0.0, None
    _, cov, cid, ann_id = best
    shifted = parent_best_shifted_box(box, gt_by_source[sid][0]["box"], strip_width) if False else None
    best_iou = 0.0
    for gt in gt_by_source.get(sid, []):
        if int(gt["source_annotation_id"]) == ann_id:
            best_iou = parent_iou(parent_best_shifted_box(box, gt["box"], strip_width), gt["box"])
            break
    if cov >= coverage_threshold or best_iou >= match_iou:
        return canonicalize(cid, family_canonical), cov, best_iou, ann_id
    return BACKGROUND, cov, best_iou, ann_id


def numeric_row_features(row: dict[str, str]) -> np.ndarray:
    box = parse_box(row["global_box"])
    w = max(1.0, float(box[2] - box[0]))
    h = max(1.0, float(box[3] - box[1]))
    strip_w = max(1.0, float(row["strip_width"]))
    return np.asarray(
        [
            float(row.get("member_count", 1)),
            float(row.get("max_reliability_v1", 0.0)),
            float(row.get("mean_reliability_v1", 0.0)),
            float(row.get("max_pred_score", 0.0)),
            float(row.get("normal_reference_score", 0.0)),
            float(row.get("frequency_evidence_score", 0.0)),
            float(row.get("reliability_v2_score", 0.0)),
            math.log(w),
            math.log(h),
            math.log(w * h),
            w / strip_w,
            h / 150.0,
        ],
        dtype=np.float32,
    )


def featurize_rows(
    parent,
    split: str,
    rows: list[dict[str, str]],
    strip_meta: dict[int, dict[str, Any]],
    strip_root: Path,
    transition_features,
    class_ids: list[int],
    use_original_class_feature: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    image_cache: dict[int, np.ndarray] = {}
    onehot_index = {cid: i for i, cid in enumerate(class_ids)}
    x: list[np.ndarray] = []
    kept_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        sid = int(row["source_image_id"])
        if sid not in strip_meta:
            continue
        row = dict(row)
        row.setdefault("strip_width", str(strip_meta[sid].get("strip_width", "")))
        row.setdefault("strip_height", str(strip_meta[sid].get("strip_height", 150)))
        if sid not in image_cache:
            image_cache[sid] = parent.read_bgr(parent.strip_file_for(split, sid, strip_root, strip_meta))
        strip = image_cache[sid]
        if strip is None:
            continue
        box = parse_box(row["global_box"])
        image_vec = parent.base_image_feature(strip, box, float(row["strip_width"]), transition_features)
        pieces = [numeric_row_features(row), image_vec]
        if use_original_class_feature:
            onehot = np.zeros(len(class_ids), dtype=np.float32)
            if int(row["class_id"]) in onehot_index:
                onehot[onehot_index[int(row["class_id"])]] = 1.0
            pieces.append(onehot)
        x.append(np.concatenate(pieces).astype(np.float32))
        kept_rows.append(dict(row))
        if idx % 1000 == 0:
            print(f"[CLASS-CORRECTOR] {split} features {idx}/{len(rows)}", flush=True)
    return np.vstack(x), kept_rows


def convert_rows_with_prediction(
    rows: list[dict[str, Any]],
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    names: dict[int, str],
    family_canonical: dict[int, int],
    mode: str,
    target_classes: set[int],
    prob_threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, label, prob in zip(rows, pred_labels, pred_probs):
        original = int(row["class_id"])
        pred = int(label)
        final = original
        action = "keep_original"
        if mode == "direct":
            final = pred
            action = "direct_reclassify"
        elif mode == "tail_or_family_override":
            if pred != BACKGROUND and canonicalize(pred, family_canonical) in target_classes and float(prob) >= prob_threshold:
                final = pred
                action = "override_tail_or_family"
        elif mode == "tail_only_override":
            if pred != BACKGROUND and canonicalize(pred, family_canonical) in target_classes and float(prob) >= prob_threshold:
                final = pred
                action = "override_pure_tail"
        else:
            raise ValueError(mode)
        if final == BACKGROUND:
            continue
        new_row = dict(row)
        new_row["original_class_id"] = original
        new_row["original_class_name"] = row.get("class_name", names.get(original, f"class_{original}"))
        new_row["class_id"] = final
        new_row["class_name"] = names.get(final, f"class_{final}")
        new_row["corrector_pred_class_id"] = pred
        new_row["corrector_pred_class_name"] = names.get(pred, f"class_{pred}")
        new_row["corrector_pred_prob"] = round(float(prob), 6)
        new_row["corrector_action"] = action
        out.append(new_row)
    return out


def aggregate(classes: list[dict[str, Any]], ids: set[int], name: str) -> dict[str, Any]:
    subset = [row for row in classes if int(row["class_id"]) in ids]
    total = sum(int(row["gt_instances"]) for row in subset)
    correct = sum(int(row["correct"]) for row in subset)
    wrong = sum(int(row["wrong"]) for row in subset)
    miss = sum(int(row["miss"]) for row in subset)
    return {
        "group": name,
        "gt_instances": total,
        "correct": correct,
        "wrong": wrong,
        "miss": miss,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "triple": f"{correct} / {wrong} / {miss}",
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def safe_label(label: int, names: dict[int, str]) -> str:
    return names.get(label, f"class_{label}").replace("/", "_").replace("\\", "_")


def plot_confusion(cm: np.ndarray, labels: list[int], names: dict[int, str], path: Path) -> None:
    fig_w = max(8, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    tick_labels = [safe_label(x, names) for x in labels]
    ax.set_xticklabels(tick_labels, rotation=80, ha="right", fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    ax.set_title("Second-stage class corrector confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent = load_parent(args.scripts_root)
    globals()["parent_best_shifted_box"] = parent.best_shifted_box
    globals()["parent_intersection_area"] = parent.intersection_area
    globals()["parent_box_area"] = parent.box_area
    globals()["parent_iou"] = parent.iou

    exclude = {int(x) for x in args.exclude_class_id}
    pure_tail = {int(x) for x in args.pure_tail_class_id}
    tail_or_family = {int(x) for x in args.tail_or_family_class_id}
    match_families = parent.parse_match_families(args.match_family)
    transition_features = parent.load_feature_functions(args.scripts_root)

    train_rows_all = [row for row in read_csv_rows(args.train_evidence) if float(row["reliability_v2_score"]) >= args.min_reliability]
    val_rows_all = [row for row in read_csv_rows(args.val_evidence) if float(row["reliability_v2_score"]) >= args.min_reliability]
    train_strip_boxes = load_json(args.train_strip_boxes)
    val_strip_boxes = load_json(args.val_strip_boxes)
    train_strip_meta = parent.strip_meta_by_source(args.train_strip_boxes)
    val_strip_meta = parent.strip_meta_by_source(args.val_strip_boxes)
    val_source = load_json(args.source_val_annotations)
    names = class_name_map(val_source, train_rows_all + val_rows_all)
    family_canonical = build_family_canonical(train_rows_all, match_families)
    canonical_match_families = [set(family) for family in match_families]

    train_gt = parent.gt_boxes_by_source(train_strip_boxes, exclude)
    val_gt = parent.gt_boxes_by_source(val_strip_boxes, exclude)

    print(f"[CLASS-CORRECTOR] train_rows={len(train_rows_all)} val_rows={len(val_rows_all)}", flush=True)
    class_ids = sorted({int(row["class_id"]) for row in train_rows_all + val_rows_all})
    use_original_class_feature = not args.disable_original_class_feature
    train_x, train_rows = featurize_rows(
        parent, "train", train_rows_all, train_strip_meta, args.strip_root, transition_features, class_ids, use_original_class_feature
    )
    val_x, val_rows = featurize_rows(
        parent, "val", val_rows_all, val_strip_meta, args.strip_root, transition_features, class_ids, use_original_class_feature
    )

    train_labels: list[int] = []
    train_label_rows: list[dict[str, Any]] = []
    for row in train_rows:
        target, cov, ov, ann_id = assign_target_class(row, train_gt, family_canonical, args.coverage_threshold, args.match_iou)
        train_labels.append(target)
        train_label_rows.append(
            {
                "split": "train",
                "merged_id": row["merged_id"],
                "source_image_id": row["source_image_id"],
                "original_class_id": row["class_id"],
                "target_class_id": target,
                "target_class_name": names.get(target, f"class_{target}"),
                "best_gt_coverage": round(cov, 6),
                "best_gt_iou": round(ov, 6),
                "source_annotation_id": "" if ann_id is None else ann_id,
            }
        )
    val_labels: list[int] = []
    val_label_rows: list[dict[str, Any]] = []
    for row in val_rows:
        target, cov, ov, ann_id = assign_target_class(row, val_gt, family_canonical, args.coverage_threshold, args.match_iou)
        val_labels.append(target)
        val_label_rows.append(
            {
                "split": "val",
                "merged_id": row["merged_id"],
                "source_image_id": row["source_image_id"],
                "original_class_id": row["class_id"],
                "target_class_id": target,
                "target_class_name": names.get(target, f"class_{target}"),
                "best_gt_coverage": round(cov, 6),
                "best_gt_iou": round(ov, 6),
                "source_annotation_id": "" if ann_id is None else ann_id,
            }
        )
    y_train = np.asarray(train_labels, dtype=np.int32)
    y_val = np.asarray(val_labels, dtype=np.int32)
    label_set = sorted(set(y_train.tolist()) | set(y_val.tolist()))
    print(f"[CLASS-CORRECTOR] label_distribution_train={dict(Counter(y_train.tolist()))}", flush=True)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=800,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    for index, label in enumerate(y_train):
        if int(label) in pure_tail:
            sample_weight[index] *= float(args.tail_sample_weight)
    model.fit(train_x, y_train, clf__sample_weight=sample_weight)
    val_pred = model.predict(val_x)
    val_prob_all = model.predict_proba(val_x)
    classes = model.named_steps["clf"].classes_
    val_prob = val_prob_all.max(axis=1)

    joblib.dump(
        {
            "model": model,
            "input_class_ids": class_ids,
            "family_canonical": family_canonical,
            "pure_tail_class_ids": sorted(pure_tail),
            "tail_or_family_class_ids": sorted(tail_or_family),
            "min_reliability": args.min_reliability,
            "override_prob_threshold": args.override_prob_threshold,
            "use_original_class_feature": use_original_class_feature,
            "tail_sample_weight": args.tail_sample_weight,
        },
        args.output_dir / "ibo_second_stage_class_corrector.joblib",
    )
    write_csv(args.output_dir / "train_candidate_target_labels.csv", train_label_rows)
    write_csv(args.output_dir / "val_candidate_target_labels.csv", val_label_rows)
    val_pred_rows = []
    for row, target, pred, prob in zip(val_rows, y_val, val_pred, val_prob):
        val_pred_rows.append(
            {
                "merged_id": row["merged_id"],
                "source_image_id": row["source_image_id"],
                "source_file": row.get("source_file", ""),
                "original_class_id": int(row["class_id"]),
                "original_class_name": row.get("class_name", ""),
                "target_class_id": int(target),
                "target_class_name": names.get(int(target), f"class_{target}"),
                "pred_class_id": int(pred),
                "pred_class_name": names.get(int(pred), f"class_{pred}"),
                "pred_prob": round(float(prob), 6),
                "reliability_v2_score": row.get("reliability_v2_score", ""),
                "global_box": row.get("global_box", ""),
            }
        )
    write_csv(args.output_dir / "val_candidate_class_corrector_predictions.csv", val_pred_rows)

    baseline_summary, baseline_classes = parent.evaluate_instance_from_selected(
        val_rows,
        val_source,
        val_strip_boxes,
        args.coverage_threshold,
        args.match_iou,
        exclude,
        match_families,
        "baseline_original_class",
    )
    direct_rows = convert_rows_with_prediction(val_rows, val_pred, val_prob, names, family_canonical, "direct", tail_or_family, args.override_prob_threshold)
    direct_summary, direct_classes = parent.evaluate_instance_from_selected(
        direct_rows,
        val_source,
        val_strip_boxes,
        args.coverage_threshold,
        args.match_iou,
        exclude,
        match_families,
        "second_stage_direct_reclassify",
    )
    tail_family_rows = convert_rows_with_prediction(val_rows, val_pred, val_prob, names, family_canonical, "tail_or_family_override", tail_or_family, args.override_prob_threshold)
    tail_family_summary, tail_family_classes = parent.evaluate_instance_from_selected(
        tail_family_rows,
        val_source,
        val_strip_boxes,
        args.coverage_threshold,
        args.match_iou,
        exclude,
        match_families,
        f"tail_family_override_p{args.override_prob_threshold:.2f}",
    )
    tail_rows = convert_rows_with_prediction(val_rows, val_pred, val_prob, names, family_canonical, "tail_only_override", pure_tail, args.override_prob_threshold)
    tail_summary, tail_classes = parent.evaluate_instance_from_selected(
        tail_rows,
        val_source,
        val_strip_boxes,
        args.coverage_threshold,
        args.match_iou,
        exclude,
        match_families,
        f"pure_tail_override_p{args.override_prob_threshold:.2f}",
    )

    method_defs = [
        ("baseline_original_class", baseline_summary, baseline_classes, val_rows),
        ("second_stage_direct_reclassify", direct_summary, direct_classes, direct_rows),
        (f"tail_family_override_p{args.override_prob_threshold:.2f}", tail_family_summary, tail_family_classes, tail_family_rows),
        (f"pure_tail_override_p{args.override_prob_threshold:.2f}", tail_summary, tail_classes, tail_rows),
    ]
    method_rows = []
    group_rows = []
    class_rows_all = []
    for method, summary, cls_rows, selected in method_defs:
        tail_group = aggregate(cls_rows, pure_tail, "pure_tail")
        non_tail_ids = {int(row["class_id"]) for row in cls_rows} - pure_tail
        non_tail_group = aggregate(cls_rows, non_tail_ids, "non_tail")
        method_rows.append(
            {
                "method": method,
                "selected_groups": len(selected),
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
            out = dict(group)
            out["method"] = method
            group_rows.append(out)
        for row in cls_rows:
            out = dict(row)
            out["method"] = method
            class_rows_all.append(out)

    write_csv(args.output_dir / "second_stage_method_summary.csv", method_rows)
    write_csv(args.output_dir / "second_stage_group_summary.csv", group_rows)
    write_csv(args.output_dir / "second_stage_by_class.csv", class_rows_all)
    write_csv(args.output_dir / "selected_predictions_direct_reclassify.csv", direct_rows)
    write_csv(args.output_dir / "selected_predictions_tail_family_override.csv", tail_family_rows)
    write_csv(args.output_dir / "selected_predictions_pure_tail_override.csv", tail_rows)

    report_dict = classification_report(
        y_val,
        val_pred,
        labels=label_set,
        target_names=[names.get(int(x), f"class_{x}") for x in label_set],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_val, val_pred, labels=label_set)
    with (args.output_dir / "candidate_classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_accuracy": accuracy_score(y_val, val_pred),
                "candidate_balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
                "label_distribution_train": dict(Counter(int(x) for x in y_train.tolist())),
                "label_distribution_val": dict(Counter(int(x) for x in y_val.tolist())),
                "classification_report": report_dict,
                "family_canonical": {str(k): v for k, v in family_canonical.items()},
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    plot_confusion(cm, label_set, names, args.output_dir / "candidate_class_corrector_confusion_matrix.png")

    best_tail = max(method_rows, key=lambda row: (float(row["pure_tail_accuracy"]), float(row["accuracy"])))
    best_total = max(method_rows, key=lambda row: (float(row["accuracy"]), float(row["pure_tail_accuracy"])))
    lines = [
        "# IBO 二阶段轻量类别校正实验报告",
        "",
        "## Material Passport",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 输入候选 | `{args.val_evidence}` |",
        f"| 训练候选 | `{args.train_evidence}` |",
        f"| 最小 IBO reliability | {args.min_reliability} |",
        f"| 轻量分类器 | ExtraTreesClassifier, class_weight=balanced, n_estimators=800 |",
        f"| 是否使用 D-FINE 原始类别 one-hot | {use_original_class_feature} |",
        f"| 尾类样本权重 | {args.tail_sample_weight} |",
        f"| 纯尾类 | {', '.join(str(x) for x in sorted(pure_tail))} |",
        f"| 类别族匹配 | {'; '.join(args.match_family)} |",
        "",
        "## 候选级分类器表现",
        "",
        f"- 验证候选级 accuracy：{fmt_pct(float(accuracy_score(y_val, val_pred)))}",
        f"- 验证候选级 balanced accuracy：{fmt_pct(float(balanced_accuracy_score(y_val, val_pred)))}",
        "",
        "## 实例级回投对比",
        "",
        "| 方法 | group数 | 总体 正确/错类/漏检 | 总准确率 | 纯尾类 正确/错类/漏检 | 纯尾类准确率 | 非尾类准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        mark = ""
        if row["method"] == best_tail["method"]:
            mark += " **尾类最好**"
        if row["method"] == best_total["method"]:
            mark += " **总体最好**"
        lines.append(
            f"| {row['method']}{mark} | {row['selected_groups']} | {row['triple']} | {fmt_pct(float(row['accuracy']))} | "
            f"{row['pure_tail_triple']} | {fmt_pct(float(row['pure_tail_accuracy']))} | {fmt_pct(float(row['non_tail_accuracy']))} |"
        )
    lines.extend(
        [
            "",
            "## 类别级结果",
            "",
            "| 方法 | 类别 | 标注实例 | 正确 / 错类 / 漏检 | 准确率 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in class_rows_all:
        cid = int(row["class_id"])
        prefix = "尾类-" if cid in pure_tail else ""
        lines.append(
            f"| {row['method']} | {prefix}{row['class_name']} | {row['gt_instances']} | {row['triple']} | {fmt_pct(float(row['accuracy']))} |"
        )
    lines.extend(
        [
            "",
            "## 阶段判断",
            "",
            "- 这个实验检验的是：在 D-FINE/IBO 已经给出候选框之后，能不能用空间、频率、正常参照距离等证据把类别纠正回来。",
            "- 如果 direct_reclassify 下降，说明分类器直接接管所有类别会破坏头部类别；如果 tail override 提升，说明它适合作为长尾专用校正模块。",
            "- 若纯尾类仍然不升，说明当前训练候选中尾类可学习样本还不够，需要进一步做尾类样本增强或人工复核尾类错误样本。",
        ]
    )
    (args.output_dir / "IBO二阶段轻量类别校正实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"method_summary": method_rows, "best_tail": best_tail["method"], "best_total": best_total["method"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
