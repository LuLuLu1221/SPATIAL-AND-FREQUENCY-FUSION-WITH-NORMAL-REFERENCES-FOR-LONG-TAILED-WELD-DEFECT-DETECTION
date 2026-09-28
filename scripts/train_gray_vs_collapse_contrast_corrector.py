#!/usr/bin/env python
"""Gray-weld vs collapse-family contrast corrector.

This experiment is deliberately narrow and interpretable:

1. Keep the existing D-FINE + IBO evidence candidate generator fixed.
2. Build a binary contrast task: 焊灰色 (class 20) vs 缺焊长塌合并类.
3. Add class-prototype contrast features so a candidate is compared against
   both the gray class and the confusable high-frequency family.
4. Only allow conservative overrides from the confusable family to 焊灰色.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"D:\1\项目论文")
DEFAULT_EVIDENCE = ROOT / "zhwk_runs" / "normal_reference_no_defect_20260907" / "03_group_evidence_external_normal_correct_eval_sameclass_family12_14_15_smoke2_3"
DEFAULT_TARGETS = ROOT / "zhwk_runs" / "normal_reference_no_defect_20260908" / "08_pure_tail_binary_then_tail_classifier_v2_withorig"
DEFAULT_OUT = ROOT / "zhwk_runs" / "normal_reference_no_defect_20260912" / "20_gray_vs_collapse_contrast_corrector_v1"

GRAY = 20
COLLAPSE_CANONICAL = 14
COLLAPSE_FAMILY = {12, 14, 15, 18}
SMOKE_FAMILY = {0, 2, 3}
EXCLUDE = {11, 17, 19, 21, 22}


CLASS_NAMES = {
    -1: "background",
    0: "小焊烟",
    1: "焊炸",
    2: "焊烟团",
    3: "焊烟",
    4: "焊渣",
    5: "长焊高裂",
    6: "点焊高",
    7: "焊洞",
    8: "焊坑",
    9: "焊洞长",
    10: "长焊高",
    11: "钢帽",
    12: "断焊",
    13: "焊高纹",
    14: "长焊高塌",
    15: "缺焊蓝黑",
    16: "焊高烟",
    17: "缺焊裂",
    18: "焊缝蓝黑",
    19: "焊盘离",
    20: "焊灰色",
    21: "焊高缝",
    22: "脱焊",
}


BASE_FEATURES = [
    "class_id",
    "member_count",
    "max_reliability_v1",
    "mean_reliability_v1",
    "max_pred_score",
    "normal_reference_score",
    "frequency_evidence_score",
    "reliability_v2_score",
    "same_cov",
    "same_iou",
    "other_cov",
    "other_iou",
    "box_w",
    "box_h",
    "box_area",
    "box_aspect",
    "box_x_center_norm",
    "box_y_center_norm",
    "patch_mean",
    "patch_std",
    "patch_min",
    "patch_max",
    "patch_p10",
    "patch_p50",
    "patch_p90",
    "patch_grad_mean",
    "patch_grad_std",
    "patch_lap_var",
    "patch_fft_low",
    "patch_fft_mid",
    "patch_fft_high",
    "patch_fft_high_low_ratio",
]

CONTRAST_FEATURES = [
    "dist_to_gray_proto",
    "dist_to_collapse_proto",
    "proto_margin_collapse_minus_gray",
    "gray_relative_similarity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts-root", type=Path, default=ROOT / "zhwk_project" / "scripts")
    parser.add_argument("--strip-root", type=Path, default=ROOT / "zhwk_unwrapped_150px_cyclic_tiles_v1")
    parser.add_argument("--train-strip-boxes", type=Path, default=ROOT / "zhwk_unwrapped_150px_cyclic_tiles_v1" / "metadata" / "strip_boxes_train.json")
    parser.add_argument("--val-strip-boxes", type=Path, default=ROOT / "zhwk_unwrapped_150px_cyclic_tiles_v1" / "metadata" / "strip_boxes_val.json")
    parser.add_argument("--source-val-annotations", type=Path, default=ROOT / "zhwk_dfine_v1" / "annotations" / "instances_val.json")
    parser.add_argument("--train-evidence", type=Path, default=DEFAULT_EVIDENCE / "ibo_group_evidence_train.csv")
    parser.add_argument("--val-evidence", type=Path, default=DEFAULT_EVIDENCE / "ibo_group_evidence_val.csv")
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TARGETS / "train_candidate_tail_targets.csv")
    parser.add_argument("--val-targets", type=Path, default=DEFAULT_TARGETS / "val_candidate_tail_targets.csv")
    parser.add_argument("--review-root", type=Path, default=ROOT / "zhwk_runs" / "normal_reference_no_defect_20260911" / "13_pure_tail_review_package_v2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-reliability", type=float, default=0.05)
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def load_parent(scripts_root: Path):
    sys.path.insert(0, str(scripts_root))
    import train_ibo_group_evidence_fusion_v2 as parent

    return parent


def load_feature_core(scripts_root: Path):
    sys.path.insert(0, str(scripts_root))
    import train_targeted_tail_corrector_patch_features as core

    return core


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def merge_data(evidence_path: Path, targets_path: Path, min_reliability: float) -> pd.DataFrame:
    ev = pd.read_csv(evidence_path)
    tg = pd.read_csv(targets_path)
    cols = ["merged_id", "target_class_id", "target_class_name", "best_gt_coverage", "best_gt_iou", "source_annotation_id"]
    df = ev.merge(tg[cols], on="merged_id", how="inner")
    df = df[df["reliability_v2_score"].astype(float) >= float(min_reliability)].copy()
    return df


def feature_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    use = df.copy()
    for col in columns:
        if col not in use.columns:
            use[col] = 0.0
    return use[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def add_proto_features(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = train_df.copy()
    val = val_df.copy()
    cols = BASE_FEATURES
    x_train = feature_matrix(train, cols)
    x_val = feature_matrix(val, cols)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(imputer.fit_transform(x_train))
    x_val_z = scaler.transform(imputer.transform(x_val))

    y_train = train["target_class_id"].astype(int).to_numpy()
    gray_mask = y_train == GRAY
    collapse_mask = np.isin(y_train, [COLLAPSE_CANONICAL, 18])
    if int(gray_mask.sum()) < 2 or int(collapse_mask.sum()) < 2:
        raise RuntimeError(f"Too few prototype samples: gray={int(gray_mask.sum())}, collapse={int(collapse_mask.sum())}")
    gray_proto = x_train_z[gray_mask].mean(axis=0)
    collapse_proto = x_train_z[collapse_mask].mean(axis=0)

    def attach(df: pd.DataFrame, xz: np.ndarray) -> pd.DataFrame:
        out = df.copy()
        d_gray = np.linalg.norm(xz - gray_proto[None, :], axis=1)
        d_col = np.linalg.norm(xz - collapse_proto[None, :], axis=1)
        out["dist_to_gray_proto"] = d_gray
        out["dist_to_collapse_proto"] = d_col
        out["proto_margin_collapse_minus_gray"] = d_col - d_gray
        out["gray_relative_similarity"] = 1.0 / (1.0 + np.maximum(0.0, d_gray)) - 1.0 / (1.0 + np.maximum(0.0, d_col))
        return out

    info = {
        "gray_train_positive_candidates": int(gray_mask.sum()),
        "collapse_train_candidates": int(collapse_mask.sum()),
        "prototype_feature_columns": cols,
    }
    return attach(train, x_train_z), attach(val, x_val_z), info


def make_binary_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    target = df["target_class_id"].astype(int)
    original = df["class_id"].astype(int)
    # Positive: matched to 焊灰色. Negative: matched to collapse family or predicted as collapse-family background.
    eligible = (
        target.eq(GRAY)
        | target.eq(COLLAPSE_CANONICAL)
        | target.eq(18)
        | (target.eq(-1) & original.isin(COLLAPSE_FAMILY | {GRAY}))
    )
    work = df[eligible].copy()
    y = (work["target_class_id"].astype(int).to_numpy() == GRAY).astype(int)
    return work, y


def make_model(seed: int) -> Pipeline:
    clf = ExtraTreesClassifier(
        n_estimators=900,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])


def positive_prob(model: Pipeline, x: np.ndarray) -> np.ndarray:
    clf = model.named_steps["clf"]
    if 1 not in clf.classes_:
        return np.zeros((x.shape[0],), dtype=np.float32)
    return model.predict_proba(x)[:, list(clf.classes_).index(1)]


def parse_box(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(v) for v in ast.literal_eval(str(value))]


def base_selected(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        packed = row.to_dict()
        packed["original_class_id"] = int(packed["class_id"])
        packed["original_class_name"] = CLASS_NAMES.get(int(packed["class_id"]), str(packed["class_id"]))
        packed["gray_contrast_prob"] = 0.0
        packed["gray_contrast_action"] = "keep_original"
        packed["global_box"] = parse_box(packed["global_box"])
        rows.append(packed)
    return rows


def corrected_selected(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    min_rel: float,
    min_margin: float,
    max_original_score: float,
    require_proto_gray_better: bool,
    allowed_from: set[int],
) -> list[dict[str, Any]]:
    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        original = int(row["class_id"])
        final = original
        action = "keep_original"
        prob = float(probs[idx])
        proto_margin = float(row.get("proto_margin_collapse_minus_gray", 0.0))
        rel = float(row["reliability_v2_score"])
        score = float(row["max_pred_score"])
        if (
            original in allowed_from
            and prob >= threshold
            and rel >= min_rel
            and score <= max_original_score
            and proto_margin >= min_margin
            and (not require_proto_gray_better or proto_margin > 0.0)
        ):
            final = GRAY
            action = "collapse_to_gray_override"
        packed = row.to_dict()
        packed["original_class_id"] = original
        packed["original_class_name"] = CLASS_NAMES.get(original, str(original))
        packed["class_id"] = final
        packed["class_name"] = CLASS_NAMES.get(final, str(final))
        packed["gray_contrast_prob"] = round(prob, 6)
        packed["gray_proto_margin"] = round(proto_margin, 6)
        packed["gray_contrast_action"] = action
        packed["global_box"] = parse_box(packed["global_box"])
        rows.append(packed)
    return rows


def aggregate(class_rows: list[dict[str, Any]], ids: set[int], name: str) -> dict[str, Any]:
    subset = [r for r in class_rows if int(r["class_id"]) in ids]
    total = sum(int(r["gt_instances"]) for r in subset)
    correct = sum(int(r["correct"]) for r in subset)
    wrong = sum(int(r["wrong"]) for r in subset)
    miss = sum(int(r["miss"]) for r in subset)
    return {
        "group": name,
        "gt_instances": total,
        "correct": correct,
        "wrong": wrong,
        "miss": miss,
        "accuracy": correct / total if total else 0.0,
        "triple": f"{correct} / {wrong} / {miss}",
    }


def evaluate(parent, selected, source_val, val_strip_boxes, args, method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    match_families = parent.parse_match_families(["0,2,3", "12,14,15,18"])
    return parent.evaluate_instance_from_selected(
        selected,
        source_val,
        val_strip_boxes,
        args.coverage_threshold,
        args.match_iou,
        EXCLUDE,
        match_families,
        method,
    )


def method_row(method: str, selected: list[dict[str, Any]], summary: dict[str, Any], class_rows: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    gray = aggregate(class_rows, {GRAY}, "gray")
    tail = aggregate(class_rows, {8, 10, 20}, "focused_tail")  # 焊坑/长焊高/焊灰色 in the revised low-frequency set with val support.
    collapse = aggregate(class_rows, COLLAPSE_FAMILY, "collapse_family")
    row = {
        "method": method,
        "selected_groups": len(selected),
        "total_triple": f"{summary['correct']} / {summary['wrong']} / {summary['miss']}",
        "total_accuracy": summary["instance_accuracy"],
        "gray_triple": gray["triple"],
        "gray_accuracy": gray["accuracy"],
        "collapse_family_triple": collapse["triple"],
        "collapse_family_accuracy": collapse["accuracy"],
        "focused_tail_triple": tail["triple"],
        "focused_tail_accuracy": tail["accuracy"],
        "overrides": sum(1 for r in selected if r.get("gray_contrast_action") == "collapse_to_gray_override"),
    }
    if extra:
        row.update(extra)
    return row


def copy_gray_review_images(output_dir: Path, selected: list[dict[str, Any]]) -> None:
    val_visuals = ROOT / "zhwk_runs" / "dataset_class_distribution_20260911" / "tail_problem_class_visuals_v1" / "dfine_score040_焊灰色"
    if not val_visuals.exists():
        return
    dst = output_dir / "gray_review_images_from_previous_visuals"
    dst.mkdir(parents=True, exist_ok=True)
    for p in val_visuals.glob("*.jpg"):
        shutil.copy2(p, dst / p.name)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent = load_parent(args.scripts_root)
    core = load_feature_core(args.scripts_root)
    source_val = load_json(args.source_val_annotations)
    val_strip_boxes = load_json(args.val_strip_boxes)
    train_meta = core.strip_meta(args.train_strip_boxes)
    val_meta = core.strip_meta(args.val_strip_boxes)

    train_df = merge_data(args.train_evidence, args.train_targets, args.min_reliability)
    val_df = merge_data(args.val_evidence, args.val_targets, args.min_reliability)
    print(f"[GRAY-CONTRAST] merged evidence train={len(train_df)} val={len(val_df)}", flush=True)
    train_df = core.add_geometry_and_patch(train_df, args.strip_root, train_meta)
    val_df = core.add_geometry_and_patch(val_df, args.strip_root, val_meta)
    train_df, val_df, proto_info = add_proto_features(train_df, val_df)
    print("[GRAY-CONTRAST] patch and prototype features ready", flush=True)

    train_work, y_train = make_binary_dataset(train_df)
    val_work, y_val = make_binary_dataset(val_df)
    feature_cols = BASE_FEATURES + CONTRAST_FEATURES
    x_train = feature_matrix(train_work, feature_cols)
    x_val_work = feature_matrix(val_work, feature_cols)
    x_val_all = feature_matrix(val_df, feature_cols)
    model = make_model(args.seed)
    model.fit(x_train, y_train)
    val_work_prob = positive_prob(model, x_val_work)
    val_all_prob = positive_prob(model, x_val_all)
    train_prob = positive_prob(model, x_train)

    diagnostics = {
        **proto_info,
        "binary_train_candidates": int(len(train_work)),
        "binary_train_gray_positive": int(y_train.sum()),
        "binary_val_candidates": int(len(val_work)),
        "binary_val_gray_positive": int(y_val.sum()),
        "train_aupr": float(average_precision_score(y_train, train_prob)) if len(set(y_train)) > 1 else None,
        "val_aupr": float(average_precision_score(y_val, val_work_prob)) if len(set(y_val)) > 1 else None,
        "val_auroc": float(roc_auc_score(y_val, val_work_prob)) if len(set(y_val)) > 1 else None,
        "feature_columns": feature_cols,
    }

    all_method_rows: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []

    baseline = base_selected(val_df)
    summary, cls = evaluate(parent, baseline, source_val, val_strip_boxes, args, "baseline_grouped_scope")
    baseline_row = method_row("baseline_grouped_scope", baseline, summary, cls)
    all_method_rows.append(baseline_row)
    all_class_rows.extend([{**r, "method": "baseline_grouped_scope"} for r in cls])

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    min_rels = [0.05, 0.45, 0.70]
    min_margins = [-0.5, 0.0, 0.50]
    max_scores = [0.65, 0.85, 1.01]
    allowed_sets = {
        "collapse_only": COLLAPSE_FAMILY,
    }
    require_flags = [False, True]

    total = len(thresholds) * len(min_rels) * len(min_margins) * len(max_scores) * len(allowed_sets) * len(require_flags)
    idx = 0
    for th in thresholds:
        for rel in min_rels:
            for margin in min_margins:
                for score in max_scores:
                    for allow_name, allow in allowed_sets.items():
                        for req in require_flags:
                            idx += 1
                            method = f"gray_contrast_th{th:.2f}_rel{rel:.2f}_pm{margin:.2f}_maxs{score:.2f}_{allow_name}_req{int(req)}"
                            selected = corrected_selected(val_df, val_all_prob, th, rel, margin, score, req, allow)
                            summary, cls = evaluate(parent, selected, source_val, val_strip_boxes, args, method)
                            row = method_row(
                                method,
                                selected,
                                summary,
                                cls,
                                {
                                    "threshold": th,
                                    "min_reliability": rel,
                                    "min_proto_margin": margin,
                                    "max_original_score": score,
                                    "allowed_from": allow_name,
                                    "require_proto_gray_better": req,
                                },
                            )
                            all_method_rows.append(row)
                            all_class_rows.extend([{**r, "method": method} for r in cls])
                            if idx % 50 == 0 or idx == total:
                                print(f"[GRAY-CONTRAST] grid {idx}/{total}", flush=True)

    baseline_acc = float(baseline_row["total_accuracy"])
    candidates = [r for r in all_method_rows[1:] if float(r["total_accuracy"]) >= baseline_acc - 0.005 and int(r["overrides"]) > 0]
    best_conservative = max(candidates, key=lambda r: (float(r["gray_accuracy"]), float(r["total_accuracy"]), -int(r["overrides"]))) if candidates else baseline_row
    within_1pp = [r for r in all_method_rows[1:] if float(r["total_accuracy"]) >= baseline_acc - 0.010 and int(r["overrides"]) > 0]
    best_within_1pp = max(within_1pp, key=lambda r: (float(r["gray_accuracy"]), float(r["total_accuracy"]), -int(r["overrides"]))) if within_1pp else best_conservative
    best_gray = max(all_method_rows, key=lambda r: (float(r["gray_accuracy"]), float(r["total_accuracy"])))
    best_total = max(all_method_rows, key=lambda r: (float(r["total_accuracy"]), float(r["gray_accuracy"])))

    def selected_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
        if row["method"] == "baseline_grouped_scope":
            return baseline
        allow = allowed_sets[str(row["allowed_from"])]
        return corrected_selected(
            val_df,
            val_all_prob,
            float(row["threshold"]),
            float(row["min_reliability"]),
            float(row["min_proto_margin"]),
            float(row["max_original_score"]),
            bool(int(row["require_proto_gray_better"])) if isinstance(row["require_proto_gray_better"], (int, str)) else bool(row["require_proto_gray_better"]),
            allow,
        )

    outputs = {
        "baseline": baseline_row,
        "best_conservative": best_conservative,
        "best_within_1pp": best_within_1pp,
        "best_gray": best_gray,
        "best_total": best_total,
    }
    for name, row in outputs.items():
        write_csv(args.output_dir / f"{name}_val_predictions.csv", selected_from_row(row))

    chosen = selected_from_row(best_within_1pp)
    override_rows = [
        {
            "merged_id": r.get("merged_id"),
            "source_image_id": r.get("source_image_id"),
            "source_file": r.get("source_file"),
            "original_class_id": r.get("original_class_id"),
            "original_class_name": r.get("original_class_name"),
            "final_class_id": r.get("class_id"),
            "final_class_name": r.get("class_name"),
            "target_class_id": r.get("target_class_id"),
            "target_class_name": r.get("target_class_name"),
            "gray_contrast_prob": r.get("gray_contrast_prob"),
            "gray_proto_margin": r.get("gray_proto_margin"),
            "max_pred_score": r.get("max_pred_score"),
            "reliability_v2_score": r.get("reliability_v2_score"),
            "normal_reference_score": r.get("normal_reference_score"),
            "frequency_evidence_score": r.get("frequency_evidence_score"),
        }
        for r in chosen
        if r.get("gray_contrast_action") == "collapse_to_gray_override"
    ]
    write_csv(args.output_dir / "best_within_1pp_overrides.csv", override_rows)
    write_csv(args.output_dir / "gray_contrast_method_summary.csv", all_method_rows)
    write_csv(args.output_dir / "gray_contrast_by_class_all_methods.csv", all_class_rows)
    (args.output_dir / "gray_contrast_metrics.json").write_text(json.dumps({"diagnostics": diagnostics, **outputs}, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "diagnostics": diagnostics, "selected_methods": outputs, "feature_columns": feature_cols}, args.output_dir / "gray_vs_collapse_contrast_corrector.joblib")
    copy_gray_review_images(args.output_dir, chosen)

    lines = [
        "# 焊灰色 vs 缺焊长塌合并类：混淆类对比纠错实验 v1",
        "",
        "## Material Passport",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        "| 日期 | 2026-09-12 |",
        "| 操作 | 固定 D-FINE + IBO evidence，不重训检测器 |",
        "| 目标 | 判断被缺焊长塌合并类吸收的焊灰色能否通过类间对比纠正 |",
        "| 评价口径 | 烟类 0/2/3 合并；缺焊长塌类 12/14/15/18 合并；排除 11/17/19/21/22 |",
        "| 测试集 | 未使用 |",
        "",
        "## 1. 方法说明",
        "",
        "本实验不再只问“候选区域是否异常”，而是增加一层类间对比：候选区域同时与 `焊灰色` 原型和 `缺焊长塌合并类` 原型比较。只有当焊灰色概率、IBO reliability、原型距离差等条件同时满足时，才允许把原本属于缺焊长塌合并类的候选纠正为 `焊灰色`。",
        "",
        "输入特征包括 D-FINE 分数、IBO reliability、正常参照距离、频率证据、框几何、patch 灰度/边缘/FFT 特征，以及 `离焊灰色原型距离 - 离缺焊长塌原型距离` 这一类间对比特征。",
        "",
        "## 2. 候选级诊断",
        "",
        f"- 训练候选总数：{diagnostics['binary_train_candidates']}",
        f"- 训练焊灰色正候选：{diagnostics['binary_train_gray_positive']}",
        f"- 验证候选总数：{diagnostics['binary_val_candidates']}",
        f"- 验证焊灰色正候选：{diagnostics['binary_val_gray_positive']}",
        f"- 候选级 val AUPR：{diagnostics['val_aupr']:.4f}" if diagnostics["val_aupr"] is not None else "- 候选级 val AUPR：NA",
        f"- 候选级 val AUROC：{diagnostics['val_auroc']:.4f}" if diagnostics["val_auroc"] is not None else "- 候选级 val AUROC：NA",
        "",
        "## 3. 实例级结果",
        "",
        "| 方法 | 总体 正确/错类/漏检 | 总准确率 | 焊灰色 正确/错类/漏检 | 焊灰色准确率 | 缺焊长塌合并类 正确/错类/漏检 | 改类数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in outputs.items():
        lines.append(
            f"| {label} | {row['total_triple']} | {pct(float(row['total_accuracy']))} | "
            f"{row['gray_triple']} | {pct(float(row['gray_accuracy']))} | {row['collapse_family_triple']} | {row['overrides']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 结论",
            "",
        ]
    )
    if float(best_within_1pp["gray_accuracy"]) > float(baseline_row["gray_accuracy"]):
        lines.append(
            f"在总准确率下降不超过 1 个百分点的约束下，焊灰色由 `{baseline_row['gray_triple']}` 提升到 `{best_within_1pp['gray_triple']}`，说明“混淆类对比”确实比单纯正常参照更能针对 `焊灰色→缺焊长塌` 的吸收问题。"
        )
    else:
        lines.append(
            "在当前候选与特征下，混淆类对比没有稳定提高焊灰色，说明现有 IBO/patch 特征仍不足以安全区分焊灰色和缺焊长塌合并类。"
        )
    lines.extend(
        [
            "",
            "## 5. 输出文件",
            "",
            "- `gray_contrast_method_summary.csv`：全部阈值组合结果。",
            "- `gray_contrast_by_class_all_methods.csv`：逐类别结果。",
            "- `best_within_1pp_val_predictions.csv`：推荐折中方案预测。",
            "- `best_within_1pp_overrides.csv`：推荐折中方案实际改类清单。",
            "- `gray_vs_collapse_contrast_corrector.joblib`：训练得到的轻量对比纠错器。",
            "- `gray_review_images_from_previous_visuals/`：焊灰色相关复核图。",
        ]
    )
    report = args.output_dir / "焊灰色_vs缺焊长塌合并类_混淆类对比纠错实验报告_v1.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostics": diagnostics, **outputs, "report": str(report)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
