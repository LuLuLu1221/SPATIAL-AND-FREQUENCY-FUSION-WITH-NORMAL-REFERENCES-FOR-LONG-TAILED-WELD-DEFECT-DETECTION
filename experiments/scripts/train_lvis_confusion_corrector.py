#!/usr/bin/env python
"""Train and apply a lightweight LVIS IBO confusion-class corrector.

The corrector is intentionally a second-stage post-processor:

* training labels are built from train candidates that overlap a GT instance;
* prediction features do not use GT overlap fields, so applying to val is not
  label-leaking;
* output is a corrected IBO manifest plus a reliability score CSV, compatible
  with evaluate_ibo_reliability_postprocess.py.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--val-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--train-reliability-threshold", type=float, default=0.10)
    parser.add_argument("--apply-reliability-threshold", type=float, default=0.15)
    parser.add_argument("--min-correction-proba", type=float, default=0.35)
    parser.add_argument("--min-correction-margin", type=float, default=0.10)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=120000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def parse_box(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    return [float(v) for v in value]


def add_box_features(df: pd.DataFrame, box_col: str, prefix: str) -> pd.DataFrame:
    boxes = df[box_col].apply(parse_box)
    arr = np.asarray(boxes.to_list(), dtype=np.float32)
    if arr.shape[1] == 4:
        x0, y0, x1_or_w, y1_or_h = arr.T
        if prefix == "pred":
            width = np.maximum(1e-6, x1_or_w - x0)
            height = np.maximum(1e-6, y1_or_h - y0)
            cx = x0 + width / 2.0
            cy = y0 + height / 2.0
        else:
            width = np.maximum(1e-6, x1_or_w)
            height = np.maximum(1e-6, y1_or_h)
            cx = x0 + width / 2.0
            cy = y0 + height / 2.0
        df[f"{prefix}_cx"] = cx
        df[f"{prefix}_cy"] = cy
        df[f"{prefix}_w"] = width
        df[f"{prefix}_h"] = height
        df[f"{prefix}_area"] = width * height
        df[f"{prefix}_aspect"] = width / np.maximum(height, 1e-6)
    return df


def manifest_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    keep = [
        "ibo_id",
        "pred_class_id",
        "pred_class_name",
        "pred_score",
        "pred_xyxy",
        "local_box_xywh",
        "matched_gt_class_id",
        "matched_gt_class_name",
        "best_coverage_any",
        "best_coverage_same",
        "candidate_outcome",
    ]
    slim: list[dict[str, Any]] = []
    for row in rows:
        slim.append({key: row.get(key) for key in keep if key in row})
    return pd.DataFrame(slim)


def build_feature_table(manifest_rows: list[dict[str, Any]], evidence_path: Path) -> pd.DataFrame:
    manifest = manifest_to_df(manifest_rows)
    evidence = pd.read_csv(evidence_path)
    evidence = evidence[
        [
            "ibo_id",
            "frequency_evidence_score",
            "normal_reference_distance",
            "reliability_fusion_score",
            "feature_resize_scale",
            "local_box_xywh",
        ]
    ]
    df = manifest.merge(evidence, on="ibo_id", how="inner")
    df = add_box_features(df, "pred_xyxy", "pred")
    df = add_box_features(df, "local_box_xywh", "local")
    for col in [
        "pred_class_id",
        "pred_score",
        "frequency_evidence_score",
        "normal_reference_distance",
        "reliability_fusion_score",
        "feature_resize_scale",
        "best_coverage_any",
        "best_coverage_same",
        "matched_gt_class_id",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


FEATURE_COLUMNS = [
    "pred_class_id",
    "pred_score",
    "frequency_evidence_score",
    "normal_reference_distance",
    "reliability_fusion_score",
    "feature_resize_scale",
    "pred_cx",
    "pred_cy",
    "pred_w",
    "pred_h",
    "pred_area",
    "pred_aspect",
    "local_cx",
    "local_cy",
    "local_w",
    "local_h",
    "local_area",
    "local_aspect",
]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    train_df = build_feature_table(train_rows, args.train_evidence)
    val_df = build_feature_table(val_rows, args.val_evidence)

    covered = train_df[
        (train_df["best_coverage_any"].fillna(0.0) >= args.coverage_threshold)
        & (train_df["reliability_fusion_score"].fillna(0.0) >= args.train_reliability_threshold)
        & (train_df["matched_gt_class_id"].notna())
    ].copy()
    covered["target_class_id"] = covered["matched_gt_class_id"].astype(int)
    if len(covered) > args.max_train:
        covered = covered.sample(n=args.max_train, random_state=args.random_state)

    labels = covered["target_class_id"].astype(int)
    min_count = labels.value_counts().min()
    stratify = labels if min_count >= 2 else None
    train_part, holdout = train_test_split(
        covered,
        test_size=0.20,
        random_state=args.random_state,
        stratify=stratify,
    )

    encoder = LabelEncoder()
    encoder.fit(covered["target_class_id"].astype(int))
    y_train = encoder.transform(train_part["target_class_id"].astype(int))
    y_holdout = encoder.transform(holdout["target_class_id"].astype(int))

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    model.fit(train_part[FEATURE_COLUMNS].fillna(0.0), y_train)

    holdout_proba = model.predict_proba(holdout[FEATURE_COLUMNS].fillna(0.0))
    holdout_pred = encoder.inverse_transform(np.argmax(holdout_proba, axis=1)).astype(int)
    holdout_acc = float(accuracy_score(holdout["target_class_id"].astype(int), holdout_pred))
    holdout_bal = float(balanced_accuracy_score(holdout["target_class_id"].astype(int), holdout_pred))

    val_proba = model.predict_proba(val_df[FEATURE_COLUMNS].fillna(0.0))
    order = np.argsort(val_proba, axis=1)
    top_idx = order[:, -1]
    second_idx = order[:, -2] if val_proba.shape[1] >= 2 else order[:, -1]
    top_proba = val_proba[np.arange(len(val_df)), top_idx]
    second_proba = val_proba[np.arange(len(val_df)), second_idx]
    corrected_class = encoder.inverse_transform(top_idx).astype(int)

    id_to_prediction: dict[str, dict[str, Any]] = {}
    for i, row in val_df.reset_index(drop=True).iterrows():
        original = int(row["pred_class_id"])
        proposed = int(corrected_class[i])
        margin = float(top_proba[i] - second_proba[i])
        apply = (
            float(row["reliability_fusion_score"]) >= args.apply_reliability_threshold
            and proposed != original
            and float(top_proba[i]) >= args.min_correction_proba
            and margin >= args.min_correction_margin
        )
        id_to_prediction[str(row["ibo_id"])] = {
            "ibo_id": str(row["ibo_id"]),
            "original_pred_class_id": original,
            "corrected_pred_class_id": proposed if apply else original,
            "proposed_class_id": proposed,
            "correction_applied": bool(apply),
            "correction_proba": float(top_proba[i]),
            "correction_margin": margin,
            "reliability_score": float(row["reliability_fusion_score"]),
        }

    corrected_rows: list[dict[str, Any]] = []
    applied = 0
    for row in val_rows:
        item = dict(row)
        pred = id_to_prediction.get(str(item["ibo_id"]))
        if pred and pred["correction_applied"]:
            item["original_pred_class_id"] = item["pred_class_id"]
            item["original_pred_class_name"] = item.get("pred_class_name", "")
            item["pred_class_id"] = int(pred["corrected_pred_class_id"])
            item["pred_class_name"] = f"corrected_class_{item['pred_class_id']}"
            item["confusion_correction_applied"] = True
            item["confusion_correction_proba"] = pred["correction_proba"]
            item["confusion_correction_margin"] = pred["correction_margin"]
            applied += 1
        else:
            item["confusion_correction_applied"] = False
        corrected_rows.append(item)

    score_rows = [
        {
            "ibo_id": pred["ibo_id"],
            "reliability_score": pred["reliability_score"],
        }
        for pred in id_to_prediction.values()
    ]
    prediction_rows = list(id_to_prediction.values())

    write_jsonl(args.output_dir / "ibo_manifest_val_confusion_corrected.jsonl", corrected_rows)
    write_csv(args.output_dir / "lvis_ibo_reliability_scores_val_confusion_corrected.csv", score_rows)
    write_csv(args.output_dir / "lvis_ibo_confusion_corrector_predictions_val.csv", prediction_rows)
    joblib.dump({"model": model, "label_encoder": encoder, "feature_columns": FEATURE_COLUMNS}, args.output_dir / "lvis_ibo_confusion_corrector.joblib")

    summary = {
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "train_evidence": str(args.train_evidence),
        "val_evidence": str(args.val_evidence),
        "coverage_threshold": args.coverage_threshold,
        "train_reliability_threshold": args.train_reliability_threshold,
        "apply_reliability_threshold": args.apply_reliability_threshold,
        "min_correction_proba": args.min_correction_proba,
        "min_correction_margin": args.min_correction_margin,
        "train_covered_candidates": int(len(covered)),
        "holdout_candidates": int(len(holdout)),
        "holdout_accuracy": holdout_acc,
        "holdout_balanced_accuracy": holdout_bal,
        "val_candidates": int(len(val_df)),
        "val_corrections_applied": int(applied),
    }
    (args.output_dir / "lvis_ibo_confusion_corrector_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "[CONF-CORRECTOR] "
        f"train={len(train_part)} holdout={len(holdout)} "
        f"holdout_acc={holdout_acc:.4f} holdout_bal={holdout_bal:.4f} "
        f"val={len(val_df)} applied={applied} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
