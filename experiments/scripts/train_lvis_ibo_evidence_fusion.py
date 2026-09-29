#!/usr/bin/env python
"""Train LVIS/SimLTD IBO evidence fusion on no-stretch candidate crops.

This script is the non-cyclic counterpart of the weld-strip IBO evidence
fusion stage.  It consumes IBO manifests produced by
build_ibo_lvis_nostretch_from_dfine_candidates.py and directly extracts:

- I/B/O spatial transition evidence;
- local Haar-frequency transition evidence;
- normal-reference distance using training background candidates;
- a lightweight reliability-fusion classifier.

It deliberately does not assume circular weld ROI, strip coordinates, or cyclic
merge.  All candidate geometry is recovered from pred_xyxy and crop_xyxy in the
IBO manifest.
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

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POSITIVE_OUTCOMES = {"correct", "partial_same_class"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--train-ibo-dir", type=Path, required=True)
    parser.add_argument("--val-ibo-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normal-samples", type=int, default=12000)
    parser.add_argument("--normal-k", type=int, default=7)
    parser.add_argument("--max-train", type=int, default=0, help="Debug cap; 0 means all train candidates.")
    parser.add_argument("--max-val", type=int, default=0, help="Debug cap; 0 means all val candidates.")
    parser.add_argument(
        "--max-feature-side",
        type=int,
        default=256,
        help="Resize large IBO crops before feature extraction to avoid memory spikes. 0 disables.",
    )
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


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


def read_bgr(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def load_transition_features(scripts_root: Path):
    sys.path.insert(0, str(scripts_root))
    from build_normal_reference_evidence_bank import transition_features

    return transition_features


def xyxy_to_local_xywh(row: dict[str, Any], crop_shape: tuple[int, int]) -> list[float]:
    pred = [float(v) for v in row["pred_xyxy"]]
    crop = [float(v) for v in row["crop_xyxy"]]
    height, width = crop_shape
    x0 = max(0.0, pred[0] - crop[0])
    y0 = max(0.0, pred[1] - crop[1])
    x1 = min(float(width), pred[2] - crop[0])
    y1 = min(float(height), pred[3] - crop[1])
    if x1 - x0 < 2 or y1 - y0 < 2:
        # Fallback to the center of the crop when a degenerate clipped box is
        # produced by padding/rounding at image borders.
        bw = max(4.0, min(float(width), abs(pred[2] - pred[0])))
        bh = max(4.0, min(float(height), abs(pred[3] - pred[1])))
        x0 = max(0.0, (width - bw) / 2.0)
        y0 = max(0.0, (height - bh) / 2.0)
        x1 = min(float(width), x0 + bw)
        y1 = min(float(height), y0 + bh)
    return [x0, y0, max(2.0, x1 - x0), max(2.0, y1 - y0)]


def resize_for_features(
    bgr: np.ndarray,
    local_box: list[float],
    max_side: int,
) -> tuple[np.ndarray, list[float], float]:
    if max_side <= 0:
        return bgr, local_box, 1.0
    height, width = bgr.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return bgr, local_box, 1.0
    scale = float(max_side) / float(largest)
    new_width = max(8, int(round(width * scale)))
    new_height = max(8, int(round(height * scale)))
    resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    scaled_box = [
        float(local_box[0]) * scale,
        float(local_box[1]) * scale,
        max(2.0, float(local_box[2]) * scale),
        max(2.0, float(local_box[3]) * scale),
    ]
    return resized, scaled_box, scale


def dct_features(gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray.astype(np.float32) / 255.0, (64, 64), interpolation=cv2.INTER_AREA)
    coeff = cv2.dct(resized)
    energy = np.square(coeff)
    yy, xx = np.indices((64, 64))
    radius = np.sqrt(np.square(xx) + np.square(yy))
    total = float(energy.sum()) + 1e-8
    return np.asarray(
        [
            float(energy[radius <= 8].sum()) / total,
            float(energy[(radius > 8) & (radius <= 24)].sum()) / total,
            float(energy[radius > 24].sum()) / total,
            float(np.max(energy) / total),
        ],
        dtype=np.float32,
    )


def crop_stats(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    norm = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    return np.concatenate(
        [
            hsv.reshape(-1, 3).mean(axis=0) / 255.0,
            hsv.reshape(-1, 3).std(axis=0) / 255.0,
            lab.reshape(-1, 3).mean(axis=0) / 255.0,
            lab.reshape(-1, 3).std(axis=0) / 255.0,
            np.asarray(
                [mag.mean(), mag.std(), np.cos(2.0 * ang).mean(), np.sin(2.0 * ang).mean()],
                dtype=np.float32,
            ),
            dct_features(gray),
        ]
    ).astype(np.float32)


def resize_for_feature(
    bgr: np.ndarray,
    local_box_xywh: list[float],
    max_side: int = 256,
) -> tuple[np.ndarray, list[float]]:
    """Limit crop size before feature extraction to avoid memory spikes.

    The I/B/O geometry is scaled together with the crop, so the extracted
    transition still represents the same candidate-relative evidence.  This is
    especially important for LVIS boxes whose context crop can be much larger
    than the original weld-strip crops.
    """
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return bgr, local_box_xywh
    scale = float(max_side) / float(longest)
    new_width = max(8, int(round(width * scale)))
    new_height = max(8, int(round(height * scale)))
    resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    scaled_box = [
        float(local_box_xywh[0]) * scale,
        float(local_box_xywh[1]) * scale,
        max(2.0, float(local_box_xywh[2]) * scale),
        max(2.0, float(local_box_xywh[3]) * scale),
    ]
    return resized, scaled_box


def image_feature_vector(bgr: np.ndarray, local_box_xywh: list[float], transition_features) -> np.ndarray:
    bgr, local_box_xywh = resize_for_feature(bgr, local_box_xywh)
    try:
        spatial, frequency = transition_features(bgr, local_box_xywh, inner_ratio=0.16, ring_ratio=0.22)
    except Exception:
        spatial = np.full(20, np.nan, dtype=np.float32)
        frequency = np.full(14, np.nan, dtype=np.float32)
    vector = np.concatenate([spatial, frequency, crop_stats(bgr)]).astype(np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_rows(
    split: str,
    rows: list[dict[str, Any]],
    ibo_dir: Path,
    transition_features,
    max_feature_side: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[np.ndarray]]:
    x_base: list[np.ndarray] = []
    y: list[int] = []
    out_rows: list[dict[str, Any]] = []
    raw_image_features: list[np.ndarray] = []
    for index, row in enumerate(rows, 1):
        crop_path = ibo_dir / str(row["crop_path"])
        bgr = read_bgr(crop_path)
        if bgr is None:
            continue
        local_box_original = xyxy_to_local_xywh(row, bgr.shape[:2])
        bgr_feature, local_box, resize_scale = resize_for_features(bgr, local_box_original, max_feature_side)
        image_vec = image_feature_vector(bgr_feature, local_box, transition_features)
        raw_image_features.append(image_vec)

        box_w = max(1.0, float(local_box[2]))
        box_h = max(1.0, float(local_box[3]))
        crop_h, crop_w = bgr_feature.shape[:2]
        pred_score = float(row["pred_score"])
        image_vec = np.nan_to_num(image_vec, nan=0.0, posinf=0.0, neginf=0.0)
        frequency_block = image_vec[20:34]
        dct_high = image_vec[-2]
        dct_conc = image_vec[-1]
        frequency_score = float(np.mean(np.abs(frequency_block)) + dct_high + dct_conc)
        metadata_vec = np.asarray(
            [
                pred_score,
                1.0 if row.get("confidence_group") == "high_conf" else 0.0,
                math.log1p(box_w),
                math.log1p(box_h),
                math.log1p(box_w * box_h),
                box_w / max(1.0, float(crop_w)),
                box_h / max(1.0, float(crop_h)),
                frequency_score,
            ],
            dtype=np.float32,
        )
        x_base.append(np.concatenate([metadata_vec, image_vec]).astype(np.float32))
        label = 1 if str(row["candidate_outcome"]) in POSITIVE_OUTCOMES else 0
        y.append(label)
        out_rows.append(
            {
                "split": split,
                "ibo_id": row["ibo_id"],
                "tile_image_id": row["tile_image_id"],
                "tile_file": row["tile_file"],
                "pred_class_id": row["pred_class_id"],
                "pred_class_name": row["pred_class_name"],
                "pred_score": row["pred_score"],
                "confidence_group": row["confidence_group"],
                "candidate_outcome": row["candidate_outcome"],
                "label": label,
                "frequency_evidence_score": round(frequency_score, 6),
                "crop_path": row["crop_path"],
                "local_box_xywh": [round(float(v), 3) for v in local_box],
                "feature_resize_scale": round(float(resize_scale), 6),
            }
        )
        if index % 5000 == 0 or index == len(rows):
            print(f"[LVIS-IBO-FUSION] {split}: features {index}/{len(rows)}", flush=True)
    return np.vstack(x_base), np.asarray(y, dtype=np.int64), out_rows, raw_image_features


def thresholds(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def summarize_thresholds(y_true: np.ndarray, scores: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds(args.threshold_start, args.threshold_stop, args.threshold_step):
        pred = (scores >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1": round(float(f1), 6),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "selected": int(tp + fp),
            }
        )
    return rows


def add_scores(rows: list[dict[str, Any]], normal_scores: np.ndarray, reliability_scores: np.ndarray) -> None:
    for row, normal_score, reliability in zip(rows, normal_scores, reliability_scores):
        row["normal_reference_distance"] = round(float(normal_score), 6)
        row["reliability_fusion_score"] = round(float(reliability), 6)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    transition_features = load_transition_features(args.scripts_root)

    train_rows = read_jsonl(args.train_manifest, args.max_train)
    val_rows = read_jsonl(args.val_manifest, args.max_val)
    print(f"[LVIS-IBO-FUSION] train_rows={len(train_rows)} val_rows={len(val_rows)}", flush=True)

    train_x_base, train_y, train_out, train_image_features = build_feature_rows(
        "train", train_rows, args.train_ibo_dir, transition_features, args.max_feature_side
    )
    val_x_base, val_y, val_out, val_image_features = build_feature_rows(
        "val", val_rows, args.val_ibo_dir, transition_features, args.max_feature_side
    )

    background_indices = [
        index for index, row in enumerate(train_out)
        if row["candidate_outcome"] == "background"
    ]
    if not background_indices:
        background_indices = [index for index, label in enumerate(train_y) if label == 0]
    sample_size = min(args.normal_samples, len(background_indices))
    chosen = rng.choice(background_indices, size=sample_size, replace=False)
    normal_bank = np.nan_to_num(
        np.vstack([train_image_features[int(index)] for index in chosen]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    normal_scaler = StandardScaler()
    normal_scaled = normal_scaler.fit_transform(normal_bank)
    normal_nn = NearestNeighbors(n_neighbors=min(args.normal_k, len(normal_bank)), metric="euclidean")
    normal_nn.fit(normal_scaled)

    train_img_scaled = normal_scaler.transform(
        np.nan_to_num(np.vstack(train_image_features), nan=0.0, posinf=0.0, neginf=0.0)
    )
    val_img_scaled = normal_scaler.transform(
        np.nan_to_num(np.vstack(val_image_features), nan=0.0, posinf=0.0, neginf=0.0)
    )
    train_normal_scores = normal_nn.kneighbors(train_img_scaled, return_distance=True)[0].mean(axis=1)
    val_normal_scores = normal_nn.kneighbors(val_img_scaled, return_distance=True)[0].mean(axis=1)

    train_x = np.nan_to_num(
        np.concatenate([train_x_base, train_normal_scores[:, None]], axis=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    val_x = np.nan_to_num(
        np.concatenate([val_x_base, val_normal_scores[:, None]], axis=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=320,
                    random_state=args.seed,
                    class_weight="balanced",
                    min_samples_leaf=3,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    train_scores = model.predict_proba(train_x)[:, 1]
    val_scores = model.predict_proba(val_x)[:, 1]

    add_scores(train_out, train_normal_scores, train_scores)
    add_scores(val_out, val_normal_scores, val_scores)
    write_csv(args.output_dir / "lvis_ibo_evidence_train.csv", train_out)
    write_csv(args.output_dir / "lvis_ibo_evidence_val.csv", val_out)
    write_csv(
        args.output_dir / "normal_reference_bank_background_candidates.csv",
        [train_out[int(index)] | {"normal_bank_index": i} for i, index in enumerate(chosen)],
    )

    metrics = {
        "train_candidates": len(train_out),
        "val_candidates": len(val_out),
        "train_positive": int(train_y.sum()),
        "train_negative": int((1 - train_y).sum()),
        "val_positive": int(val_y.sum()),
        "val_negative": int((1 - val_y).sum()),
        "normal_reference_source": "training_background_ibo_candidates",
        "normal_reference_rows": int(len(normal_bank)),
        "normal_k": int(min(args.normal_k, len(normal_bank))),
        "max_feature_side": int(args.max_feature_side),
        "train_outcome_counts": dict(Counter(row["candidate_outcome"] for row in train_out)),
        "val_outcome_counts": dict(Counter(row["candidate_outcome"] for row in val_out)),
        "val_auroc": float(roc_auc_score(val_y, val_scores)) if len(np.unique(val_y)) > 1 else None,
        "val_aupr": float(average_precision_score(val_y, val_scores)) if len(np.unique(val_y)) > 1 else None,
        "val_brier": float(brier_score_loss(val_y, val_scores)) if len(np.unique(val_y)) > 1 else None,
        "val_classification_report_t050": classification_report(
            val_y, (val_scores >= 0.5).astype(np.int64), output_dict=True, zero_division=0
        ),
    }
    (args.output_dir / "lvis_ibo_evidence_fusion_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sweep = summarize_thresholds(val_y, val_scores, args)
    write_csv(args.output_dir / "lvis_ibo_reliability_threshold_sweep.csv", sweep)
    best_f1 = max(sweep, key=lambda row: float(row["f1"])) if sweep else {}

    lines = [
        "# LVIS/SimLTD IBO 正常参照—频率证据可靠性融合结果",
        "",
        f"- 训练候选数：{len(train_out)}",
        f"- 验证候选数：{len(val_out)}",
        f"- 训练正/负样本：{int(train_y.sum())} / {int((1 - train_y).sum())}",
        f"- 验证正/负样本：{int(val_y.sum())} / {int((1 - val_y).sum())}",
        f"- 正常参照库：training background IBO candidates，样本数 {len(normal_bank)}",
        f"- Val AUROC：{metrics['val_auroc']:.6f}" if metrics["val_auroc"] is not None else "- Val AUROC：NA",
        f"- Val AUPR：{metrics['val_aupr']:.6f}" if metrics["val_aupr"] is not None else "- Val AUPR：NA",
        f"- Val Brier：{metrics['val_brier']:.6f}" if metrics["val_brier"] is not None else "- Val Brier：NA",
        "",
        "## 最佳 F1 阈值",
        "",
        json.dumps(best_f1, ensure_ascii=False, indent=2),
        "",
        "## 输出文件",
        "",
        "- `lvis_ibo_evidence_train.csv`",
        "- `lvis_ibo_evidence_val.csv`",
        "- `normal_reference_bank_background_candidates.csv`",
        "- `lvis_ibo_evidence_fusion_metrics.json`",
        "- `lvis_ibo_reliability_threshold_sweep.csv`",
    ]
    (args.output_dir / "LVIS_IBO正常参照频率证据可靠性融合报告.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model": model,
            "normal_scaler": normal_scaler,
            "normal_nn": normal_nn,
            "normal_k": int(min(args.normal_k, len(normal_bank))),
            "positive_outcomes": sorted(POSITIVE_OUTCOMES),
        },
        args.output_dir / "lvis_ibo_evidence_fusion.joblib",
    )
    print(
        f"[LVIS-IBO-FUSION] done output={args.output_dir} "
        f"val_auroc={metrics['val_auroc']} val_aupr={metrics['val_aupr']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
