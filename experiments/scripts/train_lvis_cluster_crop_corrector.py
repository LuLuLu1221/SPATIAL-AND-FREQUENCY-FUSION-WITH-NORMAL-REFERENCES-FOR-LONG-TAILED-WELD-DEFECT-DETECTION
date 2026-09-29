#!/usr/bin/env python
"""Train local confusion-cluster crop visual correctors for LVIS IBO candidates.

This is a non-leaky implementation of the paper-method component
"confusable-class contrast correction":

* confusion clusters are discovered from TRAIN candidates only;
* labels come from TRAIN candidate-to-GT coverage;
* VAL application uses only candidate-visible features: crop image descriptor,
  D-FINE class/score, reliability, normal-reference distance, frequency score,
  and geometry;
* output is a corrected val IBO manifest compatible with the existing
  evaluate_ibo_reliability_postprocess.py script.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--val-evidence", type=Path, required=True)
    parser.add_argument("--train-crop-root", type=Path, required=True)
    parser.add_argument("--val-crop-root", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--val-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--cluster-reliability-threshold", type=float, default=0.15)
    parser.add_argument("--apply-reliability-threshold", type=float, default=0.15)
    parser.add_argument("--top-confusion-edges", type=int, default=30)
    parser.add_argument("--min-edge-count", type=int, default=25)
    parser.add_argument("--min-cluster-train", type=int, default=80)
    parser.add_argument("--max-cluster-train", type=int, default=12000)
    parser.add_argument("--min-correction-proba", type=float, default=0.45)
    parser.add_argument("--min-correction-margin", type=float, default=0.10)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=22)
    parser.add_argument("--random-state", type=int, default=0)
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_box(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    return [float(v) for v in value]


def safe_hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist = cv2.calcHist([values.astype(np.float32)], [0], None, [bins], value_range).reshape(-1)
    total = float(hist.sum())
    return (hist / total if total > 0 else hist).astype(np.float32)


def dct_ring_features(gray: np.ndarray, size: int = 64) -> np.ndarray:
    resized = cv2.resize(gray.astype(np.float32) / 255.0, (size, size), interpolation=cv2.INTER_AREA)
    coeff = cv2.dct(resized)
    energy = np.square(coeff)
    yy, xx = np.indices((size, size))
    radius = np.sqrt(np.square(xx) + np.square(yy))
    total = float(energy.sum()) + 1e-8
    rings = [
        radius <= 4,
        (radius > 4) & (radius <= 8),
        (radius > 8) & (radius <= 16),
        (radius > 16) & (radius <= 24),
        radius > 24,
    ]
    feats = [float(energy[mask].sum()) / total for mask in rings]
    feats.append(float(np.max(energy)) / total)
    return np.asarray(feats, dtype=np.float32)


def imread_bgr(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def crop_visual_descriptor(crop_path: Path, source_image_path: Path | None = None, crop_box: Any = None) -> np.ndarray:
    image = None
    if crop_path.exists():
        image = imread_bgr(crop_path)
    if (image is None or image.size == 0) and source_image_path is not None and source_image_path.exists():
        source = imread_bgr(source_image_path)
        if source is not None and source.size > 0:
            box = parse_box(crop_box) if crop_box is not None else [0, 0, source.shape[1], source.shape[0]]
            x0 = int(max(0, min(source.shape[1] - 1, math.floor(box[0]))))
            y0 = int(max(0, min(source.shape[0] - 1, math.floor(box[1]))))
            x1 = int(max(x0 + 1, min(source.shape[1], math.ceil(box[2]))))
            y1 = int(max(y0 + 1, min(source.shape[0], math.ceil(box[3]))))
            image = source[y0:y1, x0:x1].copy()
    if image is None or image.size == 0:
        return np.zeros(75, dtype=np.float32)
    max_side = max(image.shape[:2])
    if max_side > 160:
        scale = 160.0 / max_side
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    angle_bins = np.floor(((ang + np.pi) / (2 * np.pi)) * 8).astype(np.int32)
    angle_bins = np.clip(angle_bins, 0, 7)
    grad_hist = np.zeros(8, dtype=np.float32)
    for idx in range(8):
        grad_hist[idx] = float(mag[angle_bins == idx].sum())
    grad_hist /= float(grad_hist.sum()) + 1e-8
    pieces = [
        hsv.reshape(-1, 3).mean(axis=0) / 255.0,
        hsv.reshape(-1, 3).std(axis=0) / 255.0,
        lab.reshape(-1, 3).mean(axis=0) / 255.0,
        lab.reshape(-1, 3).std(axis=0) / 255.0,
        safe_hist(hsv[:, :, 0], 12, (0, 180)),
        safe_hist(hsv[:, :, 1], 8, (0, 256)),
        safe_hist(hsv[:, :, 2], 8, (0, 256)),
        safe_hist(gray, 16, (0, 256)),
        np.asarray([float(mag.mean()), float(mag.std()), float(np.percentile(mag, 90)), float(np.cos(2.0 * ang).mean()), float(np.sin(2.0 * ang).mean())], dtype=np.float32),
        grad_hist,
        dct_ring_features(gray),
    ]
    return np.concatenate(pieces).astype(np.float32)


def add_box_features(df: pd.DataFrame) -> pd.DataFrame:
    boxes = np.asarray(df["pred_xyxy"].apply(parse_box).to_list(), dtype=np.float32)
    x0, y0, x1, y1 = boxes.T
    w = np.maximum(1e-6, x1 - x0)
    h = np.maximum(1e-6, y1 - y0)
    df["box_w"] = w
    df["box_h"] = h
    df["box_area"] = w * h
    df["box_aspect"] = w / h
    df["box_cx"] = x0 + w / 2.0
    df["box_cy"] = y0 + h / 2.0
    return df


def build_table(manifest_rows: list[dict[str, Any]], evidence_path: Path) -> pd.DataFrame:
    manifest = pd.DataFrame(
        [
            {
                "ibo_id": row.get("ibo_id"),
                "pred_class_id": row.get("pred_class_id"),
                "pred_class_name": row.get("pred_class_name", ""),
                "pred_score": row.get("pred_score", 0.0),
                "pred_xyxy": row.get("pred_xyxy"),
                "crop_xyxy": row.get("crop_xyxy"),
                "crop_path": row.get("crop_path", ""),
                "tile_file": row.get("tile_file", row.get("source_file", "")),
                "matched_gt_class_id": row.get("matched_gt_class_id"),
                "matched_gt_class_name": row.get("matched_gt_class_name", ""),
                "best_coverage_any": row.get("best_coverage_any", 0.0),
                "best_coverage_same": row.get("best_coverage_same", 0.0),
            }
            for row in manifest_rows
        ]
    )
    evidence = pd.read_csv(evidence_path)[
        [
            "ibo_id",
            "frequency_evidence_score",
            "normal_reference_distance",
            "reliability_fusion_score",
        ]
    ]
    df = manifest.merge(evidence, on="ibo_id", how="inner")
    for col in [
        "pred_class_id",
        "pred_score",
        "matched_gt_class_id",
        "best_coverage_any",
        "best_coverage_same",
        "frequency_evidence_score",
        "normal_reference_distance",
        "reliability_fusion_score",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_box_features(df)
    return df


def discover_clusters(train_df: pd.DataFrame, reliability_threshold: float, coverage_threshold: float, top_edges: int, min_edge_count: int) -> tuple[list[set[int]], list[dict[str, Any]]]:
    eligible = train_df[
        (train_df["reliability_fusion_score"].fillna(0.0) >= reliability_threshold)
        & (train_df["best_coverage_any"].fillna(0.0) >= coverage_threshold)
        & (train_df["matched_gt_class_id"].notna())
    ].copy()
    eligible["target_class_id"] = eligible["matched_gt_class_id"].astype(int)
    eligible["pred_class_id"] = eligible["pred_class_id"].astype(int)
    wrong = eligible[eligible["target_class_id"] != eligible["pred_class_id"]]
    counter = Counter(zip(wrong["pred_class_id"], wrong["target_class_id"]))
    edges = [
        {"pred_class_id": int(a), "gt_class_id": int(b), "count": int(c)}
        for (a, b), c in counter.most_common(top_edges)
        if c >= min_edge_count
    ]
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        union(edge["pred_class_id"], edge["gt_class_id"])
    comps: dict[int, set[int]] = defaultdict(set)
    for cls in list(parent):
        comps[find(cls)].add(cls)
    clusters = [comp for comp in comps.values() if len(comp) >= 2]
    clusters.sort(key=lambda comp: (-len(comp), sorted(comp)))
    return clusters, edges


def row_feature(row: pd.Series, crop_root: Path, image_root: Path, cluster: set[int]) -> np.ndarray:
    crop_path = crop_root / str(row["crop_path"])
    source_image_path = image_root / str(row.get("tile_file", ""))
    visual = crop_visual_descriptor(crop_path, source_image_path, row.get("crop_xyxy", row.get("pred_xyxy")))
    numeric = np.asarray(
        [
            float(row["pred_score"]),
            float(row["frequency_evidence_score"]),
            float(row["normal_reference_distance"]),
            float(row["reliability_fusion_score"]),
            math.log(float(row["box_w"]) + 1.0),
            math.log(float(row["box_h"]) + 1.0),
            math.log(float(row["box_area"]) + 1.0),
            float(row["box_aspect"]),
            float(row["box_cx"]) / 640.0,
            float(row["box_cy"]) / 640.0,
        ],
        dtype=np.float32,
    )
    members = sorted(cluster)
    onehot = np.zeros(len(members), dtype=np.float32)
    pred = int(row["pred_class_id"])
    if pred in members:
        onehot[members.index(pred)] = 1.0
    return np.concatenate([numeric, visual, onehot]).astype(np.float32)


def featurize(df: pd.DataFrame, crop_root: Path, image_root: Path, cluster: set[int]) -> tuple[np.ndarray, list[str]]:
    feats: list[np.ndarray] = []
    ids: list[str] = []
    for idx, row in df.iterrows():
        feats.append(row_feature(row, crop_root, image_root, cluster))
        ids.append(str(row["ibo_id"]))
        if len(feats) % 2000 == 0:
            print(f"[CLUSTER-CROP] features {len(feats)}/{len(df)}", flush=True)
    return np.vstack(feats), ids


def class_name_lookup(df: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for _, row in df.iterrows():
        if not pd.isna(row.get("pred_class_id")):
            names[int(row["pred_class_id"])] = str(row.get("pred_class_name") or f"class_{int(row['pred_class_id'])}")
        if not pd.isna(row.get("matched_gt_class_id")):
            names[int(row["matched_gt_class_id"])] = str(row.get("matched_gt_class_name") or f"class_{int(row['matched_gt_class_id'])}")
    return names


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    train_df = build_table(train_rows, args.train_evidence)
    val_df = build_table(val_rows, args.val_evidence)
    names = class_name_lookup(pd.concat([train_df, val_df], ignore_index=True))

    clusters, edges = discover_clusters(
        train_df,
        args.cluster_reliability_threshold,
        args.coverage_threshold,
        args.top_confusion_edges,
        args.min_edge_count,
    )
    write_csv(
        args.output_dir / "train_discovered_confusion_edges.csv",
        [
            {
                **edge,
                "pred_class_name": names.get(edge["pred_class_id"], f"class_{edge['pred_class_id']}"),
                "gt_class_name": names.get(edge["gt_class_id"], f"class_{edge['gt_class_id']}"),
            }
            for edge in edges
        ],
    )
    write_csv(
        args.output_dir / "train_discovered_confusion_clusters.csv",
        [
            {
                "cluster_id": idx,
                "class_ids": ",".join(str(v) for v in sorted(cluster)),
                "class_names": " | ".join(names.get(v, f"class_{v}") for v in sorted(cluster)),
                "size": len(cluster),
            }
            for idx, cluster in enumerate(clusters)
        ],
    )
    print(f"[CLUSTER-CROP] clusters={len(clusters)} edges={len(edges)}", flush=True)

    pred_by_id: dict[str, dict[str, Any]] = {}
    model_records: list[dict[str, Any]] = []
    model_bundle: dict[str, Any] = {"models": [], "args": vars(args)}

    for cluster_id, cluster in enumerate(clusters):
        train_cluster = train_df[
            (train_df["pred_class_id"].astype("Int64").isin(cluster))
            & (train_df["matched_gt_class_id"].astype("Int64").isin(cluster))
            & (train_df["best_coverage_any"].fillna(0.0) >= args.coverage_threshold)
            & (train_df["reliability_fusion_score"].fillna(0.0) >= args.cluster_reliability_threshold)
        ].copy()
        if len(train_cluster) < args.min_cluster_train or train_cluster["matched_gt_class_id"].nunique() < 2:
            print(f"[CLUSTER-CROP] skip cluster={cluster_id} train={len(train_cluster)} classes={train_cluster['matched_gt_class_id'].nunique()}", flush=True)
            continue
        if len(train_cluster) > args.max_cluster_train:
            train_cluster = train_cluster.sample(n=args.max_cluster_train, random_state=args.random_state)
        train_cluster["target_class_id"] = train_cluster["matched_gt_class_id"].astype(int)
        val_cluster = val_df[
            (val_df["pred_class_id"].astype("Int64").isin(cluster))
            & (val_df["reliability_fusion_score"].fillna(0.0) >= args.apply_reliability_threshold)
        ].copy()
        if val_cluster.empty:
            continue

        x, _ = featurize(train_cluster, args.train_crop_root, args.train_image_root, cluster)
        y = train_cluster["target_class_id"].astype(int).to_numpy()
        stratify = y if min(Counter(y).values()) >= 2 else None
        x_train, x_hold, y_train, y_hold = train_test_split(x, y, test_size=0.2, random_state=args.random_state, stratify=stratify)
        model = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=args.random_state + cluster_id,
        )
        model.fit(x_train, y_train)
        hold_pred = model.predict(x_hold)
        hold_acc = float(accuracy_score(y_hold, hold_pred))
        hold_bal = float(balanced_accuracy_score(y_hold, hold_pred))

        xv, val_ids = featurize(val_cluster, args.val_crop_root, args.val_image_root, cluster)
        proba = model.predict_proba(xv)
        classes = model.classes_.astype(int)
        order = np.argsort(proba, axis=1)
        top_idx = order[:, -1]
        second_idx = order[:, -2] if proba.shape[1] >= 2 else order[:, -1]
        top_cls = classes[top_idx]
        top_p = proba[np.arange(len(xv)), top_idx]
        second_p = proba[np.arange(len(xv)), second_idx]
        for ibo_id, proposed, p_top, p_second in zip(val_ids, top_cls, top_p, second_p):
            row = val_cluster[val_cluster["ibo_id"].astype(str) == ibo_id].iloc[0]
            original = int(row["pred_class_id"])
            margin = float(p_top - p_second)
            apply = proposed != original and float(p_top) >= args.min_correction_proba and margin >= args.min_correction_margin
            current = pred_by_id.get(ibo_id)
            candidate = {
                "ibo_id": ibo_id,
                "cluster_id": cluster_id,
                "original_pred_class_id": original,
                "proposed_class_id": int(proposed),
                "correction_applied": bool(apply),
                "correction_proba": float(p_top),
                "correction_margin": margin,
                "reliability_score": float(row["reliability_fusion_score"]),
            }
            if current is None or candidate["correction_proba"] > current["correction_proba"]:
                pred_by_id[ibo_id] = candidate

        model_records.append(
            {
                "cluster_id": cluster_id,
                "classes": ",".join(str(v) for v in sorted(cluster)),
                "class_names": " | ".join(names.get(v, f"class_{v}") for v in sorted(cluster)),
                "train_candidates": len(train_cluster),
                "val_candidates": len(val_cluster),
                "holdout_accuracy": hold_acc,
                "holdout_balanced_accuracy": hold_bal,
            }
        )
        model_bundle["models"].append({"cluster_id": cluster_id, "cluster": sorted(cluster), "model": model})
        print(
            f"[CLUSTER-CROP] cluster={cluster_id} classes={sorted(cluster)} "
            f"train={len(train_cluster)} val={len(val_cluster)} hold_acc={hold_acc:.4f} hold_bal={hold_bal:.4f}",
            flush=True,
        )

    corrected_rows: list[dict[str, Any]] = []
    applied = 0
    for row in val_rows:
        item = dict(row)
        pred = pred_by_id.get(str(item["ibo_id"]))
        if pred and pred["correction_applied"]:
            item["original_pred_class_id"] = item["pred_class_id"]
            item["original_pred_class_name"] = item.get("pred_class_name", "")
            item["pred_class_id"] = int(pred["proposed_class_id"])
            item["pred_class_name"] = names.get(int(pred["proposed_class_id"]), f"corrected_class_{pred['proposed_class_id']}")
            item["cluster_crop_correction_applied"] = True
            item["cluster_crop_correction_proba"] = pred["correction_proba"]
            item["cluster_crop_correction_margin"] = pred["correction_margin"]
            item["cluster_crop_correction_cluster_id"] = pred["cluster_id"]
            applied += 1
        else:
            item["cluster_crop_correction_applied"] = False
        corrected_rows.append(item)

    score_rows = [
        {"ibo_id": str(row["ibo_id"]), "reliability_score": float(row["reliability_fusion_score"])}
        for _, row in val_df.iterrows()
    ]
    write_jsonl(args.output_dir / "ibo_manifest_val_cluster_crop_corrected.jsonl", corrected_rows)
    write_csv(args.output_dir / "lvis_ibo_reliability_scores_val_cluster_crop_corrected.csv", score_rows)
    write_csv(args.output_dir / "cluster_crop_corrector_predictions_val.csv", list(pred_by_id.values()))
    write_csv(args.output_dir / "cluster_crop_corrector_model_report.csv", model_records)
    joblib.dump(model_bundle, args.output_dir / "cluster_crop_corrector_models.joblib")
    summary = {
        "clusters_discovered": len(clusters),
        "edges_used": len(edges),
        "models_trained": len(model_records),
        "val_candidates_with_cluster_prediction": len(pred_by_id),
        "val_corrections_applied": applied,
        "min_correction_proba": args.min_correction_proba,
        "min_correction_margin": args.min_correction_margin,
    }
    (args.output_dir / "cluster_crop_corrector_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[CLUSTER-CROP] done models={len(model_records)} predictions={len(pred_by_id)} applied={applied} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
