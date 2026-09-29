#!/usr/bin/env python
"""Train local confusion-cluster correctors using D-FINE query embeddings.

This is the closest LVIS/SimLTD implementation of the paper component
"detection-query based confusable-class contrast correction".

Inputs are existing IBO manifests/evidence plus query features exported by
extract_lvis_dfine_query_features.py. No validation GT is used for cluster
discovery or training.
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
    parser.add_argument("--train-query-features", type=Path, required=True)
    parser.add_argument("--val-query-features", type=Path, required=True)
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
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=24)
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
    return add_box_features(df)


def load_query_features(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    ids = data["ibo_ids"].astype(str)
    features = data["features"].astype(np.float32)
    return {ibo_id: features[i] for i, ibo_id in enumerate(ids)}


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


def class_name_lookup(df: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for _, row in df.iterrows():
        if not pd.isna(row.get("pred_class_id")):
            names[int(row["pred_class_id"])] = str(row.get("pred_class_name") or f"class_{int(row['pred_class_id'])}")
        if not pd.isna(row.get("matched_gt_class_id")):
            names[int(row["matched_gt_class_id"])] = str(row.get("matched_gt_class_name") or f"class_{int(row['matched_gt_class_id'])}")
    return names


def row_feature(row: pd.Series, query_by_id: dict[str, np.ndarray], cluster: set[int]) -> np.ndarray:
    q = query_by_id.get(str(row["ibo_id"]))
    if q is None:
        q = np.zeros(256, dtype=np.float32)
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
    return np.concatenate([numeric, q, onehot]).astype(np.float32)


def featurize(df: pd.DataFrame, query_by_id: dict[str, np.ndarray], cluster: set[int]) -> tuple[np.ndarray, list[str]]:
    feats: list[np.ndarray] = []
    ids: list[str] = []
    for _, row in df.iterrows():
        feats.append(row_feature(row, query_by_id, cluster))
        ids.append(str(row["ibo_id"]))
    return np.vstack(feats), ids


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    train_df = build_table(train_rows, args.train_evidence)
    val_df = build_table(val_rows, args.val_evidence)
    train_q = load_query_features(args.train_query_features)
    val_q = load_query_features(args.val_query_features)
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
    print(f"[CLUSTER-QUERY] clusters={len(clusters)} edges={len(edges)}", flush=True)

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
            print(f"[CLUSTER-QUERY] skip cluster={cluster_id} train={len(train_cluster)}", flush=True)
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

        x, _ = featurize(train_cluster, train_q, cluster)
        y = train_cluster["target_class_id"].astype(int).to_numpy()
        stratify = y if min(Counter(y).values()) >= 2 else None
        x_train, x_hold, y_train, y_hold = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=args.random_state,
            stratify=stratify,
        )
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

        xv, val_ids = featurize(val_cluster, val_q, cluster)
        proba = model.predict_proba(xv)
        classes = model.classes_.astype(int)
        order = np.argsort(proba, axis=1)
        top_idx = order[:, -1]
        second_idx = order[:, -2] if proba.shape[1] >= 2 else order[:, -1]
        top_cls = classes[top_idx]
        top_p = proba[np.arange(len(xv)), top_idx]
        second_p = proba[np.arange(len(xv)), second_idx]
        val_lookup = val_cluster.set_index(val_cluster["ibo_id"].astype(str), drop=False)
        for ibo_id, proposed, p_top, p_second in zip(val_ids, top_cls, top_p, second_p):
            row = val_lookup.loc[ibo_id]
            original = int(row["pred_class_id"])
            margin = float(p_top - p_second)
            apply = proposed != original and float(p_top) >= args.min_correction_proba and margin >= args.min_correction_margin
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
            current = pred_by_id.get(ibo_id)
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
            f"[CLUSTER-QUERY] cluster={cluster_id} classes={sorted(cluster)} "
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
            item["cluster_query_correction_applied"] = True
            item["cluster_query_correction_proba"] = pred["correction_proba"]
            item["cluster_query_correction_margin"] = pred["correction_margin"]
            item["cluster_query_correction_cluster_id"] = pred["cluster_id"]
            applied += 1
        else:
            item["cluster_query_correction_applied"] = False
        corrected_rows.append(item)

    score_rows = [
        {"ibo_id": str(row["ibo_id"]), "reliability_score": float(row["reliability_fusion_score"])}
        for _, row in val_df.iterrows()
    ]
    write_jsonl(args.output_dir / "ibo_manifest_val_cluster_query_corrected.jsonl", corrected_rows)
    write_csv(args.output_dir / "lvis_ibo_reliability_scores_val_cluster_query_corrected.csv", score_rows)
    write_csv(args.output_dir / "cluster_query_corrector_predictions_val.csv", list(pred_by_id.values()))
    write_csv(args.output_dir / "cluster_query_corrector_model_report.csv", model_records)
    joblib.dump(model_bundle, args.output_dir / "cluster_query_corrector_models.joblib")
    summary = {
        "clusters_discovered": len(clusters),
        "edges_used": len(edges),
        "models_trained": len(model_records),
        "val_candidates_with_cluster_prediction": len(pred_by_id),
        "val_corrections_applied": applied,
        "min_correction_proba": args.min_correction_proba,
        "min_correction_margin": args.min_correction_margin,
    }
    (args.output_dir / "cluster_query_corrector_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[CLUSTER-QUERY] done models={len(model_records)} predictions={len(pred_by_id)} applied={applied} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
