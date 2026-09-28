#!/usr/bin/env python
"""Train/evaluate IBO group-level evidence fusion v2.

This stage works after cyclic same-class candidate merging.  It builds a
training-only normal-reference feature bank from defect-free strip regions,
extracts spatial/frequency evidence for merged IBO groups, and trains a
lightweight group-level reliability classifier.
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

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, classification_report, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--strip-root", type=Path, required=True)
    parser.add_argument("--train-merged-candidates", type=Path, required=True)
    parser.add_argument("--val-merged-candidates", type=Path, required=True)
    parser.add_argument("--train-strip-boxes", type=Path, required=True)
    parser.add_argument("--val-strip-boxes", type=Path, required=True)
    parser.add_argument("--source-train-annotations", type=Path, required=True)
    parser.add_argument("--source-val-annotations", type=Path, required=True)
    parser.add_argument("--method", default="merge_protect_rel0.45_p0.25_iou0.50_sameclass")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normal-samples", type=int, default=6000)
    parser.add_argument("--normal-k", type=int, default=7)
    parser.add_argument(
        "--external-normal-summary",
        type=Path,
        default=None,
        help="Optional summary.json produced by unwrap_68jizhu_weld_annulus.py for truly defect-free strips.",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--exclude-class-id", type=int, action="append", default=[])
    parser.add_argument(
        "--match-family",
        action="append",
        default=[],
        help="Comma-separated class ids treated as one class for matching, e.g. 12,14,15. May be repeated.",
    )
    parser.add_argument("--seed", type=int, default=20260903)
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


def read_bgr(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def parse_box(value: str) -> np.ndarray:
    return np.asarray([float(v) for v in ast.literal_eval(value)], dtype=np.float32)


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


def best_shifted_box(box: np.ndarray, target: np.ndarray, strip_width: float) -> np.ndarray:
    shifts = (-strip_width, 0.0, strip_width)
    candidates = [box + np.asarray([s, 0.0, s, 0.0], dtype=np.float32) for s in shifts]
    target_center = (target[0] + target[2]) / 2.0
    return min(candidates, key=lambda b: abs(float((b[0] + b[2]) / 2.0 - target_center)))


def union_coverage(gt_box: np.ndarray, boxes: list[np.ndarray], strip_width: float) -> float:
    if not boxes:
        return 0.0
    events: list[tuple[float, int, float, float]] = []
    for raw in boxes:
        box = best_shifted_box(raw, gt_box, strip_width)
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
            start, end = intervals[0]
            total = 0.0
            for ys, ye in intervals[1:]:
                if ys <= end:
                    end = max(end, ye)
                else:
                    total += end - start
                    start, end = ys, ye
            total += end - start
            area += (x - previous_x) * total
        if flag > 0:
            active.append((y0, y1))
        else:
            try:
                active.remove((y0, y1))
            except ValueError:
                pass
        previous_x = x
    return min(1.0, area / max(1e-6, box_area(gt_box)))


def load_feature_functions(scripts_root: Path):
    sys.path.insert(0, str(scripts_root))
    from build_normal_reference_evidence_bank import transition_features

    return transition_features


def strip_file_for(split: str, source_image_id: int, strip_root: Path, strip_meta: dict[int, dict[str, Any]]) -> Path:
    meta = strip_meta[source_image_id]
    rel = str(meta.get("strip_file") or Path("strips") / split / f"{split}_{source_image_id:06d}_strip.jpg")
    return strip_root / rel


def cyclic_crop_and_local_box(strip: np.ndarray, box: np.ndarray, strip_width: float, context_scale: float = 2.0) -> tuple[np.ndarray, list[float]]:
    height = strip.shape[0]
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(4.0, x1 - x0)
    bh = max(4.0, y1 - y0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    cw = max(32.0, bw * context_scale)
    ch = max(32.0, bh * context_scale)
    crop_x0 = cx - cw / 2.0
    crop_x1 = cx + cw / 2.0
    crop_y0 = max(0, int(math.floor(cy - ch / 2.0)))
    crop_y1 = min(height, int(math.ceil(cy + ch / 2.0)))
    if crop_y1 - crop_y0 < 8:
        crop_y0, crop_y1 = 0, height
    x_start = int(math.floor(crop_x0))
    x_end = int(math.ceil(crop_x1))
    cols = np.mod(np.arange(x_start, max(x_start + 8, x_end)), int(round(strip_width))).astype(np.int64)
    crop = strip[crop_y0:crop_y1, cols]
    local_box = [x0 - x_start, y0 - crop_y0, bw, bh]
    return crop, local_box


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
            np.asarray([mag.mean(), mag.std(), np.cos(2.0 * ang).mean(), np.sin(2.0 * ang).mean()], dtype=np.float32),
            dct_features(gray),
        ]
    ).astype(np.float32)


def base_image_feature(strip: np.ndarray, box: np.ndarray, strip_width: float, transition_features) -> np.ndarray:
    crop, local = cyclic_crop_and_local_box(strip, box, strip_width)
    try:
        spatial, frequency = transition_features(crop, local, inner_ratio=0.16, ring_ratio=0.22)
    except Exception:
        spatial = np.full(20, np.nan, dtype=np.float32)
        frequency = np.full(14, np.nan, dtype=np.float32)
    return np.concatenate([spatial, frequency, crop_stats(crop)]).astype(np.float32)


def load_merged_candidates(path: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        if row.get("method") != method:
            continue
        item = {
            "merged_id": row["merged_id"],
            "source_image_id": int(row["source_image_id"]),
            "source_file": row.get("source_file", ""),
            "class_id": int(row["class_id"]),
            "class_name": row.get("class_name", f"class_{row['class_id']}"),
            "strip_width": float(row["strip_width"]),
            "global_box": parse_box(row["global_box"]),
            "member_count": int(float(row["member_count"])),
            "max_reliability": float(row["max_reliability"]),
            "mean_reliability": float(row["mean_reliability"]),
            "max_pred_score": float(row["max_pred_score"]),
            "class_vote_margin": float(row["class_vote_margin"]),
        }
        rows.append(item)
    return rows


def strip_meta_by_source(path: Path) -> dict[int, dict[str, Any]]:
    return {int(item["source_image_id"]): item for item in load_json(path)}


def gt_boxes_by_source(strip_boxes: list[dict[str, Any]], exclude: set[int]) -> dict[int, list[dict[str, Any]]]:
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in strip_boxes:
        for box in item["boxes"]:
            if int(box["category_id"]) in exclude:
                continue
            by[int(item["source_image_id"])].append(
                {
                    "source_annotation_id": int(box["source_annotation_id"]),
                    "class_id": int(box["category_id"]),
                    "box": xyxy_from_xywh([box["x"], box["y"], box["width"], box["height"]]),
                    "strip_width": float(item["strip_width"]),
                }
            )
    return by


def label_candidate(
    candidate: dict[str, Any],
    gt_by_source: dict[int, list[dict[str, Any]]],
    coverage_threshold: float,
    match_iou: float,
    match_families: list[set[int]],
) -> dict[str, Any]:
    gts = gt_by_source.get(int(candidate["source_image_id"]), [])
    same_boxes = [gt["box"] for gt in gts if class_matches(int(gt["class_id"]), int(candidate["class_id"]), match_families)]
    other_boxes = [gt["box"] for gt in gts if not class_matches(int(gt["class_id"]), int(candidate["class_id"]), match_families)]
    width = float(candidate["strip_width"])
    box = np.asarray(candidate["global_box"], dtype=np.float32)
    same_cov = max((intersection_area(best_shifted_box(box, gt, width), gt) / max(1e-6, box_area(gt)) for gt in same_boxes), default=0.0)
    same_iou = max((iou(best_shifted_box(box, gt, width), gt) for gt in same_boxes), default=0.0)
    other_cov = max((intersection_area(best_shifted_box(box, gt, width), gt) / max(1e-6, box_area(gt)) for gt in other_boxes), default=0.0)
    other_iou = max((iou(best_shifted_box(box, gt, width), gt) for gt in other_boxes), default=0.0)
    if same_cov >= coverage_threshold or same_iou >= match_iou:
        outcome = "positive"
        y = 1
    elif other_cov >= coverage_threshold or other_iou >= match_iou:
        outcome = "wrong_class"
        y = 0
    else:
        outcome = "background"
        y = 0
    return {"label": y, "candidate_outcome": outcome, "same_cov": same_cov, "same_iou": same_iou, "other_cov": other_cov, "other_iou": other_iou}


def sample_normal_bank(
    train_candidates: list[dict[str, Any]],
    train_strip_meta: dict[int, dict[str, Any]],
    train_gt: dict[int, list[dict[str, Any]]],
    strip_root: Path,
    transition_features,
    normal_samples: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    dims = np.asarray(
        [
            [
                max(8.0, float(item["global_box"][2] - item["global_box"][0])),
                max(8.0, float(item["global_box"][3] - item["global_box"][1])),
            ]
            for item in train_candidates
        ],
        dtype=np.float32,
    )
    source_ids = list(train_strip_meta)
    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    image_cache: dict[int, np.ndarray] = {}
    attempts = 0
    while len(features) < normal_samples and attempts < normal_samples * 80:
        attempts += 1
        sid = int(rng.choice(source_ids))
        meta = train_strip_meta[sid]
        width = float(meta["strip_width"])
        height = float(meta.get("strip_height", 150))
        bw, bh = dims[int(rng.integers(0, len(dims)))]
        bw = float(np.clip(bw, 12, min(512, width)))
        bh = float(np.clip(bh, 8, min(140, height)))
        x0 = float(rng.uniform(0, width))
        y0 = float(rng.uniform(0, max(1.0, height - bh)))
        box = np.asarray([x0, y0, x0 + bw, y0 + bh], dtype=np.float32)
        bad = False
        for gt in train_gt.get(sid, []):
            shifted = best_shifted_box(box, gt["box"], width)
            cov = intersection_area(shifted, gt["box"]) / max(1e-6, box_area(gt["box"]))
            if cov > 0.03 or iou(shifted, gt["box"]) > 0.03:
                bad = True
                break
        if bad:
            continue
        if sid not in image_cache:
            image_cache[sid] = read_bgr(strip_file_for("train", sid, strip_root, train_strip_meta))
        strip = image_cache[sid]
        if strip is None:
            continue
        vector = base_image_feature(strip, box, width, transition_features)
        features.append(vector)
        rows.append({"normal_id": f"normal_{len(features):06d}", "source_image_id": sid, "global_box": [round(float(v), 3) for v in box.tolist()]})
        if len(features) % 1000 == 0:
            print(f"[IBO-V2] normal bank {len(features)}/{normal_samples}", flush=True)
    return np.vstack(features), rows


def load_external_normal_strips(summary_path: Path) -> list[dict[str, Any]]:
    items = load_json(summary_path)
    strips: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        output = Path(item["output"])
        strip_path = output / "weld_annulus_unwrapped.png"
        if not strip_path.is_file():
            continue
        size = item.get("strip_size", {})
        strips.append(
            {
                "normal_source_id": index,
                "source": item.get("source", ""),
                "strip_path": strip_path,
                "strip_width": int(size.get("width", 0)),
                "strip_height": int(size.get("height", 150)),
            }
        )
    if not strips:
        raise RuntimeError(f"No external normal strips found from summary: {summary_path}")
    return strips


def sample_external_normal_bank(
    train_candidates: list[dict[str, Any]],
    external_summary: Path,
    transition_features,
    normal_samples: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    normal_strips = load_external_normal_strips(external_summary)
    dims = np.asarray(
        [
            [
                max(8.0, float(item["global_box"][2] - item["global_box"][0])),
                max(8.0, float(item["global_box"][3] - item["global_box"][1])),
            ]
            for item in train_candidates
        ],
        dtype=np.float32,
    )
    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    image_cache: dict[int, np.ndarray] = {}
    attempts = 0
    while len(features) < normal_samples and attempts < normal_samples * 80:
        attempts += 1
        meta = normal_strips[int(rng.integers(0, len(normal_strips)))]
        sid = int(meta["normal_source_id"])
        width = float(meta["strip_width"])
        height = float(meta.get("strip_height", 150))
        if width <= 0 or height <= 0:
            continue
        bw, bh = dims[int(rng.integers(0, len(dims)))]
        bw = float(np.clip(bw, 12, min(512, width)))
        bh = float(np.clip(bh, 8, min(140, height)))
        x0 = float(rng.uniform(0, width))
        y0 = float(rng.uniform(0, max(1.0, height - bh)))
        box = np.asarray([x0, y0, x0 + bw, y0 + bh], dtype=np.float32)
        if sid not in image_cache:
            image_cache[sid] = read_bgr(Path(meta["strip_path"]))
        strip = image_cache[sid]
        if strip is None:
            continue
        vector = base_image_feature(strip, box, width, transition_features)
        features.append(vector)
        rows.append(
            {
                "normal_id": f"external_normal_{len(features):06d}",
                "normal_source_id": sid,
                "source": meta["source"],
                "strip_path": str(meta["strip_path"]),
                "global_box": [round(float(v), 3) for v in box.tolist()],
            }
        )
        if len(features) % 1000 == 0:
            print(f"[IBO-V2] external normal bank {len(features)}/{normal_samples}", flush=True)
    if not features:
        raise RuntimeError("No external normal-bank features could be sampled")
    return np.vstack(features), rows


def make_group_matrix(
    split: str,
    candidates: list[dict[str, Any]],
    strip_meta: dict[int, dict[str, Any]],
    gt_by_source: dict[int, list[dict[str, Any]]],
    strip_root: Path,
    transition_features,
    normal_scaler: StandardScaler,
    normal_nn: NearestNeighbors,
    normal_k: int,
    coverage_threshold: float,
    match_iou: float,
    match_families: list[set[int]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cache: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    x: list[np.ndarray] = []
    y: list[int] = []
    for idx, item in enumerate(candidates, 1):
        sid = int(item["source_image_id"])
        if sid not in strip_meta:
            continue
        if sid not in cache:
            cache[sid] = read_bgr(strip_file_for(split, sid, strip_root, strip_meta))
        strip = cache[sid]
        if strip is None:
            continue
        box = np.asarray(item["global_box"], dtype=np.float32)
        image_vec = base_image_feature(strip, box, float(item["strip_width"]), transition_features)
        scaled = normal_scaler.transform(image_vec.reshape(1, -1))
        distances, _ = normal_nn.kneighbors(scaled, n_neighbors=normal_k)
        normal_score = float(distances.mean())
        # Frequency evidence emphasizes the Haar transition block and DCT high/concentration tail.
        frequency_block = image_vec[20:34]
        dct_high = image_vec[-2]
        dct_conc = image_vec[-1]
        frequency_score = float(np.nanmean(np.abs(frequency_block)) + dct_high + dct_conc)
        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        group_vec = np.asarray(
            [
                item["max_reliability"],
                item["mean_reliability"],
                item["max_pred_score"],
                math.log1p(item["member_count"]),
                item["class_vote_margin"],
                math.log(width),
                math.log(height),
                math.log(width * height),
                width / max(1.0, float(item["strip_width"])),
                height / max(1.0, float(strip.shape[0])),
                normal_score,
                frequency_score,
            ],
            dtype=np.float32,
        )
        label = label_candidate(item, gt_by_source, coverage_threshold, match_iou, match_families)
        vector = np.concatenate([group_vec, image_vec]).astype(np.float32)
        x.append(vector)
        y.append(int(label["label"]))
        row = {
            "split": split,
            "merged_id": item["merged_id"],
            "source_image_id": sid,
            "source_file": item.get("source_file", ""),
            "class_id": item["class_id"],
            "class_name": item["class_name"],
            "member_count": item["member_count"],
            "max_reliability_v1": round(float(item["max_reliability"]), 6),
            "mean_reliability_v1": round(float(item["mean_reliability"]), 6),
            "max_pred_score": round(float(item["max_pred_score"]), 6),
            "normal_reference_score": round(normal_score, 6),
            "frequency_evidence_score": round(frequency_score, 6),
            "label": int(label["label"]),
            "candidate_outcome": label["candidate_outcome"],
            "same_cov": round(float(label["same_cov"]), 6),
            "same_iou": round(float(label["same_iou"]), 6),
            "other_cov": round(float(label["other_cov"]), 6),
            "other_iou": round(float(label["other_iou"]), 6),
            "global_box": [round(float(v), 3) for v in box.tolist()],
        }
        rows.append(row)
        if idx % 1000 == 0 or idx == len(candidates):
            print(f"[IBO-V2] {split} groups {idx}/{len(candidates)}", flush=True)
    return np.vstack(x), np.asarray(y, dtype=np.int64), rows


def evaluate_instance_from_selected(
    selected_rows: list[dict[str, Any]],
    source_data: dict[str, Any],
    strip_boxes: list[dict[str, Any]],
    coverage_threshold: float,
    match_iou: float,
    exclude: set[int],
    match_families: list[set[int]],
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    category_names = {int(item["id"]): str(item["name"]) for item in source_data["categories"]}
    strip_gt = gt_boxes_by_source(strip_boxes, exclude)
    represented = {
        int(gt["source_annotation_id"]): (sid, gt)
        for sid, gts in strip_gt.items()
        for gt in gts
    }
    preds_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        box = np.asarray(row["global_box"] if isinstance(row["global_box"], list) else ast.literal_eval(str(row["global_box"])), dtype=np.float32)
        packed = dict(row)
        packed["box_np"] = box
        preds_by_source[int(row["source_image_id"])].append(packed)
    class_counts: dict[int, Counter[str]] = defaultdict(Counter)
    class_rows: list[dict[str, Any]] = []
    for ann in source_data["annotations"]:
        cid = int(ann["category_id"])
        if cid in exclude:
            continue
        source_annotation_id = int(ann["id"])
        if source_annotation_id not in represented:
            class_counts[cid]["miss"] += 1
            continue
        sid, gt = represented[source_annotation_id]
        cands = preds_by_source.get(int(sid), [])
        same = [cand["box_np"] for cand in cands if class_matches(cid, int(cand["class_id"]), match_families)]
        other = [cand["box_np"] for cand in cands if not class_matches(cid, int(cand["class_id"]), match_families)]
        width = float(gt["strip_width"])
        same_cov = union_coverage(gt["box"], same, width)
        other_cov = union_coverage(gt["box"], other, width)
        other_iou = max((iou(gt["box"], best_shifted_box(box, gt["box"], width)) for box in other), default=0.0)
        if same_cov >= coverage_threshold:
            outcome = "correct"
        elif other_cov >= coverage_threshold or other_iou >= match_iou:
            outcome = "wrong"
        else:
            outcome = "miss"
        class_counts[cid][outcome] += 1
    for cid in sorted(class_counts):
        c = class_counts[cid]
        total = c["correct"] + c["wrong"] + c["miss"]
        class_rows.append(
            {
                "method": method,
                "class_id": cid,
                "class_name": category_names.get(cid, f"class_{cid}"),
                "gt_instances": total,
                "correct": c["correct"],
                "wrong": c["wrong"],
                "miss": c["miss"],
                "accuracy": round(c["correct"] / total, 6) if total else 0.0,
                "triple": f"{c['correct']} / {c['wrong']} / {c['miss']}",
            }
        )
    total = sum(row["gt_instances"] for row in class_rows)
    correct = sum(row["correct"] for row in class_rows)
    wrong = sum(row["wrong"] for row in class_rows)
    miss = sum(row["miss"] for row in class_rows)
    summary = {
        "method": method,
        "selected_groups": len(selected_rows),
        "instances": total,
        "correct": correct,
        "wrong": wrong,
        "miss": miss,
        "instance_accuracy": round(correct / total, 6) if total else 0.0,
    }
    return summary, class_rows


def thresholds(start: float, stop: float, step: float) -> list[float]:
    vals: list[float] = []
    current = start
    while current <= stop + 1e-9:
        vals.append(round(current, 4))
        current += step
    return vals


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclude = {int(v) for v in args.exclude_class_id}
    match_families = parse_match_families(args.match_family)
    transition_features = load_feature_functions(args.scripts_root)

    train_strip_boxes = load_json(args.train_strip_boxes)
    val_strip_boxes = load_json(args.val_strip_boxes)
    train_strip_meta = strip_meta_by_source(args.train_strip_boxes)
    val_strip_meta = strip_meta_by_source(args.val_strip_boxes)
    train_gt = gt_boxes_by_source(train_strip_boxes, exclude)
    val_source_data = load_json(args.source_val_annotations)
    train_source_data = load_json(args.source_train_annotations)

    train_candidates = load_merged_candidates(args.train_merged_candidates, args.method)
    val_candidates = load_merged_candidates(args.val_merged_candidates, args.method)
    print(f"[IBO-V2] method={args.method} train_groups={len(train_candidates)} val_groups={len(val_candidates)}", flush=True)

    normal_source = "training_non_annotated_regions"
    if args.external_normal_summary is not None:
        normal_source = "external_defect_free_unwrapped_strips"
        normal_x, normal_rows = sample_external_normal_bank(
            train_candidates,
            args.external_normal_summary,
            transition_features,
            args.normal_samples,
            args.seed,
        )
    else:
        normal_x, normal_rows = sample_normal_bank(
            train_candidates,
            train_strip_meta,
            train_gt,
            args.strip_root,
            transition_features,
            args.normal_samples,
            args.seed,
        )
    normal_scaler = StandardScaler()
    normal_scaled = normal_scaler.fit_transform(normal_x)
    normal_nn = NearestNeighbors(n_neighbors=args.normal_k, metric="euclidean")
    normal_nn.fit(normal_scaled)
    write_csv(args.output_dir / "normal_reference_bank_train.csv", normal_rows)

    train_x, train_y, train_rows = make_group_matrix(
        "train", train_candidates, train_strip_meta, train_gt, args.strip_root,
        transition_features, normal_scaler, normal_nn, args.normal_k,
        args.coverage_threshold, args.match_iou, match_families
    )
    val_gt = gt_boxes_by_source(val_strip_boxes, exclude)
    val_x, val_y, val_rows = make_group_matrix(
        "val", val_candidates, val_strip_meta, val_gt, args.strip_root,
        transition_features, normal_scaler, normal_nn, args.normal_k,
        args.coverage_threshold, args.match_iou, match_families
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=args.seed, n_jobs=-1)),
        ]
    )
    model.fit(train_x, train_y)
    train_scores = model.predict_proba(train_x)[:, 1]
    val_scores = model.predict_proba(val_x)[:, 1]
    for rows, scores in [(train_rows, train_scores), (val_rows, val_scores)]:
        for row, score in zip(rows, scores):
            row["reliability_v2_score"] = round(float(score), 6)
    write_csv(args.output_dir / "ibo_group_evidence_train.csv", train_rows)
    write_csv(args.output_dir / "ibo_group_evidence_val.csv", val_rows)
    joblib.dump({"model": model, "normal_scaler": normal_scaler, "normal_nn": normal_nn, "normal_k": args.normal_k}, args.output_dir / "ibo_group_evidence_fusion_v2.joblib")

    metrics: dict[str, Any] = {
        "train_groups": len(train_rows),
        "val_groups": len(val_rows),
        "train_positive": int(train_y.sum()),
        "train_negative": int((1 - train_y).sum()),
        "val_positive": int(val_y.sum()),
        "val_negative": int((1 - val_y).sum()),
        "val_auroc": float(roc_auc_score(val_y, val_scores)) if len(np.unique(val_y)) > 1 else None,
        "val_aupr": float(average_precision_score(val_y, val_scores)) if len(np.unique(val_y)) > 1 else None,
        "val_brier": float(brier_score_loss(val_y, val_scores)) if len(np.unique(val_y)) > 1 else None,
        "val_classification_report_t050": classification_report(val_y, (val_scores >= 0.5).astype(int), output_dict=True, zero_division=0),
        "normal_reference_source": normal_source,
        "external_normal_summary": str(args.external_normal_summary) if args.external_normal_summary is not None else "",
        "normal_reference_rows": len(normal_rows),
    }
    (args.output_dir / "ibo_group_evidence_fusion_v2_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    sweep: list[dict[str, Any]] = []
    class_rows_all: list[dict[str, Any]] = []
    pre_summary, pre_classes = evaluate_instance_from_selected(
        val_rows,
        val_source_data,
        val_strip_boxes,
        args.coverage_threshold,
        args.match_iou,
        exclude,
        match_families,
        "v1_merge_pre_v2_all_groups",
    )
    sweep.append(pre_summary)
    class_rows_all.extend(pre_classes)
    best_summary = pre_summary
    best_classes = pre_classes
    for threshold in thresholds(args.threshold_start, args.threshold_stop, args.threshold_step):
        selected = [row for row in val_rows if float(row["reliability_v2_score"]) >= threshold]
        summary, classes = evaluate_instance_from_selected(
            selected,
            val_source_data,
            val_strip_boxes,
            args.coverage_threshold,
            args.match_iou,
            exclude,
            match_families,
            f"v2_score_ge_{threshold:.2f}",
        )
        summary["threshold"] = threshold
        sweep.append(summary)
        if (
            summary["correct"] >= 1000
            and summary["miss"] <= 40
            and summary["wrong"] < best_summary["wrong"]
        ) or (
            summary["instance_accuracy"] > best_summary["instance_accuracy"]
            and summary["miss"] <= 45
        ):
            best_summary = summary
            best_classes = classes
    class_rows_all.extend(best_classes)
    write_csv(args.output_dir / "v2_threshold_sweep_summary.csv", sweep)
    write_csv(args.output_dir / "v2_best_by_class.csv", best_classes)

    try:
        xs = [float(row.get("threshold", -1)) for row in sweep if "threshold" in row]
        acc = [float(row["instance_accuracy"]) for row in sweep if "threshold" in row]
        wrong = [int(row["wrong"]) for row in sweep if "threshold" in row]
        miss = [int(row["miss"]) for row in sweep if "threshold" in row]
        fig, ax1 = plt.subplots(figsize=(7.6, 4.6), dpi=160)
        ax1.plot(xs, acc, marker="o", label="instance accuracy")
        ax1.set_xlabel("V2 reliability threshold")
        ax1.set_ylabel("Instance accuracy")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(xs, wrong, marker="s", color="#d95f02", label="wrong")
        ax2.plot(xs, miss, marker="^", color="#7570b3", label="miss")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(args.output_dir / "v2_threshold_sweep.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[IBO-V2] plot skipped: {exc}", flush=True)

    lines = [
        "# IBO 正常参照—频率证据—可靠性融合 v2 实验报告",
        "",
        "## Material Passport",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 输入候选 | `{args.method}` |",
        f"| 正常参照库 | {normal_source}，样本数 {len(normal_rows)} |",
        f"| 输出目录 | `{args.output_dir}` |",
        "",
        "## 候选级可靠性指标",
        "",
        f"- 验证集 AUROC：{metrics['val_auroc']:.4f}",
        f"- 验证集 AUPR：{metrics['val_aupr']:.4f}",
        f"- 验证集 Brier score：{metrics['val_brier']:.4f}",
        "",
        "## 实例级回投结果",
        "",
        "| 方法 | group 数 | 正确 / 错误 / 漏检 | 实例准确率 |",
        "|---|---:|---:|---:|",
    ]
    for row in sweep:
        mark = " **推荐**" if row["method"] == best_summary["method"] else ""
        lines.append(f"| {row['method']}{mark} | {row['selected_groups']} | {row['correct']} / {row['wrong']} / {row['miss']} | {fmt_pct(float(row['instance_accuracy']))} |")
    lines.extend(
        [
            "",
            "## 推荐参数的类别级结果",
            "",
            "| 类别 | 标注实例 | 正确 / 错误 / 漏检 | 准确率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in best_classes:
        lines.append(f"| {row['class_name']} | {row['gt_instances']} | {row['triple']} | {fmt_pct(float(row['accuracy']))} |")
    lines.extend(
        [
            "",
            "## 阶段判断",
            "",
            "- v2 在 group 级候选上加入训练集正常参照距离和频率证据，但仍然只使用验证集人工框进行评价。",
            "- 如果 v2 阈值提升后错误下降但漏检上升，说明证据模块偏保守；如果错误和漏检同时下降，说明正常参照/频率证据有效。",
            "- 本阶段是后处理与证据融合实验，不改变 D-FINE 主检测器。",
        ]
    )
    (args.output_dir / "IBO正常参照频率证据可靠性融合v2报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[IBO-V2] done best={best_summary['method']} "
        f"inst={best_summary['correct']}/{best_summary['wrong']}/{best_summary['miss']} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
