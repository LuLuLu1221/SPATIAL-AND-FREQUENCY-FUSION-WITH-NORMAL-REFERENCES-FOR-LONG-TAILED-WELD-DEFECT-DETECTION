#!/usr/bin/env python
"""Train and evaluate a lightweight second-stage classifier for classes 12/14/15.

The base detector remains the 150px cyclic-window D-FINE model.  This script
trains a small classical crop classifier on ground-truth crops of:

  12 = 断焊
  14 = 长焊高塌
  15 = 缺焊蓝黑

During validation, only D-FINE predictions whose predicted class is one of these
three IDs are re-checked.  The box is kept unchanged; only the subtype label may
be changed.  The output uses the same strict-audit semantics as
evaluate_dfine_strict_detection_errors.py so results can be compared with prior
tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TARGET_CLASS_IDS = (12, 14, 15)
CLASS_NAMES = {
    12: "断焊",
    14: "长焊高塌",
    15: "缺焊蓝黑",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-root", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--source-val-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--context-scale", type=float, default=2.0)
    parser.add_argument("--jitter-copies", type=int, default=4)
    parser.add_argument("--jitter", type=float, default=0.12)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--partial-coverage", type=float, default=0.20)
    parser.add_argument("--fragmented-coverage", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--switch-thresholds", type=str, default="0.00,0.55,0.65,0.75",
                        help="Comma-separated classifier probability thresholds. "
                             "If classifier disagrees with D-FINE, switch only when prob >= threshold.")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_annotation_id(annotation: dict[str, Any]) -> int:
    return int(annotation.get("source_annotation_id", annotation["id"]))


def source_image_id(image: dict[str, Any]) -> int:
    return int(image.get("source_image_id", image["id"]))


def crop_with_context(
    image: Image.Image,
    bbox_xywh: list[float],
    context_scale: float,
    jitter: float,
    rng: random.Random,
) -> Image.Image:
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx, cy = x + w / 2.0, y + h / 2.0
    if jitter > 0:
        cx += rng.uniform(-jitter, jitter) * max(w, 1.0)
        cy += rng.uniform(-jitter, jitter) * max(h, 1.0)
        context_scale *= math.exp(rng.uniform(-0.18, 0.18))
    crop_w = max(14.0, w * context_scale)
    crop_h = max(14.0, h * context_scale)
    left = int(math.floor(cx - crop_w / 2.0))
    top = int(math.floor(cy - crop_h / 2.0))
    right = int(math.ceil(cx + crop_w / 2.0))
    bottom = int(math.ceil(cy + crop_h / 2.0))
    return image.crop((left, top, right, bottom)).convert("RGB")


def image_feature_vector(crop: Image.Image, crop_size: int) -> np.ndarray:
    crop = ImageOps.pad(crop, (crop_size, crop_size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    rgb = np.asarray(crop, dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    feats: list[np.ndarray] = []
    for channel in range(3):
        feats.append(np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0), density=True)[0])
    feats.append(np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)[0])

    # Grid colour/texture statistics.
    for grid in (2, 4, 8):
        h_step = crop_size // grid
        w_step = crop_size // grid
        stats = []
        for gy in range(grid):
            for gx in range(grid):
                block = rgb[gy * h_step:(gy + 1) * h_step, gx * w_step:(gx + 1) * w_step, :]
                gblock = gray[gy * h_step:(gy + 1) * h_step, gx * w_step:(gx + 1) * w_step]
                stats.extend(block.mean(axis=(0, 1)).tolist())
                stats.extend(block.std(axis=(0, 1)).tolist())
                stats.append(float(gblock.mean()))
                stats.append(float(gblock.std()))
        feats.append(np.asarray(stats, dtype=np.float32))

    # HOG-like gradient histogram.
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hog_stats = []
    grid = 4
    h_step = crop_size // grid
    w_step = crop_size // grid
    bins = np.linspace(0.0, 1.0, 10)
    for cy in range(grid):
        for cx in range(grid):
            a = angle[cy * h_step:(cy + 1) * h_step, cx * w_step:(cx + 1) * w_step]
            m = mag[cy * h_step:(cy + 1) * h_step, cx * w_step:(cx + 1) * w_step]
            hist, _ = np.histogram(a, bins=bins, weights=m, density=False)
            hist = hist.astype(np.float32)
            hist /= max(1e-6, float(hist.sum()))
            hog_stats.extend(hist.tolist())
    feats.append(np.asarray(hog_stats, dtype=np.float32))

    return np.concatenate([np.asarray(item, dtype=np.float32).ravel() for item in feats])


def collect_gt_crop_features(
    image_root: Path,
    annotation_path: Path,
    crop_size: int,
    context_scale: float,
    jitter_copies: int,
    jitter: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    coco = load_json(annotation_path)
    images = {int(item["id"]): item for item in coco["images"]}
    items = [
        {
            "image_id": int(ann["image_id"]),
            "bbox": ann["bbox"],
            "category_id": int(ann["category_id"]),
            "source_image_id": source_image_id(images[int(ann["image_id"])]),
            "source_annotation_id": source_annotation_id(ann),
        }
        for ann in coco["annotations"]
        if int(ann["category_id"]) in TARGET_CLASS_IDS
    ]
    class_to_index = {cid: index for index, cid in enumerate(TARGET_CLASS_IDS)}
    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    image_cache: dict[int, Image.Image] = {}
    for row_index, item in enumerate(items):
        image_id = int(item["image_id"])
        if image_id not in image_cache:
            image_cache[image_id] = Image.open(image_root / str(images[image_id]["file_name"])).convert("RGB")
        copies = max(1, jitter_copies)
        for copy_index in range(copies):
            local_jitter = 0.0 if copy_index == 0 else jitter
            rng = random.Random(seed * 1_000_003 + row_index * 997 + copy_index * 37)
            crop = crop_with_context(image_cache[image_id], item["bbox"], context_scale, local_jitter, rng)
            features.append(image_feature_vector(crop, crop_size))
            labels.append(class_to_index[int(item["category_id"])])
            metadata.append({**item, "copy_index": copy_index})
    return np.stack(features), np.asarray(labels, dtype=np.int64), metadata


def grouped_train_val_split(metadata: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({int(item["source_image_id"]) for item in metadata})
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_count = max(1, round(len(groups) * val_fraction))
    val_groups = set(groups[:val_count])
    train_indices = [i for i, item in enumerate(metadata) if int(item["source_image_id"]) not in val_groups]
    val_indices = [i for i, item in enumerate(metadata) if int(item["source_image_id"]) in val_groups]
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def fit_candidate_models(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int):
    models = {
        "logreg_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=seed),
        ),
        "svc_rbf_balanced": make_pipeline(
            StandardScaler(),
            SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt", class_weight="balanced",
            min_samples_leaf=2, random_state=seed, n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_features="sqrt", class_weight="balanced",
            min_samples_leaf=2, random_state=seed, n_jobs=-1,
        ),
    }
    rows = []
    best_name = ""
    best_model = None
    best_score = -1.0
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        acc = accuracy_score(y_val, pred)
        macro = f1_score(y_val, pred, average="macro", zero_division=0)
        rows.append({"model": name, "internal_val_accuracy": acc, "internal_val_macro_f1": macro})
        if macro > best_score:
            best_name, best_model, best_score = name, model, macro
    return best_name, best_model, rows


def load_strict_module(scripts_root: Path):
    sys.path.insert(0, str(scripts_root))
    import evaluate_dfine_strict_detection_errors as strict  # type: ignore
    return strict


def audit_with_second_stage(
    *,
    strict,
    dfine_root: Path,
    config: Path,
    checkpoint: Path,
    classifier,
    switch_threshold: float,
    source_data: dict[str, Any],
    val_images: Path,
    val_annotations: Path,
    output_dir: Path,
    threshold: float,
    match_iou: float,
    partial_coverage: float,
    fragmented_coverage: float,
    batch_size: int,
    crop_size: int,
    context_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    model_name = f"dfine_150px_second_stage_confusable3_p{switch_threshold:.2f}".replace(".", "p")
    spec = strict.ModelSpec(model_name, config, checkpoint, val_images, val_annotations)
    data = load_json(val_annotations)
    categories = {int(item["id"]): str(item["name"]) for item in source_data["categories"]}
    records = sorted(data["images"], key=lambda item: int(item["id"]))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in data["annotations"]:
        annotations_by_image[int(item["image_id"])].append(item)

    source_annotations = {int(item["id"]): item for item in source_data["annotations"]}
    source_details: dict[int, dict[str, Any]] = {
        source_id: {
            "source_annotation_id": source_id,
            "source_image_id": int(annotation["image_id"]),
            "class_id": int(annotation["category_id"]),
            "class_name": categories[int(annotation["category_id"])],
            "represented": False,
            "native_occurrences": 0,
            "matched": False,
            "true_positive_native_matches": 0,
            "max_same_class_iou": 0.0,
            "max_same_class_coverage": 0.0,
            "max_same_class_prediction_count": 0,
            "wrong_class_categories": set(),
        }
        for source_id, annotation in source_annotations.items()
    }
    for annotation in data["annotations"]:
        source_id = source_annotation_id(annotation)
        source_details[source_id]["represented"] = True
        source_details[source_id]["native_occurrences"] += 1

    dfine_model, transform, spatial_size = strict.build_model(spec, dfine_root, device)
    prediction_counts = Counter()
    source_prediction_error_counts: dict[int, Counter[str]] = defaultdict(Counter)
    native_prediction_rows: list[dict[str, Any]] = []
    predictions_by_source_image: Counter[int] = Counter()
    reclass_counts = Counter()

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start:start + batch_size]
            tensors: list[torch.Tensor] = []
            sizes: list[list[float]] = []
            originals: list[Image.Image] = []
            for record in batch_records:
                with Image.open(val_images / str(record["file_name"])) as opened:
                    image = opened.convert("RGB")
                originals.append(image)
                tensors.append(transform(image))
                sizes.append([float(image.width), float(image.height)])
            labels, boxes, scores = dfine_model(
                torch.stack(tensors).to(device),
                torch.tensor(sizes, dtype=torch.float32, device=device),
            )
            for index, record in enumerate(batch_records):
                image_id = int(record["id"])
                source_id = source_image_id(record)
                width, height = sizes[index]
                predictions: list[dict[str, Any]] = []
                for category_id, box, score in zip(labels[index].tolist(), boxes[index].tolist(), scores[index].tolist()):
                    if float(score) < threshold:
                        continue
                    x0, y0, x1, y1 = (float(value) for value in box)
                    x0, x1 = sorted((max(0.0, x0), min(width - 1.0, x1)))
                    y0, y1 = sorted((max(0.0, y0), min(height - 1.0, y1)))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    original_category_id = int(category_id)
                    final_category_id = original_category_id
                    classifier_class_id = ""
                    classifier_probability = ""
                    switched = False
                    if original_category_id in TARGET_CLASS_IDS:
                        bbox_xywh = [x0, y0, x1 - x0, y1 - y0]
                        crop = crop_with_context(
                            originals[index], bbox_xywh, context_scale, 0.0,
                            random.Random(12345),
                        )
                        feat = image_feature_vector(crop, crop_size).reshape(1, -1)
                        if hasattr(classifier, "predict_proba"):
                            proba = classifier.predict_proba(feat)[0]
                            pred_index = int(np.argmax(proba))
                            pred_prob = float(proba[pred_index])
                        else:
                            pred_index = int(classifier.predict(feat)[0])
                            pred_prob = 1.0
                        classifier_class_id = TARGET_CLASS_IDS[pred_index]
                        classifier_probability = round(pred_prob, 6)
                        if classifier_class_id != original_category_id and pred_prob >= switch_threshold:
                            final_category_id = int(classifier_class_id)
                            switched = True
                            reclass_counts[(original_category_id, final_category_id)] += 1
                        else:
                            reclass_counts[(original_category_id, original_category_id)] += 1
                    predictions.append({
                        "category_id": int(final_category_id),
                        "score": float(score),
                        "box": np.asarray([x0, y0, x1, y1], dtype=np.float32),
                        "original_category_id": original_category_id,
                        "classifier_class_id": classifier_class_id,
                        "classifier_probability": classifier_probability,
                        "second_stage_switched": switched,
                    })

                predictions_by_source_image[source_id] += len(predictions)
                ground_truth = annotations_by_image.get(image_id, [])
                local_gt, prediction_labels = strict.analyse_native_input(ground_truth, predictions, match_iou)
                for detail in local_gt.values():
                    origin = source_details[detail["source_annotation_id"]]
                    origin["matched"] = origin["matched"] or detail["matched"]
                    origin["true_positive_native_matches"] += int(detail["matched"])
                    origin["max_same_class_iou"] = max(origin["max_same_class_iou"], detail["max_same_class_iou"])
                    origin["max_same_class_coverage"] = max(origin["max_same_class_coverage"], detail["same_class_union_coverage"])
                    origin["max_same_class_prediction_count"] = max(origin["max_same_class_prediction_count"], detail["same_class_prediction_count"])
                    origin["wrong_class_categories"].update(detail["wrong_class_prediction_categories"])
                for prediction_index, (prediction, label) in enumerate(zip(predictions, prediction_labels)):
                    outcome = str(label["outcome"])
                    category_id = int(prediction["category_id"])
                    prediction_counts[(category_id, outcome)] += 1
                    anchor_source_annotation = ""
                    if label["matched_gt_index"] is not None:
                        anchor = ground_truth[int(label["matched_gt_index"])]
                        anchor_source_annotation = source_annotation_id(anchor)
                        if outcome != "true_positive":
                            source_prediction_error_counts[int(anchor_source_annotation)][outcome] += 1
                    native_prediction_rows.append({
                        "native_image_id": image_id,
                        "source_image_id": source_id,
                        "native_file": str(record["file_name"]),
                        "prediction_index": prediction_index,
                        "predicted_class_id": category_id,
                        "predicted_class_name": categories.get(category_id, f"class_{category_id}"),
                        "original_predicted_class_id": prediction["original_category_id"],
                        "original_predicted_class_name": categories.get(int(prediction["original_category_id"]), f"class_{prediction['original_category_id']}"),
                        "classifier_class_id": prediction["classifier_class_id"],
                        "classifier_probability": prediction["classifier_probability"],
                        "second_stage_switched": prediction["second_stage_switched"],
                        "confidence": round(float(prediction["score"]), 6),
                        "prediction_outcome": outcome,
                        "anchor_source_annotation_id": anchor_source_annotation,
                    })
            completed = min(start + len(batch_records), len(records))
            if completed % 100 == 0 or completed == len(records):
                print(f"[{model_name}] inferred/audited {completed}/{len(records)} native images", flush=True)

    instance_rows: list[dict[str, Any]] = []
    for source_id in sorted(source_details):
        detail = source_details[source_id]
        outcome = strict.source_outcome(detail, partial_coverage, fragmented_coverage)
        instance_rows.append({
            "source_annotation_id": source_id,
            "source_image_id": detail["source_image_id"],
            "class_id": detail["class_id"],
            "class_name": detail["class_name"],
            "represented_in_native_dataset": detail["represented"],
            "native_occurrences": detail["native_occurrences"],
            "true_positive_native_matches": detail["true_positive_native_matches"],
            "max_same_class_iou": round(float(detail["max_same_class_iou"]), 6),
            "max_same_class_union_coverage": round(float(detail["max_same_class_coverage"]), 6),
            "wrong_class_predictions": ";".join(categories[item] for item in sorted(detail["wrong_class_categories"])),
            "same_tile_duplicate_prediction_boxes": source_prediction_error_counts[source_id]["same_tile_duplicate"],
            "wrong_class_prediction_boxes": source_prediction_error_counts[source_id]["wrong_class"],
            "localization_or_fragment_prediction_boxes": source_prediction_error_counts[source_id]["localization_or_fragment"],
            "strict_gt_outcome": outcome,
        })
    instance_by_source = {int(item["source_annotation_id"]): item for item in instance_rows}

    gt_outcome_names = ("correct", "fragmented_coverage", "partial_coverage", "wrong_class", "miss", "not_represented")
    per_class_rows: list[dict[str, Any]] = []
    for category_id in sorted(categories):
        rows = [row for row in instance_rows if int(row["class_id"]) == category_id]
        counts = Counter(str(row["strict_gt_outcome"]) for row in rows)
        per_class_rows.append({
            "class_id": category_id,
            "class_name": categories[category_id],
            "ground_truth_instances": len(rows),
            "evaluable_instances": len(rows) - counts["not_represented"],
            **{name: counts[name] for name in gt_outcome_names},
            "mean_same_class_coverage": round(float(np.mean([float(row["max_same_class_union_coverage"]) for row in rows])) if rows else 0.0, 6),
        })

    prediction_outcome_names = ("true_positive", "same_tile_duplicate", "wrong_class", "localization_or_fragment", "background")
    per_prediction_class_rows: list[dict[str, Any]] = []
    for category_id in sorted(categories):
        counts = {name: prediction_counts[(category_id, name)] for name in prediction_outcome_names}
        per_prediction_class_rows.append({
            "predicted_class_id": category_id,
            "predicted_class_name": categories[category_id],
            "prediction_boxes": sum(counts.values()),
            **counts,
        })

    image_ids_to_source_annotations: dict[int, list[int]] = defaultdict(list)
    for row in instance_rows:
        image_ids_to_source_annotations[int(row["source_image_id"])].append(int(row["source_annotation_id"]))
    errors_by_source_image: Counter[int] = Counter()
    for row in native_prediction_rows:
        if row["prediction_outcome"] != "true_positive":
            errors_by_source_image[int(row["source_image_id"])] += 1
    image_rows: list[dict[str, Any]] = []
    image_outcomes = Counter()
    for source_image in sorted(source_data["images"], key=lambda item: int(item["id"])):
        image_id = int(source_image["id"])
        annotation_ids = image_ids_to_source_annotations.get(image_id, [])
        gt_rows = [instance_by_source[item] for item in annotation_ids]
        non_tp_count = sum(row["strict_gt_outcome"] != "correct" for row in gt_rows)
        fp_count = errors_by_source_image[image_id]
        cross_tile_repeats = sum(max(0, int(row["true_positive_native_matches"]) - 1) for row in gt_rows)
        if not gt_rows:
            outcome = "fully_correct" if fp_count == 0 else "prediction_error_only"
        elif non_tp_count:
            outcome = "miss_or_partial"
        elif fp_count:
            outcome = "detected_with_extra_error"
        else:
            outcome = "fully_correct"
        image_outcomes[outcome] += 1
        image_rows.append({
            "source_image_id": image_id,
            "source_file": str(source_image["file_name"]),
            "ground_truth_instances": len(gt_rows),
            "non_true_positive_ground_truth_instances": non_tp_count,
            "prediction_error_boxes": fp_count,
            "cross_tile_repeat_true_positive_boxes": cross_tile_repeats,
            "native_predictions_at_threshold": predictions_by_source_image[image_id],
            "strict_image_outcome": outcome,
        })

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    write_csv(model_dir / "per_ground_truth_class_strict_stats.csv", per_class_rows)
    write_csv(model_dir / "per_prediction_class_error_stats.csv", per_prediction_class_rows)
    write_csv(model_dir / "per_source_defect_instance_strict_outcomes.csv", instance_rows)
    write_csv(model_dir / "per_source_image_strict_outcomes.csv", image_rows)
    write_csv(model_dir / "per_native_prediction_outcomes.csv", native_prediction_rows)

    gt_counts = Counter(str(row["strict_gt_outcome"]) for row in instance_rows)
    prediction_counts_total = Counter(str(row["prediction_outcome"]) for row in native_prediction_rows)
    result = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "second_stage_switch_threshold": switch_threshold,
        "native_validation_images": len(records),
        "native_validation_annotations": len(data["annotations"]),
        "eval_spatial_size_h_w": "x".join(map(str, spatial_size)),
        "source_ground_truth_instances": len(instance_rows),
        "confidence_threshold": threshold,
        "match_iou": match_iou,
        "partial_coverage_threshold": partial_coverage,
        "fragmented_coverage_threshold": fragmented_coverage,
        **{f"gt_{name}": gt_counts[name] for name in gt_outcome_names},
        **{f"pred_{name}": prediction_counts_total[name] for name in prediction_outcome_names},
        "cross_tile_repeat_true_positive_boxes": sum(max(0, int(row["true_positive_native_matches"]) - 1) for row in instance_rows),
        "image_fully_correct": image_outcomes["fully_correct"],
        "image_detected_with_extra_error": image_outcomes["detected_with_extra_error"],
        "image_miss_or_partial": image_outcomes["miss_or_partial"],
        "image_prediction_error_only": image_outcomes["prediction_error_only"],
        "second_stage_switches": sum(count for (src, dst), count in reclass_counts.items() if src != dst),
        "second_stage_decisions": sum(reclass_counts.values()),
    }
    (model_dir / "run_metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(model_dir / "second_stage_switch_counts.csv", [
        {
            "from_class_id": src,
            "from_class_name": CLASS_NAMES.get(src, str(src)),
            "to_class_id": dst,
            "to_class_name": CLASS_NAMES.get(dst, str(dst)),
            "count": count,
        }
        for (src, dst), count in sorted(reclass_counts.items())
    ])
    del dfine_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def compact_threeway(row: dict[str, str]) -> tuple[int, int, int, int, float]:
    total = int(row["ground_truth_instances"])
    correct = int(row["correct"]) + int(row["fragmented_coverage"]) + int(row["partial_coverage"])
    wrong = int(row["wrong_class"])
    miss = int(row["miss"]) + int(row["not_represented"])
    acc = correct / total * 100.0 if total else 0.0
    return correct, wrong, miss, total, acc


def summarize_results(output_dir: Path, variant_results: list[dict[str, Any]]) -> None:
    class_name = {
        0: "小焊烟", 1: "焊炸", 2: "焊烟团", 3: "焊烟", 4: "焊渣",
        5: "长焊高裂", 6: "点焊高", 7: "焊洞", 8: "焊坑", 9: "焊洞长",
        11: "钢帽", 12: "断焊", 13: "焊高纹", 14: "长焊高塌", 15: "缺焊蓝黑",
        16: "焊高烟", 17: "缺焊裂", 18: "焊缝蓝黑", 20: "焊灰色", 21: "焊高缝",
    }
    base_dir = Path(r"D:\1\项目论文\zhwk_runs\dfine_s_unwrapped_150px_cyclic_tiles_20260826_seed0_r1\strict_error_audit_val_20260827\unwrapped_150px_cyclic")
    rows_by_model: dict[str, dict[int, tuple[int, int, int, int, float]]] = {}
    if base_dir.exists():
        with (base_dir / "per_ground_truth_class_strict_stats.csv").open("r", newline="", encoding="utf-8-sig") as handle:
            rows_by_model["新版150px循环滑窗"] = {
                int(row["class_id"]): compact_threeway(row)
                for row in csv.DictReader(handle)
                if int(row["ground_truth_instances"]) > 0
            }
    for result in variant_results:
        model_dir = output_dir / result["model"]
        with (model_dir / "per_ground_truth_class_strict_stats.csv").open("r", newline="", encoding="utf-8-sig") as handle:
            rows_by_model[result["model"]] = {
                int(row["class_id"]): compact_threeway(row)
                for row in csv.DictReader(handle)
                if int(row["ground_truth_instances"]) > 0
            }

    ordered = [13, 3, 14, 15, 7, 16, 9, 5, 1, 6, 4, 12, 0, 20, 18, 2, 8, 17, 21]
    lines = [
        "# D-FINE 150px + 轻量二阶段分类器结果",
        "",
        "统计口径：正确 = strict correct + fragmented_coverage + partial_coverage；错误 = wrong_class；漏检 = miss + not_represented。钢帽不纳入总计。",
        "",
        "## 各类别结果",
        "",
    ]
    headers = ["类别", "标注实例"] + list(rows_by_model)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|---|---:" + "|---:" * len(rows_by_model) + "|")
    for cid in ordered:
        if not any(cid in model_rows for model_rows in rows_by_model.values()):
            continue
        total = next(model_rows[cid][3] for model_rows in rows_by_model.values() if cid in model_rows)
        label = class_name.get(cid, str(cid))
        if cid in TARGET_CLASS_IDS:
            label = f"**{label}**"
        values = []
        for model_rows in rows_by_model.values():
            if cid not in model_rows:
                values.append("—")
                continue
            correct, wrong, miss, _, acc = model_rows[cid]
            values.append(f"{correct} / {wrong} / {miss}<br>{acc:.2f}%")
        lines.append("| " + " | ".join([label, str(total), *values]) + " |")

    lines.extend(["", "## 汇总", ""])
    sets = {
        "非尾类": [13, 3, 14, 15, 7, 16, 9, 5, 1, 6, 4, 12],
        "尾类": [0, 20, 18, 2, 8, 17, 21],
        "三类目标": [12, 14, 15],
        "总计，不含钢帽": ordered,
    }
    lines.append("| 范围 | " + " | ".join(rows_by_model) + " |")
    lines.append("|---" + "|---:" * len(rows_by_model) + "|")
    for label, ids in sets.items():
        values = []
        for model_rows in rows_by_model.values():
            correct = wrong = miss = total = 0
            for cid in ids:
                if cid not in model_rows:
                    continue
                c, w, m, t, _ = model_rows[cid]
                correct += c
                wrong += w
                miss += m
                total += t
            values.append(f"{correct} / {wrong} / {miss}<br>{(correct / total * 100.0 if total else 0.0):.2f}%")
        lines.append("| " + " | ".join([label, *values]) + " |")
    lines.extend(["", "## 变体元数据", ""])
    lines.append("| 变体 | 阈值 | 二阶段处理框数 | 改类次数 |")
    lines.append("|---|---:|---:|---:|")
    for row in variant_results:
        lines.append(
            f"| {row['model']} | {row['second_stage_switch_threshold']:.2f} | "
            f"{row['second_stage_decisions']} | {row['second_stage_switches']} |"
        )
    (output_dir / "second_stage_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Extracting GT crop features...", flush=True)
    x_all, y_all, metadata = collect_gt_crop_features(
        args.train_images, args.train_annotations, args.crop_size,
        args.context_scale, args.jitter_copies, args.jitter, args.seed,
    )
    train_idx, val_idx = grouped_train_val_split(metadata, 0.18, args.seed)
    best_name, classifier, model_rows = fit_candidate_models(
        x_all[train_idx], y_all[train_idx], x_all[val_idx], y_all[val_idx], args.seed,
    )
    write_csv(args.output_dir / "classifier_model_selection.csv", model_rows)
    with (args.output_dir / "best_sklearn_classifier.pkl").open("wb") as handle:
        pickle.dump({
            "classifier": classifier,
            "best_model_name": best_name,
            "target_class_ids": TARGET_CLASS_IDS,
            "class_names": CLASS_NAMES,
            "crop_size": args.crop_size,
            "context_scale": args.context_scale,
        }, handle)
    print(f"Best classifier: {best_name}", flush=True)

    print("Evaluating classifier on validation GT crops...", flush=True)
    x_val_gt, y_val_gt, _ = collect_gt_crop_features(
        args.val_images, args.val_annotations, args.crop_size,
        args.context_scale, 1, 0.0, args.seed,
    )
    pred_val_gt = classifier.predict(x_val_gt)
    val_report = {
        "gt_crop_accuracy": float(accuracy_score(y_val_gt, pred_val_gt)),
        "gt_crop_macro_f1": float(f1_score(y_val_gt, pred_val_gt, average="macro", zero_division=0)),
        "gt_crop_confusion_matrix": confusion_matrix(y_val_gt, pred_val_gt).tolist(),
        "train_augmented_crops": int(len(y_all)),
        "train_source_crops": int(len(metadata) / max(1, args.jitter_copies)),
        "val_gt_crops": int(len(y_val_gt)),
        "best_model_name": best_name,
    }
    (args.output_dir / "classifier_validation_report.json").write_text(json.dumps(val_report, ensure_ascii=False, indent=2), encoding="utf-8")

    strict = load_strict_module(args.scripts_root)
    source_data = load_json(args.source_val_annotations)
    thresholds = [float(item) for item in args.switch_thresholds.split(",") if item.strip()]
    variant_results = []
    for switch_threshold in thresholds:
        print(f"Running D-FINE + second-stage audit, switch_threshold={switch_threshold:.2f}", flush=True)
        result = audit_with_second_stage(
            strict=strict,
            dfine_root=args.dfine_root,
            config=args.config,
            checkpoint=args.checkpoint,
            classifier=classifier,
            switch_threshold=switch_threshold,
            source_data=source_data,
            val_images=args.val_images,
            val_annotations=args.val_annotations,
            output_dir=args.output_dir,
            threshold=args.threshold,
            match_iou=args.match_iou,
            partial_coverage=args.partial_coverage,
            fragmented_coverage=args.fragmented_coverage,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            context_scale=args.context_scale,
            device=torch.device(args.device),
        )
        variant_results.append(result)
    write_csv(args.output_dir / "second_stage_model_summary.csv", variant_results)
    summarize_results(args.output_dir, variant_results)
    print(f"Second-stage experiment complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
