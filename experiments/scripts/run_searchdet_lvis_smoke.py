"""Run the released SearchDet DINOv2 + SAM-HQ candidate mechanism on LVIS.

The original repository exposes single-concept demonstration scripts rather
than an LVIS evaluator.  This adapter preserves its model components and
embedding adjustment, applies them to the predeclared smoke records, and
reports class-conditioned localization at IoU 0.5.  It is intentionally not
labelled as official LVIS AP.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything_hq import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.metrics.pairwise import cosine_similarity


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    y, x = np.where(mask)
    return [float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)]


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--searchdet-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.searchdet_root)
    sys.path.insert(0, str(root))
    from heatmap_generation import DinoFeatureExtractor

    pilot = json.loads(Path(args.pilot_manifest).read_text(encoding="utf-8"))
    supports = json.loads(Path(args.support_manifest).read_text(encoding="utf-8"))
    support_by_id = {row["category_id"]: row for row in supports["classes"]}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip only GitHub API fork validation; source and weights remain official DINOv2.
    local_dino_repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
    dino = torch.hub.load(str(local_dino_repo), "dinov2_vits14", source="local").to("cuda").eval()
    extractor = DinoFeatureExtractor(dino, resize_images=True, crop_images=False)
    sam = sam_model_registry["vit_l"](checkpoint=args.sam_checkpoint).to("cuda").eval()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    support_embeddings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for category_id, row in support_by_id.items():
        pos = [Image.open(file).convert("RGB") for file in row["positive_files"]]
        neg = [Image.open(file).convert("RGB") for file in row["negative_files"]]
        if len(pos) < 5 or len(neg) < 5:
            continue
        with torch.no_grad():
            pos_vec, _ = extractor(pos)
            neg_vec, _ = extractor(neg)
        support_embeddings[category_id] = (normalize(pos_vec.cpu().numpy()), normalize(neg_vec.cpu().numpy()))

    rows = []
    for index, item in enumerate(pilot["smoke_items"][: args.limit], start=1):
        category_id = item["category_id"]
        started = time.time()
        if category_id not in support_embeddings:
            rows.append({"image_id": item["image_id"], "category_id": category_id, "status": "skipped_missing_supports"})
            continue
        image_path = Path(args.image_root) / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image)
        pos, neg = support_embeddings[category_id]
        # Exact released adjustment rule: one adjusted embedding per positive exemplar.
        adjusted = []
        for query in pos:
            pos_w = cosine_similarity(query[None, :], pos).flatten()
            neg_w = cosine_similarity(query[None, :], neg).flatten()
            adjusted.append((pos_w[:, None] * pos).sum(axis=0) - (neg_w[:, None] * neg).sum(axis=0))
        adjusted = normalize(np.asarray(adjusted, dtype=np.float32))
        masks = mask_generator.generate(image_np)
        mask_vectors = []
        valid_masks = []
        for mask in masks:
            masked = np.full_like(image_np, 255)
            masked[mask["segmentation"]] = image_np[mask["segmentation"]]
            with torch.no_grad():
                vector, _ = extractor([Image.fromarray(masked)])
            mask_vectors.append(vector.squeeze().cpu().numpy())
            valid_masks.append(mask)
        mask_vectors = normalize(np.asarray(mask_vectors, dtype=np.float32))
        scores = (mask_vectors @ adjusted.T).max(axis=1)
        order = np.argsort(-scores)
        predictions = [
            {"score": float(scores[i]), "bbox_xyxy": bbox_from_mask(valid_masks[i]["segmentation"])}
            for i in order[:10]
        ]
        target_boxes = [xywh_to_xyxy(target["bbox_xywh"]) for target in item["targets"]]
        max_iou = [max((iou(box, pred["bbox_xyxy"]) for pred in predictions), default=0.0) for box in target_boxes]
        top1_iou = [iou(box, predictions[0]["bbox_xyxy"]) if predictions else 0.0 for box in target_boxes]
        rows.append({
            "image_id": item["image_id"], "file_name": item["file_name"], "category_id": category_id,
            "category_name": item["category_name"], "status": "ok", "n_masks": len(valid_masks),
            "targets": item["targets"], "top10_predictions": predictions,
            "top1_iou_per_target": top1_iou, "top10_iou_per_target": max_iou,
            "elapsed_seconds": time.time() - started,
        })
        print(f"[{index}/{min(args.limit, len(pilot['smoke_items']))}] {item['category_name']} masks={len(valid_masks)}", flush=True)
        torch.cuda.empty_cache()

    evaluable = [value for row in rows if row.get("status") == "ok" for value in row["top10_iou_per_target"]]
    top1 = [value for row in rows if row.get("status") == "ok" for value in row["top1_iou_per_target"]]
    report = {
        "protocol": {
            "scope": "SearchDet released DINOv2 + SAM-HQ candidate mechanism; class-conditioned LVIS smoke localization.",
            "not_official_metric": "This is not full LVIS AP because released SearchDet code has no batch multi-class LVIS evaluator.",
            "retrieval_backend": supports.get("retrieval_backend"),
            "sam_checkpoint": str(Path(args.sam_checkpoint).resolve()),
            "topk": [1, 10],
            "iou_threshold": 0.5,
        },
        "summary": {
            "processed_items": len(rows), "evaluated_instances": len(evaluable),
            "top1_recall_iou50": float(np.mean(np.asarray(top1) >= 0.5)) if top1 else None,
            "top10_recall_iou50": float(np.mean(np.asarray(evaluable) >= 0.5)) if evaluable else None,
        },
        "items": rows,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
