#!/usr/bin/env python
"""Extract D-FINE detection-query features aligned to an existing LVIS IBO manifest.

The script reruns the trained D-FINE detector with the same no-stretch
resize+pad inference used to build IBO candidates. It reproduces the top-k
postprocessing indices and gathers ``raw_outputs["query_feats"]`` by those query
indices, then writes one feature vector per existing ``ibo_id``.

It does not rebuild crops or overwrite the IBO manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TVF
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--candidate-threshold", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-float16", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


def clamp_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = sorted((max(0.0, x0), min(float(width), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(height), y1)))
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def build_model(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(args.dfine_root))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config), resume=str(args.checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.deploy().to(device).eval()

    spatial_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    eval_h, eval_w = [int(value) for value in spatial_size]
    if eval_h != eval_w:
        raise ValueError(f"Expected square eval_spatial_size, got {spatial_size}")
    eval_size = eval_h

    post = cfg.postprocessor.deploy()
    num_classes = int(post.num_classes)
    num_top_queries = int(post.num_top_queries)
    use_focal_loss = bool(post.use_focal_loss)

    def transform(image: Image.Image) -> torch.Tensor:
        width, height = image.size
        scale = eval_size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = TVF.resize(image, [new_height, new_width])
        tensor = TVF.to_tensor(resized)
        return F.pad(tensor, [0, eval_size - new_width, 0, eval_size - new_height], value=0.0)

    return model, transform, eval_size, num_classes, num_top_queries, use_focal_loss


def postprocess_with_query_features(
    raw_outputs: dict[str, torch.Tensor],
    original_sizes: torch.Tensor,
    eval_size: int,
    num_classes: int,
    num_top_queries: int,
    use_focal_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = raw_outputs["pred_logits"]
    boxes = raw_outputs["pred_boxes"]
    query_feats = raw_outputs["query_feats"]
    padded_sizes = torch.full(
        (logits.shape[0], 2),
        float(eval_size),
        dtype=torch.float32,
        device=logits.device,
    )
    bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    bbox_pred *= padded_sizes.repeat(1, 2).unsqueeze(1)

    if use_focal_loss:
        flat_scores = torch.sigmoid(logits).flatten(1)
        scores, flat_index = torch.topk(flat_scores, num_top_queries, dim=-1)
        labels = flat_index % num_classes
        query_index = flat_index // num_classes
        gathered_boxes = bbox_pred.gather(
            dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
        )
        gathered_feats = query_feats.gather(
            dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, query_feats.shape[-1])
        )
    else:
        probs = torch.softmax(logits, dim=-1)[:, :, :-1]
        scores, labels = probs.max(dim=-1)
        if scores.shape[1] > num_top_queries:
            scores, query_index = torch.topk(scores, num_top_queries, dim=-1)
            labels = torch.gather(labels, dim=1, index=query_index)
            gathered_boxes = bbox_pred.gather(
                dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
            gathered_feats = query_feats.gather(
                dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, query_feats.shape[-1])
            )
        else:
            query_index = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)
            gathered_boxes = bbox_pred
            gathered_feats = query_feats

    widths = original_sizes[:, 0].to(gathered_boxes.device)
    heights = original_sizes[:, 1].to(gathered_boxes.device)
    max_sides = torch.maximum(widths, heights)
    scales = float(eval_size) / max_sides
    valid_widths = widths * scales
    valid_heights = heights * scales
    original_boxes = gathered_boxes.clone()
    original_boxes[..., 0::2] = original_boxes[..., 0::2].clamp(min=0.0)
    original_boxes[..., 1::2] = original_boxes[..., 1::2].clamp(min=0.0)
    original_boxes[..., 0::2] = torch.minimum(original_boxes[..., 0::2], valid_widths[:, None, None])
    original_boxes[..., 1::2] = torch.minimum(original_boxes[..., 1::2], valid_heights[:, None, None])
    original_boxes = original_boxes / scales[:, None, None]
    original_boxes[..., 0::2] = torch.minimum(original_boxes[..., 0::2], widths[:, None, None])
    original_boxes[..., 1::2] = torch.minimum(original_boxes[..., 1::2], heights[:, None, None])
    return labels, original_boxes, scores, query_index, gathered_feats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    data = load_json(args.annotations)
    records = sorted(data["images"], key=lambda item: int(item["id"]))
    if args.max_images > 0:
        records = records[: args.max_images]
    manifest_rows = read_jsonl(args.manifest)
    model, transform, eval_size, num_classes, num_top_queries, use_focal_loss = build_model(args, device)

    features: list[np.ndarray] = []
    ibo_ids: list[str] = []
    query_indices: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    manifest_pos = 0
    print(
        f"[QUERY-FEAT] split={args.split} images={len(records)} manifest={len(manifest_rows)} device={device}",
        flush=True,
    )
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            tensors: list[torch.Tensor] = []
            sizes: list[list[float]] = []
            images: list[Image.Image] = []
            for record in batch_records:
                image = Image.open(args.image_root / str(record["file_name"])).convert("RGB")
                images.append(image)
                tensors.append(transform(image))
                sizes.append([float(image.width), float(image.height)])
            raw_outputs = model(torch.stack(tensors).to(device))
            labels, boxes, scores, query_index, query_feats = postprocess_with_query_features(
                raw_outputs,
                torch.tensor(sizes, dtype=torch.float32, device=device),
                eval_size,
                num_classes,
                num_top_queries,
                use_focal_loss,
            )
            labels_cpu = labels.detach().cpu()
            boxes_cpu = boxes.detach().cpu()
            scores_cpu = scores.detach().cpu()
            qidx_cpu = query_index.detach().cpu()
            qfeat_cpu = query_feats.detach().cpu()
            for batch_index, record in enumerate(batch_records):
                width = int(sizes[batch_index][0])
                height = int(sizes[batch_index][1])
                tile_image_id = int(record["id"])
                for category_id, box, score, qidx, qfeat in zip(
                    labels_cpu[batch_index].tolist(),
                    boxes_cpu[batch_index].numpy(),
                    scores_cpu[batch_index].tolist(),
                    qidx_cpu[batch_index].tolist(),
                    qfeat_cpu[batch_index].numpy(),
                ):
                    score = float(score)
                    if score < args.candidate_threshold:
                        continue
                    pred_box = clamp_box(np.asarray(box, dtype=np.float32), width, height)
                    if pred_box[2] <= pred_box[0] or pred_box[3] <= pred_box[1]:
                        continue
                    if manifest_pos >= len(manifest_rows):
                        mismatch_rows.append(
                            {
                                "kind": "extra_generated_candidate",
                                "tile_image_id": tile_image_id,
                                "pred_class_id": int(category_id),
                                "pred_score": round(score, 6),
                            }
                        )
                        continue
                    row = manifest_rows[manifest_pos]
                    mismatches: list[str] = []
                    if int(row["tile_image_id"]) != tile_image_id:
                        mismatches.append("tile_image_id")
                    if int(row["pred_class_id"]) != int(category_id):
                        mismatches.append("pred_class_id")
                    if abs(float(row["pred_score"]) - score) > 2e-4:
                        mismatches.append("pred_score")
                    old_box = np.asarray(row["pred_xyxy"], dtype=np.float32)
                    if float(np.max(np.abs(old_box - pred_box))) > 0.75:
                        mismatches.append("pred_xyxy")
                    if mismatches:
                        mismatch_rows.append(
                            {
                                "kind": "alignment_mismatch",
                                "ibo_id": row.get("ibo_id", ""),
                                "fields": ",".join(mismatches),
                                "manifest_pos": manifest_pos,
                                "manifest_tile": row.get("tile_image_id", ""),
                                "generated_tile": tile_image_id,
                                "manifest_class": row.get("pred_class_id", ""),
                                "generated_class": int(category_id),
                                "manifest_score": row.get("pred_score", ""),
                                "generated_score": round(score, 6),
                                "max_box_abs_diff": round(float(np.max(np.abs(old_box - pred_box))), 6),
                            }
                        )
                    ibo_ids.append(str(row["ibo_id"]))
                    query_indices.append(int(qidx))
                    features.append(qfeat.astype(np.float32))
                    metadata_rows.append(
                        {
                            "ibo_id": row["ibo_id"],
                            "tile_image_id": tile_image_id,
                            "query_index": int(qidx),
                            "pred_class_id": int(category_id),
                            "pred_score": round(score, 6),
                        }
                    )
                    manifest_pos += 1
            for image in images:
                image.close()
            done = start + len(batch_records)
            if done == len(records) or done % max(args.batch_size * 25, 100) == 0:
                print(
                    f"[QUERY-FEAT] {args.split}: {done}/{len(records)} images features={len(features)} mismatches={len(mismatch_rows)}",
                    flush=True,
                )

    if manifest_pos != len(manifest_rows):
        mismatch_rows.append(
            {
                "kind": "missing_generated_candidates",
                "generated": manifest_pos,
                "manifest": len(manifest_rows),
            }
        )

    feat_array = np.vstack(features).astype(np.float16 if args.save_float16 else np.float32)
    out_npz = args.output_dir / f"dfine_query_features_{args.split}.npz"
    np.savez_compressed(
        out_npz,
        ibo_ids=np.asarray(ibo_ids, dtype="U32"),
        query_indices=np.asarray(query_indices, dtype=np.int32),
        features=feat_array,
    )
    write_csv(args.output_dir / f"dfine_query_feature_meta_{args.split}.csv", metadata_rows)
    write_csv(args.output_dir / f"dfine_query_feature_alignment_mismatches_{args.split}.csv", mismatch_rows)
    summary = {
        "split": args.split,
        "images": len(records),
        "manifest_candidates": len(manifest_rows),
        "features": int(feat_array.shape[0]),
        "feature_dim": int(feat_array.shape[1]),
        "dtype": str(feat_array.dtype),
        "mismatches": len(mismatch_rows),
        "output_npz": str(out_npz),
    }
    (args.output_dir / f"dfine_query_features_summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[QUERY-FEAT] done split={args.split} features={feat_array.shape} mismatches={len(mismatch_rows)} output={out_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
