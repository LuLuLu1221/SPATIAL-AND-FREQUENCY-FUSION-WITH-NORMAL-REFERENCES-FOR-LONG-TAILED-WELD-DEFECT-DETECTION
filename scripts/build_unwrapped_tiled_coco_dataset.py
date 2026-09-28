#!/usr/bin/env python
"""Build a leakage-safe COCO dataset of overlapping, unwrapped weld tiles.

The source split is chosen *before* any ROI or tiling operation.  Each source
image is unwrapped with the delivered C1 ROI model, cut into cyclic overlapping
tiles, and its original human boxes are mapped into the weld-strip coordinate
system.  Tiles are letterboxed by padding (never anisotropically stretched).

Only ``train`` and ``val`` are created by default.  The held-out test split is
deliberately not read so it cannot influence this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class StripBox:
    """One source annotation expressed in a cyclic unwrapped strip."""

    source_annotation_id: int
    category_id: int
    x: float
    y: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--roi-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=640)
    parser.add_argument("--strip-height", type=int, default=150)
    parser.add_argument("--stride", type=int, default=480, help="Horizontal stride in unwrapped-strip pixels.")
    parser.add_argument(
        "--phase-offset-mode",
        choices=("none", "per_source_hash"),
        default="none",
        help=(
            "Cyclic starting phase for the tile grid. per_source_hash applies a "
            "deterministic source-specific offset, so the artificial unwrapping seam "
            "is not always seen at the same local x position."
        ),
    )
    parser.add_argument("--min-visible", type=float, default=0.50,
                        help="Keep a tiled label only if this fraction of its strip box remains visible.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--save-full-strips",
        action="store_true",
        help="Also save one complete unwrapped strip per source image for longitudinal interval learning.",
    )
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional deterministic per-split cap, for smoke tests only.")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--debug-count", type=int, default=0,
                        help="Save this many labelled tile debug images per split.")
    parser.add_argument("--max-roi-failure-rate", type=float, default=0.02)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # faster-coco-eval on Windows opens annotation files with the active GBK
    # locale.  ASCII-escaped JSON keeps the same Chinese labels while remaining
    # readable by that loader, exactly as in the raw D-FINE baseline dataset.
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")


def sample_rectangle_boundary(bbox: list[float], samples_per_edge: int = 25) -> np.ndarray:
    """Densely sample the four edges before converting a rectangle to polar space."""
    x, y, width, height = [float(value) for value in bbox]
    values = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float32)
    top = np.column_stack((x + width * values, np.full_like(values, y)))
    right = np.column_stack((np.full_like(values, x + width), y + height * values))
    bottom = np.column_stack((x + width * (1.0 - values), np.full_like(values, y + height)))
    left = np.column_stack((np.full_like(values, x), y + height * (1.0 - values)))
    return np.vstack((top, right, bottom, left))


def map_source_box_to_strip(
    annotation: dict[str, Any],
    *,
    center_x: float,
    center_y: float,
    outer_radius: float,
    inset_px: float,
    strip_width: int,
    strip_height: int,
) -> StripBox | None:
    """Map an axis-aligned source box to a seam-safe bounding box in the strip."""
    points = sample_rectangle_boundary(annotation["bbox"])
    dx, dy = points[:, 0] - center_x, points[:, 1] - center_y
    radius = np.hypot(dx, dy)
    valid = (radius <= outer_radius) & (radius >= outer_radius - inset_px)
    if int(valid.sum()) < 4:
        return None

    # Phase zero is 12 o'clock and progresses clockwise, matching the ROI contract.
    phase = np.mod(np.arctan2(dy[valid], dx[valid]) + math.pi * 0.5, math.tau)
    mean_phase = math.atan2(float(np.sin(phase).mean()), float(np.cos(phase).mean()))
    if mean_phase < 0.0:
        mean_phase += math.tau
    center_strip_x = mean_phase * strip_width / math.tau
    phase_delta = np.angle(np.exp(1j * (phase - mean_phase)))
    xs = center_strip_x + phase_delta * strip_width / math.tau
    ys = (outer_radius - radius[valid]) * (strip_height - 1) / inset_px
    left, top, right, bottom = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    top, bottom = max(0.0, top), min(float(strip_height - 1), bottom)
    width, height = right - left, bottom - top
    if width < 1.0 or height < 1.0:
        return None
    return StripBox(
        source_annotation_id=int(annotation["id"]),
        category_id=int(annotation["category_id"]),
        x=left,
        y=top,
        width=width,
        height=height,
    )


def labels_for_tile(
    boxes: list[StripBox],
    *,
    tile_start: int,
    tile_width: int,
    strip_width: int,
    y_pad: int,
    min_visible: float,
) -> list[tuple[StripBox, list[float]]]:
    """Clip cyclic strip boxes to one tile and retain sufficiently visible labels."""
    retained: list[tuple[StripBox, list[float]]] = []
    tile_end = tile_start + tile_width
    for source_box in boxes:
        best: tuple[float, float, float] | None = None
        for shift in (-strip_width, 0, strip_width):
            left, right = source_box.x + shift, source_box.x + shift + source_box.width
            overlap_left, overlap_right = max(left, tile_start), min(right, tile_end)
            overlap_width = max(0.0, overlap_right - overlap_left)
            visible = overlap_width / max(source_box.width, 1e-6)
            if visible >= min_visible and (best is None or visible > best[0]):
                best = (visible, overlap_left, overlap_width)
        if best is None:
            continue
        _, overlap_left, overlap_width = best
        retained.append((source_box, [
            overlap_left - tile_start,
            source_box.y + y_pad,
            overlap_width,
            source_box.height,
        ]))
    return retained


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, quality))
    if not ok:
        raise RuntimeError(f"Cannot encode tile: {path}")
    encoded.tofile(str(path))


def draw_debug(image: np.ndarray, labels: list[tuple[StripBox, list[float]]]) -> np.ndarray:
    rendered = image.copy()
    for source_box, bbox in labels:
        x, y, width, height = [round(value) for value in bbox]
        colour = ((37 * source_box.category_id) % 256, (97 * source_box.category_id + 80) % 256,
                  (173 * source_box.category_id + 30) % 256)
        cv2.rectangle(rendered, (x, y), (x + width, y + height), colour, 2, cv2.LINE_AA)
        cv2.putText(rendered, str(source_box.category_id), (x, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    return rendered


def choose_records(records: list[dict[str, Any]], max_images: int | None, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: int(item["id"]))
    if max_images is None or max_images >= len(ordered):
        return ordered
    selector = random.Random(seed)
    return sorted(selector.sample(ordered, max_images), key=lambda item: int(item["id"]))


def cyclic_tile_starts(
    *,
    strip_width: int,
    stride: int,
    source_image_id: int,
    split: str,
    args: argparse.Namespace,
) -> tuple[list[int], int]:
    """Return a gap-free, cyclic grid with an optional deterministic phase shift."""
    if args.phase_offset_mode == "none":
        offset = 0
    else:
        split_seed = {"train": 0, "val": 1, "test": 2}[split]
        selector = random.Random(args.seed * 1_000_003 + source_image_id * 7_919 + split_seed)
        offset = selector.randrange(stride)
    starts = [int((offset + position) % strip_width) for position in range(0, strip_width, stride)]
    if len(starts) != len(set(starts)):
        raise RuntimeError("Cyclic tile grid unexpectedly contains duplicate starts")
    return starts, offset


def build_split(
    *,
    split: str,
    source: dict[str, Any],
    args: argparse.Namespace,
    runtime: Any,
    unwrap_c1_annulus: Any,
    read_image: Any,
    inset_px: float,
) -> dict[str, Any]:
    records = choose_records(source["images"], args.max_images, args.seed + {"train": 0, "val": 1, "test": 2}[split])
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in source["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    image_dir = args.output_dir / "images" / split
    debug_dir = args.output_dir / "debug" / split
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    tile_metadata: list[dict[str, Any]] = []
    strip_box_metadata: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source_images_requested": len(records),
        "source_annotations_requested": sum(len(annotations_by_image[int(record["id"])]) for record in records),
        "processed_source_images": 0,
        "roi_failures": [],
        "source_annotations_intersecting_weld_band": 0,
        "source_annotations_represented_in_tiles": 0,
        "generated_tiles": 0,
        "empty_tiles": 0,
        "generated_annotations": 0,
        "saved_full_strips": 0,
        "category_counts": Counter(),
    }
    next_image_id, next_annotation_id, debug_written = 1, 1, 0
    represented_source_ids: set[int] = set()

    for ordinal, record in enumerate(records, start=1):
        source_path = args.image_root / str(record["file_name"])
        try:
            prediction = runtime.predict(read_image(source_path))
            if prediction.c1 is None:
                raise RuntimeError("C1 circle fitting failed")
            strip, _, _ = unwrap_c1_annulus(prediction.image, prediction.c1, inset_px, args.strip_height)
        except Exception as exc:  # Data generation must record, rather than hide, an ROI failure.
            report["roi_failures"].append({"image_id": int(record["id"]), "source_file": str(record["file_name"]), "error": str(exc)})
            print(f"[{split} {ordinal}/{len(records)}] ROI FAILED: {record['file_name']} :: {exc}", flush=True)
            continue

        strip_height, strip_width = strip.shape[:2]
        if strip_height > args.tile_height:
            raise ValueError(f"strip height {strip_height} exceeds tile height {args.tile_height}")
        y_pad = (args.tile_height - strip_height) // 2
        mapped_boxes = [
            mapped for annotation in annotations_by_image[int(record["id"])]
            if (mapped := map_source_box_to_strip(
                annotation,
                center_x=prediction.c1.cx,
                center_y=prediction.c1.cy,
                outer_radius=prediction.c1.radius,
                inset_px=inset_px,
                strip_width=strip_width,
                strip_height=strip_height,
            )) is not None
        ]
        report["source_annotations_intersecting_weld_band"] += len(mapped_boxes)
        tile_starts, phase_offset = cyclic_tile_starts(
            strip_width=strip_width,
            stride=args.stride,
            source_image_id=int(record["id"]),
            split=split,
            args=args,
        )
        # Preserve the uncut cyclic coordinates as a side artifact.  It is the
        # supervision source for the future longitudinal interval branch; the
        # standard COCO export below still contains only tile-local boxes.
        strip_box_metadata.append({
            "source_image_id": int(record["id"]),
            "source_file": str(record["file_name"]),
            "strip_width": strip_width,
            "strip_height": strip_height,
            "phase_offset": phase_offset,
            "boxes": [asdict(box) for box in mapped_boxes],
        })
        if args.save_full_strips:
            strip_name = f"{split}_{int(record['id']):06d}_strip.jpg"
            write_image(args.output_dir / "strips" / split / strip_name, strip, args.jpeg_quality)
            strip_box_metadata[-1]["strip_file"] = str(Path("strips") / split / strip_name)
            report["saved_full_strips"] += 1
        for tile_index, tile_start in enumerate(tile_starts):
            columns = np.mod(np.arange(tile_start, tile_start + args.tile_width), strip_width)
            crop = strip[:, columns]
            tile = np.full((args.tile_height, args.tile_width, 3), 114, dtype=np.uint8)
            tile[y_pad:y_pad + strip_height, :] = crop
            labels = labels_for_tile(
                mapped_boxes,
                tile_start=tile_start,
                tile_width=args.tile_width,
                strip_width=strip_width,
                y_pad=y_pad,
                min_visible=args.min_visible,
            )
            output_name = f"{split}_{int(record['id']):06d}_tile_{tile_index:02d}.jpg"
            write_image(image_dir / output_name, tile, args.jpeg_quality)
            coco_images.append({
                "id": next_image_id,
                "file_name": output_name,
                "width": args.tile_width,
                "height": args.tile_height,
                "source_image_id": int(record["id"]),
                "source_file": str(record["file_name"]),
                "tile_start": tile_start,
                "phase_offset": phase_offset,
            })
            tile_metadata.append({
                "tile_image_id": next_image_id,
                "tile_file": output_name,
                "source_image_id": int(record["id"]),
                "source_file": str(record["file_name"]),
                "strip_width": strip_width,
                "strip_height": strip_height,
                "tile_start": tile_start,
                "phase_offset": phase_offset,
                "tile_width": args.tile_width,
                "y_pad": y_pad,
                "c1_circle": asdict(prediction.c1),
                "inset_px": inset_px,
                "correction": "none",
            })
            if not labels:
                report["empty_tiles"] += 1
            for source_box, bbox in labels:
                area = float(bbox[2] * bbox[3])
                coco_annotations.append({
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": source_box.category_id,
                    "bbox": [round(value, 4) for value in bbox],
                    "area": round(area, 4),
                    "iscrowd": 0,
                    "segmentation": [],
                    "source_annotation_id": source_box.source_annotation_id,
                })
                next_annotation_id += 1
                represented_source_ids.add(source_box.source_annotation_id)
                report["category_counts"][source_box.category_id] += 1
            if labels and debug_written < args.debug_count:
                write_image(debug_dir / output_name, draw_debug(tile, labels), args.jpeg_quality)
                debug_written += 1
            next_image_id += 1

        report["processed_source_images"] += 1
        report["generated_tiles"] += len(tile_starts)
        if ordinal % 25 == 0 or ordinal == len(records):
            print(f"[{split} {ordinal}/{len(records)}] processed={report['processed_source_images']} tiles={report['generated_tiles']} roi_failures={len(report['roi_failures'])}", flush=True)

    report["source_annotations_represented_in_tiles"] = len(represented_source_ids)
    report["generated_annotations"] = len(coco_annotations)
    report["category_counts"] = {str(key): value for key, value in sorted(report["category_counts"].items())}
    failure_rate = len(report["roi_failures"]) / max(1, len(records))
    report["roi_failure_rate"] = failure_rate
    if failure_rate > args.max_roi_failure_rate:
        raise RuntimeError(f"{split} ROI failure rate {failure_rate:.2%} exceeds {args.max_roi_failure_rate:.2%}")

    output_coco = {
        "info": {
            "description": "68-pole C1 weld-annulus unwrapped overlapping tiles",
            "source_split": split,
            "tile_geometry": {
                "tile_width": args.tile_width,
                "tile_height": args.tile_height,
                "strip_height": args.strip_height,
                "stride": args.stride,
                "min_visible": args.min_visible,
                "phase_offset_mode": args.phase_offset_mode,
                "vertical_padding": y_pad if report["processed_source_images"] else None,
                "cyclic_horizontal_boundary": True,
                "full_strips_saved": args.save_full_strips,
            },
            "roi_correction": "none",
        },
        "licenses": source.get("licenses", []),
        "categories": source["categories"],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    write_json(args.output_dir / "annotations" / f"instances_{split}.json", output_coco)
    write_json(args.output_dir / "metadata" / f"tiles_{split}.json", tile_metadata)
    write_json(args.output_dir / "metadata" / f"strip_boxes_{split}.json", strip_box_metadata)
    return report


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix outputs into non-empty directory: {args.output_dir}")
    if args.tile_width <= 0 or args.tile_height <= 0 or args.strip_height <= 1 or args.stride <= 0:
        raise ValueError("tile dimensions, strip height, and stride must be positive")
    if not 0.0 < args.min_visible <= 1.0:
        raise ValueError("--min-visible must be in (0, 1]")
    if args.tile_width < args.stride:
        raise ValueError("--stride must not exceed --tile-width; this would leave gaps in the weld ring")

    roi_src = args.roi_package / "src"
    sys.path.insert(0, str(roi_src))
    from fbf_68jizhu_roi_runtime import RoiRuntime, read_image  # type: ignore[import-not-found]
    from unwrap_68jizhu_weld_annulus import unwrap_c1_annulus  # type: ignore[import-not-found]

    runtime = RoiRuntime(args.roi_package / "configs" / "fbf_68jizhu_roi_runtime.yaml")
    inset_px = float(runtime.cfg["geometry"]["c1_inner_inset_px"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    overall: dict[str, Any] = {
        "source_image_root": str(args.image_root.resolve()),
        "source_annotations_root": str(args.annotations_root.resolve()),
        "roi_package": str(args.roi_package.resolve()),
        "roi_model": str((args.roi_package / "models" / "fbf_68jizhu_roi_mbv2_384.onnx").resolve()),
        "splits": list(args.splits),
        "parameters": {
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "strip_height": args.strip_height,
            "stride": args.stride,
            "min_visible": args.min_visible,
            "phase_offset_mode": args.phase_offset_mode,
            "save_full_strips": args.save_full_strips,
            "seed": args.seed,
            "roi_correction": "none",
        },
        "split_reports": {},
    }
    for split in args.splits:
        source_path = args.annotations_root / f"instances_{split}.json"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        print(f"Building {split} split from {source_path}", flush=True)
        overall["split_reports"][split] = build_split(
            split=split,
            source=read_json(source_path),
            args=args,
            runtime=runtime,
            unwrap_c1_annulus=unwrap_c1_annulus,
            read_image=read_image,
            inset_px=inset_px,
        )
    write_json(args.output_dir / "preparation_report.json", overall)
    print(json.dumps(overall["split_reports"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
