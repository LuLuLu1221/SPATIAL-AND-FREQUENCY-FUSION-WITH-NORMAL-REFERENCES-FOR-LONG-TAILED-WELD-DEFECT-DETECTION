"""Prepare reproducible SimLTD metadata from the existing LVIS pilot annotation.

The source LVIS pilot annotation uses zero-based category IDs.  SimLTD only
needs an ordered class-name list for each stage and category IDs for checkpoint
surgery, so this script keeps those IDs intact and records the exact 30-shot
sample used by the final stage.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shots", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def write_lines(path: Path, values: list[str | int]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda item: item["id"])
    frequencies = defaultdict(list)
    for category in categories:
        frequencies[category["frequency"]].append(category)

    head = frequencies["c"] + frequencies["f"]
    tail = frequencies["r"]
    if (len(head), len(tail), len(categories)) != (866, 337, 1203):
        raise ValueError("Unexpected LVIS v1 frequency split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, group in (("head", head), ("tail", tail), ("all", categories)):
        group = sorted(group, key=lambda item: item["id"])
        write_lines(args.output_dir / f"lvis_v1_{label}_classes.txt", [item["name"] for item in group])
        write_lines(args.output_dir / f"lvis_v1_{label}_ids.txt", [item["id"] for item in group])

    by_category: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        by_category[annotation["category_id"]].append(annotation)
    rng = random.Random(args.seed)
    sampled = []
    for category in categories:
        records = by_category[category["id"]]
        sampled.extend(records if len(records) <= args.shots else rng.sample(records, args.shots))
    sampled_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
        "images": data["images"],
        "annotations": sampled,
    }
    sample_path = args.output_dir / f"lvis_v1_train_pilot10k_seed{args.seed}@{args.shots}shots.json"
    sample_path.write_text(json.dumps(sampled_data), encoding="utf-8")
    manifest = {
        "source_annotation": str(args.annotation),
        "seed": args.seed,
        "shots": args.shots,
        "head_classes": len(head),
        "tail_classes": len(tail),
        "sampled_annotations": len(sampled),
    }
    (args.output_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
