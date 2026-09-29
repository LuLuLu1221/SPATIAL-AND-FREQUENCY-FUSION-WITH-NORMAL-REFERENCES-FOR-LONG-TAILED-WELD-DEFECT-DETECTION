#!/usr/bin/env python
"""Analyze wrong-class pairs among LVIS/SimLTD IBO candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores: dict[str, float] = {}
    with args.evidence.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scores[row["ibo_id"]] = float(row["reliability_fusion_score"])

    counts: Counter[tuple[str, str, str, str]] = Counter()
    examples: dict[tuple[str, str, str, str], str] = {}
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if scores.get(row["ibo_id"], 0.0) < args.threshold:
                continue
            gt_class = str(row.get("matched_gt_class_id", ""))
            if not gt_class:
                continue
            pred_class = str(row.get("pred_class_id", ""))
            if pred_class == gt_class:
                continue
            key = (
                pred_class,
                str(row.get("pred_class_name", "")),
                gt_class,
                str(row.get("matched_gt_class_name", "")),
            )
            counts[key] += 1
            examples.setdefault(key, str(row.get("crop_path", "")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pred_class_id", "pred_class_name", "gt_class_id", "gt_class_name", "count", "example_crop"])
        for key, count in counts.most_common(args.top_k):
            writer.writerow([*key, count, examples[key]])

    print(f"[CONFUSION] threshold={args.threshold} pairs={len(counts)} output={args.output}", flush=True)
    for key, count in counts.most_common(min(20, args.top_k)):
        print(f"[CONFUSION] pred={key[1]}({key[0]}) gt={key[3]}({key[2]}) count={count}", flush=True)


if __name__ == "__main__":
    main()
