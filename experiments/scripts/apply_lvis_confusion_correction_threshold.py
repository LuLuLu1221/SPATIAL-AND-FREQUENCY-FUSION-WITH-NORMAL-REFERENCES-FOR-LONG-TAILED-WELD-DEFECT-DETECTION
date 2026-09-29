#!/usr/bin/env python
"""Apply saved LVIS confusion-corrector predictions with new thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply-reliability-threshold", type=float, default=0.15)
    parser.add_argument("--min-correction-proba", type=float, default=0.50)
    parser.add_argument("--min-correction-margin", type=float, default=0.20)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.val_manifest)
    preds = {row["ibo_id"]: row for row in read_csv(args.prediction_csv)}

    corrected = []
    applied = 0
    for row in rows:
        item = dict(row)
        pred = preds.get(str(item["ibo_id"]))
        apply = False
        if pred:
            original = int(pred["original_pred_class_id"])
            proposed = int(pred["proposed_class_id"])
            reliability = float(pred["reliability_score"])
            proba = float(pred["correction_proba"])
            margin = float(pred["correction_margin"])
            apply = (
                reliability >= args.apply_reliability_threshold
                and proposed != original
                and proba >= args.min_correction_proba
                and margin >= args.min_correction_margin
            )
            if apply:
                item["original_pred_class_id"] = item["pred_class_id"]
                item["original_pred_class_name"] = item.get("pred_class_name", "")
                item["pred_class_id"] = proposed
                item["pred_class_name"] = f"corrected_class_{proposed}"
                item["confusion_correction_applied"] = True
                item["confusion_correction_proba"] = proba
                item["confusion_correction_margin"] = margin
                applied += 1
            else:
                item["confusion_correction_applied"] = False
        corrected.append(item)

    score_rows = [
        {"ibo_id": row["ibo_id"], "reliability_score": row["reliability_score"]}
        for row in preds.values()
    ]
    write_jsonl(args.output_dir / "ibo_manifest_val_confusion_corrected.jsonl", corrected)
    write_csv(args.output_dir / "lvis_ibo_reliability_scores_val_confusion_corrected.csv", score_rows)
    summary = {
        "apply_reliability_threshold": args.apply_reliability_threshold,
        "min_correction_proba": args.min_correction_proba,
        "min_correction_margin": args.min_correction_margin,
        "val_candidates": len(rows),
        "corrections_applied": applied,
    }
    (args.output_dir / "apply_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[APPLY-CORRECTOR] applied={applied} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
