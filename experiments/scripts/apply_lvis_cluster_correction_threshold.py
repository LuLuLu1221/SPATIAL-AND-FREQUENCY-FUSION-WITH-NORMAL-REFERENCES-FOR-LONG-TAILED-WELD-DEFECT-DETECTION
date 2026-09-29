#!/usr/bin/env python
"""Apply saved cluster-corrector predictions with new correction thresholds."""

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
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply-reliability-threshold", type=float, default=0.15)
    parser.add_argument("--min-correction-proba", type=float, default=0.55)
    parser.add_argument("--min-correction-margin", type=float, default=0.15)
    parser.add_argument("--tag", default="cluster")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.val_manifest)
    preds = {row["ibo_id"]: row for row in read_csv(args.prediction_csv)}

    corrected: list[dict[str, Any]] = []
    applied = 0
    for row in rows:
        item = dict(row)
        pred = preds.get(str(item["ibo_id"]))
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
                item["pred_class_name"] = f"{args.tag}_corrected_class_{proposed}"
                item[f"{args.tag}_correction_applied"] = True
                item[f"{args.tag}_correction_proba"] = proba
                item[f"{args.tag}_correction_margin"] = margin
                item[f"{args.tag}_correction_cluster_id"] = pred.get("cluster_id", "")
                applied += 1
            else:
                item[f"{args.tag}_correction_applied"] = False
        else:
            item[f"{args.tag}_correction_applied"] = False
        corrected.append(item)

    write_jsonl(args.output_dir / "ibo_manifest_val_threshold_corrected.jsonl", corrected)
    # Preserve complete score file so evaluator sees all candidates.
    score_text = args.score_csv.read_text(encoding="utf-8-sig")
    (args.output_dir / "lvis_ibo_reliability_scores_val_threshold_corrected.csv").write_text(score_text, encoding="utf-8-sig")
    summary = {
        "apply_reliability_threshold": args.apply_reliability_threshold,
        "min_correction_proba": args.min_correction_proba,
        "min_correction_margin": args.min_correction_margin,
        "val_candidates": len(rows),
        "cluster_predictions": len(preds),
        "corrections_applied": applied,
    }
    (args.output_dir / "apply_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[APPLY-CLUSTER] applied={applied} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
