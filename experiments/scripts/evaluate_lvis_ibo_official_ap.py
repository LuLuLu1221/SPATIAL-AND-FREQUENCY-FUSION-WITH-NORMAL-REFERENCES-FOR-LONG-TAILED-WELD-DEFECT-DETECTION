"""Export existing LVIS IBO post-processing outputs and score them with LVIS API.

This does not retrain D-FINE.  It evaluates three frozen prediction sets from
the same candidate pool: raw candidates, reliability-filtered candidates, and
reliability-filtered candidates after restricted cluster correction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Compatibility for the installed LVIS API under NumPy 2.x.
import numpy as np
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from lvis import LVIS, LVISEval, LVISResults


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_scores(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["ibo_id"]: float(row["reliability_score"]) for row in csv.DictReader(handle)}


def export_predictions(rows: list[dict], reliability: dict[str, float], threshold: float | None) -> list[dict]:
    predictions: list[dict] = []
    for row in rows:
        if threshold is not None and reliability.get(str(row["ibo_id"]), -1.0) < threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in row["pred_xyxy"])
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        predictions.append({
            "image_id": int(row["source_image_id"]),
            "category_id": int(row["pred_class_id"]),
            "bbox": [x1, y1, width, height],
            "score": float(row["pred_score"]),
        })
    return predictions


def score(ground_truth: Path, prediction_json: Path) -> dict:
    lvis_gt = LVIS(str(ground_truth))
    lvis_dt = LVISResults(lvis_gt, str(prediction_json))
    evaluation = LVISEval(lvis_gt, lvis_dt, "bbox")
    evaluation.run()
    return {key: float(value) for key, value in evaluation.get_results().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--corrected-manifest", type=Path, required=True)
    parser.add_argument("--reliability-scores", type=Path, required=True)
    parser.add_argument("--reliability-threshold", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = read_jsonl(args.baseline_manifest)
    corrected = read_jsonl(args.corrected_manifest)
    reliability = read_scores(args.reliability_scores)
    settings = {
        "dfine_candidate_raw": (baseline, None),
        "ibo_reliability": (baseline, args.reliability_threshold),
        "ibo_reliability_cluster_corrected": (corrected, args.reliability_threshold),
    }
    results = {"ground_truth": str(args.ground_truth.resolve()), "reliability_threshold": args.reliability_threshold, "settings": {}}
    for name, (rows, threshold) in settings.items():
        exported = export_predictions(rows, reliability, threshold)
        prediction_path = args.output_dir / f"{name}_predictions.json"
        prediction_path.write_text(json.dumps(exported), encoding="utf-8")
        metrics = score(args.ground_truth, prediction_path)
        results["settings"][name] = {"predictions": len(exported), "metrics": metrics}
        print(f"[LVIS-AP] {name}: predictions={len(exported)} AP={metrics.get('AP')}", flush=True)
    (args.output_dir / "official_lvis_ap_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
