"""Ablation study for D-FINE + IBO + normal/frequency evidence.

The script only consumes existing candidate/evidence CSV files and evaluates
which evidence blocks change instance-level results.  It never uses
same_cov/same_iou/other_cov/other_iou as model inputs because those are
ground-truth-derived evaluation fields.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"D:\1\项目论文")
SCRIPTS = ROOT / "zhwk_project" / "scripts"
EVIDENCE_DIR = ROOT / "zhwk_runs" / "normal_reference_no_defect_20260907" / "03_group_evidence_external_normal_correct_eval_sameclass_family12_14_15_smoke2_3"
GRAY_CORRECT_DIR = ROOT / "zhwk_runs" / "normal_reference_no_defect_20260912" / "20_gray_vs_collapse_contrast_corrector_v1"
OUT = ROOT / "zhwk_runs" / "normal_reference_no_defect_20260912" / "21_ibo_evidence_ablation_v1"

TRAIN_EVIDENCE = EVIDENCE_DIR / "ibo_group_evidence_train.csv"
VAL_EVIDENCE = EVIDENCE_DIR / "ibo_group_evidence_val.csv"
VAL_ANNOTATIONS = ROOT / "zhwk_dfine_v1" / "annotations" / "instances_val.json"
VAL_STRIP_BOXES = ROOT / "zhwk_unwrapped_150px_cyclic_tiles_v1" / "metadata" / "strip_boxes_val.json"
GRAY_OVERRIDES = GRAY_CORRECT_DIR / "best_within_1pp_overrides.csv"

EXCLUDE = {11, 17, 19, 21, 22}
MATCH_FAMILIES = ["0,2,3", "12,14,15,18"]
COVERAGE_THRESHOLD = 0.20
MATCH_IOU = 0.50
SEED = 20260912

GROUPS = [
    ("烟类合并", {0, 2, 3}, "head"),
    ("焊炸", {1}, "non_tail"),
    ("焊渣", {4}, "tail"),
    ("长焊高裂", {5}, "non_tail"),
    ("点焊高", {6}, "tail"),
    ("焊洞", {7}, "non_tail"),
    ("焊坑", {8}, "tail"),
    ("焊洞长", {9}, "tail"),
    ("长焊高", {10}, "tail_holdout_absent_in_original_val"),
    ("缺焊长塌合并", {12, 14, 15, 18}, "head_confusable"),
    ("焊高纹", {13}, "head"),
    ("焊高烟", {16}, "non_tail"),
    ("焊灰色", {20}, "tail_key"),
]

ABLATIONS = [
    {
        "id": "A0",
        "name": "D-FINE 原始分数",
        "kind": "score",
        "score_col": "max_pred_score",
        "features": [],
        "purpose": "只看检测器候选框和类别本身，不使用 IBO/正常参照/频率证据。",
    },
    {
        "id": "A1",
        "name": "D-FINE + IBO空间/组证据",
        "kind": "model",
        "features": ["max_pred_score", "member_count", "max_reliability_v1", "mean_reliability_v1"],
        "purpose": "看候选框聚合、组内可靠性、重叠支持是否带来提升。",
    },
    {
        "id": "A2",
        "name": "D-FINE + IBO + 正常参照",
        "kind": "model",
        "features": ["max_pred_score", "member_count", "max_reliability_v1", "mean_reliability_v1", "normal_reference_score"],
        "purpose": "看无缺陷正常参照距离是否能帮助过滤/确认异常。",
    },
    {
        "id": "A3",
        "name": "D-FINE + IBO + 频率证据",
        "kind": "model",
        "features": ["max_pred_score", "member_count", "max_reliability_v1", "mean_reliability_v1", "frequency_evidence_score"],
        "purpose": "看频率纹理信息是否能帮助区分纹理型、颜色/塌陷型缺陷。",
    },
    {
        "id": "A4",
        "name": "D-FINE + IBO + 正常参照 + 频率证据",
        "kind": "model",
        "features": [
            "max_pred_score",
            "member_count",
            "max_reliability_v1",
            "mean_reliability_v1",
            "normal_reference_score",
            "frequency_evidence_score",
        ],
        "purpose": "主体方法，不含焊灰色混淆类纠错。",
    },
    {
        "id": "A5",
        "name": "A4 + 焊灰色/缺焊长塌混淆类纠错",
        "kind": "model_plus_gray_override",
        "features": [
            "max_pred_score",
            "member_count",
            "max_reliability_v1",
            "mean_reliability_v1",
            "normal_reference_score",
            "frequency_evidence_score",
        ],
        "purpose": "在主体方法之后，只针对焊灰色被缺焊长塌合并类吸收的问题做二阶段纠错。",
    },
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in df.to_dict("records"):
        if not isinstance(row.get("global_box"), list):
            row["global_box"] = ast.literal_eval(str(row["global_box"]))
        rows.append(row)
    return rows


def aggregate_group(class_rows: list[dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    sub = [r for r in class_rows if int(r["class_id"]) in ids]
    gt = int(sum(int(r["gt_instances"]) for r in sub))
    correct = int(sum(int(r["correct"]) for r in sub))
    wrong = int(sum(int(r["wrong"]) for r in sub))
    miss = int(sum(int(r["miss"]) for r in sub))
    acc = correct / gt if gt else math.nan
    return {"gt": gt, "correct": correct, "wrong": wrong, "miss": miss, "accuracy": acc}


def pct(v: float) -> str:
    return "NA" if math.isnan(v) else f"{v * 100:.2f}%"


def triple(g: dict[str, Any]) -> str:
    return f"{g['correct']} / {g['wrong']} / {g['miss']}"


def evaluate(parent, selected: list[dict[str, Any]], method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return parent.evaluate_instance_from_selected(
        selected,
        load_json(VAL_ANNOTATIONS),
        load_json(VAL_STRIP_BOXES),
        COVERAGE_THRESHOLD,
        MATCH_IOU,
        EXCLUDE,
        parent.parse_match_families(MATCH_FAMILIES),
        method,
    )


def build_model(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def load_gray_overrides() -> dict[str, dict[str, Any]]:
    if not GRAY_OVERRIDES.exists():
        return {}
    df = pd.read_csv(GRAY_OVERRIDES)
    return {str(r["merged_id"]): r for r in df.to_dict("records")}


def apply_gray_overrides(rows: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        mid = str(item.get("merged_id"))
        if mid in overrides:
            item["original_class_id"] = int(item["class_id"])
            item["original_class_name"] = item.get("class_name", "")
            item["class_id"] = int(overrides[mid]["final_class_id"])
            item["class_name"] = "焊灰色"
            item["gray_contrast_action"] = "collapse_to_gray_override"
        out.append(item)
    return out


def choose_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    # Primary: total accuracy; secondary: focused tail accuracy; tertiary: fewer wrong labels.
    return max(
        candidates,
        key=lambda r: (
            float(r["total_accuracy"]),
            float(r["focused_tail_accuracy"]) if not math.isnan(float(r["focused_tail_accuracy"])) else -1.0,
            -int(r["total_wrong"]),
            -int(r["selected_groups"]),
        ),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCRIPTS))
    import train_ibo_group_evidence_fusion_v2 as parent

    train = pd.read_csv(TRAIN_EVIDENCE)
    val = pd.read_csv(VAL_EVIDENCE)
    for col in ["member_count", "max_reliability_v1", "mean_reliability_v1", "max_pred_score", "normal_reference_score", "frequency_evidence_score"]:
        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0.0)
        val[col] = pd.to_numeric(val[col], errors="coerce").fillna(0.0)
    train["label"] = pd.to_numeric(train["label"], errors="coerce").fillna(0).astype(int)

    all_summary_rows: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    selected_predictions: dict[str, list[dict[str, Any]]] = {}
    gray_overrides = load_gray_overrides()

    thresholds = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
    score_thresholds = [round(x, 2) for x in np.arange(0.25, 0.951, 0.05)]

    for ab in ABLATIONS:
        print(f"[ABLATION] {ab['id']} {ab['name']}", flush=True)
        if ab["kind"] == "score":
            scores = val[ab["score_col"]].to_numpy(dtype=float)
            ths = score_thresholds
            model = None
        else:
            features = ab["features"]
            model = build_model(features)
            model.fit(train[features].to_numpy(dtype=float), train["label"].to_numpy(dtype=int))
            scores = model.predict_proba(val[features].to_numpy(dtype=float))[:, 1]
            ths = thresholds
            joblib.dump({"ablation": ab, "model": model, "features": features}, OUT / f"{ab['id']}_model.joblib")

        ab_candidates: list[dict[str, Any]] = []
        for th in ths:
            mask = scores >= th
            selected = as_records(val[mask].copy())
            if ab["kind"] == "model_plus_gray_override":
                selected = apply_gray_overrides(selected, gray_overrides)
            summary, class_rows = evaluate(parent, selected, f"{ab['id']}_th{th:.2f}")
            total = {
                "ablation_id": ab["id"],
                "ablation_name": ab["name"],
                "threshold": th,
                "selected_groups": len(selected),
                "total_gt": int(summary["instances"]),
                "total_correct": int(summary["correct"]),
                "total_wrong": int(summary["wrong"]),
                "total_miss": int(summary["miss"]),
                "total_accuracy": float(summary["instance_accuracy"]),
                "purpose": ab["purpose"],
            }
            for gname, ids, _tag in GROUPS:
                g = aggregate_group(class_rows, ids)
                key = gname
                total[f"{key}_gt"] = g["gt"]
                total[f"{key}_correct"] = g["correct"]
                total[f"{key}_wrong"] = g["wrong"]
                total[f"{key}_miss"] = g["miss"]
                total[f"{key}_accuracy"] = g["accuracy"]
            focused_ids = {4, 6, 8, 9, 10, 20}
            focused = aggregate_group(class_rows, focused_ids)
            head_ids = {0, 2, 3, 12, 13, 14, 15, 18}
            head = aggregate_group(class_rows, head_ids)
            total["focused_tail_gt"] = focused["gt"]
            total["focused_tail_correct"] = focused["correct"]
            total["focused_tail_wrong"] = focused["wrong"]
            total["focused_tail_miss"] = focused["miss"]
            total["focused_tail_accuracy"] = focused["accuracy"]
            total["head_gt"] = head["gt"]
            total["head_correct"] = head["correct"]
            total["head_wrong"] = head["wrong"]
            total["head_miss"] = head["miss"]
            total["head_accuracy"] = head["accuracy"]
            ab_candidates.append(total)
            threshold_rows.append(total)

        best = choose_best(ab_candidates)
        best_method = f"{ab['id']}_th{best['threshold']:.2f}"
        mask = scores >= float(best["threshold"])
        selected = as_records(val[mask].copy())
        if ab["kind"] == "model_plus_gray_override":
            selected = apply_gray_overrides(selected, gray_overrides)
        summary, class_rows = evaluate(parent, selected, best_method)
        selected_predictions[ab["id"]] = selected

        summary_row = dict(best)
        summary_row["best_method"] = best_method
        all_summary_rows.append(summary_row)
        for row in class_rows:
            all_class_rows.append({"ablation_id": ab["id"], "ablation_name": ab["name"], **row})
        write_csv(OUT / f"{ab['id']}_selected_predictions.csv", selected)

    write_csv(OUT / "ablation_threshold_grid.csv", threshold_rows)
    write_csv(OUT / "ablation_summary.csv", all_summary_rows)
    write_csv(OUT / "ablation_class_rows_raw.csv", all_class_rows)

    # Human-readable grouped table.
    lines = [
        "# IBO 证据模块消融实验 v1",
        "",
        "## Material Passport",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        "| 日期 | 2026-09-12 |",
        "| 数据 | 当前原始验证集 evidence；不含从训练集临时抽出的长焊高 holdout |",
        "| 评价 | 实例级：正确 / 错类 / 漏检；GT 覆盖率阈值 20%；IoU 错类辅助阈值 0.50 |",
        "| 合并 | 烟类 0/2/3；缺焊长塌类 12/14/15/18；排除 11/17/19/21/22 |",
        "",
        "## 1. 总体与关键组结果",
        "",
        "| 编号 | 方法 | 阈值 | 总体 | 总准确率 | 头类 | 重点尾类 | 焊灰色 | 缺焊长塌合并 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_summary_rows:
        total_s = f"{row['total_correct']} / {row['total_wrong']} / {row['total_miss']}"
        head_s = f"{row['head_correct']} / {row['head_wrong']} / {row['head_miss']}<br>{pct(row['head_accuracy'])}"
        tail_s = f"{row['focused_tail_correct']} / {row['focused_tail_wrong']} / {row['focused_tail_miss']}<br>{pct(row['focused_tail_accuracy'])}"
        gray_s = f"{row['焊灰色_correct']} / {row['焊灰色_wrong']} / {row['焊灰色_miss']}<br>{pct(row['焊灰色_accuracy'])}"
        collapse_s = f"{row['缺焊长塌合并_correct']} / {row['缺焊长塌合并_wrong']} / {row['缺焊长塌合并_miss']}<br>{pct(row['缺焊长塌合并_accuracy'])}"
        lines.append(
            f"| {row['ablation_id']} | {row['ablation_name']} | {row['threshold']:.2f} | {total_s} | {pct(row['total_accuracy'])} | {head_s} | {tail_s} | {gray_s} | {collapse_s} |"
        )

    lines.extend(
        [
            "",
            "## 2. 各缺陷类别详细表",
            "",
            "| 类别 | 标注实例 | A0 D-FINE | A1 +IBO | A2 +正常参照 | A3 +频率 | A4 主体方法 | A5 +混淆纠错 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_ab = {row["ablation_id"]: row for row in all_summary_rows}
    for gname, ids, tag in GROUPS:
        gt = by_ab["A0"].get(f"{gname}_gt", 0)
        cells = []
        for aid in ["A0", "A1", "A2", "A3", "A4", "A5"]:
            row = by_ab[aid]
            cells.append(
                f"{row[f'{gname}_correct']} / {row[f'{gname}_wrong']} / {row[f'{gname}_miss']}<br>{pct(row[f'{gname}_accuracy'])}"
            )
        lines.append(f"| {gname} | {gt} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 3. 初步结论",
            "",
            "- 如果 A1 高于 A0，说明候选框聚合/IBO 组证据有效。",
            "- 如果 A2 高于 A1，说明正常参照距离有效。",
            "- 如果 A3 高于 A1，说明频率证据有效。",
            "- 如果 A4 高于 A2/A3，说明空间、正常参照与频率证据存在互补。",
            "- 如果 A5 提升焊灰色但伤害缺焊长塌合并类，说明尾类纠错有效但需要控制误伤。",
            "",
            "## 4. 输出文件",
            "",
            "- `ablation_summary.csv`：每个消融最佳阈值及核心指标。",
            "- `ablation_threshold_grid.csv`：所有阈值扫描结果。",
            "- `ablation_class_rows_raw.csv`：原始类别级结果。",
            "- `A0_selected_predictions.csv` 至 `A5_selected_predictions.csv`：每个消融最佳阈值下保留的预测候选。",
        ]
    )
    report = OUT / "IBO证据模块消融实验报告_v1.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "report": str(report), "summary": all_summary_rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
