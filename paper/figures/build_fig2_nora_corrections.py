"""Render Fig. 2 from two real A5 gray-weld correction cases.
The source cases are retained in the completed industrial A0--A5 experiment.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT = Path(r"D:\1\项目论文")
RUN = PROJECT / "zhwk_runs" / "normal_reference_no_defect_20260912" / "21_ibo_evidence_ablation_v1"
TILES = PROJECT / "zhwk_unwrapped_tiles_v1"
OUT = Path(__file__).resolve().parent / "fig2_real_nora_corrections"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CASES = ["merge_0000177", "merge_0000186"]
COLORS = {"gt": "#2A9D55", "base": "#D97706", "nora": "#176B87"}


def load_case_rows() -> dict[str, dict[str, str]]:
    with (RUN / "A5_selected_predictions.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = {r["merged_id"]: r for r in csv.DictReader(fh)}
    return {key: rows[key] for key in CASES}


def rebuild_strip(source_id: int) -> tuple[Image.Image, int]:
    entries = json.loads((TILES / "metadata" / "tiles_val.json").read_text(encoding="utf-8"))
    records = [r for r in entries if int(r["source_image_id"]) == source_id]
    if not records:
        raise RuntimeError(f"No tiles found for source image {source_id}")
    width = int(records[0]["strip_width"])
    height = int(records[0]["strip_height"])
    canvas = Image.new("RGB", (width, height), "black")
    for rec in records:
        tile = Image.open(TILES / "images" / "val" / rec["tile_file"]).convert("RGB")
        y0 = int(rec["y_pad"])
        band = tile.crop((0, y0, int(rec["tile_width"]), y0 + height))
        x0 = int(rec["tile_start"])
        canvas.paste(band, (x0, 0))
    return canvas, width


def crop_for_case(strip: Image.Image, box: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[float, float, float, float]]:
    x0, y0, x1, y1 = box
    margin = 55
    left = max(0, int(x0) - margin)
    right = min(strip.width, int(x1) + margin)
    if right - left < 720:
        centre = (x0 + x1) / 2
        left = max(0, int(centre - 360))
        right = min(strip.width, left + 720)
        left = max(0, right - 720)
    crop = strip.crop((left, 0, right, strip.height))
    return crop, (x0 - left, y0, x1 - left, y1)


def panel(ax, image: Image.Image, box: tuple[float, float, float, float], color: str, title: str) -> None:
    ax.imshow(image)
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=1.5, edgecolor=color))
    ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.45)


def main() -> None:
    rows = load_case_rows()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 2.05), constrained_layout=True)
    labels = [
        ("GT: gray weld", COLORS["gt"]),
        ("D-FINE: collapse family", COLORS["base"]),
        ("NORA: gray weld", COLORS["nora"]),
    ]
    for row_index, merged_id in enumerate(CASES):
        row = rows[merged_id]
        strip, _ = rebuild_strip(int(row["source_image_id"]))
        box = tuple(float(v) for v in ast.literal_eval(row["global_box"]))
        crop, crop_box = crop_for_case(strip, box)
        for column, (label, color) in enumerate(labels):
            panel(axes[row_index, column], crop, crop_box, color, label)
        axes[row_index, 0].text(
            -0.08, 0.5, f"Case {row_index + 1}", transform=axes[row_index, 0].transAxes,
            va="center", ha="right", fontsize=7.4, fontweight="bold", rotation=90,
        )
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()