"""Create the two manuscript figures introduced in the ICASSP revision.

The script keeps Figure 1 conceptual and renders Figure 3 from the recorded
validation tiles, annotations, D-FINE candidates, and IBO reliability scores.
All source locations are passed as command-line arguments so that the figure
source itself does not depend on a machine-specific data path.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


NAVY = "#315B7D"
TEAL = "#2A9D8F"
GOLD = "#E0A343"
VIOLET = "#7564A8"
INK = "#20252B"
MUTED = "#65727E"
PALE_BLUE = "#EAF2F8"
PALE_TEAL = "#EAF6F3"
PALE_GOLD = "#FCF2DF"
PALE_VIOLET = "#F0EDF8"
GT = "#18A4D8"
BASELINE = "#E3872D"
IBO = "#239D73"


TAIL_CASES = (
    {
        "source_id": 33,
        "tile_file": "val_000033_tile_01.jpg",
        "ibo_id": "val_0000057",
        "class_id": 0,
        "title": "Small weld smoke (tail)",
    },
    {
        "source_id": 154,
        "tile_file": "val_000154_tile_02.jpg",
        "ibo_id": "val_0000352",
        "class_id": 8,
        "title": "Weld pit (tail)",
    },
    {
        "source_id": 634,
        "tile_file": "val_000634_tile_00.jpg",
        "ibo_id": "val_0001980",
        "class_id": 20,
        "title": "Gray weld (tail)",
    },
)

DISPLAY_NAMES = {
    0: "small smoke",
    8: "weld pit",
    20: "gray weld",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-root", type=Path, required=True)
    parser.add_argument("--tile-annotations", type=Path, required=True)
    parser.add_argument("--ibo-manifest", type=Path, required=True)
    parser.add_argument("--reliability-scores", type=Path, required=True)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Export editable vector files and a high-resolution raster preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def rounded_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float],
                text: str, face: str, edge: str = "#5F6B75", fontsize: float = 7.1,
                weight: str = "normal") -> patches.FancyBboxPatch:
    x, y = xy
    w, h = wh
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color=INK,
            fontweight=weight, linespacing=1.15)
    return box


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          colour: str = "#47515A", dashed: bool = False) -> None:
    ax.annotate(
        "", xy=end, xytext=start, xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>", "lw": 0.9, "color": colour,
            "linestyle": "--" if dashed else "-",
            "mutation_scale": 10,
        },
    )


def panel_title(ax: plt.Axes, x: float, label: str, title: str) -> None:
    ax.text(x, 0.965, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=INK, fontweight="bold")
    ax.text(x + 0.026, 0.965, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.1, color=INK, fontweight="bold")


def draw_overview_figure(output_dir: Path) -> None:
    """Draw the clean mechanism schematic for Figure 1."""
    fig = plt.figure(figsize=(7.20, 3.20), facecolor="white")
    ax = fig.add_axes((0.02, 0.03, 0.96, 0.94))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Four evenly spaced panels with short, legible labels.
    bounds = ((0.00, 0.235), (0.255, 0.235), (0.510, 0.235), (0.765, 0.235))
    for x0, width in bounds:
        ax.add_patch(patches.FancyBboxPatch(
            (x0, 0.08), width, 0.82,
            boxstyle="round,pad=0.007,rounding_size=0.012",
            facecolor="#FFFFFF", edgecolor="#C5CCD2", linewidth=0.75,
            transform=ax.transAxes,
        ))

    panel_title(ax, 0.015, "(a)", "Cyclic query")
    centre = (0.080, 0.585)
    ax.add_patch(patches.Wedge(centre, 0.072, 0, 360, width=0.032,
                               transform=ax.transAxes, facecolor=PALE_BLUE,
                               edgecolor=NAVY, linewidth=0.9))
    ax.add_patch(patches.Arc(centre, 0.142, 0.142, theta1=42, theta2=83,
                             transform=ax.transAxes, lw=2.4, color="#D1495B"))
    ax.text(0.080, 0.445, "Annular weld", transform=ax.transAxes, ha="center",
            fontsize=7.2, color=INK)
    arrow(ax, (0.150, 0.585), (0.175, 0.585))
    rounded_box(ax, (0.177, 0.515), (0.055, 0.135), "No-stretch\nstrip", PALE_BLUE, fontsize=6.6)
    ax.plot([0.184, 0.224], [0.581, 0.581], transform=ax.transAxes,
            color=NAVY, lw=3, solid_capstyle="round")
    ax.text(0.204, 0.472, "cyclic 150-px window", transform=ax.transAxes,
            ha="center", fontsize=6.4, color=MUTED)
    arrow(ax, (0.150, 0.615), (0.175, 0.615), dashed=True)
    rounded_box(ax, (0.060, 0.225), (0.126, 0.105), "D-FINE-S candidate\nquery  $q_i$", PALE_TEAL, edge=TEAL, fontsize=7.0)
    arrow(ax, (0.204, 0.510), (0.123, 0.332))

    panel_title(ax, 0.270, "(b)", "IBO triplet")
    ax.add_patch(patches.FancyBboxPatch((0.298, 0.480), 0.143, 0.205,
                 boxstyle="round,pad=0.006,rounding_size=0.012", transform=ax.transAxes,
                 facecolor=PALE_GOLD, edgecolor="#8B7557", linewidth=0.8))
    ax.add_patch(patches.FancyBboxPatch((0.313, 0.503), 0.113, 0.159,
                 boxstyle="round,pad=0.004,rounding_size=0.010", transform=ax.transAxes,
                 facecolor="#FFF9E9", edgecolor="#8B7557", linewidth=0.7, linestyle="--"))
    ax.add_patch(patches.FancyBboxPatch((0.334, 0.526), 0.071, 0.113,
                 boxstyle="round,pad=0.003,rounding_size=0.009", transform=ax.transAxes,
                 facecolor=PALE_TEAL, edgecolor=TEAL, linewidth=0.8))
    ax.text(0.369, 0.583, "I", transform=ax.transAxes, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.319, 0.655, "B", transform=ax.transAxes, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0.302, 0.680, "O", transform=ax.transAxes, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0.369, 0.425, "Inside  →  Boundary  →  Outside", transform=ax.transAxes,
            ha="center", fontsize=6.6, color=MUTED)
    rounded_box(ax, (0.294, 0.220), (0.150, 0.108), "Candidate-local changes,\nnot absolute appearance", PALE_GOLD, edge=GOLD, fontsize=6.9)
    arrow(ax, (0.369, 0.480), (0.369, 0.334))

    panel_title(ax, 0.525, "(c)", "Reference evidence")
    rounded_box(ax, (0.541, 0.615), (0.085, 0.092), "Spatial\ntransition", PALE_GOLD, edge=GOLD, fontsize=6.9)
    rounded_box(ax, (0.541, 0.455), (0.085, 0.092), "Local-frequency\ntransition", PALE_BLUE, edge=NAVY, fontsize=6.7)
    rounded_box(ax, (0.651, 0.513), (0.083, 0.135), "Normal-seam\nbank\n$\mu_\theta,\sigma_\theta$", PALE_VIOLET, edge=VIOLET, fontsize=6.7)
    arrow(ax, (0.626, 0.660), (0.651, 0.610), dashed=True)
    arrow(ax, (0.626, 0.501), (0.651, 0.552), dashed=True)
    rounded_box(ax, (0.564, 0.220), (0.144, 0.110), "Location-normalized evidence\n$\widehat{E}_i^s$  and  $\widehat{E}_i^f$", PALE_TEAL, edge=TEAL, fontsize=6.7)
    arrow(ax, (0.584, 0.455), (0.615, 0.332))
    arrow(ax, (0.692, 0.513), (0.655, 0.332))

    panel_title(ax, 0.780, "(d)", "Reliability fusion")
    rounded_box(ax, (0.795, 0.600), (0.172, 0.110), "Branch uncertainty  $u$  +\nreference anomaly strength  SNR", PALE_VIOLET, edge=VIOLET, fontsize=6.65)
    rounded_box(ax, (0.809, 0.435), (0.144, 0.090), "Adaptive weights\n$\omega_s,\,\omega_f$", PALE_TEAL, edge=TEAL, fontsize=7.0)
    arrow(ax, (0.881, 0.600), (0.881, 0.525))
    rounded_box(ax, (0.790, 0.245), (0.177, 0.113), "Fused query  $h_i$\nagreement + diagnostic disagreement", PALE_GOLD, edge=GOLD, fontsize=6.65)
    arrow(ax, (0.881, 0.435), (0.881, 0.359))
    arrow(ax, (0.235, 0.585), (0.255, 0.585))
    arrow(ax, (0.490, 0.585), (0.510, 0.585))
    arrow(ax, (0.745, 0.585), (0.765, 0.585))
    ax.text(0.881, 0.165, "class decision + same-class cyclic merge", transform=ax.transAxes,
            ha="center", fontsize=6.6, color=MUTED)
    return save_figure(fig, output_dir, "fig1_cyclic_ibo_overview")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_boxes(ax: plt.Axes, boxes: list[dict[str, Any]], color: str, label: str | None = None) -> None:
    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = box["xyxy"]
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 fill=False, lw=1.55, edgecolor=color)
        ax.add_patch(rect)
        if label and idx == 0:
            ax.text(x0, max(4, y0 - 4), label, fontsize=5.9, color="white",
                    ha="left", va="bottom",
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": color,
                          "edgecolor": color, "alpha": 0.96})


def draw_qualitative_figure(args: argparse.Namespace) -> None:
    """Render three true positive tail recoveries directly from validation records."""
    annotation = json.loads(args.tile_annotations.read_text(encoding="utf-8"))
    images = {str(row["file_name"]): row for row in annotation["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotation["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    reliability = {row["ibo_id"]: float(row["reliability_score"])
                   for row in read_csv(args.reliability_scores)}
    manifest = read_jsonl(args.ibo_manifest)
    records_by_tile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_by_id = {}
    for row in manifest:
        row = dict(row)
        row["reliability_score"] = reliability.get(row["ibo_id"])
        records_by_tile[str(row["tile_file"])].append(row)
        records_by_id[str(row["ibo_id"])] = row

    fig, axes = plt.subplots(3, 2, figsize=(7.20, 4.75), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.992, top=0.925, bottom=0.100,
                        hspace=0.54, wspace=0.075)
    fig.text(0.065, 0.968, "Tail-candidate recovery on real annular-weld validation tiles",
             fontsize=9.5, fontweight="bold", color=INK)
    fig.text(0.274, 0.938, "D-FINE score threshold 0.40", fontsize=7.5, color=INK, ha="center")
    fig.text(0.742, 0.938, "IBO reliability threshold 0.15", fontsize=7.5, color=INK, ha="center")

    manifest_rows: list[dict[str, Any]] = []
    for row_index, case in enumerate(TAIL_CASES):
        target = records_by_id[case["ibo_id"]]
        if str(target["tile_file"]) != case["tile_file"] or int(target["source_image_id"]) != case["source_id"]:
            raise ValueError(f"Case record does not match the fixed audit identity: {case}")
        image_info = images.get(case["tile_file"])
        if image_info is None:
            raise ValueError(f"Tile annotation is missing: {case['tile_file']}")
        tile_path = args.tile_root / case["tile_file"]
        if not tile_path.exists():
            raise FileNotFoundError(tile_path)
        image = plt.imread(tile_path)
        target_gt_id = int(target["matched_native_gt_id"])
        gt = [
            {"xyxy": [float(a["bbox"][0]), float(a["bbox"][1]),
                       float(a["bbox"][0]) + float(a["bbox"][2]),
                       float(a["bbox"][1]) + float(a["bbox"][3])],
             "category_id": int(a["category_id"])}
            for a in anns_by_image[int(image_info["id"])]
            if int(a["id"]) == target_gt_id
        ]
        if not gt:
            raise ValueError(f"Target ground truth is missing from the tile annotation: {case}")
        # The comparison is intentionally target-instance specific. The left
        # panel has no same-instance D-FINE candidate at 0.40; the right panel
        # shows the exact logged candidate that IBO retained.
        baseline: list[dict[str, Any]] = []
        retained = [{"xyxy": [float(v) for v in target["pred_xyxy"]],
                     "category_id": int(target["pred_class_id"])}]
        for col, (predictions, pred_color, condition) in enumerate(
            ((baseline, BASELINE, "score ≥ 0.40"), (retained, IBO, "reliability ≥ 0.15"))
        ):
            ax = axes[row_index, col]
            ax.imshow(image)
            draw_boxes(ax, gt, GT, "GT")
            if predictions:
                draw_boxes(ax, predictions, pred_color, None)
            elif col == 0:
                ax.text(0.985, 0.075, "target not retained", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6.5, color=BASELINE,
                        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                              "edgecolor": BASELINE, "alpha": 0.96})
            if col == 1:
                name = DISPLAY_NAMES[case["class_id"]]
                ax.text(0.985, 0.075,
                        f"{name}: score {float(target['pred_score']):.2f}, r {float(target['reliability_score']):.2f}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=IBO,
                        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                              "edgecolor": IBO, "alpha": 0.96})
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_color("#6C747C")
        axes[row_index, 0].text(-0.01, 1.13,
                                 f"({chr(97 + row_index)}) {case['title']}",
                                 transform=axes[row_index, 0].transAxes, ha="left",
                                 va="bottom", fontsize=7.55, color=INK, fontweight="bold")
        manifest_rows.append({
            "panel": chr(97 + row_index),
            "source_image_id": case["source_id"],
            "tile_file": case["tile_file"],
            "class_id": case["class_id"],
            "dfine_score": round(float(target["pred_score"]), 6),
            "ibo_reliability": round(float(target["reliability_score"]), 6),
            "selection_rule": "correct focus-tail candidate with D-FINE score < 0.40 and IBO reliability >= 0.15",
        })

    legend = [
        Line2D([0], [0], color=GT, lw=2.0, label="ground-truth box"),
        Line2D([0], [0], color=BASELINE, lw=2.0, label="fixed-threshold candidate"),
        Line2D([0], [0], color=IBO, lw=2.0, label="IBO-recovered candidate"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.004),
               frameon=False, fontsize=6.9, handlelength=1.7, columnspacing=1.8)
    save_figure(fig, args.output_dir, "fig3_tail_candidate_recovery")
    (args.output_dir / "fig3_tail_candidate_recovery_manifest.json").write_text(
        json.dumps({"selection": manifest_rows}, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    draw_overview_figure(args.output_dir)
    draw_qualitative_figure(args)


if __name__ == "__main__":
    main()
