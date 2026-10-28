#!/usr/bin/env python3
"""Create a publication-ready overview of the locked QACLF method."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "qaclf_20260821"
COLORS = {
    "ink": "#202124",
    "muted": "#62666D",
    "line": "#C8CCD2",
    "train": "#F4F6F8",
    "path": "#E8F2FA",
    "path_edge": "#0072B2",
    "omics": "#E8F5EF",
    "omics_edge": "#009E73",
    "cal": "#FFF4DA",
    "cal_edge": "#B97800",
    "fusion": "#FCEBE4",
    "fusion_edge": "#D55E00",
    "output": "#F3EAF5",
    "output_edge": "#8C4A91",
}


def rounded_box(ax, x, y, w, h, face, edge, linewidth=1.25, radius=0.014):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def label_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    lines,
    face,
    edge,
    title_size=10.0,
    text_size=8.1,
):
    rounded_box(ax, x, y, w, h, face, edge)
    ax.text(
        x + 0.014,
        y + h - 0.030,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=edge,
        zorder=3,
    )
    line_y = y + h - 0.070
    for line in lines:
        ax.text(
            x + 0.014,
            line_y,
            line,
            ha="left",
            va="top",
            fontsize=text_size,
            color=COLORS["ink"],
            linespacing=1.22,
            zorder=3,
        )
        line_y -= 0.038 if "\n" not in line else 0.057


def arrow(ax, start, end, color=None, dashed=False, rad=0.0, width=1.35):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=width,
        color=color or COLORS["muted"],
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=1,
    )
    ax.add_patch(patch)


def chip(ax, x, y, w, text, face="white", edge=None, color=None):
    edge = edge or COLORS["line"]
    color = color or COLORS["ink"]
    rounded_box(ax, x, y, w, 0.045, face, edge, linewidth=0.9, radius=0.010)
    ax.text(
        x + w / 2,
        y + 0.0225,
        text,
        ha="center",
        va="center",
        fontsize=7.5,
        color=color,
        zorder=3,
    )


def make_figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(15.2, 8.0), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.965,
        "QACLF: Quantile-Aligned Conservative Late Fusion",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.035,
        0.925,
        "Leakage-safe late fusion of a frozen vision-language survival model and a compact molecular readout",
        ha="left",
        va="top",
        fontsize=10,
        color=COLORS["muted"],
    )

    # Outer-training-only fitting lane.
    rounded_box(ax, 0.035, 0.765, 0.93, 0.115, COLORS["train"], COLORS["line"])
    ax.text(
        0.052,
        0.855,
        "Outer-training fold only",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.052,
        0.818,
        "Fit or store",
        ha="left",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    chip(ax, 0.225, 0.793, 0.170, "matched VLSA checkpoint", COLORS["path"], COLORS["path_edge"])
    chip(ax, 0.408, 0.793, 0.205, "scaler + PCA64 + L2-Cox", COLORS["omics"], COLORS["omics_edge"])
    chip(ax, 0.626, 0.793, 0.155, "two empirical CDFs", COLORS["cal"], COLORS["cal_edge"])
    chip(ax, 0.794, 0.793, 0.150, "Cox scale + Breslow", COLORS["output"], COLORS["output_edge"])
    ax.plot([0.035, 0.965], [0.735, 0.735], color=COLORS["line"], linewidth=1.0, linestyle=(0, (5, 4)))
    ax.text(
        0.965,
        0.742,
        "Held-out labels never cross this boundary",
        ha="right",
        va="bottom",
        fontsize=7.7,
        color=COLORS["muted"],
    )

    # Inputs and two inference branches.
    label_box(
        ax,
        0.035,
        0.485,
        0.125,
        0.175,
        "Pathology",
        ["WSI", "Precomputed CONCH", "patch embeddings"],
        COLORS["path"],
        COLORS["path_edge"],
        title_size=10.2,
        text_size=8.0,
    )
    label_box(
        ax,
        0.035,
        0.245,
        0.125,
        0.175,
        "Molecular",
        ["RNA-seq + CNV", "Six curated", "signature groups"],
        COLORS["omics"],
        COLORS["omics_edge"],
        title_size=10.2,
        text_size=8.0,
    )

    label_box(
        ax,
        0.205,
        0.485,
        0.205,
        0.175,
        "Frozen pathology branch",
        ["VLSA / VLFAN + ordinal prompts", r"Survival curve $S_{p}(t)$", r"Mean predicted time $T_{p}$"],
        COLORS["path"],
        COLORS["path_edge"],
        title_size=10.0,
    )
    label_box(
        ax,
        0.205,
        0.245,
        0.205,
        0.175,
        "Compact omics branch",
        ["Signature filtering and scaling", r"PCA64 $\rightarrow$ L2-Cox", r"Time-oriented score $T_{o}=-r_{o}$"],
        COLORS["omics"],
        COLORS["omics_edge"],
        title_size=10.0,
    )
    arrow(ax, (0.160, 0.572), (0.205, 0.572), COLORS["path_edge"])
    arrow(ax, (0.160, 0.332), (0.205, 0.332), COLORS["omics_edge"])

    label_box(
        ax,
        0.470,
        0.365,
        0.175,
        0.205,
        "Inductive calibration",
        [r"$q_p=\widehat{F}_{p}^{train}(T_p)$", r"$q_o=\widehat{F}_{o}^{train}(T_o)$", "Monotone scale alignment", "No held-out labels"],
        COLORS["cal"],
        COLORS["cal_edge"],
        title_size=10.0,
    )
    arrow(ax, (0.410, 0.572), (0.470, 0.510), COLORS["path_edge"], rad=0.08)
    arrow(ax, (0.410, 0.332), (0.470, 0.420), COLORS["omics_edge"], rad=-0.08)

    label_box(
        ax,
        0.690,
        0.365,
        0.150,
        0.205,
        "Conservative fusion",
        [r"$q=0.80q_p+0.20q_o$", "Fixed across cohorts", "Missing omics:", r"$q \leftarrow q_p$"],
        COLORS["fusion"],
        COLORS["fusion_edge"],
        title_size=10.0,
    )
    arrow(ax, (0.645, 0.468), (0.690, 0.468), COLORS["cal_edge"])

    label_box(
        ax,
        0.875,
        0.365,
        0.090,
        0.205,
        "Output",
        ["Risk ranking", "and full", r"curve $S(t)$"],
        COLORS["output"],
        COLORS["output_edge"],
        title_size=9.5,
        text_size=7.5,
    )
    arrow(ax, (0.840, 0.468), (0.875, 0.468), COLORS["fusion_edge"])

    # Train-only dependencies are shown as dashed arrows.
    arrow(ax, (0.310, 0.793), (0.305, 0.660), COLORS["path_edge"], dashed=True, rad=-0.08, width=1.0)
    arrow(ax, (0.510, 0.793), (0.308, 0.420), COLORS["omics_edge"], dashed=True, rad=0.12, width=1.0)
    arrow(ax, (0.703, 0.793), (0.555, 0.570), COLORS["cal_edge"], dashed=True, rad=0.08, width=1.0)
    arrow(ax, (0.869, 0.793), (0.920, 0.570), COLORS["output_edge"], dashed=True, rad=-0.08, width=1.0)

    # Formula and protocol panel.
    rounded_box(ax, 0.205, 0.075, 0.650, 0.100, "white", COLORS["line"], linewidth=1.0)
    ax.text(
        0.225,
        0.145,
        "Why the calibration matters",
        ha="left",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.225,
        0.108,
        "Each empirical CDF maps a modality-specific time score to its training-fold percentile before fusion,",
        ha="left",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    ax.text(
        0.225,
        0.083,
        "preserving within-modality order while preventing arbitrary score scales from dominating the patient-level rank.",
        ha="left",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )

    chip(ax, 0.035, 0.075, 0.145, "no joint retraining", COLORS["train"], COLORS["line"])
    chip(ax, 0.875, 0.125, 0.090, "65 Cox params", COLORS["omics"], COLORS["omics_edge"])
    chip(ax, 0.875, 0.075, 0.090, "fixed protocol", COLORS["fusion"], COLORS["fusion_edge"])

    ax.text(
        0.035,
        0.025,
        "At inference, higher q indicates longer predicted survival. A positive monotone Cox-Breslow projection provides full curves without changing C-index.",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=COLORS["muted"],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"qaclf_method_overview.{extension}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.06}
        if extension == "png":
            kwargs["dpi"] = 450
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    manifest_path = OUT / "figure_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"outputs": {}}
    manifest["method_implementation"] = str(
        ROOT / "tools" / "evaluate_frozen_pathomic_fusion_outer5.py"
    )
    manifest.setdefault("outputs", {})["method_overview"] = [
        str(path) for path in outputs
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    make_figure()
