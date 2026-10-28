#!/usr/bin/env python3
"""Create publication-ready QACLF result, ablation, and robustness figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "qaclf_20260821"
COLORS = {
    "paper": "#202124",
    "matched": "#8A8F98",
    "mcat": "#0072B2",
    "cat": "#009E73",
    "qaclf": "#D55E00",
    "omics": "#CC79A7",
    "raw": "#56B4E9",
    "equal": "#E69F00",
}
DATASET_COLORS = {
    "BLCA": "#0072B2",
    "BRCA": "#7A7F87",
    "GBMLGG": "#009E73",
    "LUAD": "#E69F00",
    "UCEC": "#D55E00",
    "MACRO": "#202124",
}
COHORTS = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]
DATASET_MARKERS = {
    "BLCA": "o",
    "BRCA": "s",
    "GBMLGG": "^",
    "LUAD": "v",
    "UCEC": "P",
    "MACRO": "D",
}
DATASET_LINESTYLES = {
    "BLCA": "-",
    "BRCA": "--",
    "GBMLGG": "-.",
    "LUAD": ":",
    "UCEC": (0, (3, 1, 1, 1)),
    "MACRO": "-",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#DADCE0", linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label):
    ax.text(
        -0.09,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def save_all(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.06}
        if extension == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def make_main_figure():
    summary_path = (
        ROOT
        / "result"
        / "external_multimodal_comparison_20260821"
        / "unified"
        / "dataset_summary_unified.csv"
    )
    bootstrap_path = summary_path.parent / "paired_bootstrap_unified.csv"
    summary = pd.read_csv(summary_path).set_index("dataset")
    order = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC", "MACRO"]
    summary = summary.loc[order]
    methods = [
        ("paper_vlsa", "Published VLSA", COLORS["paper"], "x"),
        ("matched_vlsa", "Matched VLSA", COLORS["matched"], "o"),
        ("mcat_conch", "MCAT-CONCH", COLORS["mcat"], "s"),
        ("cat_tpl_conch", "CAT-TPL-CONCH", COLORS["cat"], "^"),
        ("qaclf", "QACLF", COLORS["qaclf"], "D"),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.35),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    ax = axes[0]
    x = np.arange(len(order), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(methods))
    for offset, (column, label, color, marker) in zip(offsets, methods):
        values = summary[column].to_numpy(dtype=float)
        scatter_kwargs = {
            "x": x + offset,
            "y": values,
            "label": label,
            "color": color,
            "marker": marker,
            "s": 34 if column != "qaclf" else 42,
            "linewidth": 0.9,
            "zorder": 3,
        }
        if marker != "x":
            scatter_kwargs["edgecolor"] = color
        ax.scatter(**scatter_kwargs)
    ax.axvline(4.5, color="#B8BABC", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Macro" if item == "MACRO" else item for item in order])
    ax.set_ylabel("C-index")
    ax.set_ylim(0.55, 0.84)
    ax.set_title("Unified outer 5-fold comparison")
    style_axis(ax, "y")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        columnspacing=0.9,
        handletextpad=0.4,
    )
    panel_label(ax, "A")

    ax = axes[1]
    boot = pd.read_csv(bootstrap_path)
    boot = boot[boot["comparison"] == "qaclf_vs_matched"].set_index("dataset").loc[order]
    y = np.arange(len(order), dtype=float)
    delta = boot["observed_delta"].to_numpy(dtype=float)
    lower = boot["delta_ci025"].to_numpy(dtype=float)
    upper = boot["delta_ci975"].to_numpy(dtype=float)
    significant = (lower > 0.0) | (upper < 0.0)
    for index, (yi, value, lo, hi, is_significant) in enumerate(
        zip(y, delta, lower, upper, significant)
    ):
        color = COLORS["qaclf"] if index < len(order) - 1 else COLORS["paper"]
        ax.errorbar(
            value,
            yi,
            xerr=np.array([[value - lo], [hi - value]]),
            fmt="o",
            markersize=5.4,
            markerfacecolor=color if is_significant else "white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.4,
            zorder=3,
        )
    ax.axvline(0.0, color="#4A4D50", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(["Macro" if item == "MACRO" else item for item in order])
    ax.invert_yaxis()
    ax.set_xlabel("C-index gain vs matched VLSA (95% CI)")
    ax.set_title("Paired bootstrap effect")
    ax.set_xlim(-0.025, 0.034)
    style_axis(ax, "x")
    panel_label(ax, "B")

    fig.tight_layout(w_pad=1.6)
    fig.subplots_adjust(top=0.73)
    return save_all(fig, "qaclf_main_comparison")


def make_ablation_figure():
    ablation_dir = ROOT / "result" / "qaclf-ablation-sensitivity-20260821"
    component = pd.read_csv(ablation_dir / "dataset_summary_ablation.csv")
    alpha = pd.read_csv(ablation_dir / "dataset_summary_alpha_sensitivity.csv")
    pca = pd.read_csv(ROOT / "result" / "qaclf-pca-sensitivity-20260821" / "dataset_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.55))
    ax = axes[0]
    component_order = [
        ("pathology_only", "Pathology", COLORS["matched"]),
        ("omics_only", "Omics", COLORS["omics"]),
        ("raw_fixed_fusion", "Raw fusion", COLORS["raw"]),
        ("ecdf_equal_fusion", "ECDF equal", COLORS["equal"]),
        ("qaclf_locked", "QACLF", COLORS["qaclf"]),
    ]
    macro = component[component["dataset"] == "MACRO"].set_index("method")
    cohort_component = component[component["dataset"].isin(COHORTS)]
    y = np.arange(len(component_order), dtype=float)
    values = np.array([macro.loc[key, "mean_c_index"] for key, _, _ in component_order])
    colors = [color for _, _, color in component_order]
    labels = [label for _, label, _ in component_order]
    for yi, (method, _, color) in zip(y, component_order):
        cohort_values = (
            cohort_component[cohort_component["method"] == method]
            .set_index("dataset")
            .loc[COHORTS, "mean_c_index"]
            .to_numpy(dtype=float)
        )
        jitter = np.linspace(-0.13, 0.13, len(COHORTS))
        ax.scatter(
            cohort_values,
            yi + jitter,
            color="#FFFFFF",
            edgecolor=color,
            marker="o",
            s=19,
            linewidth=0.8,
            zorder=2,
        )
        ax.plot(
            [cohort_values.min(), cohort_values.max()],
            [yi, yi],
            color=color,
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )
    ax.scatter(
        values,
        y,
        color=colors,
        marker="D",
        s=38,
        zorder=4,
        edgecolor="#202124",
        linewidth=0.45,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.54, 0.84)
    ax.set_xlabel("C-index")
    ax.set_title("Component ablation")
    style_axis(ax, "x")
    panel_label(ax, "A")

    ax = axes[1]
    macro_alpha = alpha[alpha["dataset"] == "MACRO"].sort_values("alpha")
    for dataset in COHORTS:
        cohort_alpha = alpha[alpha["dataset"] == dataset].sort_values("alpha")
        ax.plot(
            cohort_alpha["alpha"],
            cohort_alpha["mean_c_index"],
            color=DATASET_COLORS[dataset],
            marker=DATASET_MARKERS[dataset],
            linestyle=DATASET_LINESTYLES[dataset],
            markersize=2.8,
            linewidth=0.75,
            alpha=0.62,
        )
    ax.plot(
        macro_alpha["alpha"],
        macro_alpha["mean_c_index"],
        color=COLORS["paper"],
        marker="D",
        markersize=4.2,
        linewidth=1.8,
    )
    locked = macro_alpha[np.isclose(macro_alpha["alpha"], 0.20)].iloc[0]
    ax.scatter(
        [locked["alpha"]],
        [locked["mean_c_index"]],
        marker="D",
        s=52,
        color=COLORS["paper"],
        zorder=4,
    )
    ax.axvspan(0.10, 0.30, color="#E8F1F7", alpha=0.8, zorder=0)
    ax.set_xticks(macro_alpha["alpha"])
    ax.set_xlabel("Omics fusion weight")
    ax.set_ylabel("C-index")
    alpha_min = alpha[alpha["dataset"].isin(COHORTS)]["mean_c_index"].min()
    alpha_max = alpha[alpha["dataset"].isin(COHORTS)]["mean_c_index"].max()
    alpha_pad = max(0.01, 0.06 * (alpha_max - alpha_min))
    ax.set_ylim(alpha_min - alpha_pad, alpha_max + alpha_pad)
    ax.set_title("Fixed-weight sensitivity")
    style_axis(ax, "y")
    panel_label(ax, "B")

    ax = axes[2]
    macro_pca = pca[pca["dataset"] == "MACRO"].sort_values("requested_pca_components")
    positions = np.arange(len(macro_pca), dtype=float)
    for dataset in COHORTS:
        cohort_pca = pca[pca["dataset"] == dataset].sort_values("requested_pca_components")
        ax.plot(
            positions,
            cohort_pca["mean_fused_c_index"],
            color=DATASET_COLORS[dataset],
            marker=DATASET_MARKERS[dataset],
            linestyle=DATASET_LINESTYLES[dataset],
            markersize=2.8,
            linewidth=0.75,
            alpha=0.62,
        )
    ax.plot(
        positions,
        macro_pca["mean_fused_c_index"],
        color=COLORS["paper"],
        marker="D",
        markersize=4.2,
        linewidth=1.8,
    )
    locked_index = int(np.flatnonzero(macro_pca["requested_pca_components"].to_numpy() == 64)[0])
    ax.scatter(
        [locked_index],
        [macro_pca.iloc[locked_index]["mean_fused_c_index"]],
        marker="D",
        s=52,
        color=COLORS["paper"],
        zorder=4,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(macro_pca["requested_pca_components"].astype(int).astype(str))
    ax.set_xlabel("PCA components")
    ax.set_ylabel("C-index")
    pca_min = pca[pca["dataset"].isin(COHORTS)]["mean_fused_c_index"].min()
    pca_max = pca[pca["dataset"].isin(COHORTS)]["mean_fused_c_index"].max()
    pca_pad = max(0.01, 0.06 * (pca_max - pca_min))
    ax.set_ylim(pca_min - pca_pad, pca_max + pca_pad)
    ax.set_title("Representation dimension")
    style_axis(ax, "y")
    panel_label(ax, "C")

    fig.text(
        0.5,
        0.01,
        "Small symbols/lines show cohort means; black diamonds show the macro mean; enlarged diamonds mark locked settings.",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#4A4D50",
    )
    fig.tight_layout(w_pad=1.15, rect=(0.0, 0.045, 1.0, 1.0))
    return save_all(fig, "qaclf_ablation_sensitivity")


def make_robustness_figure():
    source = (
        ROOT
        / "result"
        / "qaclf-omics-robustness-20260821"
        / "dataset_summary.csv"
    )
    data = pd.read_csv(source)
    repeat_data = pd.read_csv(source.with_name("repeat_metrics.csv"))
    datasets = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC", "MACRO"]
    conditions = [
        ("feature_dropout", "Feature dropout", [0.0, 0.25, 0.50, 0.75], ["0", "25", "50", "75"]),
        ("gaussian_noise", "Gaussian noise", [0.0, 0.10, 0.25, 0.50], ["0", "0.10", "0.25", "0.50"]),
        ("patient_missing", "Patient-level missing", [0.0, 0.25, 0.50, 0.75, 1.0], ["0", "25", "50", "75", "100"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.55), sharey=True)
    for panel_index, (ax, (condition, title, severity, tick_labels)) in enumerate(
        zip(axes, conditions)
    ):
        positions = np.arange(len(severity), dtype=float)
        for dataset in datasets:
            target_datasets = COHORTS if dataset == "MACRO" else [dataset]
            clean_rows = repeat_data[
                (repeat_data["dataset"].isin(target_datasets))
                & (repeat_data["condition"] == "clean")
            ].copy()
            clean_delta = float(
                (clean_rows["perturbed_fused_c_index"] - clean_rows["pathology_c_index"]).mean()
            )
            means = [clean_delta]
            standard_deviations = [0.0]
            for value in severity[1:]:
                subset = repeat_data[
                    (repeat_data["dataset"].isin(target_datasets))
                    & (repeat_data["condition"] == condition)
                    & np.isclose(repeat_data["severity"], value)
                ].copy()
                subset["delta"] = (
                    subset["perturbed_fused_c_index"] - subset["pathology_c_index"]
                )
                repeat_means = subset.groupby("repeat", sort=True)["delta"].mean()
                means.append(float(repeat_means.mean()))
                standard_deviations.append(float(repeat_means.std(ddof=1)))
            values = np.asarray(means, dtype=float)
            sd = np.asarray(standard_deviations, dtype=float)
            is_macro = dataset == "MACRO"
            color = DATASET_COLORS[dataset]
            ax.fill_between(
                positions,
                values - sd,
                values + sd,
                color=color,
                alpha=0.16 if is_macro else 0.055,
                linewidth=0,
                zorder=3 if is_macro else 1,
            )
            ax.plot(
                positions,
                values,
                color=color,
                marker=DATASET_MARKERS[dataset],
                linestyle=DATASET_LINESTYLES[dataset],
                markersize=4.2 if is_macro else 3.2,
                linewidth=2.2 if is_macro else 1.0,
                alpha=1.0 if is_macro else 0.75,
                label="Macro" if is_macro else dataset,
                zorder=4 if is_macro else 2,
            )
        ax.axhline(0.0, color="#4A4D50", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        if condition == "gaussian_noise":
            ax.set_xlabel("Noise standard deviation")
        else:
            ax.set_xlabel("Missing rate (%)")
        ax.set_title(title)
        ax.set_ylim(-0.030, 0.0215)
        style_axis(ax, "y")
        panel_label(ax, chr(ord("A") + panel_index))
    axes[0].set_ylabel("C-index gain vs pathology")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.text(
        0.5,
        0.01,
        "Shading denotes +/-1 SD across 20 corruption repeats; the clean point is deterministic.",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#4A4D50",
    )
    fig.tight_layout(w_pad=1.2, rect=(0.0, 0.05, 1.0, 0.91))
    return save_all(fig, "qaclf_omics_robustness")


def make_curve_quality_figure(vlsa_label="Matched VLSA", audit_prefix=None):
    source = (
        ROOT
        / "result"
        / "frozen-pathomic-fusion-five-dataset-summary-20260820"
        / "curve_dataset_summary.csv"
    )
    data = pd.read_csv(source).set_index("dataset")
    datasets = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]
    data = data.loc[datasets]
    x = np.arange(len(datasets), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    panels = [
        (
            axes[0, 0],
            "baseline_ibs",
            "qaclf_ibs",
            "Integrated Brier score",
            "IBS (lower is better)",
            (0.08, 0.29),
        ),
        (
            axes[0, 1],
            "baseline_km_calibration",
            "qaclf_km_calibration",
            "Kaplan-Meier calibration",
            "Calibration error (lower is better)",
            (0.0, 0.066),
        ),
        (
            axes[1, 0],
            "baseline_mae",
            "qaclf_mae",
            "Absolute-time accuracy",
            "Hinge MAE in months (lower is better)",
            (15.0, 57.0),
        ),
    ]

    for panel_index, (ax, baseline_col, qaclf_col, title, ylabel, ylim) in enumerate(
        panels
    ):
        baseline = data[baseline_col].to_numpy(dtype=float)
        candidate = data[qaclf_col].to_numpy(dtype=float)
        for xi, base_value, candidate_value in zip(x, baseline, candidate):
            ax.plot(
                [xi - 0.12, xi + 0.12],
                [base_value, candidate_value],
                color="#B8BABC",
                linewidth=1.0,
                zorder=1,
            )
        ax.scatter(
            x - 0.12,
            baseline,
            color=COLORS["matched"],
            marker="o",
            s=34,
            label=vlsa_label,
            zorder=3,
        )
        ax.scatter(
            x + 0.12,
            candidate,
            color=COLORS["qaclf"],
            marker="D",
            s=38,
            label="QACLF",
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        style_axis(ax, "y")
        panel_label(ax, chr(ord("A") + panel_index))

    ax = axes[1, 1]
    width = 0.34
    baseline_pass = data["baseline_d_calibration_pass_folds"].to_numpy(dtype=float)
    candidate_pass = data["qaclf_d_calibration_pass_folds"].to_numpy(dtype=float)
    ax.bar(
        x - width / 2,
        baseline_pass,
        width=width,
        color=COLORS["matched"],
        label=vlsa_label,
    )
    ax.bar(
        x + width / 2,
        candidate_pass,
        width=width,
        color=COLORS["qaclf"],
        label="QACLF",
    )
    for xi, base_value, candidate_value in zip(x, baseline_pass, candidate_pass):
        ax.text(xi - width / 2, base_value + 0.09, f"{int(base_value)}", ha="center", fontsize=7.5)
        ax.text(xi + width / 2, candidate_value + 0.09, f"{int(candidate_value)}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("D-calibration not rejected (folds / 5)")
    ax.set_ylim(0.0, 5.7)
    ax.set_title("Distributional calibration")
    style_axis(ax, "y")
    panel_label(ax, "D")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.6,
    )
    fig.tight_layout(w_pad=2.0, h_pad=2.0, rect=(0.0, 0.0, 1.0, 0.96))
    if audit_prefix is not None:
        import sys
        sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, axes=list(axes.flat), panel_ids=["A", "B", "C", "D"],
            json_out=str(audit_prefix) + ".alignment.json",
            overlay_svg=str(audit_prefix) + ".alignment.svg", strict=True)
    return save_all(fig, "qaclf_survival_curve_quality")


def main():
    setup_style()
    outputs = {
        "main_comparison": make_main_figure(),
        "ablation_sensitivity": make_ablation_figure(),
        "survival_curve_quality": make_curve_quality_figure(),
        "omics_robustness": make_robustness_figure(),
    }
    method_outputs = [
        OUT / f"qaclf_method_overview.{extension}"
        for extension in ("png", "pdf", "svg")
    ]
    if all(path.exists() for path in method_outputs):
        outputs = {
            "method_overview": [str(path) for path in method_outputs],
            **outputs,
        }
    manifest = {
        "metric_contract": "SurvivalEVAL_mean_continuous_comparable_all_ties",
        "main_result_source": str(
            ROOT
            / "result"
            / "external_multimodal_comparison_20260821"
            / "unified"
        ),
        "ablation_source": str(
            ROOT / "result" / "qaclf-ablation-sensitivity-20260821"
        ),
        "pca_source": str(ROOT / "result" / "qaclf-pca-sensitivity-20260821"),
        "robustness_source": str(
            ROOT / "result" / "qaclf-omics-robustness-20260821"
        ),
        "survival_curve_source": str(
            ROOT
            / "result"
            / "frozen-pathomic-fusion-five-dataset-summary-20260820"
            / "curve_dataset_summary.csv"
        ),
        "method_implementation": str(
            ROOT / "tools" / "evaluate_frozen_pathomic_fusion_outer5.py"
        ),
        "outputs": outputs,
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
