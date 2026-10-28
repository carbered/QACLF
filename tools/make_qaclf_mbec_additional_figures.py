#!/usr/bin/env python3
"""Create the MBEC graphical abstract and OOF risk-stratification figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures" / "qaclf_20260821"
RESULT_DIR = ROOT / "result" / "qaclf-km-risk-stratification-20260828"
CLINICAL_ADJUSTMENT_PATH = (
    ROOT
    / "result"
    / "qaclf-clinical-adjustment-20260828"
    / "clinical_adjustment_summary.csv"
)
PREDICTION_FILES = {
    "BLCA": ROOT / "result" / "blca-frozen-pathomic-fusion-outer5-20260820" / "patient_predictions.csv",
    "BRCA": ROOT / "result" / "brca-frozen-pathomic-fusion-outer5-20260820" / "patient_predictions.csv",
    "GBMLGG": ROOT / "result" / "gbmlgg-frozen-pathomic-fusion-outer5-20260820" / "patient_predictions.csv",
    "LUAD": ROOT
    / "result"
    / "luad-frozen-pathomic-fusion-cloud-baseline-outer5-20260820"
    / "patient_predictions.csv",
    "UCEC": ROOT / "result" / "ucec-frozen-pathomic-fusion-outer5-20260820" / "patient_predictions.csv",
}
COLORS = {
    "ink": "#202124",
    "muted": "#62666D",
    "line": "#C8CCD2",
    "high": "#D55E00",
    "low": "#0072B2",
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


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, stem, png_dpi=450, pad=0.05, tight=True):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = FIGURE_DIR / f"{stem}.{extension}"
        kwargs = {}
        if tight:
            kwargs.update({"bbox_inches": "tight", "pad_inches": pad})
        if extension == "png":
            kwargs["dpi"] = png_dpi
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def kaplan_meier(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    event_times = np.unique(time[event])
    x = [0.0]
    y = [1.0]
    survival = 1.0
    event_survival = []
    for current_time in event_times:
        at_risk = np.sum(time >= current_time)
        deaths = np.sum((time == current_time) & event)
        if at_risk <= 0:
            continue
        survival *= 1.0 - deaths / at_risk
        x.append(float(current_time))
        y.append(float(survival))
        event_survival.append(float(survival))
    return np.asarray(x), np.asarray(y), event_times, np.asarray(event_survival)


def survival_at_times(query_times, event_times, event_survival):
    query_times = np.asarray(query_times, dtype=float)
    indices = np.searchsorted(event_times, query_times, side="right") - 1
    output = np.ones_like(query_times, dtype=float)
    valid = indices >= 0
    output[valid] = event_survival[indices[valid]]
    return output


def logrank_test(time, event, high_risk):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    high_risk = np.asarray(high_risk, dtype=bool)
    observed_minus_expected = 0.0
    variance = 0.0
    for current_time in np.unique(time[event]):
        at_risk = time >= current_time
        deaths = (time == current_time) & event
        n_high = float(np.sum(at_risk & high_risk))
        n_low = float(np.sum(at_risk & ~high_risk))
        d_high = float(np.sum(deaths & high_risk))
        d_total = float(np.sum(deaths))
        n_total = n_high + n_low
        if n_total <= 1.0 or d_total <= 0.0:
            continue
        expected_high = d_total * n_high / n_total
        current_variance = (
            n_high
            * n_low
            * d_total
            * (n_total - d_total)
            / (n_total * n_total * (n_total - 1.0))
        )
        observed_minus_expected += d_high - expected_high
        variance += current_variance
    statistic = observed_minus_expected**2 / variance if variance > 0.0 else np.nan
    p_value = float(chi2.sf(statistic, 1)) if np.isfinite(statistic) else np.nan
    return float(statistic), p_value


def cox_binary_hr(time, event, high_risk):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    x = np.asarray(high_risk, dtype=float)
    beta = 0.0
    information = np.nan
    for _ in range(50):
        score = 0.0
        information = 0.0
        for current_time in np.unique(time[event]):
            deaths = (time == current_time) & event
            at_risk = time >= current_time
            death_count = int(np.sum(deaths))
            weights = np.exp(np.clip(beta * x[at_risk], -40.0, 40.0))
            denominator = float(np.sum(weights))
            if death_count == 0 or denominator <= 0.0:
                continue
            mean_x = float(np.sum(weights * x[at_risk]) / denominator)
            mean_x2 = float(np.sum(weights * x[at_risk] ** 2) / denominator)
            score += float(np.sum(x[deaths])) - death_count * mean_x
            information += death_count * max(mean_x2 - mean_x**2, 0.0)
        if information <= 1e-12:
            return np.nan, np.nan, np.nan
        step = float(np.clip(score / information, -1.0, 1.0))
        beta += step
        if abs(step) < 1e-9:
            break
    standard_error = np.sqrt(1.0 / information)
    hazard_ratio = float(np.exp(beta))
    lower = float(np.exp(beta - 1.96 * standard_error))
    upper = float(np.exp(beta + 1.96 * standard_error))
    return hazard_ratio, lower, upper


def format_p_statement(value):
    if value < 0.001:
        return "log-rank p<0.001"
    return f"log-rank p={value:.3f}"


def plot_km_group(ax, frame, mask, label, color):
    group = frame.loc[mask]
    x, y, event_times, event_survival = kaplan_meier(group["event_time"], group["event"])
    ax.step(x, y, where="post", color=color, label=f"{label} (n={len(group)})")
    censor_times = group.loc[group["event"] < 0.5, "event_time"].to_numpy(dtype=float)
    if len(censor_times) > 45:
        censor_times = np.sort(censor_times)[np.linspace(0, len(censor_times) - 1, 45).astype(int)]
    if len(censor_times):
        censor_survival = survival_at_times(censor_times, event_times, event_survival)
        ax.plot(
            censor_times,
            censor_survival,
            linestyle="none",
            marker="|",
            markersize=4.2,
            markeredgewidth=0.65,
            color=color,
            alpha=0.75,
        )


def make_risk_stratification_figure():
    rows = []
    frames = {}
    for cohort, path in PREDICTION_FILES.items():
        frame = pd.read_csv(path)
        required = {"patient_id", "event_time", "event", "fused_predicted_time"}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.dropna(subset=list(required)).copy()
        frame["high_risk"] = frame["fused_predicted_time"] < 0.5
        statistic, p_value = logrank_test(frame["event_time"], frame["event"], frame["high_risk"])
        hr, lower, upper = cox_binary_hr(frame["event_time"], frame["event"], frame["high_risk"])
        rows.append(
            {
                "cohort": cohort,
                "n": len(frame),
                "events": int(frame["event"].sum()),
                "high_risk_n": int(frame["high_risk"].sum()),
                "low_risk_n": int((~frame["high_risk"]).sum()),
                "high_risk_events": int(frame.loc[frame["high_risk"], "event"].sum()),
                "low_risk_events": int(frame.loc[~frame["high_risk"], "event"].sum()),
                "logrank_chi2": statistic,
                "logrank_p": p_value,
                "hazard_ratio_high_vs_low": hr,
                "hr_ci025": lower,
                "hr_ci975": upper,
                "fixed_quantile_threshold": 0.5,
            }
        )
        frames[cohort] = frame

    stats = pd.DataFrame(rows)
    if not CLINICAL_ADJUSTMENT_PATH.is_file():
        raise FileNotFoundError(
            f"Run tools/evaluate_qaclf_clinical_adjustment.py first: "
            f"{CLINICAL_ADJUSTMENT_PATH}"
        )
    clinical_adjustment = pd.read_csv(CLINICAL_ADJUSTMENT_PATH)
    adjusted_columns = [
        "cohort",
        "n_complete",
        "events",
        "adjustment_covariates",
        "adjusted_hr",
        "adjusted_ci025",
        "adjusted_ci975",
        "adjusted_wald_p",
        "risk_likelihood_ratio_p",
    ]
    clinical_adjustment = clinical_adjustment[adjusted_columns].rename(
        columns={"events": "adjusted_events"}
    )
    stats = stats.merge(clinical_adjustment, on="cohort", validate="one_to_one")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = RESULT_DIR / "risk_stratification_stats.csv"
    stats.to_csv(stats_path, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.85))
    axes = axes.ravel()
    for panel_index, cohort in enumerate(PREDICTION_FILES):
        ax = axes[panel_index]
        frame = frames[cohort]
        row = stats.loc[stats["cohort"] == cohort].iloc[0]
        plot_km_group(ax, frame, frame["high_risk"], "High risk", COLORS["high"])
        plot_km_group(ax, frame, ~frame["high_risk"], "Low risk", COLORS["low"])
        ax.set_title(f"{cohort}  (n={row['n']:.0f}, events={row['events']:.0f})")
        ax.text(
            0.04,
            0.08,
            f"{format_p_statement(row['logrank_p'])}\n"
            f"HR={row['hazard_ratio_high_vs_low']:.2f} "
            f"[{row['hr_ci025']:.2f}, {row['hr_ci975']:.2f}]",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color=COLORS["ink"],
        )
        ax.set_ylim(0.0, 1.03)
        ax.set_xlim(left=0.0)
        ax.grid(color="#DADCE0", linewidth=0.5, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel_index in (0, 3):
            ax.set_ylabel("Survival probability")
        if panel_index >= 3:
            ax.set_xlabel("Time (months)")
        ax.legend(frameon=False, loc="upper right", handlelength=1.6)
        ax.text(
            -0.16,
            1.04,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
        )

    ax = axes[-1]
    y = np.arange(len(stats), dtype=float)
    hr = stats["hazard_ratio_high_vs_low"].to_numpy(dtype=float)
    lower = stats["hr_ci025"].to_numpy(dtype=float)
    upper = stats["hr_ci975"].to_numpy(dtype=float)
    ax.errorbar(
        hr,
        y - 0.12,
        xerr=np.vstack([hr - lower, upper - hr]),
        fmt="o",
        markersize=4.6,
        color=COLORS["high"],
        ecolor=COLORS["high"],
        capsize=2.2,
        linewidth=1.0,
        label="Unadjusted",
    )
    adjusted_hr = stats["adjusted_hr"].to_numpy(dtype=float)
    adjusted_lower = stats["adjusted_ci025"].to_numpy(dtype=float)
    adjusted_upper = stats["adjusted_ci975"].to_numpy(dtype=float)
    ax.errorbar(
        adjusted_hr,
        y + 0.12,
        xerr=np.vstack(
            [adjusted_hr - adjusted_lower, adjusted_upper - adjusted_hr]
        ),
        fmt="s",
        markersize=4.0,
        color=COLORS["output_edge"],
        ecolor=COLORS["output_edge"],
        capsize=2.2,
        linewidth=1.0,
        label="Clinically adjusted",
    )
    ax.axvline(1.0, color=COLORS["muted"], linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(stats["cohort"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.9, 8.5)
    ax.set_xticks([1.0, 2.0, 4.0, 8.0])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlabel("Hazard ratio: high vs low risk")
    ax.set_title("Fixed-threshold risk separation")
    ax.text(
        0.98,
        0.98,
        "Unadjusted",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=COLORS["high"],
    )
    ax.text(
        0.98,
        0.91,
        "Clinically adjusted",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=COLORS["output_edge"],
    )
    ax.grid(axis="x", color="#DADCE0", linewidth=0.5, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.16, 1.04, "F", transform=ax.transAxes, fontsize=9.5, fontweight="bold")

    fig.tight_layout(w_pad=1.3, h_pad=1.2)
    paths = save_figure(fig, "qaclf_oof_km_risk_stratification")
    return paths, stats_path, stats


def rounded_box(ax, x, y, width, height, face, edge, radius=0.025, linewidth=0.85):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=COLORS["muted"]):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=6.5,
            linewidth=0.8,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def make_graphical_abstract():
    target_dpi = 600
    width_in = round(32.93 / 25.4 * target_dpi) / target_dpi
    height_in = round(37.63 / 25.4 * target_dpi) / target_dpi
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(0.5, 0.985, "QACLF", ha="center", va="top", fontsize=7.0, fontweight="bold", color=COLORS["ink"])
    ax.text(0.5, 0.915, "Frozen calibrated pathomic fusion", ha="center", va="top", fontsize=3.4, color=COLORS["muted"])

    rounded_box(ax, 0.04, 0.69, 0.40, 0.16, COLORS["path"], COLORS["path_edge"])
    for index, shade in enumerate(("#D9A9C5", "#B779A5", "#F0C6D8")):
        ax.add_patch(Rectangle((0.07 + 0.055 * index, 0.775 - 0.018 * index), 0.12, 0.045, facecolor=shade, edgecolor="white", linewidth=0.35))
    ax.text(0.24, 0.730, "WSI\nFrozen VLSA", ha="center", va="center", fontsize=4.0, fontweight="bold", color=COLORS["path_edge"])

    rounded_box(ax, 0.56, 0.69, 0.40, 0.16, COLORS["omics"], COLORS["omics_edge"])
    for index, height in enumerate((0.035, 0.075, 0.050, 0.095, 0.060)):
        ax.add_patch(Rectangle((0.60 + index * 0.052, 0.775), 0.028, height * 0.72, facecolor=COLORS["omics_edge"], edgecolor="none", alpha=0.85))
    ax.text(0.76, 0.730, "RNA/CNV\nPCA + L2-Cox", ha="center", va="center", fontsize=4.0, fontweight="bold", color=COLORS["omics_edge"])

    arrow(ax, (0.24, 0.68), (0.42, 0.600), COLORS["path_edge"])
    arrow(ax, (0.76, 0.68), (0.58, 0.600), COLORS["omics_edge"])
    rounded_box(ax, 0.18, 0.49, 0.64, 0.12, COLORS["cal"], COLORS["cal_edge"])
    ax.text(0.50, 0.560, "Training-fold ECDF", ha="center", va="center", fontsize=4.6, fontweight="bold", color=COLORS["cal_edge"])
    ax.text(0.50, 0.515, "monotone scale alignment", ha="center", va="center", fontsize=3.4, color=COLORS["ink"])

    arrow(ax, (0.50, 0.485), (0.50, 0.425), COLORS["fusion_edge"])
    rounded_box(ax, 0.18, 0.315, 0.64, 0.12, COLORS["fusion"], COLORS["fusion_edge"])
    ax.text(0.50, 0.390, "Fixed fusion", ha="center", va="center", fontsize=4.3, fontweight="bold", color=COLORS["fusion_edge"])
    ax.text(0.50, 0.345, "0.80 pathology + 0.20 omics", ha="center", va="center", fontsize=3.5, color=COLORS["ink"])

    arrow(ax, (0.50, 0.310), (0.50, 0.255), COLORS["output_edge"])
    rounded_box(ax, 0.10, 0.070, 0.80, 0.19, COLORS["output"], COLORS["output_edge"])
    ax.text(0.50, 0.215, "Calibrated survival risk", ha="center", va="center", fontsize=4.8, fontweight="bold", color=COLORS["output_edge"])
    ax.text(0.50, 0.158, "5 TCGA cohorts", ha="center", va="center", fontsize=3.6, color=COLORS["ink"])
    ax.text(0.50, 0.112, "C-index 0.705  |  +0.009", ha="center", va="center", fontsize=3.8, fontweight="bold", color=COLORS["fusion_edge"])

    paths = save_figure(
        fig,
        "qaclf_graphical_abstract_mbec",
        png_dpi=target_dpi,
        pad=0.0,
        tight=False,
    )
    return paths


def update_manifest(km_paths, stats_path, graphical_abstract_paths):
    manifest_path = FIGURE_DIR / "figure_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest["risk_stratification_source"] = str(stats_path)
    manifest["clinical_adjustment_source"] = str(CLINICAL_ADJUSTMENT_PATH)
    manifest["outputs"]["oof_km_risk_stratification"] = km_paths
    manifest["outputs"]["graphical_abstract_mbec"] = graphical_abstract_paths
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main():
    setup_style()
    km_paths, stats_path, stats = make_risk_stratification_figure()
    graphical_abstract_paths = make_graphical_abstract()
    update_manifest(km_paths, stats_path, graphical_abstract_paths)
    print("===== QACLF OOF risk stratification =====")
    print(stats.to_string(index=False))
    print(f"stats={stats_path}")
    print("km_outputs=" + json.dumps(km_paths))
    print("graphical_abstract_outputs=" + json.dumps(graphical_abstract_paths))


if __name__ == "__main__":
    main()
