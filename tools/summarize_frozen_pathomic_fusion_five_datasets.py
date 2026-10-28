#!/usr/bin/env python3
"""Summarize the locked five-cohort frozen pathomic fusion evaluation."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "result/frozen-pathomic-fusion-five-dataset-summary-20260820"
EXPERIMENTS = {
    "BLCA": (
        0.6176,
        "result/blca-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    ),
    "BRCA": (0.6652, "result/brca-frozen-pathomic-fusion-outer5-20260820"),
    "GBMLGG": (
        0.8002,
        "result/gbmlgg-frozen-pathomic-fusion-outer5-20260820",
    ),
    "LUAD": (
        0.6370,
        "result/luad-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    ),
    "UCEC": (0.7571, "result/ucec-frozen-pathomic-fusion-outer5-20260820"),
}
CURVE_EXPERIMENTS = {
    "BLCA": "result/blca-qaclf-survival-curves-20260820",
    "BRCA": "result/brca-qaclf-survival-curves-20260820",
    "GBMLGG": "result/gbmlgg-qaclf-survival-curves-20260820",
    "LUAD": "result/luad-qaclf-survival-curves-20260820",
    "UCEC": "result/ucec-qaclf-survival-curves-20260820",
}


def main():
    rows = []
    fold_frames = []
    for dataset, (paper, result_path) in EXPERIMENTS.items():
        result_dir = ROOT / result_path
        summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
        folds = pd.read_csv(result_dir / "fold_metrics.csv")
        folds.insert(0, "dataset", dataset)
        fold_frames.append(folds)
        rows.append({
            "dataset": dataset,
            "paper_reported": paper,
            "matched_baseline": summary["fold_mean_baseline_c_index"],
            "omics_only": summary["fold_mean_omics_c_index"],
            "fused": summary["fold_mean_fused_c_index"],
            "delta_vs_baseline": summary["fold_mean_delta"],
            "delta_vs_paper": summary["fold_mean_fused_c_index"] - paper,
            "positive_folds": summary["positive_folds"],
            "ci025": summary["delta_ci025"],
            "ci975": summary["delta_ci975"],
            "probability_delta_gt_zero": summary["probability_delta_gt_zero"],
            "selected_omics": summary["selected_omics"],
        })

    table = pd.DataFrame(rows)
    overall = {
        "method": "frozen_inductively_calibrated_pathomic_fusion",
        "alpha": 0.20,
        "pca_components": 64,
        "weight_selection": "fixed_before_outer_test_evaluation",
        "calibration_fit": "outer_training_empirical_cdf_only",
        "paper_macro_mean": float(table["paper_reported"].mean()),
        "matched_baseline_macro_mean": float(table["matched_baseline"].mean()),
        "omics_only_macro_mean": float(table["omics_only"].mean()),
        "fused_macro_mean": float(table["fused"].mean()),
        "macro_delta_vs_baseline": float(table["delta_vs_baseline"].mean()),
        "macro_delta_vs_paper": float(table["delta_vs_paper"].mean()),
        "positive_datasets": int((table["delta_vs_baseline"] > 0).sum()),
        "datasets_above_paper": int((table["delta_vs_paper"] > 0).sum()),
        "positive_folds": int(table["positive_folds"].sum()),
        "total_folds": 25,
        "datasets_with_ci_excluding_zero": int(
            ((table["ci025"] > 0) | (table["ci975"] < 0)).sum()
        ),
    }

    curve_rows = []
    for dataset, result_path in CURVE_EXPERIMENTS.items():
        summary = json.loads(
            (ROOT / result_path / "curve_summary.json").read_text(encoding="utf-8")
        )
        curve_rows.append({
            "dataset": dataset,
            "baseline_ibs": summary["baseline_ibs_mean"],
            "qaclf_ibs": summary["qaclf_ibs_mean"],
            "ibs_gain": summary["ibs_gain_mean"],
            "baseline_mae": summary["baseline_mae_mean"],
            "qaclf_mae": summary["qaclf_mae_mean"],
            "mae_gain": summary["mae_gain_mean"],
            "baseline_d_calibration_pass_folds": summary[
                "baseline_d_calibration_pass_folds"
            ],
            "qaclf_d_calibration_pass_folds": summary[
                "qaclf_d_calibration_pass_folds"
            ],
            "baseline_km_calibration": summary["baseline_km_calibration_mean"],
            "qaclf_km_calibration": summary["qaclf_km_calibration_mean"],
        })
    curve_table = pd.DataFrame(curve_rows)
    curve_overall = {
        "curve_projection": "positive_monotone_cox_scale_plus_breslow_outer_train_only",
        "baseline_ibs_macro_mean": float(curve_table["baseline_ibs"].mean()),
        "qaclf_ibs_macro_mean": float(curve_table["qaclf_ibs"].mean()),
        "ibs_gain_macro_mean": float(curve_table["ibs_gain"].mean()),
        "ibs_positive_datasets": int((curve_table["ibs_gain"] > 0).sum()),
        "baseline_mae_macro_mean": float(curve_table["baseline_mae"].mean()),
        "qaclf_mae_macro_mean": float(curve_table["qaclf_mae"].mean()),
        "mae_gain_macro_mean": float(curve_table["mae_gain"].mean()),
        "mae_positive_datasets": int((curve_table["mae_gain"] > 0).sum()),
        "baseline_d_calibration_pass_folds": int(
            curve_table["baseline_d_calibration_pass_folds"].sum()
        ),
        "qaclf_d_calibration_pass_folds": int(
            curve_table["qaclf_d_calibration_pass_folds"].sum()
        ),
        "baseline_km_calibration_macro_mean": float(
            curve_table["baseline_km_calibration"].mean()
        ),
        "qaclf_km_calibration_macro_mean": float(
            curve_table["qaclf_km_calibration"].mean()
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "fold_metrics.csv", index=False
    )
    (OUTPUT_DIR / "overall_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    curve_table.to_csv(OUTPUT_DIR / "curve_dataset_summary.csv", index=False)
    (OUTPUT_DIR / "curve_overall_summary.json").write_text(
        json.dumps(curve_overall, indent=2), encoding="utf-8"
    )
    print("===== Frozen pathomic fusion five-dataset summary =====")
    print(table.to_string(index=False))
    print("===== Overall summary =====")
    print(json.dumps(overall, indent=2))
    print("===== Survival-curve summary =====")
    print(curve_table.to_string(index=False))
    print(json.dumps(curve_overall, indent=2))
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
