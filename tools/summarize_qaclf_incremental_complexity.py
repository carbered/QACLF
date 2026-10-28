#!/usr/bin/env python3
"""Summarize the incremental CPU and model-state cost of frozen QACLF."""

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robustness-dir", default="result/qaclf-omics-robustness-20260821"
    )
    parser.add_argument(
        "--output-dir", default="result/qaclf-incremental-complexity-20260821"
    )
    args = parser.parse_args()

    source_dir = resolve(args.robustness_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold = pd.read_csv(source_dir / "fold_complexity.csv")
    summary = (
        fold.groupby("dataset", sort=False)
        .agg(
            folds=("fold", "count"),
            mean_train_n=("train_n", "mean"),
            mean_test_n=("test_n", "mean"),
            selected_omics=("selected_omics", "first"),
            pca_components=("pca_components", "first"),
            cox_parameters=("cox_parameters", "first"),
            deep_trainable_parameters_added=(
                "deep_trainable_parameters_added",
                "first",
            ),
            mean_state_mb_float64=("minimal_state_mb_float64", "mean"),
            total_fit_seconds=("fit_seconds", "sum"),
            mean_fit_seconds_per_fold=("fit_seconds", "mean"),
            mean_inference_ms_per_patient=("inference_ms_per_patient", "mean"),
            mean_pca_explained_variance=("pca_explained_variance", "mean"),
        )
        .reset_index()
    )
    macro = pd.DataFrame(
        [
            {
                "dataset": "MACRO",
                "folds": int(summary["folds"].sum()),
                "mean_train_n": summary["mean_train_n"].mean(),
                "mean_test_n": summary["mean_test_n"].mean(),
                "selected_omics": summary["selected_omics"].mean(),
                "pca_components": summary["pca_components"].mean(),
                "cox_parameters": summary["cox_parameters"].mean(),
                "deep_trainable_parameters_added": 0,
                "mean_state_mb_float64": summary["mean_state_mb_float64"].mean(),
                "total_fit_seconds": summary["total_fit_seconds"].sum(),
                "mean_fit_seconds_per_fold": summary[
                    "mean_fit_seconds_per_fold"
                ].mean(),
                "mean_inference_ms_per_patient": summary[
                    "mean_inference_ms_per_patient"
                ].mean(),
                "mean_pca_explained_variance": summary[
                    "mean_pca_explained_variance"
                ].mean(),
            }
        ]
    )
    summary = pd.concat([summary, macro], ignore_index=True)
    summary.to_csv(output_dir / "complexity_summary.csv", index=False)
    protocol = {
        "source": str(source_dir / "fold_complexity.csv"),
        "incremental_gpu_training_required": False,
        "incremental_deep_trainable_parameters": 0,
        "cox_parameters_per_fold": 65,
        "timing_scope": (
            "CPU StandardScaler, randomized PCA64, Cox fitting, and locked "
            "PCA-Cox/ECDF/fusion inference; VLSA pathology inference is excluded."
        ),
        "minimal_state_scope": (
            "Scaler mean/scale, PCA mean/components, Cox weight/bias, and the "
            "training-fold sorted Cox predictions required by ECDF calibration."
        ),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("===== QACLF incremental complexity =====")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
