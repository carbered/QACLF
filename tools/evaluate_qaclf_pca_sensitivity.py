#!/usr/bin/env python3
"""Refit fold-local PCA-Cox models for fixed QACLF PCA sensitivity analysis."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn
import torch

from compare_paired_survival_predictions import c_index
from probe_cat_omics_complementarity import train_empirical_cdf
from probe_cat_omics_signal import fit_cox, signature_features


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("blca", "brca", "gbmlgg", "luad", "ucec")
QACLF_DIRS = {
    "blca": "blca-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    "brca": "brca-frozen-pathomic-fusion-outer5-20260820",
    "gbmlgg": "gbmlgg-frozen-pathomic-fusion-outer5-20260820",
    "luad": "luad-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    "ucec": "ucec-frozen-pathomic-fusion-outer5-20260820",
}
METRIC_CONTRACT = "SurvivalEVAL_mean_continuous_comparable_all_ties"
LOCKED_ALPHA = 0.20
LOCKED_PCA = 64


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="result/qaclf-pca-sensitivity-20260821"
    )
    parser.add_argument("--omics-table-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(COHORTS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument(
        "--pca-components", type=int, nargs="+", default=[16, 32, 64, 128]
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    invalid = sorted(set(args.datasets) - set(COHORTS))
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}")
    if not set(args.folds).issubset(range(5)):
        raise ValueError("Folds must be in [0, 4].")
    dimensions = sorted(set(args.pca_components))
    if any(value <= 0 for value in dimensions):
        raise ValueError("PCA dimensions must be positive.")

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cat_root = ROOT / "external" / "CAT-with-TPL-master"
    if args.omics_table_dir is not None:
        table_dir = resolve(args.omics_table_dir)
        candidates = [table_dir]
    else:
        candidates = [cat_root / "dataset_csv", cat_root / "dataset_csv_sig"]
    table_dir = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "signatures.csv").exists()
            and all(
                (candidate / f"tcga_{cohort}_all_clean.csv.zip").exists()
                for cohort in args.datasets
            )
        ),
        None,
    )
    if table_dir is None:
        checked = [str(path) for path in candidates]
        raise FileNotFoundError(
            "No complete CAT omics table directory was found. "
            f"Checked: {checked}"
        )
    signatures_path = table_dir / "signatures.csv"
    signatures = pd.read_csv(signatures_path)

    rows = []
    patient_rows = []
    audit = {
        "metric_contract": METRIC_CONTRACT,
        "locked_primary": {"pca_components": LOCKED_PCA, "alpha": LOCKED_ALPHA},
        "random_seed": args.seed,
        "evaluated_pca_components": dimensions,
        "sensitivity_for_reporting_only": True,
        "standardization_fit": "outer_training_patients_only",
        "pca_fit": "outer_training_patients_only",
        "cox_fit": "outer_training_patients_only",
        "ecdf_fit": "outer_training_predictions_only",
        "test_labels_used_for_parameter_selection": False,
        "runtime": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "omics_table_dir": str(table_dir),
        "signatures_sha256": hashlib.sha256(signatures_path.read_bytes()).hexdigest(),
        "all_patient_sets_equal": True,
        "all_labels_equal": True,
        "pca64_omics_prediction_max_abs_delta": 0.0,
        "pca64_omics_ecdf_max_abs_delta": 0.0,
        "pca64_fused_prediction_max_abs_delta": 0.0,
        "datasets": {},
    }

    for cohort in args.datasets:
        qaclf_path = ROOT / "result" / QACLF_DIRS[cohort] / "patient_predictions.csv"
        qaclf = pd.read_csv(qaclf_path, dtype={"patient_id": str})
        table_path = table_dir / f"tcga_{cohort}_all_clean.csv.zip"
        table = (
            pd.read_csv(table_path, low_memory=False)
            .drop_duplicates("case_id")
            .set_index("case_id")
        )
        features, group_sizes = signature_features(table.columns, signatures)
        if not features:
            raise RuntimeError(f"No signature-linked omics features for {cohort}.")
        raw_by_patient = table[features]
        cohort_audit = {
            "qaclf_source": str(qaclf_path),
            "selected_omics": len(features),
            "omics_table": str(table_path),
            "omics_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            "signature_group_sizes": group_sizes,
            "splits": {},
        }

        for fold_index in args.folds:
            split_path = (
                ROOT
                / "data_split"
                / "5foldcv"
                / f"tcga_{cohort}"
                / f"splits_{fold_index}.csv"
            )
            split = pd.read_csv(split_path, dtype=str)
            train_ids = split["train"].dropna().tolist()
            test_ids = split["val"].dropna().tolist()
            cohort_audit["splits"][str(fold_index)] = {
                "path": str(split_path),
                "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "train_patients": len(train_ids),
                "test_patients": len(test_ids),
            }
            target = qaclf[qaclf["fold"] == fold_index].copy().set_index("patient_id")
            patient_sets_equal = set(test_ids) == set(target.index)
            audit["all_patient_sets_equal"] &= patient_sets_equal
            if not patient_sets_equal:
                raise RuntimeError(f"Patient sets differ for {cohort} fold {fold_index}.")
            required = set(train_ids) | set(test_ids)
            if not required.issubset(raw_by_patient.index):
                missing = sorted(required - set(raw_by_patient.index))
                raise RuntimeError(
                    f"Omics table misses {len(missing)} patients for {cohort}: {missing[:5]}"
                )
            target = target.loc[test_ids]
            test_table = table.loc[test_ids]
            labels_equal = (
                np.allclose(
                    target["event_time"].to_numpy(dtype=float),
                    test_table["survival_months"].to_numpy(dtype=float),
                    atol=1e-4,
                )
                and np.allclose(
                    target["event"].to_numpy(dtype=float),
                    1.0 - test_table["censorship"].to_numpy(dtype=float),
                )
            )
            audit["all_labels_equal"] &= labels_equal
            if not labels_equal:
                raise RuntimeError(f"Labels differ for {cohort} fold {fold_index}.")

            train_raw = raw_by_patient.loc[train_ids].to_numpy(dtype=np.float64)
            test_raw = raw_by_patient.loc[test_ids].to_numpy(dtype=np.float64)
            if not np.isfinite(train_raw).all() or not np.isfinite(test_raw).all():
                raise RuntimeError(f"Non-finite omics values for {cohort} fold {fold_index}.")
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_raw)
            test_scaled = scaler.transform(test_raw)
            train_time = table.loc[train_ids, "survival_months"].to_numpy(dtype=float)
            train_event = 1.0 - table.loc[train_ids, "censorship"].to_numpy(dtype=float)
            test_time = target["event_time"].to_numpy(dtype=float)
            test_event = target["event"].to_numpy(dtype=float)
            pathology_ecdf = target["pathology_calibrated"].to_numpy(dtype=float)

            for requested_components in dimensions:
                components = min(
                    requested_components,
                    train_scaled.shape[0] - 1,
                    train_scaled.shape[1],
                )
                started = time.perf_counter()
                pca = PCA(
                    n_components=components,
                    svd_solver="randomized",
                    random_state=args.seed,
                )
                train_x = pca.fit_transform(train_scaled)
                test_x = pca.transform(test_scaled)
                weight, bias, final_loss = fit_cox(
                    train_x,
                    train_time,
                    train_event,
                    args.steps,
                    args.lr,
                    args.l2,
                )
                elapsed = time.perf_counter() - started
                train_omics_time = -(train_x @ weight + bias)
                test_omics_time = -(test_x @ weight + bias)
                omics_ecdf = train_empirical_cdf(train_omics_time, test_omics_time)
                fused = (
                    (1.0 - LOCKED_ALPHA) * pathology_ecdf
                    + LOCKED_ALPHA * omics_ecdf
                )
                evaluation_args = (test_time, test_event, train_time, train_event)
                omics_c = c_index(test_omics_time, *evaluation_args)
                fused_c = c_index(fused, *evaluation_args)
                rows.append(
                    {
                        "dataset": cohort.upper(),
                        "fold": fold_index,
                        "pca_random_seed": args.seed,
                        "n": len(test_ids),
                        "events": int(test_event.sum()),
                        "selected_omics": len(features),
                        "requested_pca_components": requested_components,
                        "actual_pca_components": components,
                        "pca_explained_variance": float(
                            pca.explained_variance_ratio_.sum()
                        ),
                        "cox_final_loss": final_loss,
                        "fit_seconds": elapsed,
                        "omics_c_index": omics_c,
                        "fused_c_index": fused_c,
                    }
                )
                patient_rows.extend(
                    {
                        "dataset": cohort.upper(),
                        "fold": fold_index,
                        "patient_id": patient_id,
                        "pca_random_seed": args.seed,
                        "requested_pca_components": requested_components,
                        "omics_predicted_time": test_omics_time[index],
                        "omics_calibrated": omics_ecdf[index],
                        "fused_predicted_time": fused[index],
                    }
                    for index, patient_id in enumerate(test_ids)
                )

                if requested_components == LOCKED_PCA:
                    audit["pca64_omics_prediction_max_abs_delta"] = max(
                        audit["pca64_omics_prediction_max_abs_delta"],
                        float(
                            np.max(
                                np.abs(
                                    test_omics_time
                                    - target["omics_predicted_time"].to_numpy(dtype=float)
                                )
                            )
                        ),
                    )
                    audit["pca64_omics_ecdf_max_abs_delta"] = max(
                        audit["pca64_omics_ecdf_max_abs_delta"],
                        float(
                            np.max(
                                np.abs(
                                    omics_ecdf
                                    - target["omics_calibrated"].to_numpy(dtype=float)
                                )
                            )
                        ),
                    )
                    audit["pca64_fused_prediction_max_abs_delta"] = max(
                        audit["pca64_fused_prediction_max_abs_delta"],
                        float(
                            np.max(
                                np.abs(
                                    fused
                                    - target["fused_predicted_time"].to_numpy(dtype=float)
                                )
                            )
                        ),
                    )
        audit["datasets"][cohort] = cohort_audit

    fold_metrics = pd.DataFrame(rows)
    dataset_summary = (
        fold_metrics.groupby(
            ["dataset", "requested_pca_components"], sort=False
        ).agg(
            mean_omics_c_index=("omics_c_index", "mean"),
            std_omics_c_index=("omics_c_index", lambda x: x.std(ddof=0)),
            mean_fused_c_index=("fused_c_index", "mean"),
            std_fused_c_index=("fused_c_index", lambda x: x.std(ddof=0)),
            mean_explained_variance=("pca_explained_variance", "mean"),
            total_fit_seconds=("fit_seconds", "sum"),
        ).reset_index()
    )
    macro = (
        dataset_summary.groupby("requested_pca_components", sort=False).agg(
            mean_omics_c_index=("mean_omics_c_index", "mean"),
            std_omics_c_index=("mean_omics_c_index", lambda x: x.std(ddof=0)),
            mean_fused_c_index=("mean_fused_c_index", "mean"),
            std_fused_c_index=("mean_fused_c_index", lambda x: x.std(ddof=0)),
            mean_explained_variance=("mean_explained_variance", "mean"),
            total_fit_seconds=("total_fit_seconds", "sum"),
        ).reset_index()
    )
    macro.insert(0, "dataset", "MACRO")
    dataset_summary = pd.concat([dataset_summary, macro], ignore_index=True)
    for dataset in dataset_summary["dataset"].unique():
        mask = dataset_summary["dataset"] == dataset
        locked_rows = dataset_summary[
            mask
            & (dataset_summary["requested_pca_components"] == LOCKED_PCA)
        ]
        if len(locked_rows) == 1:
            reference = float(locked_rows["mean_fused_c_index"].iloc[0])
            dataset_summary.loc[mask, "delta_vs_pca64"] = (
                dataset_summary.loc[mask, "mean_fused_c_index"] - reference
            )
        else:
            dataset_summary.loc[mask, "delta_vs_pca64"] = np.nan

    audit["pca64_evaluated"] = LOCKED_PCA in dimensions
    audit["pca64_exact_reproduction"] = (
        bool(
            audit["pca64_omics_prediction_max_abs_delta"] <= 1e-10
            and audit["pca64_omics_ecdf_max_abs_delta"] <= 1e-10
            and audit["pca64_fused_prediction_max_abs_delta"] <= 1e-10
        )
        if audit["pca64_evaluated"]
        else None
    )

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(output_dir / "patient_predictions.csv", index=False)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("===== QACLF PCA sensitivity =====")
    print(
        dataset_summary.pivot(
            index="dataset",
            columns="requested_pca_components",
            values="mean_fused_c_index",
        ).to_string(float_format=lambda value: f"{value:.6f}")
    )
    print("===== PCA64 reproduction audit =====")
    print(
        json.dumps(
            {
                key: value
                for key, value in audit.items()
                if key.startswith("pca64_")
            },
            indent=2,
        )
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
