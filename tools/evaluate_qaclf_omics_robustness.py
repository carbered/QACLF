#!/usr/bin/env python3
"""Evaluate QACLF robustness to label-independent test-time omics corruption."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def stable_seed(base_seed, cohort_index, fold, condition_index, repeat):
    return (
        int(base_seed)
        + 100000 * cohort_index
        + 10000 * fold
        + 100 * condition_index
        + repeat
    )


def calibrated_omics(train_x, test_scaled, pca, weight, bias, train_reference):
    test_x = pca.transform(test_scaled)
    test_time = -(test_x @ weight + bias)
    return test_time, train_empirical_cdf(train_reference, test_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="result/qaclf-omics-robustness-20260821"
    )
    parser.add_argument("--omics-table-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(COHORTS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--feature-dropout", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--noise-sigma", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument(
        "--patient-missing", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00]
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inference-repeats", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    invalid = sorted(set(args.datasets) - set(COHORTS))
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}")
    if not set(args.folds).issubset(range(5)):
        raise ValueError("Folds must be in [0, 4].")
    if args.repeats < 1 or args.inference_repeats < 1:
        raise ValueError("Repeat counts must be positive.")
    for name, values in (
        ("feature_dropout", args.feature_dropout),
        ("patient_missing", args.patient_missing),
    ):
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{name} values must be in [0, 1].")
    if any(value < 0.0 for value in args.noise_sigma):
        raise ValueError("Noise standard deviations must be non-negative.")

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cat_root = ROOT / "external" / "CAT-with-TPL-master"
    candidates = (
        [resolve(args.omics_table_dir)]
        if args.omics_table_dir is not None
        else [cat_root / "dataset_csv", cat_root / "dataset_csv_sig"]
    )
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
        raise FileNotFoundError(
            "No complete CAT omics table directory was found. Checked: "
            f"{[str(path) for path in candidates]}"
        )
    signatures_path = table_dir / "signatures.csv"
    signatures = pd.read_csv(signatures_path)

    repeat_rows = []
    complexity_rows = []
    condition_specs = [("clean", 0.0)]
    condition_specs.extend(("feature_dropout", value) for value in args.feature_dropout)
    condition_specs.extend(("gaussian_noise", value) for value in args.noise_sigma)
    condition_specs.extend(("patient_missing", value) for value in args.patient_missing)

    audit = {
        "metric_contract": METRIC_CONTRACT,
        "locked_method": {"pca_components": LOCKED_PCA, "alpha": LOCKED_ALPHA},
        "conditions": [
            {"condition": condition, "severity": severity}
            for condition, severity in condition_specs
        ],
        "feature_dropout_definition": (
            "MCAR test entries are replaced by the outer-training feature mean, "
            "which is zero after the locked training-only scaler."
        ),
        "gaussian_noise_definition": (
            "Independent zero-mean Gaussian noise is added in the locked "
            "training-standardized feature space."
        ),
        "patient_missing_definition": (
            "An exact random fraction of test patients has the omics branch replaced "
            "by its pathology calibration, so fusion falls back to pathology."
        ),
        "perturbation_scope": "test_omics_only",
        "perturbations_independent_of_survival_labels": True,
        "training_models_refit_per_perturbation": False,
        "test_labels_used_for_fit_calibration_or_perturbation": False,
        "repeats": args.repeats,
        "random_seed": args.seed,
        "runtime": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "omics_table_dir": str(table_dir),
        "signatures_sha256": hashlib.sha256(signatures_path.read_bytes()).hexdigest(),
        "datasets": {},
    }

    for cohort_index, cohort in enumerate(args.datasets):
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
            "omics_table": str(table_path),
            "omics_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            "selected_omics": len(features),
            "signature_group_sizes": group_sizes,
            "splits": {},
        }

        for fold in args.folds:
            split_path = (
                ROOT / "data_split" / "5foldcv" / f"tcga_{cohort}" / f"splits_{fold}.csv"
            )
            split = pd.read_csv(split_path, dtype=str)
            train_ids = split["train"].dropna().tolist()
            test_ids = split["val"].dropna().tolist()
            target = qaclf[qaclf["fold"] == fold].copy().set_index("patient_id")
            if set(test_ids) != set(target.index):
                raise RuntimeError(f"Patient sets differ for {cohort} fold {fold}.")
            required = set(train_ids) | set(test_ids)
            if not required.issubset(raw_by_patient.index):
                missing = sorted(required - set(raw_by_patient.index))
                raise RuntimeError(
                    f"Omics table misses {len(missing)} patients for {cohort}: {missing[:5]}"
                )
            target = target.loc[test_ids]
            labels_equal = (
                np.allclose(
                    target["event_time"].to_numpy(dtype=float),
                    table.loc[test_ids, "survival_months"].to_numpy(dtype=float),
                    atol=1e-4,
                )
                and np.allclose(
                    target["event"].to_numpy(dtype=float),
                    1.0 - table.loc[test_ids, "censorship"].to_numpy(dtype=float),
                )
            )
            if not labels_equal:
                raise RuntimeError(f"Labels differ for {cohort} fold {fold}.")

            train_raw = raw_by_patient.loc[train_ids].to_numpy(dtype=np.float64)
            test_raw = raw_by_patient.loc[test_ids].to_numpy(dtype=np.float64)
            if not np.isfinite(train_raw).all() or not np.isfinite(test_raw).all():
                raise RuntimeError(f"Non-finite omics values for {cohort} fold {fold}.")
            train_time = table.loc[train_ids, "survival_months"].to_numpy(dtype=float)
            train_event = 1.0 - table.loc[train_ids, "censorship"].to_numpy(dtype=float)
            test_time = target["event_time"].to_numpy(dtype=float)
            test_event = target["event"].to_numpy(dtype=float)
            pathology = target["pathology_calibrated"].to_numpy(dtype=float)

            fit_started = time.perf_counter()
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_raw)
            test_scaled = scaler.transform(test_raw)
            components = min(LOCKED_PCA, len(train_ids) - 1, len(features))
            pca = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=42,
            )
            train_x = pca.fit_transform(train_scaled)
            weight, bias, final_loss = fit_cox(
                train_x,
                train_time,
                train_event,
                args.steps,
                args.lr,
                args.l2,
            )
            fit_seconds = time.perf_counter() - fit_started
            train_omics_time = -(train_x @ weight + bias)
            clean_time, clean_omics = calibrated_omics(
                train_x,
                test_scaled,
                pca,
                weight,
                bias,
                train_omics_time,
            )
            clean_fused = (1.0 - LOCKED_ALPHA) * pathology + LOCKED_ALPHA * clean_omics
            evaluation_args = (test_time, test_event, train_time, train_event)
            clean_c = c_index(clean_fused, *evaluation_args)
            pathology_c = c_index(pathology, *evaluation_args)
            clean_omics_c = c_index(clean_time, *evaluation_args)

            benchmark_started = time.perf_counter()
            for _ in range(args.inference_repeats):
                _, benchmark_omics = calibrated_omics(
                    train_x,
                    test_scaled,
                    pca,
                    weight,
                    bias,
                    train_omics_time,
                )
                benchmark_fused = (
                    (1.0 - LOCKED_ALPHA) * pathology + LOCKED_ALPHA * benchmark_omics
                )
            inference_seconds = (
                time.perf_counter() - benchmark_started
            ) / args.inference_repeats
            if not np.allclose(benchmark_fused, clean_fused):
                raise RuntimeError("Inference benchmark changed clean predictions.")

            minimal_state_floats = (
                2 * len(features)
                + len(features)
                + components * len(features)
                + components
                + 1
                + len(train_ids)
            )
            complexity_rows.append(
                {
                    "dataset": cohort.upper(),
                    "fold": fold,
                    "train_n": len(train_ids),
                    "test_n": len(test_ids),
                    "selected_omics": len(features),
                    "pca_components": components,
                    "cox_parameters": components + 1,
                    "deep_trainable_parameters_added": 0,
                    "minimal_persistent_float_values": minimal_state_floats,
                    "minimal_state_mb_float64": minimal_state_floats * 8 / (1024 ** 2),
                    "fit_seconds": fit_seconds,
                    "inference_seconds_per_fold": inference_seconds,
                    "inference_ms_per_patient": 1000 * inference_seconds / len(test_ids),
                    "cox_final_loss": final_loss,
                    "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                }
            )

            for condition_index, (condition, severity) in enumerate(condition_specs):
                repeats = 1 if condition == "clean" or severity == 1.0 else args.repeats
                for repeat in range(repeats):
                    seed = stable_seed(
                        args.seed, cohort_index, fold, condition_index, repeat
                    )
                    rng = np.random.RandomState(seed)
                    missing_entries = 0
                    missing_patients = 0
                    if condition == "clean":
                        omics_time = clean_time
                        omics = clean_omics
                        fused = clean_fused
                    elif condition == "feature_dropout":
                        perturbed = test_scaled.copy()
                        mask = rng.random_sample(perturbed.shape) < severity
                        perturbed[mask] = 0.0
                        missing_entries = int(mask.sum())
                        omics_time, omics = calibrated_omics(
                            train_x,
                            perturbed,
                            pca,
                            weight,
                            bias,
                            train_omics_time,
                        )
                        fused = (1.0 - LOCKED_ALPHA) * pathology + LOCKED_ALPHA * omics
                    elif condition == "gaussian_noise":
                        perturbed = test_scaled + rng.normal(
                            loc=0.0, scale=severity, size=test_scaled.shape
                        )
                        omics_time, omics = calibrated_omics(
                            train_x,
                            perturbed,
                            pca,
                            weight,
                            bias,
                            train_omics_time,
                        )
                        fused = (1.0 - LOCKED_ALPHA) * pathology + LOCKED_ALPHA * omics
                    elif condition == "patient_missing":
                        omics_time = clean_time.copy()
                        omics = clean_omics.copy()
                        missing_patients = int(round(severity * len(test_ids)))
                        missing_index = rng.choice(
                            len(test_ids), size=missing_patients, replace=False
                        )
                        omics[missing_index] = pathology[missing_index]
                        fused = (1.0 - LOCKED_ALPHA) * pathology + LOCKED_ALPHA * omics
                    else:
                        raise RuntimeError(f"Unknown condition: {condition}")
                    repeat_rows.append(
                        {
                            "dataset": cohort.upper(),
                            "fold": fold,
                            "condition": condition,
                            "severity": severity,
                            "repeat": repeat,
                            "seed": seed,
                            "n": len(test_ids),
                            "events": int(test_event.sum()),
                            "missing_entries": missing_entries,
                            "missing_patients": missing_patients,
                            "pathology_c_index": pathology_c,
                            "clean_omics_c_index": clean_omics_c,
                            "clean_fused_c_index": clean_c,
                            "perturbed_omics_c_index": c_index(
                                omics_time, *evaluation_args
                            ),
                            "perturbed_fused_c_index": c_index(fused, *evaluation_args),
                        }
                    )

            cohort_audit["splits"][str(fold)] = {
                "path": str(split_path),
                "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "train_patients": len(train_ids),
                "test_patients": len(test_ids),
                "events": int(test_event.sum()),
            }
        audit["datasets"][cohort] = cohort_audit

    repeat_metrics = pd.DataFrame(repeat_rows)
    fold_summary = (
        repeat_metrics.groupby(
            ["dataset", "fold", "condition", "severity"], sort=False
        )
        .agg(
            repeats=("repeat", "count"),
            n=("n", "first"),
            events=("events", "first"),
            pathology_c_index=("pathology_c_index", "first"),
            clean_fused_c_index=("clean_fused_c_index", "first"),
            mean_perturbed_fused_c_index=("perturbed_fused_c_index", "mean"),
            std_perturbed_fused_c_index=(
                "perturbed_fused_c_index",
                lambda values: values.std(ddof=0),
            ),
        )
        .reset_index()
    )
    fold_summary["delta_vs_clean"] = (
        fold_summary["mean_perturbed_fused_c_index"]
        - fold_summary["clean_fused_c_index"]
    )
    fold_summary["delta_vs_pathology"] = (
        fold_summary["mean_perturbed_fused_c_index"]
        - fold_summary["pathology_c_index"]
    )
    dataset_summary = (
        fold_summary.groupby(["dataset", "condition", "severity"], sort=False)
        .agg(
            folds=("fold", "count"),
            mean_pathology_c_index=("pathology_c_index", "mean"),
            mean_clean_fused_c_index=("clean_fused_c_index", "mean"),
            mean_perturbed_fused_c_index=("mean_perturbed_fused_c_index", "mean"),
            std_across_folds=(
                "mean_perturbed_fused_c_index",
                lambda values: values.std(ddof=0),
            ),
        )
        .reset_index()
    )
    dataset_summary["delta_vs_clean"] = (
        dataset_summary["mean_perturbed_fused_c_index"]
        - dataset_summary["mean_clean_fused_c_index"]
    )
    dataset_summary["delta_vs_pathology"] = (
        dataset_summary["mean_perturbed_fused_c_index"]
        - dataset_summary["mean_pathology_c_index"]
    )
    macro = (
        dataset_summary.groupby(["condition", "severity"], sort=False)
        .agg(
            folds=("folds", "sum"),
            mean_pathology_c_index=("mean_pathology_c_index", "mean"),
            mean_clean_fused_c_index=("mean_clean_fused_c_index", "mean"),
            mean_perturbed_fused_c_index=("mean_perturbed_fused_c_index", "mean"),
            std_across_folds=(
                "mean_perturbed_fused_c_index",
                lambda values: values.std(ddof=0),
            ),
        )
        .reset_index()
    )
    macro.insert(0, "dataset", "MACRO")
    macro["delta_vs_clean"] = (
        macro["mean_perturbed_fused_c_index"] - macro["mean_clean_fused_c_index"]
    )
    macro["delta_vs_pathology"] = (
        macro["mean_perturbed_fused_c_index"] - macro["mean_pathology_c_index"]
    )
    dataset_summary = pd.concat([dataset_summary, macro], ignore_index=True)

    repeat_metrics.to_csv(output_dir / "repeat_metrics.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    pd.DataFrame(complexity_rows).to_csv(output_dir / "fold_complexity.csv", index=False)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("===== QACLF omics robustness: macro C-index =====")
    print(
        dataset_summary[dataset_summary["dataset"] == "MACRO"]
        .loc[
            :,
            [
                "condition",
                "severity",
                "mean_perturbed_fused_c_index",
                "delta_vs_clean",
                "delta_vs_pathology",
            ],
        ]
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
