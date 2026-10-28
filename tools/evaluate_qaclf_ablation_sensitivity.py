#!/usr/bin/env python3
"""Leakage-safe component ablation and fixed alpha sensitivity for QACLF."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_paired_survival_predictions import c_index
from evaluate_external_multimodal_unified import (
    QACLF_DIRS,
    METRIC_CONTRACT,
    find_external_run,
    load_training_labels,
)


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("blca", "brca", "gbmlgg", "luad", "ucec")
LOCKED_ALPHA = 0.20
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.5)
METHODS = (
    "pathology_only",
    "omics_only",
    "pathology_ecdf_only",
    "omics_ecdf_only",
    "raw_fixed_fusion",
    "ecdf_equal_fusion",
    "qaclf_locked",
)
COMPARISONS = (
    ("omics_vs_pathology", "omics_only", "pathology_only"),
    ("raw_fusion_vs_pathology", "raw_fixed_fusion", "pathology_only"),
    ("equal_fusion_vs_pathology", "ecdf_equal_fusion", "pathology_only"),
    ("qaclf_vs_pathology", "qaclf_locked", "pathology_only"),
    ("qaclf_vs_raw_fusion", "qaclf_locked", "raw_fixed_fusion"),
    ("qaclf_vs_equal_fusion", "qaclf_locked", "ecdf_equal_fusion"),
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def score(values, fold):
    return c_index(
        values,
        fold["time"],
        fold["event"],
        fold["train_time"],
        fold["train_event"],
    )


def method_predictions(target):
    pathology = target["pathology_predicted_time"].to_numpy(dtype=float)
    omics = target["omics_predicted_time"].to_numpy(dtype=float)
    pathology_ecdf = target["pathology_calibrated"].to_numpy(dtype=float)
    omics_ecdf = target["omics_calibrated"].to_numpy(dtype=float)
    return {
        "pathology_only": pathology,
        "omics_only": omics,
        "pathology_ecdf_only": pathology_ecdf,
        "omics_ecdf_only": omics_ecdf,
        "raw_fixed_fusion": (
            (1.0 - LOCKED_ALPHA) * pathology + LOCKED_ALPHA * omics
        ),
        "ecdf_equal_fusion": 0.5 * pathology_ecdf + 0.5 * omics_ecdf,
        "qaclf_locked": (
            (1.0 - LOCKED_ALPHA) * pathology_ecdf
            + LOCKED_ALPHA * omics_ecdf
        ),
    }


def paired_bootstrap(folds, samples, seed):
    rng = np.random.RandomState(seed)
    deltas = {name: [] for name, _, _ in COMPARISONS}
    for _ in range(samples):
        current = {name: [] for name, _, _ in COMPARISONS}
        valid = True
        for fold in folds:
            size = len(fold["event"])
            sampled = rng.randint(0, size, size=size)
            sampled_event = fold["event"][sampled]
            if not np.any(sampled_event):
                valid = False
                break
            args = (
                fold["time"][sampled],
                sampled_event,
                fold["train_time"],
                fold["train_event"],
            )
            try:
                scores = {
                    method: c_index(fold[method][sampled], *args)
                    for method in METHODS
                }
            except (ValueError, ZeroDivisionError):
                valid = False
                break
            for name, candidate, reference in COMPARISONS:
                current[name].append(scores[candidate] - scores[reference])
        if valid:
            for name, _, _ in COMPARISONS:
                deltas[name].append(float(np.mean(current[name])))
    if not all(deltas.values()):
        raise RuntimeError("No valid paired bootstrap samples were generated.")
    return {name: np.asarray(values, dtype=float) for name, values in deltas.items()}


def bootstrap_row(dataset, name, candidate, reference, observed, positive, values):
    return {
        "dataset": dataset,
        "comparison": name,
        "candidate": candidate,
        "reference": reference,
        "observed_delta": observed,
        "positive_folds": positive,
        "bootstrap_samples": len(values),
        "delta_ci025": float(np.quantile(values, 0.025)),
        "delta_ci975": float(np.quantile(values, 0.975)),
        "probability_delta_gt_zero": float(np.mean(values > 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="result/external_multimodal_comparison_20260821/raw/results-conch",
    )
    parser.add_argument(
        "--output-dir", default="result/qaclf-ablation-sensitivity-20260821"
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = resolve(args.raw_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cat_root = ROOT / "external" / "CAT-with-TPL-master"

    fold_rows = []
    alpha_rows = []
    shuffle_rows = []
    bootstrap_rows = []
    dataset_folds = {}
    dataset_bootstraps = {name: [] for name, _, _ in COMPARISONS}
    audit = {
        "metric_contract": METRIC_CONTRACT,
        "locked_primary": {"alpha": LOCKED_ALPHA, "pca_components": 64},
        "alpha_sensitivity": list(ALPHAS),
        "sensitivity_for_reporting_only": True,
        "test_labels_used_for_parameter_selection": False,
        "shuffle_control_uses_labels_for_permutation": False,
        "recomputed_locked_prediction_max_abs_delta": 0.0,
        "recomputed_locked_c_index_max_abs_delta": 0.0,
        "pathology_ecdf_c_index_max_abs_delta": 0.0,
        "omics_ecdf_c_index_max_abs_delta": 0.0,
        "ecdf_monotonic_no_inversions": True,
        "all_patient_sets_equal": True,
        "all_labels_equal": True,
        "datasets": {},
    }

    for cohort_index, cohort in enumerate(COHORTS):
        qaclf_dir = ROOT / "result" / QACLF_DIRS[cohort]
        qaclf_path = qaclf_dir / "patient_predictions.csv"
        saved_fold_metrics = pd.read_csv(qaclf_dir / "fold_metrics.csv").set_index("fold")
        qaclf = pd.read_csv(qaclf_path, dtype={"patient_id": str})
        mcat_dir = find_external_run(raw_root, cohort, "mcat_conch")
        table = pd.read_csv(
            cat_root / "dataset_csv_sig" / f"tcga_{cohort}_all_clean.csv.zip",
            low_memory=False,
        ).drop_duplicates("case_id").set_index("case_id")

        folds = []
        cohort_audit = {"qaclf_source": str(qaclf_path), "split_source": str(mcat_dir)}
        fold_shuffle_values = []
        for fold_index in range(5):
            target = qaclf[qaclf["fold"] == fold_index].copy()
            if target.empty or target["patient_id"].duplicated().any():
                raise RuntimeError(f"Invalid QACLF patients for {cohort} fold {fold_index}.")
            target = target.set_index("patient_id")
            train_time, train_event, split_test_ids = load_training_labels(
                mcat_dir, fold_index, table
            )
            patient_sets_equal = set(split_test_ids) == set(target.index)
            audit["all_patient_sets_equal"] &= patient_sets_equal
            if not patient_sets_equal:
                raise RuntimeError(f"Patient sets differ for {cohort} fold {fold_index}.")
            target = target.loc[split_test_ids]
            table_labels = table.loc[split_test_ids]
            labels_equal = (
                np.allclose(
                    target["event_time"].to_numpy(dtype=float),
                    table_labels["survival_months"].to_numpy(dtype=float),
                    atol=1e-4,
                )
                and np.allclose(
                    target["event"].to_numpy(dtype=float),
                    1.0 - table_labels["censorship"].to_numpy(dtype=float),
                )
            )
            audit["all_labels_equal"] &= labels_equal
            if not labels_equal:
                raise RuntimeError(f"Labels differ for {cohort} fold {fold_index}.")

            fold = {
                "time": target["event_time"].to_numpy(dtype=float),
                "event": target["event"].to_numpy(dtype=float),
                "train_time": train_time,
                "train_event": train_event,
                **method_predictions(target),
            }
            if not all(np.isfinite(fold[method]).all() for method in METHODS):
                raise RuntimeError(f"Non-finite predictions for {cohort} fold {fold_index}.")

            saved_fused = target["fused_predicted_time"].to_numpy(dtype=float)
            prediction_delta = float(np.max(np.abs(fold["qaclf_locked"] - saved_fused)))
            audit["recomputed_locked_prediction_max_abs_delta"] = max(
                audit["recomputed_locked_prediction_max_abs_delta"], prediction_delta
            )
            scores = {method: score(fold[method], fold) for method in METHODS}
            locked_metric_delta = abs(
                scores["qaclf_locked"]
                - float(saved_fold_metrics.loc[fold_index, "fused_c_index"])
            )
            audit["recomputed_locked_c_index_max_abs_delta"] = max(
                audit["recomputed_locked_c_index_max_abs_delta"], locked_metric_delta
            )
            audit["pathology_ecdf_c_index_max_abs_delta"] = max(
                audit["pathology_ecdf_c_index_max_abs_delta"],
                abs(scores["pathology_only"] - scores["pathology_ecdf_only"]),
            )
            audit["omics_ecdf_c_index_max_abs_delta"] = max(
                audit["omics_ecdf_c_index_max_abs_delta"],
                abs(scores["omics_only"] - scores["omics_ecdf_only"]),
            )
            for raw_key, ecdf_key in (
                ("pathology_only", "pathology_ecdf_only"),
                ("omics_only", "omics_ecdf_only"),
            ):
                order = np.argsort(fold[raw_key], kind="mergesort")
                monotonic = bool(np.all(np.diff(fold[ecdf_key][order]) >= -1e-12))
                audit["ecdf_monotonic_no_inversions"] &= monotonic

            fold_row = {
                "dataset": cohort.upper(),
                "fold": fold_index,
                "n": len(target),
                "events": int(fold["event"].sum()),
                "pathology_ecdf_unique_ratio": float(
                    len(np.unique(fold["pathology_ecdf_only"])) / len(target)
                ),
                "omics_ecdf_unique_ratio": float(
                    len(np.unique(fold["omics_ecdf_only"])) / len(target)
                ),
            }
            fold_row.update({f"{method}_c_index": scores[method] for method in METHODS})
            fold_rows.append(fold_row)

            pathology_ecdf = fold["pathology_ecdf_only"]
            omics_ecdf = fold["omics_ecdf_only"]
            for alpha in ALPHAS:
                values = (1.0 - alpha) * pathology_ecdf + alpha * omics_ecdf
                alpha_rows.append(
                    {
                        "dataset": cohort.upper(),
                        "fold": fold_index,
                        "alpha": alpha,
                        "c_index": score(values, fold),
                    }
                )

            rng = np.random.RandomState(args.seed + cohort_index * 100 + fold_index)
            shuffled_scores = []
            for _ in range(args.permutations):
                shuffled = omics_ecdf[rng.permutation(len(omics_ecdf))]
                values = (
                    (1.0 - LOCKED_ALPHA) * pathology_ecdf
                    + LOCKED_ALPHA * shuffled
                )
                shuffled_scores.append(score(values, fold))
            fold_shuffle_values.append(np.asarray(shuffled_scores, dtype=float))
            folds.append(fold)

        dataset_folds[cohort] = folds
        shuffled_fold_mean = np.mean(np.stack(fold_shuffle_values, axis=0), axis=0)
        observed_qaclf = float(
            np.mean([score(fold["qaclf_locked"], fold) for fold in folds])
        )
        shuffle_rows.append(
            {
                "dataset": cohort.upper(),
                "permutations": len(shuffled_fold_mean),
                "qaclf_c_index": observed_qaclf,
                "shuffled_omics_mean_c_index": float(shuffled_fold_mean.mean()),
                "shuffled_omics_ci025": float(np.quantile(shuffled_fold_mean, 0.025)),
                "shuffled_omics_ci975": float(np.quantile(shuffled_fold_mean, 0.975)),
                "qaclf_minus_shuffle_mean": float(
                    observed_qaclf - shuffled_fold_mean.mean()
                ),
                "probability_shuffle_ge_qaclf": float(
                    np.mean(shuffled_fold_mean >= observed_qaclf)
                ),
            }
        )

        current = pd.DataFrame(fold_rows)
        current = current[current["dataset"] == cohort.upper()]
        bootstraps = paired_bootstrap(
            folds, args.bootstrap, args.seed + 1000 + cohort_index * 100
        )
        for name, candidate, reference in COMPARISONS:
            values = bootstraps[name]
            dataset_bootstraps[name].append(values)
            observed = float(
                current[f"{candidate}_c_index"].mean()
                - current[f"{reference}_c_index"].mean()
            )
            bootstrap_rows.append(
                bootstrap_row(
                    cohort.upper(),
                    name,
                    candidate,
                    reference,
                    observed,
                    int(
                        (
                            current[f"{candidate}_c_index"]
                            > current[f"{reference}_c_index"]
                        ).sum()
                    ),
                    values,
                )
            )
        cohort_audit["patients"] = int(sum(len(fold["event"]) for fold in folds))
        cohort_audit["events"] = int(sum(fold["event"].sum() for fold in folds))
        audit["datasets"][cohort] = cohort_audit

    fold_metrics = pd.DataFrame(fold_rows)
    dataset_rows = []
    for dataset, current in fold_metrics.groupby("dataset", sort=False):
        for method in METHODS:
            values = current[f"{method}_c_index"]
            pathology = current["pathology_only_c_index"]
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mean_c_index": float(values.mean()),
                    "std_c_index": float(values.std(ddof=0)),
                    "delta_vs_pathology": float(values.mean() - pathology.mean()),
                    "positive_folds_vs_pathology": int((values > pathology).sum()),
                }
            )
    dataset_summary = pd.DataFrame(dataset_rows)
    for method in METHODS:
        current = dataset_summary[dataset_summary["method"] == method]
        dataset_summary = pd.concat(
            [
                dataset_summary,
                pd.DataFrame(
                    [
                        {
                            "dataset": "MACRO",
                            "method": method,
                            "mean_c_index": float(current["mean_c_index"].mean()),
                            "std_c_index": float(current["mean_c_index"].std(ddof=0)),
                            "delta_vs_pathology": float(
                                current["delta_vs_pathology"].mean()
                            ),
                            "positive_folds_vs_pathology": int(
                                current["positive_folds_vs_pathology"].sum()
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    alpha_folds = pd.DataFrame(alpha_rows)
    alpha_summary = (
        alpha_folds.groupby(["dataset", "alpha"], sort=False)["c_index"]
        .agg(mean_c_index="mean", std_c_index=lambda x: x.std(ddof=0))
        .reset_index()
    )
    alpha_macro = (
        alpha_summary.groupby("alpha", sort=False)["mean_c_index"]
        .agg(mean_c_index="mean", std_c_index=lambda x: x.std(ddof=0))
        .reset_index()
    )
    alpha_macro.insert(0, "dataset", "MACRO")
    alpha_summary = pd.concat([alpha_summary, alpha_macro], ignore_index=True)
    locked_macro = float(
        alpha_summary[
            (alpha_summary["dataset"] == "MACRO")
            & np.isclose(alpha_summary["alpha"], LOCKED_ALPHA)
        ]["mean_c_index"].iloc[0]
    )
    alpha_summary["delta_vs_locked_alpha020"] = np.nan
    # Explicit matching prevents any accidental dependence on row ordering.
    for dataset in alpha_summary["dataset"].unique():
        mask = alpha_summary["dataset"] == dataset
        reference = float(
            alpha_summary[mask & np.isclose(alpha_summary["alpha"], LOCKED_ALPHA)][
                "mean_c_index"
            ].iloc[0]
        )
        alpha_summary.loc[mask, "delta_vs_locked_alpha020"] = (
            alpha_summary.loc[mask, "mean_c_index"] - reference
        )

    for name, candidate, reference in COMPARISONS:
        aligned = min(len(values) for values in dataset_bootstraps[name])
        macro_values = np.mean(
            np.stack([values[:aligned] for values in dataset_bootstraps[name]], axis=0),
            axis=0,
        )
        current_candidate = dataset_summary[
            (dataset_summary["dataset"] == "MACRO")
            & (dataset_summary["method"] == candidate)
        ].iloc[0]
        current_reference = dataset_summary[
            (dataset_summary["dataset"] == "MACRO")
            & (dataset_summary["method"] == reference)
        ].iloc[0]
        bootstrap_rows.append(
            bootstrap_row(
                "MACRO",
                name,
                candidate,
                reference,
                float(
                    current_candidate["mean_c_index"]
                    - current_reference["mean_c_index"]
                ),
                int(
                    (
                        fold_metrics[f"{candidate}_c_index"]
                        > fold_metrics[f"{reference}_c_index"]
                    ).sum()
                ),
                macro_values,
            )
        )

    if audit["recomputed_locked_prediction_max_abs_delta"] > 1e-12:
        raise RuntimeError("Locked QACLF predictions were not reproduced exactly.")
    if audit["recomputed_locked_c_index_max_abs_delta"] > 1e-12:
        raise RuntimeError("Locked QACLF fold metrics were not reproduced exactly.")
    if not audit["ecdf_monotonic_no_inversions"]:
        raise RuntimeError("Training ECDF calibration introduced a rank inversion.")
    audit["locked_macro_c_index"] = locked_macro

    fold_metrics.to_csv(output_dir / "fold_metrics_ablation.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_summary_ablation.csv", index=False)
    alpha_folds.to_csv(output_dir / "fold_metrics_alpha_sensitivity.csv", index=False)
    alpha_summary.to_csv(output_dir / "dataset_summary_alpha_sensitivity.csv", index=False)
    pd.DataFrame(shuffle_rows).to_csv(
        output_dir / "shuffled_omics_negative_control.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "paired_bootstrap_ablation.csv", index=False
    )
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("===== QACLF component ablation =====")
    print(
        dataset_summary.pivot(
            index="dataset", columns="method", values="mean_c_index"
        ).to_string(float_format=lambda value: f"{value:.6f}")
    )
    print("===== Fixed alpha sensitivity =====")
    print(
        alpha_summary.pivot(
            index="dataset", columns="alpha", values="mean_c_index"
        ).to_string(float_format=lambda value: f"{value:.6f}")
    )
    print("===== Shuffled-omics negative control =====")
    print(pd.DataFrame(shuffle_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
