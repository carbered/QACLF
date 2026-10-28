#!/usr/bin/env python3
"""Re-evaluate MCAT, CAT-TPL, matched VLSA, and QACLF under one C-index contract."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from compare_paired_survival_predictions import c_index


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("blca", "brca", "gbmlgg", "luad", "ucec")
PAPER_VLSA = {
    "blca": 0.6176,
    "brca": 0.6652,
    "gbmlgg": 0.8002,
    "luad": 0.6370,
    "ucec": 0.7571,
}
QACLF_DIRS = {
    "blca": "blca-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    "brca": "brca-frozen-pathomic-fusion-outer5-20260820",
    "gbmlgg": "gbmlgg-frozen-pathomic-fusion-outer5-20260820",
    "luad": "luad-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    "ucec": "ucec-frozen-pathomic-fusion-outer5-20260820",
}
METHODS = ("matched_vlsa", "mcat_conch", "cat_tpl_conch", "qaclf")
COMPARISONS = (
    ("mcat_vs_matched", "mcat_conch", "matched_vlsa"),
    ("cat_vs_matched", "cat_tpl_conch", "matched_vlsa"),
    ("cat_vs_mcat", "cat_tpl_conch", "mcat_conch"),
    ("qaclf_vs_matched", "qaclf", "matched_vlsa"),
    ("qaclf_vs_mcat", "qaclf", "mcat_conch"),
    ("qaclf_vs_cat", "qaclf", "cat_tpl_conch"),
)
METRIC_CONTRACT = "SurvivalEVAL_mean_continuous_comparable_all_ties"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def find_external_run(raw_root, cohort, model):
    summaries = []
    for path in raw_root.rglob("summary.csv"):
        text = str(path).replace("\\", "/")
        if f"/tcga_{cohort}_" not in text:
            continue
        is_cat = "/MCAT_Clip_nll_surv_" in text
        if (model == "cat_tpl_conch" and is_cat) or (
            model == "mcat_conch" and not is_cat and "/MCAT_nll_surv_" in text
        ):
            summaries.append(path)
    if len(summaries) != 1:
        raise RuntimeError(
            f"Expected one {cohort} {model} result, found {len(summaries)}: {summaries}"
        )
    return summaries[0].parent


def load_external_fold(run_dir, fold):
    path = run_dir / f"split_latest_val_{fold}_results.pkl"
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    rows = []
    for patient_id, values in raw.items():
        rows.append(
            {
                "patient_id": str(patient_id),
                "event_time": float(values["survival"]),
                "event": 1.0 - float(values["censorship"]),
                "predicted_time": -float(values["risk"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["patient_id"].duplicated().any():
        raise RuntimeError(f"Duplicate patient IDs in {path}.")
    return frame.set_index("patient_id"), path


def load_training_labels(run_dir, fold, table):
    split = pd.read_csv(run_dir / f"splits_{fold}.csv", dtype=str)
    train_ids = split["train"].dropna().tolist()
    missing = sorted(set(train_ids) - set(table.index))
    if missing:
        raise RuntimeError(f"Training patients are absent from the omics table: {missing}")
    train = table.loc[train_ids]
    return (
        train["survival_months"].to_numpy(dtype=float),
        1.0 - train["censorship"].to_numpy(dtype=float),
        split["val"].dropna().tolist(),
    )


def score(values, actual_time, actual_event, train_time, train_event):
    return c_index(values, actual_time, actual_event, train_time, train_event)


def paired_bootstrap_all(folds, samples, seed):
    rng = np.random.RandomState(seed)
    deltas = {name: [] for name, _, _ in COMPARISONS}
    for _ in range(samples):
        fold_deltas = {name: [] for name, _, _ in COMPARISONS}
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
                    method: score(fold[method][sampled], *args) for method in METHODS
                }
            except (ValueError, ZeroDivisionError):
                valid = False
                break
            for name, candidate, reference in COMPARISONS:
                fold_deltas[name].append(scores[candidate] - scores[reference])
        if valid:
            for name, _, _ in COMPARISONS:
                deltas[name].append(float(np.mean(fold_deltas[name])))
    if not all(deltas.values()):
        raise RuntimeError("No valid bootstrap samples were generated.")
    return {name: np.asarray(values, dtype=float) for name, values in deltas.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="result/external_multimodal_comparison_20260821/raw/results-conch",
    )
    parser.add_argument(
        "--output-dir", default="result/external_multimodal_comparison_20260821/unified"
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = resolve(args.raw_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cat_root = ROOT / "external" / "CAT-with-TPL-master"

    fold_rows = []
    patient_rows = []
    dataset_rows = []
    bootstrap_rows = []
    bootstrap_values = {name: [] for name, _, _ in COMPARISONS}
    audit = {
        "metric_contract": METRIC_CONTRACT,
        "bootstrap_samples_requested": args.bootstrap,
        "seed": args.seed,
        "datasets": {},
        "all_patient_sets_equal": True,
        "all_labels_equal": True,
    }

    for cohort_index, cohort in enumerate(COHORTS):
        qaclf_path = ROOT / "result" / QACLF_DIRS[cohort] / "patient_predictions.csv"
        qaclf = pd.read_csv(qaclf_path, dtype={"patient_id": str})
        mcat_dir = find_external_run(raw_root, cohort, "mcat_conch")
        cat_dir = find_external_run(raw_root, cohort, "cat_tpl_conch")
        table = pd.read_csv(
            cat_root / "dataset_csv_sig" / f"tcga_{cohort}_all_clean.csv.zip",
            low_memory=False,
        ).drop_duplicates("case_id").set_index("case_id")

        dataset_folds = []
        author_metric_max_delta = 0.0
        cohort_audit = {"folds": [], "qaclf_source": str(qaclf_path)}
        for fold in range(5):
            target = qaclf[qaclf["fold"] == fold].copy().set_index("patient_id")
            mcat, mcat_path = load_external_fold(mcat_dir, fold)
            cat, cat_path = load_external_fold(cat_dir, fold)
            patient_sets_equal = set(target.index) == set(mcat.index) == set(cat.index)
            audit["all_patient_sets_equal"] &= patient_sets_equal
            if not patient_sets_equal:
                raise RuntimeError(f"Patient sets differ for {cohort} fold {fold}.")

            patient_ids = target.index.tolist()
            mcat = mcat.loc[patient_ids]
            cat = cat.loc[patient_ids]
            label_equal = (
                np.allclose(target["event_time"], mcat["event_time"], atol=1e-4)
                and np.allclose(target["event_time"], cat["event_time"], atol=1e-4)
                and np.allclose(target["event"], mcat["event"])
                and np.allclose(target["event"], cat["event"])
            )
            audit["all_labels_equal"] &= label_equal
            if not label_equal:
                raise RuntimeError(f"Survival labels differ for {cohort} fold {fold}.")

            train_time, train_event, split_val_ids = load_training_labels(
                mcat_dir, fold, table
            )
            if set(split_val_ids) != set(patient_ids):
                raise RuntimeError(f"Saved split differs for {cohort} fold {fold}.")

            record = {
                "time": target["event_time"].to_numpy(dtype=float),
                "event": target["event"].to_numpy(dtype=float),
                "train_time": train_time,
                "train_event": train_event,
                "matched_vlsa": target["pathology_predicted_time"].to_numpy(dtype=float),
                "mcat_conch": mcat["predicted_time"].to_numpy(dtype=float),
                "cat_tpl_conch": cat["predicted_time"].to_numpy(dtype=float),
                "qaclf": target["fused_predicted_time"].to_numpy(dtype=float),
            }
            scores = {
                method: score(
                    record[method], record["time"], record["event"], train_time, train_event
                )
                for method in METHODS
            }
            dataset_folds.append(record)
            fold_rows.append(
                {
                    "dataset": cohort.upper(),
                    "fold": fold,
                    "n": len(patient_ids),
                    "events": int(record["event"].sum()),
                    **{f"{method}_c_index": scores[method] for method in METHODS},
                    "qaclf_vs_matched": scores["qaclf"] - scores["matched_vlsa"],
                    "qaclf_vs_mcat": scores["qaclf"] - scores["mcat_conch"],
                    "qaclf_vs_cat": scores["qaclf"] - scores["cat_tpl_conch"],
                }
            )
            for idx, patient_id in enumerate(patient_ids):
                patient_rows.append(
                    {
                        "dataset": cohort.upper(),
                        "fold": fold,
                        "patient_id": patient_id,
                        "event_time": record["time"][idx],
                        "event": record["event"][idx],
                        **{f"{method}_predicted_time": record[method][idx] for method in METHODS},
                    }
                )

            for model, run_dir, values in (
                ("mcat_conch", mcat_dir, scores["mcat_conch"]),
                ("cat_tpl_conch", cat_dir, scores["cat_tpl_conch"]),
            ):
                summary = pd.read_csv(run_dir / "summary.csv")
                author_value = float(summary.loc[summary["folds"].astype(str) == str(fold), "val_cindex"].iloc[0])
                author_metric_max_delta = max(
                    author_metric_max_delta, abs(values - author_value)
                )
            cohort_audit["folds"].append(
                {
                    "fold": fold,
                    "patients": len(patient_ids),
                    "events": int(record["event"].sum()),
                    "patient_sets_equal": patient_sets_equal,
                    "labels_equal": label_equal,
                    "mcat_prediction": str(mcat_path),
                    "cat_prediction": str(cat_path),
                }
            )

        current_folds = pd.DataFrame(fold_rows)
        current_folds = current_folds[current_folds["dataset"] == cohort.upper()]
        row = {
            "dataset": cohort.upper(),
            "paper_vlsa": PAPER_VLSA[cohort],
        }
        for method in METHODS:
            values = current_folds[f"{method}_c_index"].to_numpy(dtype=float)
            row[method] = float(values.mean())
            row[f"{method}_std"] = float(values.std(ddof=0))
        row.update(
            {
                "qaclf_vs_matched": row["qaclf"] - row["matched_vlsa"],
                "qaclf_vs_mcat": row["qaclf"] - row["mcat_conch"],
                "qaclf_vs_cat": row["qaclf"] - row["cat_tpl_conch"],
            }
        )
        dataset_rows.append(row)

        cohort_bootstrap = paired_bootstrap_all(
            dataset_folds, args.bootstrap, args.seed + cohort_index * 100
        )
        for name, candidate, reference in COMPARISONS:
            deltas = cohort_bootstrap[name]
            bootstrap_values[name].append(deltas)
            observed = row[candidate] - row[reference]
            bootstrap_rows.append(
                {
                    "dataset": cohort.upper(),
                    "comparison": name,
                    "candidate": candidate,
                    "reference": reference,
                    "observed_delta": observed,
                    "positive_folds": int(
                        (
                            current_folds[f"{candidate}_c_index"]
                            > current_folds[f"{reference}_c_index"]
                        ).sum()
                    ),
                    "bootstrap_samples": len(deltas),
                    "delta_ci025": float(np.quantile(deltas, 0.025)),
                    "delta_ci975": float(np.quantile(deltas, 0.975)),
                    "probability_delta_gt_zero": float(np.mean(deltas > 0.0)),
                }
            )
        cohort_audit["external_author_vs_unified_max_abs_delta"] = author_metric_max_delta
        audit["datasets"][cohort] = cohort_audit

    dataset_summary = pd.DataFrame(dataset_rows)
    macro = {"dataset": "MACRO", "paper_vlsa": float(dataset_summary["paper_vlsa"].mean())}
    for method in METHODS:
        macro[method] = float(dataset_summary[method].mean())
        macro[f"{method}_std"] = float(dataset_summary[method].std(ddof=0))
    macro.update(
        {
            "qaclf_vs_matched": macro["qaclf"] - macro["matched_vlsa"],
            "qaclf_vs_mcat": macro["qaclf"] - macro["mcat_conch"],
            "qaclf_vs_cat": macro["qaclf"] - macro["cat_tpl_conch"],
        }
    )
    dataset_summary = pd.concat([dataset_summary, pd.DataFrame([macro])], ignore_index=True)

    for name, candidate, reference in COMPARISONS:
        aligned = min(len(values) for values in bootstrap_values[name])
        macro_deltas = np.mean(
            np.stack([values[:aligned] for values in bootstrap_values[name]], axis=0),
            axis=0,
        )
        fold_frame = pd.DataFrame(fold_rows)
        bootstrap_rows.append(
            {
                "dataset": "MACRO",
                "comparison": name,
                "candidate": candidate,
                "reference": reference,
                "observed_delta": macro[candidate] - macro[reference],
                "positive_folds": int(
                    (fold_frame[f"{candidate}_c_index"] > fold_frame[f"{reference}_c_index"]).sum()
                ),
                "bootstrap_samples": aligned,
                "delta_ci025": float(np.quantile(macro_deltas, 0.025)),
                "delta_ci975": float(np.quantile(macro_deltas, 0.975)),
                "probability_delta_gt_zero": float(np.mean(macro_deltas > 0.0)),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    paired_predictions = pd.DataFrame(patient_rows)
    bootstrap_summary = pd.DataFrame(bootstrap_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics_unified.csv", index=False)
    paired_predictions.to_csv(output_dir / "paired_patient_predictions.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_summary_unified.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "paired_bootstrap_unified.csv", index=False)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("===== Unified five-dataset comparison =====")
    print(
        dataset_summary[
            ["dataset", "paper_vlsa", "matched_vlsa", "mcat_conch", "cat_tpl_conch", "qaclf"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("===== QACLF paired bootstrap =====")
    print(
        bootstrap_summary[
            bootstrap_summary["comparison"].str.startswith("qaclf_vs_")
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
