#!/usr/bin/env python3
"""Run fixed QACLF PCA64-Cox evaluation across prespecified PCA seeds."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("blca", "brca", "gbmlgg", "luad", "ucec")
LOCKED_SEED = 42


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="result/qaclf-seed-stability-20260821"
    )
    parser.add_argument("--omics-table-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(COHORTS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 3404, 3405])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.10)
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    if LOCKED_SEED not in seeds:
        raise ValueError(f"The locked seed {LOCKED_SEED} must be included.")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    fold_frames = []
    dataset_frames = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        command = [
            sys.executable,
            str(ROOT / "tools" / "evaluate_qaclf_pca_sensitivity.py"),
            "--output-dir",
            str(seed_dir),
            "--pca-components",
            "64",
            "--seed",
            str(seed),
            "--steps",
            str(args.steps),
            "--lr",
            str(args.lr),
            "--l2",
            str(args.l2),
            "--datasets",
            *args.datasets,
            "--folds",
            *[str(fold) for fold in args.folds],
        ]
        if args.omics_table_dir is not None:
            command.extend(["--omics-table-dir", str(resolve(args.omics_table_dir))])
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (output_dir / f"seed_{seed}.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-30:])
            raise RuntimeError(f"Seed {seed} failed:\n{tail}")
        fold = pd.read_csv(seed_dir / "fold_metrics.csv")
        fold["pca_random_seed"] = seed
        fold_frames.append(fold)
        dataset = pd.read_csv(seed_dir / "dataset_summary.csv")
        dataset = dataset[dataset["dataset"] != "MACRO"].copy()
        dataset["pca_random_seed"] = seed
        dataset_frames.append(dataset)
        print(f"[seed stability] completed seed={seed}", flush=True)

    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    seed_dataset = pd.concat(dataset_frames, ignore_index=True)
    seed_dataset = seed_dataset[
        [
            "dataset",
            "pca_random_seed",
            "mean_omics_c_index",
            "std_omics_c_index",
            "mean_fused_c_index",
            "std_fused_c_index",
            "mean_explained_variance",
            "total_fit_seconds",
        ]
    ]
    seed_macro = (
        seed_dataset.groupby("pca_random_seed", sort=False)
        .agg(
            mean_omics_c_index=("mean_omics_c_index", "mean"),
            std_omics_c_index=("mean_omics_c_index", lambda values: values.std(ddof=0)),
            mean_fused_c_index=("mean_fused_c_index", "mean"),
            std_fused_c_index=("mean_fused_c_index", lambda values: values.std(ddof=0)),
            mean_explained_variance=("mean_explained_variance", "mean"),
            total_fit_seconds=("total_fit_seconds", "sum"),
        )
        .reset_index()
    )
    seed_macro.insert(0, "dataset", "MACRO")
    seed_dataset = pd.concat([seed_dataset, seed_macro], ignore_index=True)

    stability = (
        seed_dataset.groupby("dataset", sort=False)
        .agg(
            seeds=("pca_random_seed", "count"),
            mean_fused_c_index=("mean_fused_c_index", "mean"),
            seed_std_fused_c_index=(
                "mean_fused_c_index", lambda values: values.std(ddof=0)
            ),
            min_fused_c_index=("mean_fused_c_index", "min"),
            max_fused_c_index=("mean_fused_c_index", "max"),
        )
        .reset_index()
    )
    locked = seed_dataset[seed_dataset["pca_random_seed"] == LOCKED_SEED][
        ["dataset", "mean_fused_c_index"]
    ].rename(columns={"mean_fused_c_index": "locked_seed42_c_index"})
    stability = stability.merge(locked, on="dataset", how="left")
    stability["seed_range"] = (
        stability["max_fused_c_index"] - stability["min_fused_c_index"]
    )
    stability["mean_delta_vs_locked_seed42"] = (
        stability["mean_fused_c_index"] - stability["locked_seed42_c_index"]
    )

    fold_metrics.to_csv(output_dir / "fold_metrics_all_seeds.csv", index=False)
    seed_dataset.to_csv(output_dir / "seed_dataset_summary.csv", index=False)
    stability.to_csv(output_dir / "stability_summary.csv", index=False)
    protocol = {
        "locked_primary_seed": LOCKED_SEED,
        "evaluated_seeds": seeds,
        "seed_selected_from_outer_test": False,
        "best_seed_used_for_primary_result": False,
        "all_other_parameters_fixed": True,
        "pca_components": 64,
        "alpha": 0.20,
        "single_thread": True,
        "purpose": "randomized_PCA_numerical_stability_only",
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("===== QACLF seed-level macro C-index =====")
    print(
        seed_dataset[seed_dataset["dataset"] == "MACRO"]
        .loc[:, ["pca_random_seed", "mean_fused_c_index", "total_fit_seconds"]]
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("===== QACLF seed stability =====")
    print(stability.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
