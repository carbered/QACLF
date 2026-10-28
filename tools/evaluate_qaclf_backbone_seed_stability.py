#!/usr/bin/env python3
"""Evaluate QACLF across fixed VLSA training seeds.

The pathology predictions vary with the VLSA training seed. The molecular
branch, PCA seed, Cox configuration, fusion weight, and outer folds remain
fixed so the resulting differences isolate pathology-backbone variability.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn


WORKSPACE = Path(__file__).resolve().parents[1]
COHORTS = ("blca", "brca", "gbmlgg", "luad", "ucec")
LOCKED_PATHOLOGY_SEED = 42
LOCKED_OMICS_SEED = 42


def resolve(path, base=WORKSPACE):
    path = Path(path)
    return path if path.is_absolute() else base / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_seed_inputs(seed_root, seed):
    expected = []
    for cohort in COHORTS:
        for fold in range(5):
            matches = list(
                seed_root.glob(
                    f"{cohort}-VLSA-CONCH-fold_{fold}-*-coop_rank"
                )
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {cohort} fold {fold} directory for seed "
                    f"{seed}, found {len(matches)}."
                )
            run_dir = matches[0]
            for filename in (
                "config.yaml",
                "train_metrics-last.txt",
                "bag_clf_train_last_pred_train.csv",
                "bag_clf_train_last_pred_test.csv",
            ):
                path = run_dir / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                expected.append(path)
    return expected


def run_evaluator(
    python,
    analysis_root,
    baseline_root,
    cohort,
    output_dir,
    bootstrap,
    steps,
    lr,
    l2,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(analysis_root / "tools" / "evaluate_frozen_pathomic_fusion_outer5.py"),
        "--dataset",
        f"tcga_{cohort}",
        "--baseline-root",
        str(baseline_root),
        "--alpha",
        "0.2",
        "--pca-components",
        "64",
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--l2",
        str(l2),
        "--seed",
        str(LOCKED_OMICS_SEED),
        "--bootstrap",
        str(bootstrap),
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command,
        cwd=str(analysis_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (output_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-50:])
        raise RuntimeError(f"QACLF evaluation failed for {cohort}:\n{tail}")
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def load_reference(analysis_root):
    path = (
        analysis_root
        / "result"
        / "qaclf-pca64-fulltable-reproduction-20260821"
        / "dataset_summary.csv"
    )
    reference = pd.read_csv(path)
    reference = reference[reference["dataset"] != "MACRO"].copy()
    return path, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root",
        default="release/QACLF_CMIG_clean_20260821_v6",
    )
    parser.add_argument(
        "--multiseed-root",
        default=(
            "tmp/vlsa_matched_multiseed_complete_20260829/result/"
            "matched-vlsa-multiseed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="result/qaclf-backbone-seed-stability-20260829",
    )
    parser.add_argument(
        "--archive",
        default=(
            "C:/Users/lzj/Desktop/"
            "vlsa_matched_multiseed_complete_20260829_080422.tar.gz"
        ),
    )
    parser.add_argument(
        "--pathology-seeds", type=int, nargs="+", default=[42, 3404, 3405]
    )
    parser.add_argument("--datasets", nargs="+", default=list(COHORTS))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.10)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    analysis_root = resolve(args.analysis_root)
    multiseed_root = resolve(args.multiseed_root)
    output_dir = resolve(args.output_dir)
    archive = resolve(args.archive)
    datasets = [dataset.lower().removeprefix("tcga_") for dataset in args.datasets]
    invalid = sorted(set(datasets) - set(COHORTS))
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}")
    seeds = list(dict.fromkeys(args.pathology_seeds))
    if LOCKED_PATHOLOGY_SEED not in seeds:
        raise ValueError(
            f"Locked pathology seed {LOCKED_PATHOLOGY_SEED} must be included."
        )
    evaluator = analysis_root / "tools" / "evaluate_frozen_pathomic_fusion_outer5.py"
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_audit = []
    for seed in seeds:
        seed_root = multiseed_root / f"seed_{seed}"
        files = audit_seed_inputs(seed_root, seed)
        input_audit.append(
            {
                "pathology_seed": seed,
                "run_directories": 25,
                "required_files": len(files),
            }
        )

    summary_rows = []
    fold_frames = []
    for seed in seeds:
        seed_root = multiseed_root / f"seed_{seed}"
        for cohort in datasets:
            cohort_dir = output_dir / f"seed_{seed}" / cohort
            summary_path = cohort_dir / "summary.json"
            if args.reuse_existing and summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                print(f"[reuse] pathology_seed={seed} dataset={cohort}", flush=True)
            else:
                print(f"[run] pathology_seed={seed} dataset={cohort}", flush=True)
                summary = run_evaluator(
                    Path(sys.executable),
                    analysis_root,
                    seed_root,
                    cohort,
                    cohort_dir,
                    args.bootstrap,
                    args.steps,
                    args.lr,
                    args.l2,
                )
            summary_rows.append(
                {
                    "pathology_seed": seed,
                    "dataset": cohort.upper(),
                    "pathology_c_index": summary["fold_mean_baseline_c_index"],
                    "omics_c_index": summary["fold_mean_omics_c_index"],
                    "fused_c_index": summary["fold_mean_fused_c_index"],
                    "delta": summary["fold_mean_delta"],
                    "positive_folds": summary["positive_folds"],
                    "bootstrap_samples": summary["bootstrap_samples"],
                    "delta_ci025": summary["delta_ci025"],
                    "delta_ci975": summary["delta_ci975"],
                    "probability_delta_gt_zero": summary[
                        "probability_delta_gt_zero"
                    ],
                }
            )
            folds = pd.read_csv(cohort_dir / "fold_metrics.csv")
            folds.insert(0, "pathology_seed", seed)
            folds.insert(1, "dataset", cohort.upper())
            fold_frames.append(folds)
            print(
                f"[done] pathology_seed={seed} dataset={cohort} "
                f"pathology={summary['fold_mean_baseline_c_index']:.6f} "
                f"fused={summary['fold_mean_fused_c_index']:.6f} "
                f"delta={summary['fold_mean_delta']:+.6f}",
                flush=True,
            )

    dataset_seed = pd.DataFrame(summary_rows)
    seed_macro = (
        dataset_seed.groupby("pathology_seed", sort=False)
        .agg(
            completed_datasets=("dataset", "count"),
            pathology_macro_c_index=("pathology_c_index", "mean"),
            fused_macro_c_index=("fused_c_index", "mean"),
            macro_delta=("delta", "mean"),
            positive_datasets=("delta", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    across_seed = (
        dataset_seed.groupby("dataset", sort=False)
        .agg(
            pathology_seeds=("pathology_seed", "count"),
            mean_pathology_c_index=("pathology_c_index", "mean"),
            std_pathology_c_index=(
                "pathology_c_index", lambda values: values.std(ddof=0)
            ),
            mean_fused_c_index=("fused_c_index", "mean"),
            std_fused_c_index=("fused_c_index", lambda values: values.std(ddof=0)),
            mean_delta=("delta", "mean"),
            std_delta=("delta", lambda values: values.std(ddof=0)),
            positive_seed_dataset_pairs=(
                "delta", lambda values: int((values > 0).sum())
            ),
        )
        .reset_index()
    )
    macro_across_seed = {
        "pathology_seeds": len(seeds),
        "dataset_seed_pairs": len(dataset_seed),
        "mean_pathology_macro_c_index": float(
            seed_macro["pathology_macro_c_index"].mean()
        ),
        "std_pathology_macro_c_index": float(
            seed_macro["pathology_macro_c_index"].std(ddof=0)
        ),
        "mean_fused_macro_c_index": float(seed_macro["fused_macro_c_index"].mean()),
        "std_fused_macro_c_index": float(
            seed_macro["fused_macro_c_index"].std(ddof=0)
        ),
        "mean_macro_delta": float(seed_macro["macro_delta"].mean()),
        "std_macro_delta": float(seed_macro["macro_delta"].std(ddof=0)),
        "positive_seed_dataset_pairs": int((dataset_seed["delta"] > 0).sum()),
        "positive_pathology_seeds": int((seed_macro["macro_delta"] > 0).sum()),
    }

    reference_path, reference = load_reference(analysis_root)
    locked = dataset_seed[
        dataset_seed["pathology_seed"] == LOCKED_PATHOLOGY_SEED
    ].merge(
        reference[["dataset", "mean_fused_c_index"]],
        on="dataset",
        how="left",
        validate="one_to_one",
    )
    locked["fused_abs_delta_vs_reference"] = np.abs(
        locked["fused_c_index"] - locked["mean_fused_c_index"]
    )
    reproduction_audit = {
        "reference": reference_path.relative_to(analysis_root).as_posix(),
        "locked_pathology_seed": LOCKED_PATHOLOGY_SEED,
        "max_abs_fused_c_index_delta": float(
            locked["fused_abs_delta_vs_reference"].max()
        ),
        "bitwise_exact_within_1e12": bool(
            locked["fused_abs_delta_vs_reference"].max() <= 1e-12
        ),
        "documented_dataset_mean_tolerance": 0.001,
        "within_documented_dataset_mean_tolerance": bool(
            locked["fused_abs_delta_vs_reference"].max() <= 0.001
        ),
    }

    pd.concat(fold_frames, ignore_index=True).to_csv(
        output_dir / "fold_metrics_all_seeds.csv", index=False
    )
    dataset_seed.to_csv(output_dir / "dataset_seed_summary.csv", index=False)
    seed_macro.to_csv(output_dir / "seed_macro_summary.csv", index=False)
    across_seed.to_csv(output_dir / "dataset_across_seed_summary.csv", index=False)
    locked.to_csv(output_dir / "locked_seed42_reproduction.csv", index=False)
    protocol = {
        "purpose": "pathology_backbone_training_seed_stability",
        "pathology_seeds": seeds,
        "locked_pathology_seed": LOCKED_PATHOLOGY_SEED,
        "fixed_omics_seed": LOCKED_OMICS_SEED,
        "datasets": [cohort.upper() for cohort in datasets],
        "folds": list(range(5)),
        "pca_components": 64,
        "cox_steps": args.steps,
        "cox_lr": args.lr,
        "cox_l2": args.l2,
        "pathology_weight": 0.80,
        "omics_weight": 0.20,
        "bootstrap_per_dataset_seed": args.bootstrap,
        "single_thread": True,
        "outer_test_used_for_selection": False,
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "input_audit": input_audit,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "reproduction_audit": reproduction_audit,
        "macro_across_seed": macro_across_seed,
    }
    (output_dir / "protocol_and_audit.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("===== QACLF pathology-backbone seed summary =====")
    print(seed_macro.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("===== Across-seed summary =====")
    print(json.dumps(macro_across_seed, indent=2))
    print("===== Locked seed reproduction =====")
    print(json.dumps(reproduction_audit, indent=2))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
