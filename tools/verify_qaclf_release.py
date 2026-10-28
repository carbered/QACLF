#!/usr/bin/env python3
"""Verify QACLF integrity; retain this historical entry point for compatibility."""

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_RELEASE_BYTES = 50 * 1024 * 1024
MAX_FILE_BYTES = 25 * 1024 * 1024
EXPECTED_MACRO = 0.7050168897889689
EXPECTED_BACKBONE_SEED_MACRO = {
    42: (0.695781808041706, 0.7052604568447455, 0.009478648803039577),
    3404: (0.6862988123962727, 0.6996368963272379, 0.013338083930965236),
    3405: (0.6796182431304525, 0.6936821721937219, 0.014063929063269425),
}
PRIMARY_RESULTS = {
    "blca": "blca-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    "brca": "brca-frozen-pathomic-fusion-outer5-20260820",
    "gbmlgg": "gbmlgg-frozen-pathomic-fusion-outer5-20260820",
    "luad": "luad-frozen-pathomic-fusion-cloud-baseline-outer5-20260820",
    "ucec": "ucec-frozen-pathomic-fusion-outer5-20260820",
}
FORBIDDEN_DIRS = {"wandb", "logs", ".idea", "venv"}
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt"}
OMITTED_OMICS_TABLES = {
    f"tcga_{cohort}_all_clean.csv.zip" for cohort in PRIMARY_RESULTS
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest():
    manifest_path = ROOT / "manifest" / "SHA256SUMS.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    checked = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            path = ROOT / row["relative_path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(row["size_bytes"]):
                raise RuntimeError(f"Size mismatch: {path}")
            actual = sha256(path)
            if actual != row["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            checked += 1
    return checked


def verify_scope():
    violations = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        # Python may create untracked __pycache__ directories after first use.
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            violations.append(str(relative))
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(str(relative))
        if path.is_file() and path.name in OMITTED_OMICS_TABLES:
            violations.append(str(relative))
        if "smoke" in path.name.lower():
            violations.append(str(relative))
    if violations:
        raise RuntimeError(f"Release contains excluded artifacts: {violations[:10]}")


def verify_package_size():
    files = [path for path in ROOT.rglob("*") if path.is_file()
             and ".git" not in path.relative_to(ROOT).parts
             and "__pycache__" not in path.relative_to(ROOT).parts
             and path.suffix not in {".pyc", ".pyo"}]
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes >= MAX_RELEASE_BYTES:
        raise RuntimeError(
            f"GitHub package is {total_bytes / 1024 / 1024:.2f} MiB; "
            f"expected less than {MAX_RELEASE_BYTES / 1024 / 1024:.0f} MiB."
        )
    oversized = [path for path in files if path.stat().st_size >= MAX_FILE_BYTES]
    if oversized:
        raise RuntimeError(f"Files exceed the 25 MiB limit: {oversized}")
    return len(files), total_bytes


def verify_primary_result():
    values = {}
    for cohort, directory in PRIMARY_RESULTS.items():
        path = ROOT / "result" / directory / "summary.json"
        with path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        values[cohort] = float(summary["fold_mean_fused_c_index"])
    macro = sum(values.values()) / len(values)
    if not math.isclose(macro, EXPECTED_MACRO, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Locked macro mismatch: expected {EXPECTED_MACRO:.12f}, got {macro:.12f}"
        )
    return values, macro


def verify_required_inputs():
    required = [
        ROOT / "README.md",
        ROOT / "README_中文.md",
        ROOT / "DATA.md",
        ROOT / "NOTICE.md",
        ROOT / "docs" / "CURRENT_RESULTS.md",
        ROOT / "docs" / "references.bib",
        ROOT / "figures" / "qaclf_current" / "QACLF_method_overview.pdf",
        ROOT / "figures" / "qaclf_current" / "QACLF_eight_method_comparison.pdf",
        ROOT / "figures" / "qaclf_current" / "QACLF_ablation_sensitivity.pdf",
        ROOT / "figures" / "qaclf_current" / "QACLF_unified_probability.pdf",
        ROOT / "figures" / "qaclf_current" / "QACLF_omics_robustness.pdf",
        ROOT / "figures" / "qaclf_current" / "QACLF_oof_km_risk_stratification.pdf",
        ROOT / "result" / "qaclf-km-risk-stratification-20260828" / "risk_stratification_stats.csv",
        ROOT / "result" / "qaclf-clinical-adjustment-20260828" / "clinical_adjustment_summary.csv",
        ROOT / "result" / "qaclf-clinical-adjustment-20260828" / "clinical_adjustment_audit.json",
        ROOT / "result" / "external_multimodal_comparison_20260821" / "unified" / "dataset_summary_unified.csv",
        ROOT / "result" / "qaclf-ablation-sensitivity-20260821" / "dataset_summary_ablation.csv",
        ROOT / "result" / "qaclf-pca-sensitivity-20260821" / "dataset_summary.csv",
        ROOT / "result" / "qaclf-omics-robustness-20260821" / "dataset_summary.csv",
        ROOT / "result" / "qaclf-seed-stability-20260821" / "stability_summary.csv",
        ROOT / "result" / "qaclf-backbone-seed-stability-20260829" / "dataset_seed_summary.csv",
        ROOT / "result" / "qaclf-backbone-seed-stability-20260829" / "seed_macro_summary.csv",
        ROOT / "result" / "qaclf-backbone-seed-stability-20260829" / "protocol_and_audit.json",
        ROOT / "tools" / "evaluate_qaclf_backbone_seed_stability.py",
        ROOT / "external" / "CAT-with-TPL-master" / "dataset_csv" / "signatures.csv",
        ROOT / "manifest" / "omitted_data_sha256.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required release files: {missing}")


def verify_backbone_seed_stability():
    result_dir = ROOT / "result" / "qaclf-backbone-seed-stability-20260829"
    with (result_dir / "seed_macro_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(EXPECTED_BACKBONE_SEED_MACRO):
        raise RuntimeError(f"Unexpected pathology-seed rows: {len(rows)}")
    for row in rows:
        seed = int(row["pathology_seed"])
        if seed not in EXPECTED_BACKBONE_SEED_MACRO:
            raise RuntimeError(f"Unexpected pathology seed: {seed}")
        actual = (
            float(row["pathology_macro_c_index"]),
            float(row["fused_macro_c_index"]),
            float(row["macro_delta"]),
        )
        expected = EXPECTED_BACKBONE_SEED_MACRO[seed]
        if any(
            not math.isclose(a, e, rel_tol=0.0, abs_tol=1e-12)
            for a, e in zip(actual, expected)
        ):
            raise RuntimeError(
                f"Pathology-seed summary mismatch for seed {seed}: "
                f"expected {expected}, got {actual}"
            )
        if int(row["positive_datasets"]) != 5:
            raise RuntimeError(f"Seed {seed} does not retain 5/5 positive cohorts.")
    with (result_dir / "dataset_seed_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        dataset_rows = list(csv.DictReader(stream))
    if len(dataset_rows) != 15 or not all(float(row["delta"]) > 0 for row in dataset_rows):
        raise RuntimeError("Expected 15/15 positive pathology-seed-by-cohort deltas.")
    with (result_dir / "protocol_and_audit.json").open(
        "r", encoding="utf-8"
    ) as stream:
        audit = json.load(stream)
    reproduction = audit["reproduction_audit"]
    if not reproduction["within_documented_dataset_mean_tolerance"]:
        raise RuntimeError("Backbone-seed refit exceeded the documented tolerance.")
    return rows


def main():
    verify_required_inputs()
    verify_scope()
    file_count, total_bytes = verify_package_size()
    checked = verify_manifest()
    values, macro = verify_primary_result()
    backbone_rows = verify_backbone_seed_stability()
    from verify_qaclf_results import verify
    current = verify()
    print(f"current_evidence_verified={current}")
    print("QACLF clean release verification passed.")
    print(f"checked_files={checked}")
    print(f"package_files={file_count}")
    print(f"package_size_mib={total_bytes / 1024 / 1024:.2f}")
    print(f"locked_dataset_c_indices={values}")
    print(f"locked_macro_c_index={macro:.12f}")
    print(f"verified_pathology_training_seeds={len(backbone_rows)}")
    print("positive_pathology_seed_dataset_pairs=15/15")
    print("large_omics_tables_bundled=0")
    print("forbidden_training_artifacts=0")


if __name__ == "__main__":
    main()
