"""Audit locked-patient pathology inputs without changing experimental results."""

import hashlib
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/qaclf-dataset-resource-audit"
COHORTS = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=Path("E:/VLSA-Datasets"))
    parser.add_argument("--config-root", type=Path,
                        default=ROOT / "downloads/QACLF_DSMIL_results_no_weights/result/additional-vlsa-baselines/seed_42/dsmil")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    locked_path = ROOT / "result/external_multimodal_comparison_20260821/unified/paired_patient_predictions.csv"
    locked = pd.read_csv(locked_path)
    complexity = pd.read_csv(ROOT / "result/qaclf-incremental-complexity-20260821/complexity_summary.csv").set_index("dataset")
    summaries, patient_rows, slide_rows, checks, omitted = [], [], [], [], []
    for name in COHORTS:
        cohort = name.lower()
        cases = locked[locked.dataset == name].copy()
        assert not cases.patient_id.duplicated().any()
        label_path = ROOT / f"data_split/5foldcv/tcga_{cohort}/mahmoodlab_tcga_{cohort}_survival.csv"
        labels = pd.read_csv(label_path)
        labels = labels[labels.patient_id.isin(cases.patient_id)].copy()
        assert set(labels.patient_id) == set(cases.patient_id)
        assert not labels.pathology_id.duplicated().any()
        counts = {}
        for i, row in enumerate(labels.itertuples()):
            base = args.features_root / f"tcga_{cohort}"
            feature = base / "feats-CONCH-vl-proj/pt_files" / f"{row.pathology_id}.pt"
            coord = base / "patches" / f"{row.pathology_id}.h5"
            if not feature.is_file():
                omitted.append({"dataset": name, "slide": row.pathology_id,
                                "reason": "No feature file; skipped by WSIPatchSurv.__getitem__."})
                continue
            feat = torch.load(feature, map_location="cpu", mmap=True, weights_only=True)
            n = int(feat.shape[0])
            assert tuple(feat.shape) == (n, 512), (feature, feat.shape)
            if coord.is_file():
                with h5py.File(coord, "r") as handle:
                    assert int(handle["coords"].shape[0]) == n
            assert n > 0
            # Cross-check the first and last slide and every hundredth slide.
            if i % 100 == 0 or i == len(labels) - 1:
                checks.append({"slide": row.pathology_id, "shape": list(feat.shape)})
            counts[row.pathology_id] = n
            slide_rows.append({"dataset": name, "patient_id": row.patient_id,
                               "slide_id": row.pathology_id, "patches": n})
        labels = labels[labels.pathology_id.isin(counts)].copy()
        assert set(labels.patient_id) == set(cases.patient_id)
        labels["patches"] = labels.pathology_id.map(counts)
        grouped = labels.groupby("patient_id").agg(slides=("pathology_id", "size"), patches=("patches", "sum"))
        for pid, row in grouped.iterrows():
            patient_rows.append({"dataset": name, "patient_id": pid, "slides": int(row.slides), "patches": int(row.patches)})
        bins = []
        for fold in range(5):
            cfgpath = args.config_root / f"{cohort}-DSMIL-CONCH-fold_{fold}/config.yaml"
            config = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
            bins.append(int(config["time_bins"]))
            curves = np.load(ROOT / f"result/{cohort}-qaclf-survival-curves-20260820/fold_{fold}_qaclf_survival_curves.npz")
            assert len(curves["time_coordinates"]) == bins[-1]
        assert len(set(bins)) == 1
        summaries.append({"dataset": name, "patients": len(cases), "events": int(cases.event.sum()),
                          "slides": len(labels), "slides_per_patient": len(labels) / len(cases),
                          "patches_total": int(grouped.patches.sum()),
                          "patches_per_patient_mean": float(grouped.patches.mean()),
                          "patches_per_patient_median": float(grouped.patches.median()),
                          "patches_per_patient_min": int(grouped.patches.min()),
                          "patches_per_patient_max": int(grouped.patches.max()),
                          "time_bins": bins[0], "selected_omics": int(complexity.loc[name, "selected_omics"]),
                          "label_sha256": sha(label_path)})
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    pd.DataFrame(summaries).to_csv(out / "dataset_statistics.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(out / "patient_pathology_counts.csv", index=False)
    pd.DataFrame(slide_rows).to_csv(out / "slide_patch_counts.csv", index=False)
    report = {"locked_predictions_sha256": sha(locked_path), "patients": len(locked),
              "events": int(locked.event.sum()), "slides": len(slide_rows),
              "patches": sum(x["patches"] for x in slide_rows), "feature_shape_checks": checks,
              "omitted_slides": omitted,
              "count_source": "Every existing feature tensor shape; cross-checked against every available coordinate HDF5.",
              "scope": "Locked held-out patient union only; no training or primary result changes."}
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
