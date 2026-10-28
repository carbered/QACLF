"""Verify the downloaded ILRA runs before extending the manuscript comparison."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import sys
import tarfile
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import IntegrationWarning
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.label_converter import MetaSurvData
from eval.SurvivalEVAL.Evaluations.Concordance import concordance
from eval.SurvivalEVAL.Evaluations.util import predict_mean_survival_time

ARCHIVE = ROOT / "downloads/QACLF_ILRA_results.tar.gz"
EXTRACT = ROOT / "downloads/QACLF_ILRA_results_no_weights"
OUT = ROOT / "result/qaclf-seven-method-comparison"
COHORTS = ["blca", "brca", "gbmlgg", "luad", "ucec"]
EXPECTED_SHA = "a621c06dde207c9794eed548f472503d4bac0d77a961394ba7561cc5d7a3b6e2"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_inputs():
    assert sha(ARCHIVE) == EXPECTED_SHA
    EXTRACT.mkdir(parents=True, exist_ok=True)
    skipped = []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            target = (EXTRACT / member.name).resolve()
            assert target.is_relative_to(EXTRACT.resolve()), member.name
            assert member.isfile() or member.isdir(), member.name
            if member.name.endswith((".pth", ".pt")):
                skipped.append(member.name)
                continue
            if target.is_file():
                with archive.extractfile(member) as source:
                    assert hashlib.sha256(source.read()).hexdigest() == sha(target)
            else:
                archive.extract(member, path=EXTRACT, filter="data")
    return skipped


def main():
    skipped = extract_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    references = pd.read_csv(ROOT / "result/external_multimodal_comparison_20260821/unified/paired_patient_predictions.csv",
                             dtype={"patient_id": str})
    records, patients = [], []
    audit = {"archive_sha256": EXPECTED_SHA, "checkpoint_files_kept_in_original_archive": len(skipped),
             "evaluator": "Vendored SurvivalEVAL Mean/Linear, continuous labels, Comparable/All",
             "metric_tolerance": 1e-6, "curve_bound_tolerance": 1e-6,
             "curves_modified_for_recomputation": False, "fold_standard_deviation_ddof": 0,
             "source_sha256": {}, "source_hashes": None}
    run_root = EXTRACT / "result/additional-vlsa-baselines/seed_42/ilra"
    for cohort in COHORTS:
        for fold in range(5):
            folder = run_root / (cohort + "-ILRA-CONCH-fold_" + str(fold))
            protocol = json.loads((folder / "additional_baseline_protocol.json").read_text())
            config = yaml.safe_load((folder / "config.yaml").read_text())
            generated = EXTRACT / "config/IFMLE/additional_vlsa_baselines/ilra" / ("cfg_ilra_conch_" + cohort + "_fold" + str(fold) + "_seed42.yaml")
            split_path = Path("data_split/5foldcv") / ("tcga_" + cohort) / ("splits_" + str(fold) + ".csv")
            label_path = Path("data_split/5foldcv") / ("tcga_" + cohort) / ("mahmoodlab_tcga_" + cohort + "_survival.csv")
            assert sha(generated) == protocol["config_sha256"]
            assert sha(EXTRACT / split_path) == sha(ROOT / split_path) == protocol["split_sha256"]
            assert sha(EXTRACT / label_path) == sha(ROOT / label_path) == protocol["labels_sha256"]
            assert protocol["model"] == "ILRA" and protocol["dataset"] == cohort and protocol["fold"] == fold
            assert protocol["seed"] == 42 and protocol["checkpoint"] == "raw_last"
            if audit["source_hashes"] is None:
                audit["source_hashes"] = protocol["source_hashes"]
            assert protocol["source_hashes"] == audit["source_hashes"]
            for name, digest in protocol["source_hashes"].items():
                if (EXTRACT / name).is_file():
                    assert sha(EXTRACT / name) == digest, name
            assert sha(EXTRACT / "tools/run_vlsa_ilra_baseline.py") == protocol["runner_sha256"]
            assert config["deepmil_network"] == "ILRA" and config["arch"] == "DeepMIL"
            assert config["seed"] == 42 and config["ckpt_for_eval"] == "last"
            assert config["deepmil_use_feat_proj"] is False and not config["es"] and not config["lrs"]
            assert config["epochs"] == (5 if cohort == "gbmlgg" else 10)
            assert config["bp_every_batch"] == (16 if cohort == "gbmlgg" else 32)
            assert config["loss_type"] == "SurvIFMLE" and config["evaluator"] == "NLL-IF"
            assert config["opt_lr"] == .0002 and config["opt_weight_decay"] == .00001
            assert "feats-CONCH-vl-proj/pt_files" in config["path_patch"]
            assert [config[k] for k in ["deepmil_num_layers", "deepmil_num_heads", "deepmil_topk", "deepmil_ln"]] == [2, 8, 1, False]
            split = pd.read_csv(EXTRACT / split_path, dtype=str)
            columns = [c for c in split if not c.startswith("Unnamed")]
            train_ids = split[columns[0]].dropna().tolist()
            test_ids = split[columns[1]].dropna().tolist()
            assert not set(train_ids) & set(test_ids)
            target = references[(references.dataset == cohort.upper()) & (references.fold == fold)].set_index("patient_id")
            assert target.index.is_unique and set(target.index) == set(test_ids)
            prediction = folder / "bag_clf_train_last_pred_test.csv"
            pred = pd.read_csv(prediction, dtype={"patient_id": str}).set_index("patient_id")
            assert pred.index.is_unique and set(pred.index) == set(test_ids)
            pred = pred.loc[target.index]
            assert np.array_equal(pred.e.to_numpy(), target.event.to_numpy())
            with redirect_stdout(StringIO()):
                meta = MetaSurvData(str(EXTRACT / label_path), data_split={"train": train_ids, "test": test_ids}, verbose=False)
                meta.generate_discrete_label(num_bins=config["time_bins"], use_quantiles=False, summary=False)
            labels = meta.pat_data.set_index("patient_id")
            assert np.allclose(labels.loc[test_ids, "t"], target.loc[test_ids, "event_time"], atol=1e-4, rtol=0)
            assert np.array_equal(pred.t.to_numpy(), labels.loc[pred.index, "y_t"].to_numpy())
            curves = pred[[c for c in pred if c.startswith("surf_")]].to_numpy()
            assert curves.shape[1] == len(meta.time_coordinates)
            assert np.isfinite(curves).all() and curves.min() >= -1e-6 and curves.max() <= 1 + 1e-6
            assert np.diff(curves, axis=1).max() <= 1e-6
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IntegrationWarning)
                times = np.asarray([predict_mean_survival_time(curve, meta.time_coordinates, interpolation="Linear") for curve in curves])
            assert np.isfinite(times).all()
            train = labels.loc[train_ids]
            metric = float(concordance(times, target.event_time.to_numpy(), target.event.to_numpy(), train.t.to_numpy(), train.e.to_numpy(), pair_method="Comparable", ties="All")[0])
            logged = float(re.search(r"test/pred_c_index\s+-->\s+([0-9.eE+-]+)", (folder / "train_metrics-last.txt").read_text()).group(1))
            assert abs(metric - logged) < 1e-6, (cohort, fold, metric, logged)
            log = (EXTRACT / "logs/additional_vlsa_baselines/ilra" / (cohort + "_fold" + str(fold) + "_seed42.log")).read_text(errors="replace")
            epoch = config["epochs"]
            assert re.search(r"Epoch\s*\[" + str(epoch) + r"/" + str(epoch) + r"\]", log), (cohort, fold, "final epoch missing")
            row = {"model": "ILRA-CONCH", "dataset": cohort.upper(), "fold": fold, "n_test": len(pred),
                   "events": int(target.event.sum()), "c_index": metric, "logged_c_index": logged,
                   "metric_abs_difference": abs(metric - logged), "epochs": epoch,
                   "integration_warnings": sum(issubclass(w.category, IntegrationWarning) for w in caught)}
            records.append(row)
            for patient, predicted_time in zip(target.index, times):
                patients.append({"model": "ILRA-CONCH", "dataset": cohort.upper(), "fold": fold, "patient_id": patient,
                                 "event_time": target.loc[patient, "event_time"], "event": target.loc[patient, "event"], "predicted_time": predicted_time})
            audit["source_sha256"][str(prediction.relative_to(ROOT))] = sha(prediction)
            print("PASS ILRA", cohort, fold, "C-index", metric, flush=True)
    folds = pd.DataFrame(records)
    summary = folds.groupby(["model", "dataset"], as_index=False).agg(completed_folds=("fold", "count"), mean_c_index=("c_index", "mean"), std_c_index=("c_index", lambda x: float(np.std(x, ddof=0))))
    original = pd.read_csv(EXTRACT / "result/additional-vlsa-baselines/summary/ilra_dataset_summary.csv").set_index("dataset")
    for row in summary.itertuples():
        assert row.completed_folds == 5
        assert abs(row.mean_c_index - original.loc[row.dataset, "mean_c_index"]) < 1e-6
        assert abs(row.std_c_index - original.loc[row.dataset, "std_c_index"]) < 1e-6
    assert abs(summary.mean_c_index.mean() - .6363855177059747) < 1e-6
    folds.to_csv(OUT / "ilra_fold_metrics.csv", index=False)
    summary.to_csv(OUT / "ilra_dataset_summary.csv", index=False)
    pd.DataFrame(patients).to_csv(OUT / "ilra_patient_predictions.csv", index=False)
    six = pd.read_csv(ROOT / "result/qaclf-six-method-comparison/six_method_dataset_summary.csv")
    row = {"method": "ILRA-CONCH", "modality": "pathology"}
    for cohort in COHORTS:
        result = summary[summary.dataset == cohort.upper()].iloc[0]
        row[cohort.upper()] = result.mean_c_index
        row[cohort.upper() + "_std"] = result.std_c_index
    row["MACRO"] = float(summary.mean_c_index.mean())
    seven = pd.concat([six.iloc[:2], pd.DataFrame([row]), six.iloc[2:]], ignore_index=True)
    seven.to_csv(OUT / "seven_method_dataset_summary.csv", index=False)
    audit.update({"complete_folds": len(folds), "unique_test_patients": len(patients),
                  "max_metric_abs_difference": float(folds.metric_abs_difference.max()),
                  "macro_c_index": row["MACRO"], "folds": records,
                  "MOTCat": "Not in five-cohort comparison; BLCA only, paused for computational budget.",
                  "external_protocol_note": "Existing MCAT/CAT full-cohort binning qualification retained unchanged.",
                  "inference": "No new p-value or confidence interval computed; ILRA comparison is descriptive."})
    (OUT / "ilra_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("Verified 25 ILRA folds; seven-method comparison exported.")


if __name__ == "__main__":
    main()
