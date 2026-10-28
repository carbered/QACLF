"""Audit verified cloud DSMIL predictions before adding manuscript results."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
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
from eval.SurvivalEVAL.Evaluations.Concordance import concordance
from eval.SurvivalEVAL.Evaluations.util import predict_mean_survival_time

ARCHIVE = ROOT / "downloads/QACLF_DSMIL_results.tar.gz"
EXTRACT = ROOT / "downloads/QACLF_DSMIL_results_no_weights"
OUT = ROOT / "result/qaclf-eight-method-comparison"
COHORTS = ["blca", "brca", "gbmlgg", "luad", "ucec"]
EXPECTED_SHA = "90c304a9067f7158d61b4ba00f729c8a074d077c2adb7858e0bbb56f42504a66"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_inputs():
    assert sha(ARCHIVE) == EXPECTED_SHA, "Cloud and local archive hashes must match"
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
    references = pd.read_csv(
        ROOT / "result/external_multimodal_comparison_20260821/unified/paired_patient_predictions.csv",
        dtype={"patient_id": str})
    audit = {"archive_sha256": EXPECTED_SHA, "remote_archive_sha256_verified": True,
             "checkpoint_files_kept_in_original_archive": len(skipped),
             "evaluator": "Vendored SurvivalEVAL Mean/Linear, continuous labels, Comparable/All",
             "metric_tolerance": 1e-6, "curve_bound_tolerance": 1e-6,
             "curves_modified_for_recomputation": False, "fold_standard_deviation_ddof": 0,
             "source_hashes": None, "prediction_sha256": {}}
    for name in ("eval/SurvivalEVAL/Evaluations/Concordance.py",
                 "eval/SurvivalEVAL/Evaluations/util.py"):
        assert sha(EXTRACT / name) == sha(ROOT / name), "Evaluator/label code drift: " + name
    # The local label converter has a later optional open-ended-bin extension.
    # Reproduce this run with the archived, hash-verified cloud implementation.
    spec = importlib.util.spec_from_file_location("verified_dsmil_cloud_labels", EXTRACT / "dataset/label_converter.py")
    archived_labels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived_labels)
    MetaSurvData = archived_labels.MetaSurvData
    audit["label_converter"] = "Exact archived cloud source; local optional extension not used."
    records, patients = [], []
    for cohort in COHORTS:
        for fold in range(5):
            folder = EXTRACT / "result/additional-vlsa-baselines/seed_42/dsmil" / (
                cohort + "-DSMIL-CONCH-fold_" + str(fold))
            protocol = json.loads((folder / "additional_baseline_protocol.json").read_text())
            done = json.loads((folder / "dsmil_completed.json").read_text())
            for name, digest in done["sha256"].items():
                assert sha(folder / name) == digest, name
            config = yaml.safe_load((folder / "config.yaml").read_text())
            generated = EXTRACT / "config/IFMLE/additional_vlsa_baselines/dsmil" / (
                "cfg_dsmil_conch_{}_fold{}_seed42.yaml".format(cohort, fold))
            split_path = Path("data_split/5foldcv") / ("tcga_" + cohort) / ("splits_" + str(fold) + ".csv")
            label_path = Path("data_split/5foldcv") / ("tcga_" + cohort) / ("mahmoodlab_tcga_" + cohort + "_survival.csv")
            assert sha(generated) == protocol["config_sha256"]
            assert sha(EXTRACT / split_path) == sha(ROOT / split_path) == protocol["split_sha256"]
            assert sha(EXTRACT / label_path) == sha(ROOT / label_path) == protocol["labels_sha256"]
            assert protocol["model"] == "DSMIL" and protocol["dataset"] == cohort and protocol["fold"] == fold
            assert protocol["seed"] == 42 and protocol["checkpoint"] == "raw_last"
            if audit["source_hashes"] is None:
                audit["source_hashes"] = protocol["source_hashes"]
            assert protocol["source_hashes"] == audit["source_hashes"]
            for name, digest in protocol["source_hashes"].items():
                mapped = {"deployment/driver": "tools/run_vlsa_dsmil_baseline.py",
                          "deployment/ilra_helpers": "tools/run_vlsa_ilra_baseline.py"}.get(name, name)
                assert sha(EXTRACT / mapped) == digest, mapped
            assert config["deepmil_network"] == "DSMIL" and config["arch"] == "DeepMIL"
            assert config["seed"] == 42 and config["ckpt_for_eval"] == "last"
            assert config["deepmil_use_feat_proj"] is False and not config["es"] and not config["lrs"]
            assert config["deepmil_drop_rate"] == .25
            assert config["epochs"] == (5 if cohort == "gbmlgg" else 10)
            assert config["bp_every_batch"] == (16 if cohort == "gbmlgg" else 32)
            assert config["loss_type"] == "SurvIFMLE" and config["evaluator"] == "NLL-IF"
            assert config["opt_lr"] == .0002 and config["opt_weight_decay"] == .00001
            assert "feats-CONCH-vl-proj/pt_files" in config["path_patch"]
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
                meta = MetaSurvData(str(EXTRACT / label_path),
                                    data_split={"train": train_ids, "test": test_ids}, verbose=False)
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
                times = np.asarray([predict_mean_survival_time(curve, meta.time_coordinates,
                                                              interpolation="Linear") for curve in curves])
            assert np.isfinite(times).all()
            train = labels.loc[train_ids]
            metric = float(concordance(times, target.event_time.to_numpy(), target.event.to_numpy(),
                                       train.t.to_numpy(), train.e.to_numpy(),
                                       pair_method="Comparable", ties="All")[0])
            logged = float(re.search(r"test/pred_c_index\s+-->\s+([0-9.eE+-]+)",
                                     (folder / "train_metrics-last.txt").read_text()).group(1))
            assert abs(metric - logged) < 1e-6 and abs(logged - done["c_index"]) < 1e-12
            log = (EXTRACT / "logs/additional_vlsa_baselines/dsmil" / (
                cohort + "_fold" + str(fold) + "_seed42.log")).read_text(errors="replace")
            epoch = config["epochs"]
            assert re.search(r"Epoch\s*\[" + str(epoch) + r"/" + str(epoch) + r"\]", log)
            row = {"model": "DSMIL-CONCH", "dataset": cohort.upper(), "fold": fold,
                   "n_test": len(pred), "events": int(target.event.sum()), "c_index": metric,
                   "logged_c_index": logged, "metric_abs_difference": abs(metric - logged),
                   "epochs": epoch, "wall_seconds": done["wall_seconds"],
                   "integration_warnings": sum(issubclass(w.category, IntegrationWarning) for w in caught)}
            records.append(row)
            for patient, predicted_time in zip(target.index, times):
                patients.append({"model": "DSMIL-CONCH", "dataset": cohort.upper(), "fold": fold,
                                 "patient_id": patient, "event_time": target.loc[patient, "event_time"],
                                 "event": target.loc[patient, "event"], "predicted_time": predicted_time})
            audit["prediction_sha256"][str(prediction.relative_to(ROOT))] = sha(prediction)
            print("PASS DSMIL", cohort, fold, "C-index", metric, flush=True)
    folds = pd.DataFrame(records)
    summary = folds.groupby(["model", "dataset"], as_index=False).agg(
        completed_folds=("fold", "count"), mean_c_index=("c_index", "mean"),
        std_c_index=("c_index", lambda x: float(np.std(x, ddof=0))))
    original = pd.read_csv(EXTRACT / "result/additional-vlsa-baselines/summary/dsmil_dataset_summary_blca_brca_gbmlgg_luad_ucec_folds_0_1_2_3_4.csv").set_index("dataset")
    for row in summary.itertuples():
        assert row.completed_folds == 5
        assert abs(row.mean_c_index - original.loc[row.dataset, "mean_c_index"]) < 1e-6
        assert abs(row.std_c_index - original.loc[row.dataset, "std_c_index"]) < 1e-6
    assert abs(summary.mean_c_index.mean() - .6591261613970387) < 1e-6
    patient_frame = pd.DataFrame(patients)
    assert len(patient_frame) == 2831 and not patient_frame.duplicated(["dataset", "patient_id"]).any()
    folds.to_csv(OUT / "dsmil_fold_metrics.csv", index=False)
    summary.to_csv(OUT / "dsmil_dataset_summary.csv", index=False)
    patient_frame.to_csv(OUT / "dsmil_patient_predictions.csv", index=False)
    seven = pd.read_csv(ROOT / "result/qaclf-seven-method-comparison/seven_method_dataset_summary.csv")
    row = {"method": "DSMIL-CONCH", "modality": "pathology"}
    for cohort in COHORTS:
        result = summary[summary.dataset == cohort.upper()].iloc[0]
        row[cohort.upper()] = result.mean_c_index
        row[cohort.upper() + "_std"] = result.std_c_index
    row["MACRO"] = float(summary.mean_c_index.mean())
    eight = pd.concat([seven.iloc[:3], pd.DataFrame([row]), seven.iloc[3:]], ignore_index=True)
    eight.to_csv(OUT / "eight_method_dataset_summary.csv", index=False)
    audit.update({"complete_folds": len(folds), "unique_test_patients": len(patients),
                  "max_metric_abs_difference": float(folds.metric_abs_difference.max()),
                  "macro_c_index": row["MACRO"], "folds": records,
                  "MOTCat": "BLCA only; excluded from five-cohort comparison. Original PID no longer live.",
                  "external_protocol_note": "Existing MCAT/CAT cohort-wide binning qualification retained.",
                  "inference": "Descriptive DSMIL comparison; no new p-value or confidence interval."})
    (OUT / "dsmil_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("Verified 25 DSMIL folds; eight-method comparison exported.")


if __name__ == "__main__":
    main()
