"""Verify the updated public evidence without retraining or changing results."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ["BLCA", "BRCA", "GBMLGG", "LUAD", "UCEC"]
EXPECTED = {"ABMIL-CONCH": .6215081007645225, "TransMIL-CONCH": .6447394829142735,
            "ILRA-CONCH": .6363855177059747, "DSMIL-CONCH": .6591261613970387,
            "Matched VLSA": .695781808041706, "MCAT-CONCH": .649507380758086,
            "CAT-TPL-CONCH": .6614402367838548, "FICPF": .7050168897889689}


def rows(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def near(a, b, tolerance=1e-9):
    assert math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=0), (a, b)


def verify(recompute=False):
    comparison = rows("result/ficpf-eight-method-comparison/eight_method_dataset_summary.csv")
    assert len(comparison) == 8
    for row in comparison:
        near(statistics.fmean(float(row[c]) for c in COHORTS), row["MACRO"])
        near(row["MACRO"], EXPECTED[row["method"]], 2e-8)
    folds = rows("result/ficpf-six-method-curve-comparison/fold_curve_metrics.csv")
    dataset_metrics = rows("result/ficpf-six-method-curve-comparison/dataset_curve_metrics.csv")
    macro_metrics = rows("result/ficpf-six-method-curve-comparison/macro_curve_metrics.csv")
    assert len(folds) == 150 and len(dataset_metrics) == 30 and len(macro_metrics) == 6
    metrics = ["c_index", "ibs", "mae_hinge", "km_calibration"]
    for row in dataset_metrics:
        part = [x for x in folds if x["method"] == row["method"] and x["dataset"] == row["dataset"]]
        assert len(part) == 5 and {int(x["fold"]) for x in part} == set(range(5))
        for metric in metrics:
            near(statistics.fmean(float(x[metric]) for x in part), row[metric])
        assert sum(float(x["d_calibration_p"]) >= .05 for x in part) == int(row["d_calibration_not_rejected"])
    for row in macro_metrics:
        part = [x for x in dataset_metrics if x["method"] == row["method"]]
        assert len(part) == 5
        for metric in metrics:
            near(statistics.fmean(float(x[metric]) for x in part), row[metric])
        near(row["c_index"], EXPECTED[row["method"]], 2e-8)
        assert sum(int(x["d_calibration_not_rejected"]) for x in part) == int(row["d_calibration_not_rejected"])
    audit = json.loads((ROOT / "result/ficpf-six-method-curve-comparison/curve_audit.json").read_text())
    assert audit["complete"] and audit["fold_metric_rows"] == 150
    verified = 0
    for entry in audit["folds"]:
        if entry["method"] in {"Matched VLSA", "FICPF"}:
            continue
        model = entry["method"].removesuffix("-CONCH")
        run = ROOT / "result/retained-mil-inputs" / model.lower() / f"{entry['dataset']}-{model}-CONCH-fold_{entry['fold']}"
        for name, key in [("bag_clf_train_last_pred_test.csv", "prediction_sha256"), ("config.yaml", "config_sha256")]:
            assert hashlib.sha256((run / name).read_bytes()).hexdigest() == entry[key], (run, name)
        verified += 1
    assert verified == 100
    stats = rows("result/ficpf-dataset-resource-audit/dataset_statistics.csv")
    patients = rows("result/ficpf-dataset-resource-audit/patient_pathology_counts.csv")
    slides = rows("result/ficpf-dataset-resource-audit/slide_patch_counts.csv")
    assert len(patients) == 2831 and len(slides) == 3529
    assert len({(x["dataset"], x["patient_id"]) for x in patients}) == 2831
    assert sum(int(x["patches"]) for x in slides) == 15615143
    for row in stats:
        pp = [x for x in patients if x["dataset"] == row["dataset"]]
        ss = [x for x in slides if x["dataset"] == row["dataset"]]
        near(len(pp), row["patients"])
        near(len(ss), row["slides"])
        counts = [int(x["patches"]) for x in pp]
        near(statistics.fmean(counts), row["patches_per_patient_mean"])
        near(statistics.median(counts), row["patches_per_patient_median"])
        near(min(counts), row["patches_per_patient_min"])
        near(max(counts), row["patches_per_patient_max"])
        near(sum(counts), row["patches_total"])
        near(sum(int(x["slides"]) for x in pp), row["slides"])
    bench = json.loads((ROOT / "result/ficpf-dataset-resource-audit/pathology_benchmark.json").read_text())
    assert bench["trainable_pathology_parameters"] == 284161 and bench["incremental_cox_parameters"] == 65
    assert bench["reproduced_curve_max_absolute_error"] < 2e-5 and bench["repeats"] == 100
    recomputed = 0
    if recompute:
        sys.path.insert(0, str(ROOT))
        from compare_paired_survival_predictions import c_index
        import numpy as np
        sources = ["result/ficpf-six-method-comparison/mil_patient_predictions.csv",
                   "result/ficpf-seven-method-comparison/ilra_patient_predictions.csv",
                   "result/ficpf-eight-method-comparison/dsmil_patient_predictions.csv"]
        raw = [x for path in sources for x in rows(path)]
        for row in raw:
            row["canonical"] = row["model"].removesuffix("-CONCH")
        for model in ["ABMIL", "TransMIL", "ILRA", "DSMIL"]:
            cohort_values = []
            for cohort in COHORTS:
                values = []
                for fold in range(5):
                    part = [x for x in raw if x["canonical"] == model and x["dataset"] == cohort and int(x["fold"]) == fold]
                    assert len(part) > 0 and len({x["patient_id"] for x in part}) == len(part)
                    array = lambda key: np.asarray([float(x[key]) for x in part])
                    labels = rows(f"data_split/5foldcv/tcga_{cohort.lower()}/mahmoodlab_tcga_{cohort.lower()}_survival.csv")
                    split = rows(f"data_split/5foldcv/tcga_{cohort.lower()}/splits_{fold}.csv")
                    train_ids = {x["train"] for x in split if x["train"]}
                    test_ids = {x["val"] for x in split if x["val"]}
                    assert {x["patient_id"] for x in part} == test_ids
                    assert not test_ids & train_ids
                    train = {x["patient_id"]: x for x in labels if x["patient_id"] in train_ids}
                    train_t = np.asarray([float(x["t"]) for x in train.values()])
                    train_e = np.asarray([float(x["e"]) for x in train.values()])
                    value = c_index(array("predicted_time"), array("event_time"), array("event"), train_t, train_e)
                    retained = next(x for x in folds if x["method"] == model + "-CONCH" and x["dataset"] == cohort and int(x["fold"]) == fold)
                    near(value, retained["c_index"], 1e-6)
                    values.append(value)
                    recomputed += 1
                cohort_values.append(statistics.fmean(values))
            near(statistics.fmean(cohort_values), EXPECTED[model + "-CONCH"], 2e-8)
    return {"eight_methods_verified": 8, "curve_metric_rows_verified": 150,
            "raw_curve_and_config_hash_pairs_verified": verified, "patients": 2831,
            "slides": 3529, "patches": 15615143, "c_index_folds_recomputed": recomputed,
            "new_training": False, "primary_results_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-c-index", action="store_true")
    print(json.dumps(verify(parser.parse_args().recompute_c_index), indent=2))
