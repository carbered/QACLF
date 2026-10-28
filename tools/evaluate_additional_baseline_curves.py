"""Evaluate retained MIL curves without training or modifying predictions."""

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.SurvivalEVAL import SurvivalEvaluator

# Load the exact metric functions without unrelated feature-loading dependencies.
metric_path = ROOT / "tools/evaluate_qaclf_survival_curves.py"
metric_tree = ast.parse(metric_path.read_text(encoding="utf-8"))
metric_nodes = [node for node in metric_tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"common_ibs_times", "finite_metric", "evaluate_curve"}]
assert len(metric_nodes) == 3
exec(compile(ast.Module(body=metric_nodes, type_ignores=[]), str(metric_path), "exec"), globals())

OUT = ROOT / "result/qaclf-six-method-curve-comparison"
MIL = ROOT / "downloads/matched_mil_backbones_complete_20260903/result/matched-mil-backbones/seed_42"
RUNS = {
    "ABMIL-CONCH": (MIL / "abmil", "ABMIL"),
    "TransMIL-CONCH": (MIL / "transmil", "TransMIL"),
    "ILRA-CONCH": (ROOT / "downloads/QACLF_ILRA_results_no_weights/result/additional-vlsa-baselines/seed_42/ilra", "ILRA"),
    "DSMIL-CONCH": (ROOT / "downloads/QACLF_DSMIL_results_no_weights/result/additional-vlsa-baselines/seed_42/dsmil", "DSMIL"),
}
COHORTS = ["blca", "brca", "gbmlgg", "luad", "ucec"]
METRICS = ["c_index", "ibs", "mae_hinge", "d_calibration_p", "km_calibration"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_class(retained_root=None):
    path = (retained_root / "label_converter.py" if retained_root else
            ROOT / "downloads/QACLF_DSMIL_results_no_weights/dataset/label_converter.py")
    spec = importlib.util.spec_from_file_location("retained_cloud_curve_labels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MetaSurvData


def evaluate_with_diagnostics(curves, coordinates, actual, train, ibs_times):
    with warnings.catch_warnings(record=True) as caught, redirect_stdout(StringIO()):
        warnings.simplefilter("always")
        values = evaluate_curve(curves, coordinates, actual, train, ibs_times)
    assert all(np.isfinite(values[name]) for name in METRICS), values
    messages = {}
    for entry in caught:
        name = entry.category.__name__ + ": " + str(entry.message)
        messages[name] = messages.get(name, 0) + 1
    return values, messages


def validate_curves(curves, coordinates):
    assert curves.ndim == 2 and curves.shape[1] == len(coordinates)
    assert np.isfinite(curves).all() and np.isfinite(coordinates).all()
    assert curves.min() >= -1e-6 and curves.max() <= 1 + 1e-6
    assert np.max(np.diff(curves, axis=1)) <= 1e-6
    assert np.all(np.diff(coordinates) > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=COHORTS, default=COHORTS)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--retained-root", type=Path,
                        help="Public retained-input directory, containing model subdirectories and label_converter.py")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    retained_root = args.retained_root
    if retained_root is None and (ROOT / "result/retained-mil-inputs/label_converter.py").is_file():
        retained_root = ROOT / "result/retained-mil-inputs"
    runs = (RUNS if retained_root is None else
            {key: (retained_root / name.lower(), name) for key, (_, name) in RUNS.items()})
    Meta = label_class(retained_root)
    expected = pd.concat([
        pd.read_csv(ROOT / "result/qaclf-six-method-comparison/mil_fold_metrics.csv").assign(
            method=lambda d: d.model + "-CONCH"),
        pd.read_csv(ROOT / "result/qaclf-seven-method-comparison/ilra_fold_metrics.csv").assign(method="ILRA-CONCH"),
        pd.read_csv(ROOT / "result/qaclf-eight-method-comparison/dsmil_fold_metrics.csv").assign(method="DSMIL-CONCH"),
    ], ignore_index=True)
    reference = pd.read_csv(ROOT / "result/external_multimodal_comparison_20260821/unified/paired_patient_predictions.csv")
    evaluator_files = [ROOT / "tools/evaluate_qaclf_survival_curves.py"] + list(
        (ROOT / "eval/SurvivalEVAL").rglob("*.py"))
    evaluator_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(evaluator_files)}
    records, audits = [], []
    for cohort in args.datasets:
        retained = pd.read_csv(ROOT / f"result/{cohort}-qaclf-survival-curves-20260820/curve_metrics.csv").set_index("fold")
        for fold in args.folds:
            split_path = ROOT / f"data_split/5foldcv/tcga_{cohort}/splits_{fold}.csv"
            label_path = ROOT / f"data_split/5foldcv/tcga_{cohort}/mahmoodlab_tcga_{cohort}_survival.csv"
            split = pd.read_csv(split_path, dtype=str)
            columns = [c for c in split if not c.startswith("Unnamed")]
            train_ids, test_ids = [split[c].dropna().tolist() for c in columns[:2]]
            assert not set(train_ids) & set(test_ids)
            npz_path = ROOT / f"result/{cohort}-qaclf-survival-curves-20260820/fold_{fold}_qaclf_survival_curves.npz"
            with np.load(npz_path, allow_pickle=False) as saved:
                coordinates = saved["time_coordinates"].copy()
                qaclf_curves = saved["survival_curves"].copy()
                assert saved["patient_ids"].tolist() == test_ids
            for method, (base, network) in runs.items():
                started = time.monotonic()
                run = base / f"{cohort}-{network}-CONCH-fold_{fold}"
                config_path = run / "config.yaml"
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                assert config["deepmil_network"] == network and config["ckpt_for_eval"] == "last"
                assert not config.get("open_ended_last_time_bin", False)
                with redirect_stdout(StringIO()):
                    meta = Meta(str(label_path), data_split={"train": train_ids, "test": test_ids}, verbose=False)
                    meta.generate_discrete_label(num_bins=config["time_bins"], use_quantiles=False, summary=False)
                assert np.array_equal(coordinates, meta.time_coordinates), (method, cohort, fold, "time grid")
                actual = meta.get_patient_data(pids=test_ids, ret_columns=["t", "e"])
                train = meta.get_patient_data(split="train", ret_columns=["t", "e"])
                target = reference[(reference.dataset == cohort.upper()) & (reference.fold == fold)].set_index("patient_id").loc[test_ids]
                assert np.allclose(actual.t.to_numpy(), target.event_time.to_numpy(), rtol=0, atol=1e-4)
                assert np.array_equal(actual.e.to_numpy(), target.event.to_numpy())
                ibs_times = common_ibs_times(coordinates, coordinates, train.t.to_numpy(), actual.t.to_numpy())
                assert ",".join(f"{value:.6g}" for value in ibs_times) == retained.loc[fold, "ibs_times"]
                pred_path = run / "bag_clf_train_last_pred_test.csv"
                pred = pd.read_csv(pred_path, dtype={"patient_id": str}).set_index("patient_id")
                assert pred.index.is_unique and set(pred.index) == set(test_ids)
                pred = pred.loc[test_ids]
                assert np.array_equal(pred.e.to_numpy(), actual.e.to_numpy())
                curves = pred[[c for c in pred if c.startswith("surf_")]].to_numpy(dtype=float)
                validate_curves(curves, coordinates)
                values, warning_counts = evaluate_with_diagnostics(curves, coordinates, actual, train, ibs_times)
                old = expected[(expected.method == method) & (expected.dataset == cohort.upper()) & (expected.fold == fold)]
                assert len(old) == 1
                delta = abs(values["c_index"] - old.iloc[0].c_index)
                assert delta < 1e-6, (method, cohort, fold, delta)
                row = {"method": method, "dataset": cohort.upper(), "fold": fold,
                       "n_test": len(test_ids), "events": int(actual.e.sum()), **values}
                records.append(row)
                audits.append({"method": method, "dataset": cohort, "fold": fold,
                               "prediction_path": str(pred_path.relative_to(ROOT)), "prediction_sha256": sha(pred_path),
                               "config_sha256": sha(config_path), "split_sha256": sha(split_path),
                               "label_sha256": sha(label_path), "c_index_abs_delta": delta,
                               "ibs_times": ibs_times.tolist(), "warnings": warning_counts})
                print(f"PASS {method} {cohort} {fold}: IBS={values['ibs']:.4f} MAE={values['mae_hinge']:.2f} ({time.monotonic()-started:.1f}s)", flush=True)
                if method == "ABMIL-CONCH":
                    validate_curves(qaclf_curves, coordinates)
                    qaclf, notes = evaluate_with_diagnostics(qaclf_curves, coordinates, actual, train, ibs_times)
                    for key in METRICS:
                        assert np.isclose(qaclf[key], retained.loc[fold, "qaclf_" + key], atol=1e-10, rtol=1e-9), (cohort, fold, key)
                    audits.append({"method": "QACLF", "dataset": cohort, "fold": fold,
                                   "prediction_sha256": sha(npz_path), "retained_metrics_reproduced": True, "warnings": notes})
            for method, prefix in [("Matched VLSA", "baseline"), ("QACLF", "qaclf")]:
                records.append({"method": method, "dataset": cohort.upper(), "fold": fold,
                                "n_test": len(test_ids), "events": int(actual.e.sum()),
                                **{key: float(retained.loc[fold, prefix + "_" + key]) for key in METRICS}})
            pd.DataFrame(records).to_csv(out / "fold_curve_metrics.csv", index=False)
            (out / "curve_audit.json").write_text(json.dumps({"folds": audits, "complete": False}, indent=2), encoding="utf-8")
    frame = pd.DataFrame(records)
    summary = frame.groupby(["method", "dataset"], sort=False)[["c_index", "ibs", "mae_hinge", "km_calibration"]].mean().reset_index()
    counts = frame.assign(d_calibration_not_rejected=frame.d_calibration_p >= .05).groupby(["method", "dataset"], sort=False).agg(
        d_calibration_not_rejected=("d_calibration_not_rejected", "sum"), folds=("fold", "size")).reset_index()
    summary = summary.merge(counts, on=["method", "dataset"])
    summary.to_csv(out / "dataset_curve_metrics.csv", index=False)
    macro = summary.groupby("method", sort=False)[["c_index", "ibs", "mae_hinge", "km_calibration"]].mean().reset_index()
    totals = summary.groupby("method", sort=False)[["d_calibration_not_rejected", "folds"]].sum().reset_index()
    macro = macro.merge(totals, on="method")
    macro.to_csv(out / "macro_curve_metrics.csv", index=False)
    complete = set(args.datasets) == set(COHORTS) and set(args.folds) == set(range(5))
    audit = {"complete": complete, "new_training": False, "curves_modified": False,
             "methods": list(macro.method), "fold_metric_rows": len(frame),
             "evaluator_sha256": evaluator_hashes, "folds": audits,
             "comparison_contract": "Mean/Linear; common original per-fold IBS grid; train-fold censoring; continuous outcomes; ten-bin D-calibration.",
             "aggregation": "Equal fold average within each cohort, then equal cohort average; D-calibration counts are descriptive fold counts, not pooled p values.",
             "matched_vlsa_metrics": "Retained previously verified fold metrics on identical patient labels, time coordinates, and IBS evaluation grid.",
             "excluded_methods": {"MCAT-CONCH": "Retained prediction files contain scalar risk, not curves.",
                                  "CAT-TPL-CONCH": "Retained prediction files contain scalar risk, not curves."}}
    (out / "curve_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(macro.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
