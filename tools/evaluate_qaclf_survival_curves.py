#!/usr/bin/env python3
"""Project locked QACLF scores to survival curves using outer-train data only."""

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from compare_paired_survival_predictions import (
    build_meta,
    c_index,
    load_config,
    load_predictions,
    predicted_event_times,
)
from eval.SurvivalEVAL import SurvivalEvaluator
from eval.utils_coxph import BreslowEstimator
from evaluate_frozen_pathomic_fusion_outer5 import (
    one_path,
    relocate_saved_data_paths,
    resolve_path,
)
from probe_cat_omics_complementarity import train_empirical_cdf
from probe_cat_omics_signal import fit_cox, signature_features


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


def align_predictions(path, patient_ids, meta):
    frame, survival, _ = load_predictions(path)
    if set(frame["patient_id"]) != set(patient_ids):
        raise RuntimeError(f"Prediction patient IDs do not match {path}.")
    frame = frame.set_index("patient_id").loc[patient_ids]
    aligned = survival[frame["row_index"].to_numpy(dtype=int)]
    return predicted_event_times(aligned, meta.time_coordinates), aligned


def fused_scores(train_path, test_path, train_omics, test_omics, alpha):
    train_path_q = train_empirical_cdf(train_path, train_path)
    test_path_q = train_empirical_cdf(train_path, test_path)
    train_omics_q = train_empirical_cdf(train_omics, train_omics)
    test_omics_q = train_empirical_cdf(train_omics, test_omics)
    return (
        (1.0 - alpha) * train_path_q + alpha * train_omics_q,
        (1.0 - alpha) * test_path_q + alpha * test_omics_q,
    )


def score_to_risk_feature(score, n_train):
    clip = 0.5 / max(int(n_train), 2)
    probability = np.clip(np.asarray(score, dtype=np.float64), clip, 1.0 - clip)
    return -np.log(probability / (1.0 - probability))


def cox_scale_loss(scale, feature, time, event):
    predictor = float(scale) * feature
    losses = []
    for index in np.flatnonzero(event > 0.5):
        risk_set = time >= time[index]
        losses.append(logsumexp(predictor[risk_set]) - predictor[index])
    return float(np.mean(losses)) if losses else 0.0


def monotone_breslow_projection(
    train_score,
    test_score,
    train_time,
    train_event,
    output_coordinates,
):
    train_feature = score_to_risk_feature(train_score, len(train_score))
    test_feature = score_to_risk_feature(test_score, len(train_score))
    result = minimize_scalar(
        cox_scale_loss,
        bounds=(0.05, 5.0),
        args=(train_feature, train_time, train_event),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Monotone Cox scale fit failed: {result.message}")
    scale = float(result.x)
    train_predictor = scale * train_feature
    test_predictor = scale * test_feature
    estimator = BreslowEstimator().fit(train_predictor, train_event, train_time)
    time_coordinates = np.asarray(output_coordinates, dtype=np.float64)
    source_time = np.asarray(estimator.baseline_survival_.x, dtype=np.float64)
    source_survival = np.asarray(estimator.baseline_survival_.y, dtype=np.float64)
    source_indices = np.searchsorted(source_time, time_coordinates, side="right") - 1
    baseline_survival = np.ones_like(time_coordinates, dtype=np.float64)
    available = source_indices >= 0
    baseline_survival[available] = source_survival[
        np.minimum(source_indices[available], len(source_survival) - 1)
    ]
    risk_score = np.exp(test_predictor)
    survival = np.power(
        baseline_survival.reshape(1, -1), risk_score.reshape(-1, 1)
    )
    # Match VLSA's finite discrete-time support instead of extrapolating a
    # nonzero Cox tail beyond the predefined final time coordinate.
    survival[:, 0] = 1.0
    survival[:, -1] = 0.0
    return time_coordinates, survival, scale, float(result.fun)


def common_ibs_times(path_coordinates, qaclf_coordinates, train_time, test_time):
    upper = min(
        float(np.max(path_coordinates)),
        float(np.max(qaclf_coordinates)),
        float(np.max(train_time)),
        float(np.max(test_time)),
    )
    candidates = np.asarray(path_coordinates, dtype=np.float64)
    candidates = candidates[(candidates > 0.0) & (candidates < upper)]
    if len(candidates) < 2:
        candidates = np.linspace(0.0, upper, 13, dtype=np.float64)[1:-1]
    if len(candidates) > 12:
        indices = np.linspace(0, len(candidates) - 1, 12).round().astype(int)
        candidates = candidates[np.unique(indices)]
    return candidates


def finite_metric(function):
    try:
        value = float(function())
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def evaluate_curve(curves, coordinates, actual, train_actual, ibs_times):
    evaluator = SurvivalEvaluator(
        curves,
        coordinates,
        actual.t.to_numpy(dtype=float),
        actual.e.to_numpy(dtype=float),
        train_actual.t.to_numpy(dtype=float),
        train_actual.e.to_numpy(dtype=float),
        predict_time_method="Mean",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brier = np.asarray(
            evaluator.brier_score_multiple_points(ibs_times, IPCW_weighted=True),
            dtype=np.float64,
        )
    finite = np.isfinite(brier)
    if finite.sum() >= 2:
        x = ibs_times[finite]
        ibs = float(np.trapz(brier[finite], x) / (x[-1] - x[0]))
    else:
        ibs = np.nan
    try:
        d_calibration, _ = evaluator.d_calibration(num_bins=10)
        d_calibration = float(d_calibration)
    except Exception:
        d_calibration = np.nan
    return {
        "c_index": finite_metric(lambda: evaluator.concordance(ties="All")[0]),
        "ibs": ibs,
        "mae_hinge": finite_metric(lambda: evaluator.mae(method="Hinge")),
        "d_calibration_p": d_calibration,
        "km_calibration": finite_metric(evaluator.km_calibration),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cohort = args.dataset.replace("tcga_", "")
    baseline_root = resolve_path(args.baseline_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_root = ROOT / "external/CAT-with-TPL-master"
    cat = pd.read_csv(
        cat_root / f"dataset_csv/tcga_{cohort}_all_clean.csv.zip",
        low_memory=False,
    ).drop_duplicates("case_id").set_index("case_id")
    signatures = pd.read_csv(cat_root / "dataset_csv/signatures.csv")
    features, _ = signature_features(cat.columns, signatures)
    raw = cat[features].to_numpy(dtype=np.float64)
    raw_by_patient = {patient: raw[index] for index, patient in enumerate(cat.index)}

    metric_rows = []
    for fold in range(5):
        result_dir = one_path(
            baseline_root,
            f"{cohort}-VLSA-CONCH-fold_{fold}-*",
            f"{cohort} fold{fold} baseline result directory",
            required_suffix="coop_rank",
        )
        train_prediction_path = result_dir / "bag_clf_train_last_pred_train.csv"
        test_prediction_path = result_dir / "bag_clf_train_last_pred_test.csv"
        config, _ = load_config(test_prediction_path)
        config = relocate_saved_data_paths(config)
        meta = build_meta(config)
        split = pd.read_csv(config["data_split_path"], dtype=str)
        train_ids = split["train"].dropna().tolist()
        test_column = "test" if "test" in split.columns else "val"
        test_ids = split[test_column].dropna().tolist()

        train_raw = np.stack([raw_by_patient[patient] for patient in train_ids])
        test_raw = np.stack([raw_by_patient[patient] for patient in test_ids])
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)
        components = min(args.pca_components, len(train_ids) - 1, len(features))
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=args.seed,
        )
        train_x = pca.fit_transform(train_scaled)
        test_x = pca.transform(test_scaled)
        train_time = cat.loc[train_ids, "survival_months"].to_numpy(dtype=float)
        train_event = 1.0 - cat.loc[train_ids, "censorship"].to_numpy(dtype=float)
        omics_weight, omics_bias, _ = fit_cox(
            train_x, train_time, train_event, args.steps, args.lr, args.l2
        )
        train_omics_time = -(train_x @ omics_weight + omics_bias)
        test_omics_time = -(test_x @ omics_weight + omics_bias)

        train_path_time, _ = align_predictions(
            train_prediction_path, train_ids, meta
        )
        test_path_time, test_path_curves = align_predictions(
            test_prediction_path, test_ids, meta
        )
        train_score, test_score = fused_scores(
            train_path_time,
            test_path_time,
            train_omics_time,
            test_omics_time,
            args.alpha,
        )
        qaclf_coordinates, qaclf_curves, cox_scale, cox_loss = (
            monotone_breslow_projection(
                train_score,
                test_score,
                train_time,
                train_event,
                meta.time_coordinates,
            )
        )

        actual = meta.get_patient_data(pids=test_ids, ret_columns=["t", "e"])
        train_actual = meta.get_patient_data(split="train", ret_columns=["t", "e"])
        evaluation_args = (
            actual.t.to_numpy(dtype=float),
            actual.e.to_numpy(dtype=float),
            train_actual.t.to_numpy(dtype=float),
            train_actual.e.to_numpy(dtype=float),
        )
        scalar_c_index = c_index(test_score, *evaluation_args)
        ibs_times = common_ibs_times(
            meta.time_coordinates,
            qaclf_coordinates,
            train_actual.t.to_numpy(dtype=float),
            actual.t.to_numpy(dtype=float),
        )
        baseline_metrics = evaluate_curve(
            test_path_curves,
            meta.time_coordinates,
            actual,
            train_actual,
            ibs_times,
        )
        qaclf_metrics = evaluate_curve(
            qaclf_curves,
            qaclf_coordinates,
            actual,
            train_actual,
            ibs_times,
        )
        if abs(qaclf_metrics["c_index"] - scalar_c_index) > 1e-10:
            raise AssertionError(
                "Monotone curve projection changed C-index: "
                f"scalar={scalar_c_index}, curve={qaclf_metrics['c_index']}"
            )
        if np.any(np.diff(qaclf_curves, axis=1) > 1e-12):
            raise AssertionError("Projected QACLF curves are not monotone.")

        row = {
            "fold": fold,
            "n": len(test_ids),
            "events": int(actual.e.sum()),
            "scalar_qaclf_c_index": scalar_c_index,
            "cox_scale": cox_scale,
            "cox_scale_loss": cox_loss,
            "ibs_times": ",".join(f"{value:.6g}" for value in ibs_times),
        }
        for name, metrics in (
            ("baseline", baseline_metrics),
            ("qaclf", qaclf_metrics),
        ):
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        row["delta_c_index"] = qaclf_metrics["c_index"] - baseline_metrics["c_index"]
        row["ibs_gain"] = baseline_metrics["ibs"] - qaclf_metrics["ibs"]
        row["mae_gain"] = (
            baseline_metrics["mae_hinge"] - qaclf_metrics["mae_hinge"]
        )
        metric_rows.append(row)
        np.savez_compressed(
            output_dir / f"fold_{fold}_qaclf_survival_curves.npz",
            patient_ids=np.asarray(test_ids),
            time_coordinates=qaclf_coordinates,
            survival_curves=qaclf_curves,
            fused_scores=test_score,
            cox_scale=np.asarray([cox_scale]),
        )

    metrics = pd.DataFrame(metric_rows)
    summary = {
        "dataset": args.dataset,
        "curve_projection": "positive_monotone_cox_scale_plus_breslow_outer_train_only",
        "c_index_projection_max_abs_delta": float(
            np.max(
                np.abs(metrics["qaclf_c_index"] - metrics["scalar_qaclf_c_index"])
            )
        ),
        "baseline_c_index_mean": float(metrics["baseline_c_index"].mean()),
        "qaclf_c_index_mean": float(metrics["qaclf_c_index"].mean()),
        "baseline_ibs_mean": float(metrics["baseline_ibs"].mean()),
        "qaclf_ibs_mean": float(metrics["qaclf_ibs"].mean()),
        "ibs_gain_mean": float(metrics["ibs_gain"].mean()),
        "baseline_mae_mean": float(metrics["baseline_mae_hinge"].mean()),
        "qaclf_mae_mean": float(metrics["qaclf_mae_hinge"].mean()),
        "mae_gain_mean": float(metrics["mae_gain"].mean()),
        "baseline_d_calibration_pass_folds": int(
            (metrics["baseline_d_calibration_p"] >= 0.05).sum()
        ),
        "qaclf_d_calibration_pass_folds": int(
            (metrics["qaclf_d_calibration_p"] >= 0.05).sum()
        ),
        "baseline_km_calibration_mean": float(
            metrics["baseline_km_calibration"].mean()
        ),
        "qaclf_km_calibration_mean": float(metrics["qaclf_km_calibration"].mean()),
    }
    metrics.to_csv(output_dir / "curve_metrics.csv", index=False)
    (output_dir / "curve_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("===== QACLF survival-curve metrics =====")
    print(metrics.to_string(index=False))
    print("===== QACLF survival-curve summary =====")
    print(json.dumps(summary, indent=2))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
