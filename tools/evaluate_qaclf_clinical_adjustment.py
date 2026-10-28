#!/usr/bin/env python3
"""Evaluate whether fixed-threshold QACLF risk remains prognostic after adjustment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "result" / "qaclf-clinical-adjustment-20260828"
PREDICTION_FILES = {
    "BLCA": ROOT / "result" / "blca-frozen-pathomic-fusion-outer5-20260820" / "patient_predictions.csv",
    "BRCA": ROOT / "result" / "brca-frozen-pathomic-fusion-outer5-20260820" / "patient_predictions.csv",
    "GBMLGG": ROOT / "result" / "gbmlgg-frozen-pathomic-fusion-outer5-20260820" / "patient_predictions.csv",
    "LUAD": ROOT
    / "result"
    / "luad-frozen-pathomic-fusion-cloud-baseline-outer5-20260820"
    / "patient_predictions.csv",
    "UCEC": ROOT / "result" / "ucec-frozen-pathomic-fusion-outer5-20260820" / "patient_predictions.csv",
}
CLINICAL_DIR = ROOT / "external" / "CAT-with-TPL-master" / "dataset_csv"
MIN_CATEGORY_PATIENTS = 20


def cox_components(beta, time, event, design):
    """Return Breslow log partial likelihood, score, and observed information."""
    beta = np.asarray(beta, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    design = np.asarray(design, dtype=float)
    eta = np.clip(design @ beta, -40.0, 40.0)
    weights = np.exp(eta)
    log_likelihood = 0.0
    score = np.zeros(design.shape[1], dtype=float)
    information = np.zeros((design.shape[1], design.shape[1]), dtype=float)

    for current_time in np.unique(time[event]):
        deaths = (time == current_time) & event
        at_risk = time >= current_time
        death_count = int(np.sum(deaths))
        risk_weights = weights[at_risk]
        risk_design = design[at_risk]
        denominator = float(np.sum(risk_weights))
        if death_count == 0 or denominator <= 0.0:
            continue
        mean_x = np.sum(risk_weights[:, None] * risk_design, axis=0) / denominator
        second_moment = (
            np.einsum("n,ni,nj->ij", risk_weights, risk_design, risk_design)
            / denominator
        )
        log_likelihood += float(np.sum(eta[deaths])) - death_count * np.log(denominator)
        score += np.sum(design[deaths], axis=0) - death_count * mean_x
        information += death_count * (second_moment - np.outer(mean_x, mean_x))
    return log_likelihood, score, information


def fit_cox(time, event, design, max_iter=100, tolerance=1e-9):
    design = np.asarray(design, dtype=float)
    if design.ndim != 2 or design.shape[1] == 0:
        raise ValueError("Cox design must contain at least one covariate.")
    beta = np.zeros(design.shape[1], dtype=float)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        log_likelihood, score, information = cox_components(
            beta, time, event, design
        )
        step = np.linalg.pinv(information, rcond=1e-10) @ score
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break
        scale = 1.0
        accepted = False
        while scale >= 1.0 / 1024.0:
            candidate = beta + scale * step
            candidate_likelihood, _, _ = cox_components(
                candidate, time, event, design
            )
            if candidate_likelihood >= log_likelihood - 1e-10:
                beta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        if np.max(np.abs(scale * step)) < tolerance:
            converged = True
            break

    log_likelihood, score, information = cox_components(beta, time, event, design)
    covariance = np.linalg.pinv(information, rcond=1e-10)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "beta": beta,
        "standard_error": standard_error,
        "log_likelihood": float(log_likelihood),
        "score_max_abs": float(np.max(np.abs(score))),
        "converged": bool(converged or np.max(np.abs(score)) < 1e-6),
        "iterations": int(iterations),
    }


def collapse_subtypes(series):
    series = series.fillna("Unknown").astype(str)
    counts = series.value_counts()
    retained = set(counts[counts >= MIN_CATEGORY_PATIENTS].index)
    collapsed = series.where(series.isin(retained), "Other")
    return collapsed, counts.to_dict()


def build_design(frame):
    columns = ["high_risk", "age_per_10y"]
    values = [
        frame["high_risk"].astype(float).to_numpy(),
        ((frame["age"] - frame["age"].mean()) / 10.0).to_numpy(dtype=float),
    ]

    sex_counts = frame["is_female"].value_counts()
    sex_included = len(sex_counts) == 2 and int(sex_counts.min()) >= 10
    if sex_included:
        columns.append("female")
        values.append(frame["is_female"].astype(float).to_numpy())

    collapsed_subtype, raw_subtype_counts = collapse_subtypes(frame["oncotree_code"])
    subtype_counts = collapsed_subtype.value_counts()
    subtype_reference = str(subtype_counts.index[0])
    subtype_levels = [
        str(level) for level in subtype_counts.index if str(level) != subtype_reference
    ]
    for level in subtype_levels:
        columns.append(f"subtype_{level}")
        values.append((collapsed_subtype == level).astype(float).to_numpy())

    design = np.column_stack(values)
    audit = {
        "design_columns": columns,
        "sex_included": sex_included,
        "sex_counts": {str(key): int(value) for key, value in sex_counts.items()},
        "raw_subtype_counts": {
            str(key): int(value) for key, value in raw_subtype_counts.items()
        },
        "collapsed_subtype_counts": {
            str(key): int(value) for key, value in subtype_counts.items()
        },
        "subtype_reference": subtype_reference,
        "minimum_subtype_patients": MIN_CATEGORY_PATIENTS,
        "site_excluded_reason": "30+ sparse site categories would overfit low-event cohorts",
    }
    return design, columns, audit


def effect_summary(fit, index):
    beta = float(fit["beta"][index])
    standard_error = float(fit["standard_error"][index])
    hazard_ratio = float(np.exp(beta))
    lower = float(np.exp(beta - 1.96 * standard_error))
    upper = float(np.exp(beta + 1.96 * standard_error))
    p_value = float(chi2.sf((beta / standard_error) ** 2, 1))
    return hazard_ratio, lower, upper, p_value


def analyze_cohort(cohort, prediction_path):
    prediction = pd.read_csv(prediction_path)
    clinical_path = CLINICAL_DIR / f"tcga_{cohort.lower()}_all_clean.csv.zip"
    clinical = pd.read_csv(
        clinical_path,
        usecols=["case_id", "age", "is_female", "oncotree_code", "site"],
        low_memory=False,
    )
    consistency = clinical.groupby("case_id").agg(
        {column: "nunique" for column in ("age", "is_female", "oncotree_code", "site")}
    )
    if (consistency > 1).any().any():
        raise RuntimeError(f"Inconsistent patient-level clinical values in {clinical_path}")
    clinical = clinical.drop_duplicates("case_id")
    frame = prediction.merge(
        clinical, left_on="patient_id", right_on="case_id", how="left", validate="one_to_one"
    )
    matched = int(frame["case_id"].notna().sum())
    frame["high_risk"] = frame["fused_predicted_time"] < 0.5
    complete = frame.dropna(
        subset=["event_time", "event", "high_risk", "age", "is_female", "oncotree_code"]
    ).copy()
    design, columns, audit = build_design(complete)
    full_fit = fit_cox(complete["event_time"], complete["event"], design)
    reduced_fit = fit_cox(complete["event_time"], complete["event"], design[:, 1:])
    unadjusted_fit = fit_cox(
        complete["event_time"], complete["event"], design[:, :1]
    )
    adjusted = effect_summary(full_fit, 0)
    unadjusted = effect_summary(unadjusted_fit, 0)
    likelihood_ratio = max(
        0.0, 2.0 * (full_fit["log_likelihood"] - reduced_fit["log_likelihood"])
    )
    likelihood_ratio_p = float(chi2.sf(likelihood_ratio, 1))
    audit.update(
        {
            "prediction_path": str(prediction_path),
            "clinical_path": str(clinical_path),
            "prediction_patients": int(len(prediction)),
            "matched_patients": matched,
            "complete_case_patients": int(len(complete)),
            "excluded_missing_clinical": int(len(frame) - len(complete)),
            "events": int(complete["event"].sum()),
            "full_fit": {
                key: value
                for key, value in full_fit.items()
                if key not in {"beta", "standard_error"}
            },
            "reduced_fit": {
                key: value
                for key, value in reduced_fit.items()
                if key not in {"beta", "standard_error"}
            },
        }
    )
    row = {
        "cohort": cohort,
        "n_total": int(len(frame)),
        "n_complete": int(len(complete)),
        "events": int(complete["event"].sum()),
        "adjustment_covariates": ";".join(columns[1:]),
        "unadjusted_hr": unadjusted[0],
        "unadjusted_ci025": unadjusted[1],
        "unadjusted_ci975": unadjusted[2],
        "unadjusted_p": unadjusted[3],
        "adjusted_hr": adjusted[0],
        "adjusted_ci025": adjusted[1],
        "adjusted_ci975": adjusted[2],
        "adjusted_wald_p": adjusted[3],
        "risk_likelihood_ratio_chi2": likelihood_ratio,
        "risk_likelihood_ratio_p": likelihood_ratio_p,
        "full_fit_converged": full_fit["converged"],
        "score_max_abs": full_fit["score_max_abs"],
    }
    return row, audit


def main():
    rows = []
    audits = {}
    for cohort, prediction_path in PREDICTION_FILES.items():
        row, audit = analyze_cohort(cohort, prediction_path)
        rows.append(row)
        audits[cohort] = audit
    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "clinical_adjustment_summary.csv"
    audit_path = OUTPUT_DIR / "clinical_adjustment_audit.json"
    summary.to_csv(summary_path, index=False)
    audit_path.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print("===== QACLF clinical adjustment =====")
    print(summary.to_string(index=False))
    print(f"summary={summary_path}")
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()
