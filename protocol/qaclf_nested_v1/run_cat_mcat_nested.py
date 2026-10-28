"""Versioned MCAT/CAT retraining, Python 3.8 compatible, existing cloud env only.

Epoch selection uses inner validation NLL; outer evaluation occurs once after
fresh refitting. Historical outputs and source files are never overwritten.
"""

import argparse
import ast
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import recheck_cat_mcat_cloud as helper

HERE = Path(__file__).resolve().parent
COHORTS = helper.COHORTS
METHODS = ["mcat", "cat_tpl"]
SPEC = dict(version="qaclf-multimodal-nested-v1", seed=1, inner_split_seed=1729,
            inner_validation_fraction=.2, maximum_selection_epochs=20,
            epoch_selection="minimum inner-validation unweighted NLL; earliest exact tie",
            refit="fresh initialization on all outer-training patients for selected epoch count",
            time_bins="four event-quantile bins fitted on the current fitting patients only",
            first_boundary=0., finite_end="max observation time among current fitting patients + 1e-6",
            beyond_fit_end="administrative censoring only for the encoded discrete NLL target",
            sampling="uniform shuffled patients, no inverse-class resampling",
            optimizer="Adam", learning_rate=1e-4, weight_decay=1e-5, accumulation=16,
            loss="unchanged native hazard NLL, alpha=0", architecture="unchanged CONCH-adapted native hazard models",
            primary_curve_metrics="IPCW Brier at 12/36/60 months and IBS[12,60], 97 fixed grid points",
            native_ranking="negative sum of four native survival outputs; canonical untruncated observations",
            continuous_curve="S(0)=1 plus post-interval probabilities at finite right boundaries; linear interpolation",
            full_mean_mae="diagnostic only, additional Mean/Linear tail convention, nonfinite values retained as flags",
            outer_test_selection=False, historical_results_replaced=False,
            limitation="Post-hoc revised protocol; not a replication of VLSA Table 9 or an independent untouched test")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def unique_ids(values, name):
    ids = list(map(str, values))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Missing/duplicate " + name)
    return ids


def canonical_patients(raw):
    columns = ["case_id", "survival_months", "censorship"]
    if not set(columns + ["slide_id"]).issubset(raw):
        raise ValueError("Required original input columns absent")
    if raw.groupby("case_id")[["survival_months", "censorship"]].nunique().max().max() != 1:
        raise ValueError("Conflicting within-patient outcomes")
    patients = raw[columns].drop_duplicates("case_id").set_index("case_id")
    patients.index = patients.index.astype(str)
    if (not np.isfinite(patients.to_numpy(float)).all()
            or (patients.survival_months <= 0).any()
            or not set(patients.censorship.unique()).issubset({0., 1.})):
        raise ValueError("Invalid canonical monthly outcomes")
    return patients


def fit_bins(patients, fit_ids):
    subset = patients.loc[unique_ids(fit_ids, "fitting IDs")]
    events = subset.loc[subset.censorship == 0, "survival_months"]
    if len(events) < 8:
        raise ValueError("Too few fitting events for four event-quantile bins")
    _, bins = pd.qcut(events, q=4, retbins=True, labels=False)
    bins[0], bins[-1] = 0., float(subset.survival_months.max()) + 1e-6
    if not np.isfinite(bins).all() or not np.all(np.diff(bins) > 0):
        raise ValueError("Degenerate fitting-only time bins; do not silently merge")
    return bins


def encode_outcomes(patients, bins):
    time = patients.survival_months.to_numpy(float)
    overflow = time >= bins[-1]
    encoded_time = np.minimum(time, np.nextafter(bins[-1], 0.))
    encoded_c = np.where(overflow, 1., patients.censorship.to_numpy(float))
    labels = pd.cut(encoded_time, bins=bins, labels=False, right=False, include_lowest=True)
    if pd.isna(labels).any():
        raise ValueError("Out-of-domain NLL labels")
    return encoded_time, encoded_c, np.asarray(labels, int), overflow


def make_plan(patients, train_ids, test_ids, cohort, fold):
    train_ids = sorted(unique_ids(train_ids, "outer training IDs"))
    test_ids = sorted(unique_ids(test_ids, "outer test IDs"))
    if set(train_ids) & set(test_ids):
        raise ValueError("Outer patient overlap")
    inner_fit, inner_val = train_test_split(
        sorted(train_ids), test_size=SPEC["inner_validation_fraction"],
        random_state=SPEC["inner_split_seed"] + fold,
        stratify=patients.loc[sorted(train_ids), "censorship"].to_numpy(int))
    if set(inner_fit) & set(inner_val) or set(inner_fit) | set(inner_val) != set(train_ids):
        raise ValueError("Invalid inner patient partition")
    inner_bins, outer_bins = fit_bins(patients, inner_fit), fit_bins(patients, train_ids)
    if outer_bins[-1] <= 60:
        raise ValueError("Fixed 60-month endpoint unsupported by fitting data; no selective fold dropping")
    return dict(dataset=cohort, fold=fold, spec_sha256=json_hash(SPEC), outer_train=train_ids,
                outer_test=test_ids, inner_fit=inner_fit, inner_validation=inner_val,
                inner_bins=inner_bins.tolist(), refit_bins=outer_bins.tolist())


def prepare_stage_frame(raw, fit_ids, evaluation_ids, bins):
    if set(fit_ids) & set(evaluation_ids):
        raise ValueError("Stage overlap")
    ids = set(fit_ids) | set(evaluation_ids)
    frame = raw.loc[raw.case_id.astype(str).isin(ids)].copy()
    if set(frame.case_id.astype(str)) != ids:
        raise ValueError("Input table does not cover all stage patients")
    patients = canonical_patients(frame)
    time, c, _, _ = encode_outcomes(patients, bins)
    time_map, c_map = dict(zip(patients.index, time)), dict(zip(patients.index, c))
    frame["survival_months"] = frame.case_id.astype(str).map(time_map)
    frame["censorship"] = frame.case_id.astype(str).map(c_map)
    return frame


def patched_constructor(original_source, namespace, stage_frame, bins):
    tree = ast.parse(original_source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Generic_WSI_Survival_Dataset")
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    counts = dict(read=0, quantiles=0, bounds=0)

    class ReplaceFit(ast.NodeTransformer):
        def visit_Assign(self, statement):
            target = statement.targets[0]
            value = statement.value
            if isinstance(target, ast.Name) and target.id == "slide_data" and isinstance(value, ast.Call):
                if isinstance(value.func, ast.Attribute) and value.func.attr == "read_csv":
                    statement.value = ast.Call(func=ast.Name(id="_nested_stage_frame", ctx=ast.Load()), args=[], keywords=[])
                    counts["read"] += 1
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "qcut":
                statement.targets = [ast.Name(id="q_bins", ctx=ast.Store())]
                statement.value = ast.Call(func=ast.Name(id="_nested_fit_bins", ctx=ast.Load()), args=[], keywords=[])
                counts["quantiles"] += 1
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "q_bins":
                counts["bounds"] += 1
                return None
            return statement

    init = ReplaceFit().visit(init)
    if counts != dict(read=1, quantiles=1, bounds=2):
        raise ValueError("Unexpected upstream constructor; refusing unverified patch: " + str(counts))
    scope = dict(namespace, _nested_stage_frame=lambda: stage_frame.copy(deep=True),
                 _nested_fit_bins=lambda: np.asarray(bins, float).copy())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[])),
                 "<nested-fit-only-constructor>", "exec"), scope)
    return scope["__init__"]


def stage_splits(raw, fit_ids, evaluation_ids, bins, args, module):
    frame = prepare_stage_frame(raw, fit_ids, evaluation_ids, bins)
    source = (args.repo_root / "datasets/dataset_survival.py").read_text(encoding="utf-8")
    original = module.Generic_WSI_Survival_Dataset.__init__
    module.Generic_WSI_Survival_Dataset.__init__ = patched_constructor(source, module.__dict__, frame, bins)
    try:
        dataset = module.Generic_MIL_Survival_Dataset(
            csv_path="unused-fit-only-frame", mode="coattn", apply_sig=True,
            signatures_path=str(args.repo_root / "dataset_csv_sig/signatures.csv"),
            data_dir=str(helper.feature_directory(args.features_root, args.dataset)),
            print_info=False, shuffle=False, seed=1, n_bins=4, label_col="survival_months")
    finally:
        module.Generic_WSI_Survival_Dataset.__init__ = original
    # Preserve the historical model's gene vocabulary, but never scale outcome
    # or CSV bookkeeping columns (the upstream positional metadata slice varies).
    eligible = set(dataset.slide_data.columns) - set(dataset.metadata)
    signature_genes = set()
    for column in dataset.signatures:
        for gene in dataset.signatures[column].dropna().astype(str).unique():
            signature_genes.update(gene + suffix for suffix in ["_mut", "_cnv", "_rnaseq"])
    input_genes = eligible & signature_genes
    dataset.metadata = pd.Index([column for column in dataset.slide_data if column not in input_genes])
    groups = dict(train=pd.Series(fit_ids), val=pd.Series(evaluation_ids))
    fitting = dataset.get_split_from_df(groups, "train")
    evaluation = dataset.get_split_from_df(groups, "val")
    scaler = fitting.get_scaler()[0]
    fitting.apply_scaler((scaler,))
    evaluation.apply_scaler((scaler,))
    if (set(fitting.slide_data.slide_id) != set(fit_ids)
            or set(evaluation.slide_data.slide_id) != set(evaluation_ids)
            or not np.array_equal(dataset.bins, bins)):
        raise ValueError("Stage loader changed fitting/evaluation membership or bins")
    return fitting, evaluation, scaler


def select_epoch(losses):
    values = np.asarray(losses, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid inner validation losses")
    return int(np.argmin(values)) + 1


def preflight(args, manifest, contract):
    for name, expected in contract["source_sha256_lf"].items():
        if helper.sha(args.repo_root / name, source=True) != expected:
            raise ValueError("Pinned vendor source changed: " + name)
    for name, expected in contract["runtime_sha256"].items():
        if helper.sha(HERE / name) != expected:
            raise ValueError("Bundled runtime changed: " + name)
    selected = [r for r in manifest["folds"] if r["cohort"] in args.datasets
                and r["fold"] in args.folds and r["method"] in args.methods]
    if len(selected) != len(args.datasets) * len(args.folds) * len(args.methods):
        raise ValueError("Manifest does not cover selected jobs")
    for name, expected in manifest["input_sha256"].items():
        if "tcga_" in name and not any("tcga_" + d + "_" in name for d in args.datasets):
            continue
        if helper.sha(args.repo_root / name) != expected:
            raise ValueError("Pinned input changed: " + name)
    for item in selected:
        split = args.repo_root / ("splits/5foldcv/tcga_%s/splits_%d.csv" % (item["cohort"], item["fold"]))
        if helper.sha(split) != item["split_sha256"]:
            raise ValueError("Frozen outer split changed")
        if not helper.feature_directory(args.features_root, item["cohort"]).is_dir():
            raise FileNotFoundError("Feature directory missing")
    if "cat_tpl" in args.methods:
        path = Path.home() / ".cache/clip/RN50.pt"
        if not path.is_file() or helper.sha(path) != contract["rn50_sha256"]:
            raise ValueError("Existing RN50 cache missing/corrupt; do not install/download automatically")
    return selected


def raw_input(args, cohort):
    return pd.read_csv(args.repo_root / ("dataset_csv_sig/tcga_%s_all_clean.csv.zip" % cohort), low_memory=False)


def freeze_plans(args, items):
    args.output_root.mkdir(parents=True, exist_ok=True)
    spec_path = args.output_root / "frozen_protocol.json"
    frozen = dict(spec=SPEC, runtime_contract_sha256=helper.sha(HERE / "nested_contract.json"))
    if spec_path.exists() and json.loads(spec_path.read_text()) != frozen:
        raise ValueError("Output contains a different protocol/runtime")
    write_json(spec_path, frozen)
    folder = args.output_root / "plans"
    folder.mkdir(exist_ok=True)
    plans = {}
    locked_plans = json.loads((HERE / "locked_inner_plans.json").read_text())
    for cohort in args.datasets:
        patients = canonical_patients(raw_input(args, cohort))
        for fold in args.folds:
            item = next(r for r in items if r["cohort"] == cohort and r["fold"] == fold)
            plan = make_plan(patients, item["train_ids"], item["test_ids"], cohort, fold)
            if plan != locked_plans["%s/%d" % (cohort, fold)]:
                raise ValueError("Cloud split/bin construction differs from the packaged locked plan")
            path = folder / ("%s_fold_%d.json" % (cohort, fold))
            if path.exists() and json.loads(path.read_text()) != plan:
                raise ValueError("Inner plan changed; use a new output root")
            write_json(path, plan)
            plans[cohort, fold] = plan
    return plans


def seed_all(torch):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def guarded_optimizer(torch, model, record):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-5)
    original_step = optimizer.step
    counter = {"updates": 0}
    def step(*args, **kwargs):
        total, tensors = 0., 0
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                if not bool(torch.isfinite(grad).all().item()):
                    raise FloatingPointError("Nonfinite gradient; optimizer update refused")
                total += float(torch.sum(grad.double() ** 2).item())
                tensors += 1
        if not tensors:
            raise RuntimeError("Optimizer step has no gradients")
        value = original_step(*args, **kwargs)
        counter["updates"] += 1
        record(dict(update=counter["updates"], gradient_norm=math.sqrt(total), gradient_tensors=tensors))
        return value
    optimizer.step = step
    return optimizer


def infer(model, dataset, loss_fn, torch):
    model.eval()
    device = next(model.parameters()).device
    hazards, survival, risks, losses = [], [], [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            inputs = dict(x_path=sample[0].to(device))
            inputs.update({"x_omic%d" % j: sample[j].float().to(device) for j in range(1, 7)})
            h, s, _, _ = model(**inputs)
            loss = loss_fn(h, s, torch.tensor([int(sample[7])], device=device),
                           torch.tensor([float(sample[9])], device=device), alpha=0.)
            if not bool(torch.isfinite(loss).item()) or not bool(torch.isfinite(s).all().item()):
                raise FloatingPointError("Nonfinite inference output")
            hazards.append(h.cpu().numpy().reshape(-1))
            survival.append(s.cpu().numpy().reshape(-1))
            risks.append(float((-s.sum(dim=1)).cpu().item()))
            losses.append(float(loss.item()))
    return np.asarray(hazards), np.asarray(survival), np.asarray(risks), float(np.mean(losses))


def make_model(torch, core, item, sizes):
    from export_cat_mcat_curves import initialise_model
    model = initialise_model(core, item, sizes)
    if hasattr(model, "text_encoder"):
        for parameter in model.text_encoder.parameters():
            parameter.requires_grad_(False)
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def fixed_metrics(curves, x, actual, train):
    from eval.SurvivalEVAL.Evaluations.BrierScore import brier_multiple_points
    from eval.SurvivalEVAL.Evaluations.D_Calibration import d_calibration
    from eval.SurvivalEVAL.Evaluations.MeanError import mean_error
    from eval.SurvivalEVAL.Evaluations.util import KaplanMeier
    grid = np.linspace(12., 60., 97)
    if x[0] != 0 or x[-1] < 60 or train[:, 0].max() < 60:
        raise ValueError("Fixed probability endpoint unsupported; no fold deletion")
    points = np.array([12., 36., 60.])
    probs = lambda times: np.stack([np.interp(times, x, row) for row in curves])
    censor = KaplanMeier(train[:, 0], 1 - train[:, 1])
    needed = np.r_[grid, actual[(actual[:, 1] == 1) & (actual[:, 0] <= 60), 0]]
    if np.min(censor.predict(needed)) <= 0:
        raise ValueError("Undefined IPCW; no silent dropping")
    bs = brier_multiple_points(probs(grid), actual[:, 0], actual[:, 1], train[:, 0], train[:, 1], grid)
    point_bs = brier_multiple_points(probs(points), actual[:, 0], actual[:, 1], train[:, 0], train[:, 1], points)
    if not np.isfinite(bs).all() or not np.isfinite(point_bs).all():
        raise FloatingPointError("Nonfinite Brier score")
    capped_t = np.minimum(actual[:, 0], 60)
    capped_e = actual[:, 1] * (actual[:, 0] <= 60)
    dp = np.array([np.interp(t, x, row) for t, row in zip(capped_t, curves)])
    result = dict(ibs_12_60=float(np.trapz(bs, grid) / 48),
                  d_calibration_p_admin60=float(d_calibration(dp, capped_e, 10)[0]),
                  train_censor_survival_60=float(censor.predict(points)[-1]),
                  test_at_risk_60=int((actual[:, 0] > 60).sum()))
    result.update({"brier_%d" % m: float(v) for m, v in zip(points.astype(int), point_bs)})
    terminal = curves[:, -1]
    if np.any(terminal >= 1):
        result.update(diagnostic_full_mean_mae=None, diagnostic_unbounded_means=int((terminal >= 1).sum()))
    else:
        means = np.trapz(curves, x, axis=1) + .5 * x[-1] * terminal ** 2 / (1 - terminal)
        value = float(mean_error(means, actual[:, 0], actual[:, 1], train[:, 0], train[:, 1], method="Hinge"))
        result.update(diagnostic_full_mean_mae=value if np.isfinite(value) else None,
                      diagnostic_unbounded_means=int((~np.isfinite(means)).sum()))
    return result


def save_scaler(path, scaler, fitting):
    np.savez_compressed(path, mean=scaler.mean_, scale=scaler.scale_, variance=scaler.var_,
                        columns=np.asarray(fitting.genomic_features.columns.astype(str)))


def worker(args, items, plans):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE / "metric_runtime"))
    sys.path.insert(0, str(args.repo_root))
    helper.bind_vendor_packages(args.repo_root)
    import torch
    import datasets.dataset_survival as module
    from utils import core_utils as core
    from clip import clip
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA missing; use the existing vlsa_env")
    def no_download(url, *a, **kw):
        cached = Path.home() / ".cache/clip/RN50.pt"
        if not url.endswith("/RN50.pt") or not cached.is_file():
            raise RuntimeError("Automatic download disabled")
        return str(cached)
    clip._download = no_download
    torch.set_num_threads(1)
    plan = plans[args.dataset, args.fold]
    item = next(r for r in items if r["cohort"] == args.dataset and r["fold"] == args.fold and r["method"] == args.method)
    job = args.output_root / args.method / args.dataset / ("fold_%d" % args.fold)
    job.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    raw = raw_input(args, args.dataset)
    patients = canonical_patients(raw)
    loss_fn = core.NLLSurvLoss(alpha=0.)
    fitting, validation, scaler = stage_splits(raw, plan["inner_fit"], plan["inner_validation"], plan["inner_bins"], args, module)
    save_scaler(job / "inner_scaler.npz", scaler, fitting)
    seed_all(torch)
    model = make_model(torch, core, item, fitting.omic_sizes)
    initial_sha = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.classifier.parameters())).hexdigest()
    optimizer = guarded_optimizer(torch, model, lambda value: helper.append_record(job / "optimizer_steps.jsonl", dict(value, phase="inner_fit")))
    if args.mode == "_smoke":
        sample = fitting[0]
        inputs = dict(x_path=sample[0].to(next(model.parameters()).device))
        inputs.update({"x_omic%d" % j: sample[j].float().to(next(model.parameters()).device) for j in range(1, 7)})
        h, s, _, _ = model.train()(**inputs)
        loss = loss_fn(h, s, torch.tensor([int(sample[7])], device=h.device),
                       torch.tensor([float(sample[9])], device=h.device))
        loss.backward()
        if not bool(torch.isfinite(loss)) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite smoke loss/gradient")
        write_json(job / "smoke.json", dict(mechanics_only=True, finite_loss=True, shape=list(h.shape),
                                            scope="one inner-fitting patient; not an experimental result"))
        print("SMOKE PASS", args.method, args.dataset, args.fold, flush=True)
        return
    record_path = job / "training.jsonl"
    phase = {"name": "inner_fit"}
    def record(value):
        helper.append_record(record_path, dict(value, phase=phase["name"]))
    train_epoch = helper.make_fixed_train(torch, np, core, record)
    loader = core.get_split_loader(fitting, training=True, weighted=False, mode="coattn", batch_size=1)
    losses = []
    for epoch in range(20):
        train_epoch(epoch, model, loader, optimizer, 4, loss_fn=loss_fn, gc=16)
        with helper.diagnostic_rng(torch, np):
            _, _, _, value = infer(model, validation, loss_fn, torch)
        losses.append(value)
        helper.append_record(record_path, dict(phase="inner_validation_only", epoch=epoch, nll=value))
        print("INNER", args.method, args.dataset, args.fold, epoch + 1, "NLL", value, flush=True)
    selected = select_epoch(losses)
    selection = dict(epochs=selected, inner_validation_nll=losses, rule=SPEC["epoch_selection"],
                     plan_sha256=json_hash(plan), initial_classifier_sha256=initial_sha,
                     outer_evaluations_so_far=0)
    write_json(job / "selection_frozen_before_outer_evaluation.json", selection)
    del model, optimizer, fitting, validation, scaler, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # The outer loader is first constructed after epoch selection is frozen.
    fitting, outer, scaler = stage_splits(raw, plan["outer_train"], plan["outer_test"], plan["refit_bins"], args, module)
    save_scaler(job / "outer_refit_scaler.npz", scaler, fitting)
    seed_all(torch)
    model = make_model(torch, core, item, fitting.omic_sizes)
    refit_initial = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.classifier.parameters())).hexdigest()
    if initial_sha != refit_initial:
        raise ValueError("Fresh refit initialization differs from the locked seed")
    optimizer = guarded_optimizer(torch, model, lambda value: helper.append_record(job / "optimizer_steps.jsonl", dict(value, phase="outer_train_refit")))
    loader = core.get_split_loader(fitting, training=True, weighted=False, mode="coattn", batch_size=1)
    phase["name"] = "outer_train_refit_no_test_evaluation"
    for epoch in range(selected):
        train_epoch(epoch, model, loader, optimizer, 4, loss_fn=loss_fn, gc=16)
    checkpoint = job / "final_refit.pt"
    torch.save(model.state_dict(), str(checkpoint))
    # Save deterministic training predictions for later calibration controls.
    train_h, train_s, train_risks, train_nll = infer(model, fitting, loss_fn, torch)
    h, s, risks, outer_nll = infer(model, outer, loss_fn, torch)
    ids = outer.slide_data.slide_id.astype(str).tolist()
    train_ids = fitting.slide_data.slide_id.astype(str).tolist()
    actual = np.column_stack([patients.loc[ids].survival_months, 1 - patients.loc[ids].censorship])
    train_actual = np.column_stack([patients.loc[train_ids].survival_months, 1 - patients.loc[train_ids].censorship])
    if not np.allclose(s, np.cumprod(1 - h, axis=1), atol=1e-6, rtol=0):
        raise ValueError("Native hazards and survival disagree")
    x = np.r_[0., np.asarray(plan["refit_bins"])[1:]]
    curves = np.column_stack([np.ones(len(ids)), s])
    values = fixed_metrics(curves, x, actual, train_actual)
    # Keep the model's torch risk summation, not a time-mean surrogate.
    ci = float(core.concordance_index_censored(actual[:, 1].astype(bool), actual[:, 0], risks, tied_tol=1e-8)[0])
    prediction_path = job / "final_predictions.npz"
    np.savez_compressed(prediction_path, patient_ids=np.asarray(ids), train_ids=np.asarray(train_ids),
                        time_coordinates=x, finite_bins=np.asarray(plan["refit_bins"]),
                        hazards=h, survival_curves=curves, native_risk=risks,
                        event_time=actual[:, 0], event=actual[:, 1], train_hazards=train_h,
                        train_survival=train_s, train_native_risk=train_risks,
                        train_time=train_actual[:, 0], train_event=train_actual[:, 1])
    summary = dict(complete=True, protocol=SPEC["version"], spec_sha256=json_hash(SPEC),
                   runtime_contract_sha256=helper.sha(HERE / "nested_contract.json"),
                   plan_sha256=json_hash(plan), method=args.method, dataset=args.dataset, fold=args.fold,
                   selected_epochs=selected, native_c_index=ci, **values,
                   encoded_nll_train=train_nll, encoded_nll_outer_diagnostic=outer_nll,
                   outer_patient_count=len(ids), outer_evaluation_count=1,
                   checkpoint_sha256=helper.sha(checkpoint), prediction_sha256=helper.sha(prediction_path),
                   wall_seconds=time.monotonic() - started, historical_results_overwritten=False)
    write_json(job / "summary.json", summary)
    print("COMPLETE", json.dumps(summary), flush=True)


def verified_complete(job, plan, contract_hash):
    summary_path = job / "summary.json"
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text())
    if (not data.get("complete") or data["spec_sha256"] != json_hash(SPEC)
            or data["plan_sha256"] != json_hash(plan) or data["runtime_contract_sha256"] != contract_hash
            or data["method"] != job.parent.parent.name or data["dataset"] != job.parent.name
            or data["fold"] != int(job.name.split("_")[-1])
            or data["outer_patient_count"] != len(plan["outer_test"])
            or data["prediction_sha256"] != helper.sha(job / "final_predictions.npz")
            or data["checkpoint_sha256"] != helper.sha(job / "final_refit.pt")):
        raise ValueError("Existing job failed verification; no overwrite")
    return True


@contextmanager
def run_lock(root):
    if os.name != "posix":
        raise RuntimeError("Run orchestration is for the Linux cloud; local check/tests remain available")
    import fcntl
    with (root / "run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A job already owns this output root; do not launch a duplicate")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def summarize(args, items, plans):
    rows = []
    contract_hash = helper.sha(HERE / "nested_contract.json")
    for item in items:
        job = args.output_root / item["method"] / item["cohort"] / ("fold_%d" % item["fold"])
        if verified_complete(job, plans[item["cohort"], item["fold"]], contract_hash):
            rows.append(json.loads((job / "summary.json").read_text()))
    if not rows:
        print("No complete verified jobs; no experimental summary generated.")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "verified_fold_metrics.csv", index=False)
    keys = ["native_c_index", "ibs_12_60", "brier_12", "brier_36", "brier_60"]
    complete = frame.groupby(["method", "dataset"]).filter(lambda group: set(group.fold) == set(range(5)))
    print("Verified complete jobs:", len(rows), "/", len(items))
    if len(complete):
        dataset = complete.groupby(["method", "dataset"])[keys].mean().reset_index()
        dataset.to_csv(args.output_root / "complete_cohort_metrics.csv", index=False)
        print(dataset.to_string(index=False))
        for method in METHODS:
            subset = dataset[dataset.method == method]
            if set(subset.dataset) == set(COHORTS):
                print("FULL FIVE-COHORT MACRO", method, subset[keys].mean().to_dict())
    print("Partial runs are not five-cohort macro results; diagnostic full-mean MAE is not a primary metric.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "smoke", "run", "summarize", "_worker", "_smoke"])
    parser.add_argument("--repo-root", type=Path, default=Path("/tmp/pycharm_project_456/external/CAT-with-TPL-master"))
    parser.add_argument("--features-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/QACLF_MCAT_CAT_nested_v1_results"))
    parser.add_argument("--datasets", nargs="+", choices=COHORTS, default=list(COHORTS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--dataset", choices=COHORTS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--allow-cpu", action="store_true", help="Local mechanics only")
    args = parser.parse_args()
    if args.mode in ["_worker", "_smoke"]:
        if args.dataset is None or args.fold is None or args.method is None:
            parser.error("Worker requires dataset, fold and method")
        args.datasets, args.folds, args.methods = [args.dataset], [args.fold], [args.method]
    for values in [args.datasets, args.folds, args.methods]:
        if len(values) != len(set(values)):
            parser.error("Duplicate selection")
    args.repo_root = args.repo_root.resolve()
    args.output_root = helper.safe_output(args.output_root, args.repo_root)
    manifest = json.loads((HERE / "original_run_manifest.json").read_text())
    contract = json.loads((HERE / "nested_contract.json").read_text())
    if contract["spec"] != SPEC:
        raise ValueError("Bundled scientific protocol differs from the driver")
    items = preflight(args, manifest, contract)
    plans = freeze_plans(args, items)
    if args.mode == "check":
        print("CHECK PASS:", len(plans), "frozen fit-only plans; no GPU training or old checkpoint loading.")
        return
    if args.mode in ["_worker", "_smoke"]:
        worker(args, items, plans)
        return
    if args.mode == "summarize":
        summarize(args, items, plans)
        return
    with run_lock(args.output_root):
        for item in items:
            job = args.output_root / item["method"] / item["cohort"] / ("fold_%d" % item["fold"])
            if args.mode == "run" and verified_complete(job, plans[item["cohort"], item["fold"]], helper.sha(HERE / "nested_contract.json")):
                print("SKIP verified complete", item["method"], item["cohort"], item["fold"], flush=True)
                continue
            if job.exists():
                raise FileExistsError("Incomplete/existing job is preserved; use a new output root: " + str(job))
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "_worker" if args.mode == "run" else "_smoke",
                       "--repo-root", str(args.repo_root), "--features-root", str(args.features_root),
                       "--output-root", str(args.output_root), "--dataset", item["cohort"],
                       "--fold", str(item["fold"]), "--method", item["method"]]
            if args.allow_cpu:
                command.append("--allow-cpu")
            print("START", args.mode, item["method"], item["cohort"], item["fold"], flush=True)
            subprocess.run(command, check=True, cwd=str(args.repo_root),
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        if args.mode == "run":
            summarize(args, items, plans)
    preflight(args, manifest, contract)
    print("ALL REQUESTED JOBS VERIFIED. No guarantee of improvement; do not cherry-pick old/new versions.")


if __name__ == "__main__":
    main()
