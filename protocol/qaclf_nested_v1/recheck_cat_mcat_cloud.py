"""Fresh, isolated MCAT/CAT training audit for the existing Python 3.8 cloud env.

The legacy arm preserves historical training. The gc_fixed arm only corrects
gradient accumulation. Neither arm changes model heads, time bins, sampling,
tail conventions, learning rates, epochs, or selects an outer-test checkpoint.
"""

import argparse
import ast
from contextlib import contextmanager, redirect_stdout
import csv
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types


COHORTS = ("blca", "brca", "gbmlgg", "luad", "ucec")
HERE = Path(__file__).resolve().parent


def sha(path, source=False):
    if source:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def append_record(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def window_size(index, total, gc):
    if gc < 1 or total < 1 or index < 0 or index >= total:
        raise ValueError("Invalid accumulation index, epoch length, or gc")
    return min(gc, total - (index // gc) * gc)


def optimizer_list(optimizer):
    return optimizer if isinstance(optimizer, list) else [optimizer]


def clear_gradients(optimizer):
    for part in optimizer_list(optimizer):
        part.zero_grad()


def step_then_clear(optimizer):
    for part in optimizer_list(optimizer):
        part.step()
    clear_gradients(optimizer)


@contextmanager
def diagnostic_rng(torch, np):
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


def select_items(manifest, args):
    selected = [r for r in manifest["folds"] if r["cohort"] == args.dataset
                and r["fold"] == args.fold and r["method"] in args.methods]
    if len(selected) != len(args.methods) or len({r["method"] for r in selected}) != len(selected):
        raise ValueError("Missing or duplicate selected folds")
    return sorted(selected, key=lambda r: args.methods.index(r["method"]))


def safe_output(output, repo):
    output, repo = output.resolve(), repo.resolve()
    old = (repo / "results-conch").resolve()
    if output in (repo, old) or repo in output.parents or old in output.parents:
        raise ValueError("Output must be outside original repository and retained results")
    return output


def feature_directory(root, dataset):
    base = root if dataset == "luad" else root / ("tcga_" + dataset)
    return base / "feats-CONCH-vl-proj/pt_files"


def bind_vendor_packages(repo):
    # These generic names otherwise collide with VLSA or installed HF packages.
    for name in ["utils", "models", "datasets", "clip"]:
        folder = repo / name
        initializer = folder / "__init__.py"
        if initializer.is_file():
            spec = importlib.util.spec_from_file_location(name, initializer,
                                                          submodule_search_locations=[str(folder)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        else:
            module = types.ModuleType(name)
            module.__path__ = [str(folder)]
            module.__package__ = name
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            module.__spec__.submodule_search_locations = [str(folder)]
            sys.modules[name] = module


def preflight(args, manifest, contract):
    repo = args.repo_root.resolve()
    for name, expected in contract["source_sha256_lf"].items():
        if sha(repo / name, source=True) != expected:
            raise ValueError("Original source changed: " + name)
    for name, expected in manifest["input_sha256"].items():
        if "tcga_" in name and ("tcga_" + args.dataset + "_") not in name:
            continue
        if sha(repo / name) != expected:
            raise ValueError("Original input changed: " + name)
    features = feature_directory(args.features_root, args.dataset)
    if not features.is_dir():
        raise FileNotFoundError(str(features))
    items = select_items(manifest, args)
    for item in items:
        split = repo / "splits/5foldcv" / ("tcga_" + args.dataset) / ("splits_%d.csv" % args.fold)
        if sha(split) != item["split_sha256"]:
            raise ValueError("Original split hash changed")
        if set(item["train_ids"]) & set(item["test_ids"]):
            raise ValueError("Overlapping train and held-out patients")
        if item["settings"]["max_epochs"] != 20 or item["settings"]["seed"] != 1:
            raise ValueError("Unexpected historical training protocol")
    for name, expected in contract["runtime_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Bundled diagnostic runtime changed: " + name)
    if "cat_tpl" in args.methods:
        cached = Path.home() / ".cache/clip/RN50.pt"
        if not cached.is_file() or sha(cached) != contract["rn50_sha256"]:
            raise ValueError("Existing RN50 cache missing/corrupt; no automatic download or installation")
    print("CHECK PASS: source, omics, signatures, frozen split, feature directory, and metric runtime", flush=True)
    return items


def disable_unused_test_best_saves(core):
    tree = ast.parse(inspect.getsource(core.train))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if names == {"c_index", "best_c_index"}:
                node.test = ast.Constant(False)
                changed += 1
    if changed != 1:
        raise ValueError("Unexpected checkpoint selection code; refusing to patch")
    exec(compile(ast.fix_missing_locations(tree), "<diagnostic-last-checkpoint-only>", "exec"), core.__dict__)


def make_fixed_train(torch, np, core, record):
    def train(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None,
              reg_fn=None, lambda_reg=0., gc=16):
        device = next(model.parameters()).device
        model.train()
        clear_gradients(optimizer)
        risks, times, censored = [], [], []
        losses, totals = [], []
        n_batches, updates = len(loader), 0
        for index, sample in enumerate(loader):
            wsi, omics, label, observed, c = sample[0], sample[1:7], sample[7], sample[8], sample[9]
            inputs = {"x_path": wsi.to(device)}
            inputs.update({"x_omic%d" % j: v.float().to(device) for j, v in enumerate(omics, 1)})
            hazards, survival, _, _ = model(**inputs)
            loss = loss_fn(hazards=hazards, S=survival, Y=label.long().to(device), c=c.float().to(device))
            penalty = 0 if reg_fn is None else reg_fn(model) * lambda_reg
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite training loss")
            losses.append(float(loss.item()))
            totals.append(float((loss + penalty).item()))
            risks.append(float((-survival.sum(dim=1)).detach().cpu().item()))
            times.append(float(np.asarray(observed).item()))
            censored.append(float(c.item()))
            ((loss + penalty) / window_size(index, n_batches, gc)).backward()
            if (index + 1) % gc == 0 or index + 1 == n_batches:
                step_then_clear(optimizer)
                updates += 1
            if (index + 1) % 100 == 0:
                print("training patients %d/%d" % (index + 1, n_batches), flush=True)
        assert updates == math.ceil(n_batches / gc)
        ci = core.concordance_index_censored((1 - np.asarray(censored)).astype(bool),
                                           np.asarray(times), np.asarray(risks), tied_tol=1e-8)[0]
        record(dict(phase="train", epoch=epoch, loss_surv=float(np.mean(losses)),
                    loss=float(np.mean(totals)), c_index=float(ci), optimizer_updates=updates,
                    n_batches=n_batches, remainder=n_batches % gc))
        print("Epoch: %d, train_loss_surv: %.4f, train_loss: %.4f, train_c_index: %.4f" %
              (epoch, np.mean(losses), np.mean(totals), ci), flush=True)
    return train


def native_metrics(np, survival, bins, actual, train, grid):
    from eval.SurvivalEVAL.Evaluations.BrierScore import brier_multiple_points
    from eval.SurvivalEVAL.Evaluations.MeanError import mean_error
    from eval.SurvivalEVAL.Evaluations.util import KaplanMeier
    x = np.r_[0., bins[1:]]
    curves = np.column_stack([np.ones(len(survival)), survival])
    terminal = survival[:, -1].astype(float)
    within = np.trapz(curves, x, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = .5 * x[-1] * terminal ** 2 / (1 - terminal)
        means = within + tail
    mae = float(mean_error(means, actual[:, 0], actual[:, 1], train[:, 0], train[:, 1], method="Hinge"))
    if grid[0] < x[0] or grid[-1] > x[-1]:
        raise ValueError("Historical IBS grid not covered; no silent truncation")
    probabilities = np.stack([np.interp(grid, x, c) for c in curves])
    bs = brier_multiple_points(probabilities, actual[:, 0], actual[:, 1], train[:, 0], train[:, 1], grid)
    if not np.isfinite(bs).all():
        raise FloatingPointError("Nonfinite Brier score")
    censor_km = KaplanMeier(train[:, 0], 1 - train[:, 1])
    return dict(ibs=float(np.trapz(bs, grid) / (grid[-1] - grid[0])),
                mae_hinge_mean_linear=mae if math.isfinite(mae) else None,
                nonfinite_mean_predictions=int((~np.isfinite(means)).sum()),
                predicted_mean_max=float(means.max()) if np.isfinite(means).all() else None,
                terminal_probability_median=float(np.median(terminal)),
                terminal_probability_p95=float(np.quantile(terminal, .95)),
                terminal_probability_max=float(terminal.max()),
                censoring_survival_at_ibs_end=float(censor_km.predict(np.asarray([grid[-1]]))[0]))


def worker(args, manifest, contract):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE / "metric_runtime"))
    sys.path.insert(0, str(args.repo_root.resolve()))
    bind_vendor_packages(args.repo_root.resolve())
    import numpy as np
    import torch
    from datasets.dataset_survival import Generic_MIL_Survival_Dataset
    from utils import core_utils as core
    from clip import clip

    def no_download(url, *unused, **unused_kw):
        path = Path.home() / ".cache/clip/RN50.pt"
        if not url.endswith("/RN50.pt") or not path.is_file():
            raise RuntimeError("Automatic model download disabled")
        return str(path)
    clip._download = no_download
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; existing vlsa_env and GPU required")
    torch.set_num_threads(1)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    item = select_items(manifest, args)[0]
    model_name = item["method"]
    destination = args.output_root.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    dataset = Generic_MIL_Survival_Dataset(
        csv_path=str(args.repo_root / ("dataset_csv_sig/tcga_%s_all_clean.csv.zip" % args.dataset)),
        mode="coattn", apply_sig=True, signatures_path=str(args.repo_root / "dataset_csv_sig/signatures.csv"),
        data_dir=str(feature_directory(args.features_root, args.dataset)), shuffle=False, seed=1,
        print_info=False, patient_strat=False, n_bins=4, label_col="survival_months", ignore=[])
    split_path = args.repo_root / "splits/5foldcv" / ("tcga_" + args.dataset) / ("splits_%d.csv" % args.fold)
    train, heldout = dataset.return_splits(from_id=False, csv_path=str(split_path))
    ids = heldout.slide_data.slide_id.astype(str).tolist()
    train_ids = train.slide_data.slide_id.astype(str).tolist()
    assert set(ids) == set(item["test_ids"]) and set(train_ids) == set(item["train_ids"])
    assert not set(ids) & set(train_ids)
    for row in heldout.slide_data.itertuples():
        old = item["predictions"][str(row.slide_id)]
        assert abs(float(row.survival_months) - old["time"]) < 1e-4
        assert int(row.disc_label) == old["disc_label"] and float(row.censorship) == old["censorship"]
    bins = np.asarray(dataset.bins, float)
    grid = np.asarray(contract["ibs_grids"]["%s/%d" % (args.dataset, args.fold)], float)
    actual = np.column_stack([heldout.slide_data.survival_months.to_numpy(float),
                              1 - heldout.slide_data.censorship.to_numpy(float)])
    train_labels = np.column_stack([train.slide_data.survival_months.to_numpy(float),
                                    1 - train.slide_data.censorship.to_numpy(float)])
    configuration = dict(item["settings"], task_type="survival", mode="coattn", n_classes=4,
                         fusion="concat", wsi_input_dim=512, data_name="tcga_" + args.dataset,
                         instanceLevel_n_ctx=16, all_ctx_trainable=False, csc=True, clip_model="RN50",
                         p_drop_out=.5, drop_out=True, log_data=False, results_dir=str(destination),
                         alpha_surv=0., reg_type="None", lambda_reg=1e-4, reg=1e-5, opt="adam",
                         weighted_sample=True, testing=False, batch_size=1, early_stopping=False,
                         omic_sizes=train.omic_sizes)
    namespace = argparse.Namespace(**configuration)
    audit = dict(scope="smoke only" if args._smoke else "fresh one-fold training audit",
                 variant=args._variant, method=model_name, dataset=args.dataset, fold=args.fold,
                 original_source_unchanged=True, architecture="original adapted native sigmoid hazard head",
                 configuration=configuration, native_bins=bins.tolist(),
                 split_sha256=sha(split_path), contract_sha256=sha(HERE / "recheck_contract.json"),
                 metric_tail="original full Mean/Linear analytical equivalent; no forced tail zero",
                 time_bins="historical cohort-wide bins retained for controlled diagnosis, NOT train-only",
                 sampling="original inverse joint bin/censorship class weights; not changed",
                 checkpoint="fixed last epoch only; unused outer-test best saves disabled",
                 epochs_selected_from_test=False, python=sys.version, torch=torch.__version__,
                 numpy=np.__version__, train_patients=len(train_ids), test_patients=len(ids))
    (destination / "protocol.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    record = lambda r: append_record(destination / "epoch_metrics.jsonl", r)
    step_log = destination / "optimizer_steps.jsonl"
    updates = {"count": 0}
    get_original_optimizer = core.get_optim

    def checked_optimizer(model, settings):
        optimizer = get_original_optimizer(model, settings)
        h = hashlib.sha256()
        for p in model.classifier.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        audit["initial_classifier_sha256"] = h.hexdigest()
        (destination / "protocol.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        for part in optimizer_list(optimizer):
            original_step = part.step
            def checked_step(*a, _part=part, _original=original_step, **kw):
                squared, tensors = 0., 0
                for group in _part.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            g = p.grad.detach().float()
                            if not bool(torch.isfinite(g).all().item()):
                                raise FloatingPointError("Nonfinite gradient before optimizer step")
                            squared += float(torch.sum(g.double() ** 2).item())
                            tensors += 1
                if not tensors:
                    raise RuntimeError("Optimizer step has no gradients")
                value = _original(*a, **kw)
                updates["count"] += 1
                append_record(step_log, dict(update=updates["count"], gradient_norm=math.sqrt(squared),
                                             gradient_tensors=tensors))
                return value
            part.step = checked_step
        return optimizer
    core.get_optim = checked_optimizer
    original_train = core.train_loop_survival_coattn

    if args._variant == "gc_fixed":
        core.train_loop_survival_coattn = make_fixed_train(torch, np, core, record)
    else:
        class Capture:
            def __init__(self, stream):
                self.stream, self.text = stream, ""
            def write(self, text):
                self.text += text
                return self.stream.write(text)
            def flush(self):
                self.stream.flush()
        def historical_train(epoch, model, loader, optimizer, *a, **kw):
            before = updates["count"]
            capture = Capture(sys.stdout)
            with redirect_stdout(capture):
                value = original_train(epoch, model, loader, optimizer, *a, **kw)
            import re
            match = re.search(r"train_loss_surv: (\S+), train_loss: (\S+), train_c_index: (\S+)", capture.text)
            if not match:
                raise ValueError("Original training metrics missing")
            record(dict(phase="train", epoch=epoch, loss_surv=float(match[1]), loss=float(match[2]),
                        c_index=float(match[3]), recorded_precision="original stdout four decimals",
                        optimizer_updates=updates["count"] - before, n_batches=len(loader),
                        remainder=len(loader) % namespace.gc))
            return value
        core.train_loop_survival_coattn = historical_train

    final_curves = {}
    def validation(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None,
                   writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
        model.eval()
        device = next(model.parameters()).device
        survival_rows, risk_rows, hazard_rows, losses = [], [], [], []
        with torch.no_grad():
            for sample in loader:
                inputs = {"x_path": sample[0].to(device)}
                inputs.update({"x_omic%d" % j: sample[j].float().to(device) for j in range(1, 7)})
                hazards, survival, _, _ = model(**inputs)
                loss = loss_fn(hazards=hazards, S=survival, Y=sample[7].long().to(device),
                               c=sample[9].float().to(device), alpha=0)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Nonfinite held-out loss")
                losses.append(float(loss.item()))
                survival_rows.append(survival.cpu().numpy().reshape(-1))
                hazard_rows.append(hazards.cpu().numpy().reshape(-1))
                risk_rows.append(float((-survival.sum(dim=1)).cpu().item()))
        survival = np.asarray(survival_rows)
        assert len(survival) == len(ids) and np.isfinite(survival).all()
        assert np.allclose(survival, np.cumprod(1 - np.asarray(hazard_rows), axis=1), atol=1e-6, rtol=0)
        # The historical collate function preserves observation times as float64.
        native_c = float(core.concordance_index_censored(actual[:, 1].astype(bool),
            actual[:, 0], np.asarray(risk_rows), tied_tol=1e-8)[0])
        with diagnostic_rng(torch, np):
            values = native_metrics(np, survival, bins, actual, train_labels, grid)
        final_curves.update(survival=survival, hazards=np.asarray(hazard_rows), risk=np.asarray(risk_rows))
        record(dict(phase="heldout_diagnostic_not_selection", epoch=epoch,
                    loss_surv=float(np.mean(losses)), c_index=native_c, **values))
        print("Epoch %d heldout NLL=%.6f C-index=%.6f MAE=%s tail_max=%.6f" %
              (epoch, np.mean(losses), native_c, values["mae_hinge_mean_linear"],
               values["terminal_probability_max"]), flush=True)
        return False, native_c
    core.validate_survival_coattn = validation
    disable_unused_test_best_saves(core)

    if args._smoke:
        from export_cat_mcat_curves import initialise_model
        model = initialise_model(core, item, train.omic_sizes)
        if hasattr(model, "text_encoder"):
            for p in model.text_encoder.parameters():
                p.requires_grad_(False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).train()
        sample = train[0]
        inputs = {"x_path": sample[0].to(device)}
        inputs.update({"x_omic%d" % j: sample[j].float().to(device) for j in range(1, 7)})
        hazards, survival, _, _ = model(**inputs)
        loss = core.NLLSurvLoss(alpha=0.)(hazards, survival,
            torch.tensor([int(sample[7])], device=device), torch.tensor([float(sample[9])], device=device))
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Smoke loss/gradient nonfinite")
        summary = dict(scope="one-patient mechanics only, not an experimental result", finite_loss=True,
                       device=str(device), hazard_shape=list(hazards.shape), method=model_name)
        (destination / "smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("SMOKE PASS", json.dumps(summary), flush=True)
        return

    started = time.monotonic()
    predictions, native_c = core.train((train, heldout), args.fold, namespace)
    checkpoint = destination / ("s_%d_checkpoint_last.pt" % args.fold)
    assert checkpoint.is_file()
    epochs = [json.loads(s) for s in (destination / "epoch_metrics.jsonl").read_text().splitlines()]
    train_epochs = [r for r in epochs if r["phase"] == "train"]
    heldout_epochs = [r for r in epochs if r["phase"].startswith("heldout")]
    assert [r["epoch"] for r in train_epochs] == list(range(20))
    assert [r["epoch"] for r in heldout_epochs] == list(range(20))
    assert abs(native_c - heldout_epochs[-1]["c_index"]) < 1e-10
    np.savez_compressed(destination / "final_curves.npz", patient_ids=np.asarray(ids),
                        time_coordinates=np.r_[0., bins[1:]], native_bins=bins,
                        survival_curves=np.column_stack([np.ones(len(ids)), final_curves["survival"]]),
                        hazards=final_curves["hazards"], native_risk=final_curves["risk"],
                        event_time=actual[:, 0], event=actual[:, 1],
                        train_ids=np.asarray(train_ids), train_time=train_labels[:, 0], train_event=train_labels[:, 1])
    old_risk = np.asarray([item["predictions"][pid]["risk"] for pid in ids])
    new_risk = np.asarray([float(predictions[pid]["risk"]) for pid in ids])
    summary = dict(complete=True, new_training=True, method=model_name, variant=args._variant,
                   dataset=args.dataset, fold=args.fold, epochs=20, checkpoint_sha256=sha(checkpoint),
                   initial_classifier_sha256=audit["initial_classifier_sha256"],
                   final_native_c_index=float(native_c), wall_seconds=time.monotonic() - started,
                   old_new_risk_max_abs_delta=float(np.max(abs(old_risk - new_risk))),
                   old_new_risk_delta_is_diagnostic_not_pass_criterion=True,
                   final_metrics=heldout_epochs[-1], original_results_not_overwritten=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("COMPLETE", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "smoke", "run", "_worker"])
    parser.add_argument("--repo-root", type=Path, default=Path("/tmp/pycharm_project_456/external/CAT-with-TPL-master"))
    parser.add_argument("--features-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/QACLF_MCAT_CAT_recheck_output"))
    parser.add_argument("--dataset", choices=COHORTS, default="brca")
    parser.add_argument("--fold", choices=range(5), type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=["mcat", "cat_tpl"], default=["mcat", "cat_tpl"])
    parser.add_argument("--variants", nargs="+", choices=["legacy", "gc_fixed"], default=["legacy", "gc_fixed"])
    parser.add_argument("--allow-cpu", action="store_true", help="Local mechanics tests only")
    parser.add_argument("--_variant", choices=["legacy", "gc_fixed"], default="gc_fixed")
    parser.add_argument("--_smoke", action="store_true")
    args = parser.parse_args()
    if len(args.methods) != len(set(args.methods)) or len(args.variants) != len(set(args.variants)):
        parser.error("Duplicate methods or variants")
    args.repo_root = args.repo_root.resolve()
    args.output_root = safe_output(args.output_root, args.repo_root)
    manifest = json.loads((HERE / "original_run_manifest.json").read_text())
    contract = json.loads((HERE / "recheck_contract.json").read_text())
    items = preflight(args, manifest, contract)
    if args.mode == "check":
        print("No training, installation, checkpoint loading, or old-result modification")
        return
    if args.mode == "_worker":
        worker(args, manifest, contract)
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    variants = ["gc_fixed"] if args.mode == "smoke" else args.variants
    for variant in variants:
        for item in items:
            job = args.output_root / variant / item["method"] / args.dataset / ("fold_%d" % args.fold)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "_worker",
                       "--repo-root", str(args.repo_root), "--features-root", str(args.features_root),
                       "--output-root", str(job), "--dataset", args.dataset, "--fold", str(args.fold),
                       "--methods", item["method"], "--_variant", variant]
            if args.mode == "smoke":
                command.append("--_smoke")
            if args.allow_cpu:
                command.append("--allow-cpu")
            print("START", variant, item["method"], args.dataset, args.fold, flush=True)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            subprocess.run(command, check=True, env=env, cwd=str(args.repo_root))
    if args.mode == "run":
        comparisons = []
        for variant in variants:
            for item in items:
                job = args.output_root / variant / item["method"] / args.dataset / ("fold_%d" % args.fold)
                result = json.loads((job / "summary.json").read_text())
                comparisons.append(dict(variant=variant, method=item["method"], dataset=args.dataset,
                                        fold=args.fold, native_c_index=result["final_native_c_index"],
                                        ibs=result["final_metrics"]["ibs"],
                                        mae=result["final_metrics"]["mae_hinge_mean_linear"],
                                        tail_max=result["final_metrics"]["terminal_probability_max"],
                                        initial_classifier_sha256=result["initial_classifier_sha256"]))
        for item in items:
            if len({r["initial_classifier_sha256"] for r in comparisons if r["method"] == item["method"]}) != 1:
                raise ValueError("Initial classifier differs across arms; do not interpret as a controlled comparison")
        with (args.output_root / "one_fold_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        print(json.dumps(comparisons, indent=2), flush=True)
    for name, expected in contract["source_sha256_lf"].items():
        if sha(args.repo_root / name, source=True) != expected:
            raise RuntimeError("Original source changed during audit: " + name)
    print("ALL REQUESTED AUDIT JOBS COMPLETE. Do not treat a single held-out fold as a new macro result.", flush=True)


if __name__ == "__main__":
    main()
