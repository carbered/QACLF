"""Export retained MCAT/CAT native curves; never train or select checkpoints.

Python 3.8 compatible. Run inside the existing AutoDL vlsa_env. The companion
manifest pins patient labels, scalar risks, splits, inputs and source files.
Only use your own trusted feature tensors and retained checkpoints.
"""

import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time


COHORTS = ("blca", "brca", "gbmlgg", "luad", "ucec")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_split(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = ("train", "val")
    if not rows or not all(key in rows[0] for key in keys):
        raise ValueError("Expected saved train/val split: " + str(path))
    groups = [[row[key].strip() for row in rows if row[key].strip()] for key in keys]
    if any(len(group) != len(set(group)) for group in groups):
        raise ValueError("Duplicate patients in saved split")
    if set(groups[0]) & set(groups[1]):
        raise ValueError("Training and held-out patients overlap")
    return groups


def feature_dir(root, cohort):
    base = root if cohort == "luad" else root / ("tcga_" + cohort)
    return base / "feats-CONCH-vl-proj/pt_files"


def checkpoint_for(results, item):
    # Do not fall back to the test-selected 'best' checkpoint.
    return results / item["run_relative_path"] / ("s_%d_checkpoint_last.pt" % item["fold"])


def preflight(args, manifest):
    repo, results = args.repo_root.resolve(), args.results_root.resolve()
    if not repo.is_dir() or not results.is_dir():
        raise FileNotFoundError("Existing CAT repository/results directory is absent")
    for name, expected in manifest["source_sha256_lf"].items():
        if source_sha(repo / name) != expected:
            raise ValueError("Retained source differs from packaged contract: " + name)
    selected = [item for item in manifest["folds"]
                if item["cohort"] in args.datasets and item["fold"] in args.folds
                and item["method"] in args.methods]
    expected_count = len(args.datasets) * len(args.folds) * len(args.methods)
    if len(selected) != expected_count:
        raise ValueError("Duplicate or missing manifest folds")
    for name, expected in manifest["input_sha256"].items():
        if name.startswith("dataset_csv_sig/tcga_"):
            if not any("tcga_" + cohort + "_" in name for cohort in args.datasets):
                continue
        if sha(repo / name) != expected:
            raise ValueError("Input differs from original run: " + name)
    for cohort in args.datasets:
        if not feature_dir(args.features_root, cohort).is_dir():
            raise FileNotFoundError("Original CONCH feature directory absent: " +
                                    str(feature_dir(args.features_root, cohort)))
    if "cat_tpl" in args.methods and not (Path.home() / ".cache/clip/RN50.pt").is_file():
        raise FileNotFoundError("Existing ~/.cache/clip/RN50.pt required; no automatic download")
    for item in selected:
        run = results / item["run_relative_path"]
        saved_split = run / ("splits_%d.csv" % item["fold"])
        if sha(saved_split) != item["split_sha256"]:
            raise ValueError("Saved split hash changed: " + str(saved_split))
        train, test = read_split(saved_split)
        if set(train) != set(item["train_ids"]) or set(test) != set(item["test_ids"]):
            raise ValueError("Split membership differs from locked result")
        settings = ast.literal_eval((run / item["settings_file"]).read_text().strip())
        for key, value in item["settings"].items():
            if settings.get(key) != value:
                raise ValueError("Original experiment setting differs: " + key)
        if not checkpoint_for(results, item).is_file():
            raise FileNotFoundError("Original last checkpoint absent: " +
                                    str(checkpoint_for(results, item)))
    print("CHECK PASS: %d last checkpoints; matching source, inputs and saved splits." % len(selected), flush=True)
    print("Native historical cohort-wide time bins will be retained and labelled; no retraining.", flush=True)
    return selected


def initialise_model(core, item, sizes):
    kwargs = dict(fusion="concat", omic_sizes=sizes, n_classes=4, wsi_input_dim=512)
    if item["method"] == "mcat":
        return core.MCAT_Surv(**kwargs)
    _, _, prompts = core.get_pathological_tissue_level_prompts(
        multi_templates=False, name="tcga_" + item["cohort"])
    ctx = [prompt + " * * * * * * * * * *" for prompt in prompts]
    learner = core.PromptLearner(
        n_ctx=16, ctx_init=ctx, all_ctx_trainable=False, csc=True,
        classnames=["Prototype %d" % i for i in range(len(ctx))],
        clip_model="RN50", p_drop_out=0.5)
    return core.MCAT_Surv_Clip(prompt_learner_instanceLevel=learner,
                              clip_model="RN50", **kwargs)


def validate_outputs(np, survival, hazards, bins):
    if survival.ndim != 2 or survival.shape[1] != 4 or hazards.shape != survival.shape:
        raise ValueError("Expected four native survival intervals")
    if not np.isfinite(survival).all() or not np.isfinite(hazards).all():
        raise ValueError("Nonfinite model output")
    if np.any(survival < 0) or np.any(survival > 1) or np.any(hazards < 0) or np.any(hazards > 1):
        raise ValueError("Invalid probabilities")
    if np.any(np.diff(survival, axis=1) > 1e-6):
        raise ValueError("Survival curve is not non-increasing")
    if not np.allclose(survival, np.cumprod(1 - hazards, axis=1), atol=1e-6, rtol=0):
        raise ValueError("Hazards and survival probabilities disagree")
    if bins.shape != (5,) or not np.isfinite(bins).all() or not np.all(np.diff(bins) > 0):
        raise ValueError("Invalid native time-bin boundaries")
    if bins[1] <= 0:
        raise ValueError("First positive time coordinate required")


def run_fold(args, item, manifest_sha, torch, np, dataset_class, core):
    checkpoint = checkpoint_for(args.results_root, item)
    tag = "%s/%s/fold_%d" % (item["method"], item["cohort"], item["fold"])
    out = args.output_dir / tag
    data_path, audit_path = out / "curves.npz", out / "audit.json"
    checkpoint_hash = sha(checkpoint)
    if args.mode == "export" and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if (audit.get("complete") and audit.get("manifest_sha256") == manifest_sha
                and audit.get("checkpoint_sha256") == checkpoint_hash
                and data_path.is_file() and audit.get("curves_sha256") == sha(data_path)):
            print("SKIP verified " + tag, flush=True)
            return audit
        raise FileExistsError("Unverified prior output; use a new --output-dir: " + str(out))
    if args.mode == "export" and data_path.exists():
        raise FileExistsError("Orphaned curve file; use a new --output-dir: " + str(out))
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cohort = item["cohort"]
    dataset = dataset_class(
        csv_path=str(args.repo_root / ("dataset_csv_sig/tcga_%s_all_clean.csv.zip" % cohort)),
        mode="coattn", apply_sig=True,
        signatures_path=str(args.repo_root / "dataset_csv_sig/signatures.csv"),
        data_dir=str(feature_dir(args.features_root, cohort)), shuffle=False,
        seed=1, print_info=False, patient_strat=False, n_bins=4,
        label_col="survival_months", ignore=[])
    run = args.results_root / item["run_relative_path"]
    train, heldout = dataset.return_splits(
        from_id=False, csv_path=str(run / ("splits_%d.csv" % item["fold"])))
    ids = heldout.slide_data.slide_id.astype(str).tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(item["test_ids"]):
        raise ValueError("Dataset loader changed held-out patients")
    bins = np.asarray(dataset.bins, dtype=np.float64)
    sizes = [len(names) for names in train.omic_names]
    if len(sizes) != 6:
        raise ValueError("Expected six genomic signatures")
    model = initialise_model(core, item, sizes)
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    state = torch.load(str(checkpoint), **load_kwargs)
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    probabilities, hazards_all, risks, times, events, labels = [], [], [], [], [], []
    reference = item["predictions"]
    maximum_delta = 0.0
    started = time.monotonic()
    count = min(1, len(ids)) if args.mode == "smoke" else len(ids)
    with torch.no_grad():
        for index in range(count):
            sample = heldout[index]
            if len(sample) != 10 or sample[0].ndim != 2 or sample[0].shape[1] != 512:
                raise ValueError("Unexpected original co-attention input")
            inputs = {"x_path": sample[0].to(args.device)}
            inputs.update({"x_omic%d" % j: sample[j].float().to(args.device) for j in range(1, 7)})
            hazard, survival, _, _ = model(**inputs)
            native = survival.detach().cpu().numpy().reshape(1, -1)
            native_hazards = hazard.detach().cpu().numpy().reshape(1, -1)
            validate_outputs(np, native, native_hazards, bins)
            risk = float((-survival.sum(dim=1)).item())
            original = reference[ids[index]]
            delta = abs(risk - original["risk"])
            maximum_delta = max(maximum_delta, delta)
            if delta > 1e-5:
                raise ValueError("Original scalar risk not reproduced for %s: delta=%g" % (ids[index], delta))
            if (abs(float(sample[8]) - original["time"]) > 1e-4
                    or float(sample[9]) != original["censorship"]
                    or int(sample[7]) != original["disc_label"]):
                raise ValueError("Patient labels/native bins changed")
            probabilities.append(native[0])
            hazards_all.append(native_hazards[0])
            risks.append(risk)
            times.append(float(sample[8]))
            events.append(1.0 - float(sample[9]))
            labels.append(int(sample[7]))
            if (index + 1) % 50 == 0:
                print("%s patients %d/%d" % (tag, index + 1, count), flush=True)
    if sha(checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during inference")
    audit = {"complete": args.mode == "export", "scope": args.mode,
             "method": item["method"], "cohort": cohort, "fold": item["fold"],
             "n": count, "new_training": False, "checkpoint": checkpoint.name,
             "checkpoint_sha256": checkpoint_hash, "manifest_sha256": manifest_sha,
             "max_scalar_risk_abs_delta": maximum_delta, "risk_tolerance": 1e-5,
             "native_bins": bins.tolist(), "signature_sizes": sizes,
             "time_coordinates": "[0] + native upper interval boundaries; S(0)=1",
             "historical_time_bins": "cohort-wide event quantiles; not train-only",
             "tail": "Native final probability preserved; no forced zero endpoint or recalibration",
             "seconds": time.monotonic() - started,
             "torch": torch.__version__, "numpy": np.__version__, "python": sys.version}
    if args.mode == "export":
        out.mkdir(parents=True, exist_ok=True)
        survival = np.column_stack([np.ones(count), np.asarray(probabilities)])
        with data_path.open("xb") as stream:
            np.savez_compressed(stream, patient_ids=np.asarray(ids),
                                time_coordinates=np.r_[0.0, bins[1:]], native_bins=bins,
                                survival_curves=survival, hazards=np.asarray(hazards_all),
                                native_risk=np.asarray(risks), event_time=np.asarray(times),
                                event=np.asarray(events), discrete_label=np.asarray(labels),
                                train_ids=np.asarray(train.slide_data.slide_id.astype(str).tolist()),
                                train_time=train.slide_data.survival_months.to_numpy(dtype=float),
                                train_event=1.0 - train.slide_data.censorship.to_numpy(dtype=float))
        audit["curves_sha256"] = sha(data_path)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(("COMPLETE " if args.mode == "export" else "SMOKE ONLY; NOT AN EXPERIMENTAL RESULT ") +
          tag + " risk_delta=%g" % maximum_delta, flush=True)
    del model, state, dataset, train, heldout
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "smoke", "export"])
    parser.add_argument("--repo-root", type=Path,
                        default=Path("/tmp/pycharm_project_456/external/CAT-with-TPL-master"))
    parser.add_argument("--results-root", type=Path,
                        help="Defaults to repo-root/results-conch, including its existing archive symlink")
    parser.add_argument("--features-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().with_name("original_run_manifest.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/autodl-tmp/qaclf_external_curves"))
    parser.add_argument("--datasets", nargs="+", choices=COHORTS, default=list(COHORTS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--methods", nargs="+", choices=["mcat", "cat_tpl"], default=["mcat", "cat_tpl"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    for name in ["datasets", "folds", "methods"]:
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error("Duplicate selections are not allowed")
    args.repo_root = args.repo_root.resolve()
    args.results_root = (args.results_root or args.repo_root / "results-conch").resolve()
    if args.mode != "check" and (args.output_dir.resolve() == args.results_root
                                or args.results_root in args.output_dir.resolve().parents):
        parser.error("Output must be separate from retained training results")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = preflight(args, manifest)
    if args.mode == "check":
        return
    import numpy as np
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in current environment; activate original vlsa_env")
    os.chdir(str(args.repo_root))
    sys.path.insert(0, str(args.repo_root))
    from datasets.dataset_survival import Generic_MIL_Survival_Dataset
    from utils import core_utils as core
    audits = [run_fold(args, item, sha(args.manifest), torch, np,
                       Generic_MIL_Survival_Dataset, core) for item in selected]
    if args.mode == "export":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        full = len(audits) == 50 and all(a["complete"] for a in audits)
        summary = {"all_50_folds_complete": full, "requested_folds_complete": True,
                   "folds": audits, "manifest_sha256": sha(args.manifest), "new_training": False}
        name = "export_summary_%s_%s.json" % ("_".join(args.methods), "_".join(args.datasets))
        (args.output_dir / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Export verified. Fold count=%d; full fifty=%s" % (len(audits), full), flush=True)


if __name__ == "__main__":
    main()
