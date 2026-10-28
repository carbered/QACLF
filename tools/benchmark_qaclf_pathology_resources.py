"""Measure a retained pathology checkpoint and train-fold-only QACLF state.

This benchmark never changes the locked scientific predictions.
"""

import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from model.vlsa import VLSA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from scipy.integrate import IntegrationWarning
from compare_paired_survival_predictions import build_meta, predicted_event_times
from probe_cat_omics_signal import fit_cox, signature_features
from probe_cat_omics_complementarity import train_empirical_cdf
from evaluate_qaclf_survival_curves import monotone_breslow_projection, score_to_risk_feature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True,
                        help="Directory containing BLCA CONCH .pt files")
    parser.add_argument("--output", type=Path, default=ROOT / "result/qaclf-dataset-resource-audit/pathology_benchmark.json")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    folder = args.baseline_dir
    cfg = yaml.safe_load((folder / "config.yaml").read_text())
    def sub(prefix):
        return {k[len(prefix) + 1:]: v for k, v in cfg.items() if k.startswith(prefix + "_")}
    prompt = sub("vlsa_pmt_learner_coop")
    prompt.update(name="CoOp", pretrained=False)
    model = VLSA(text_encoder_cfg=sub("vlsa_txt_encoder"),
                 image_encoder_cfg=sub("vlsa_img_encoder"), prompt_learner_cfg=prompt,
                 vlsa_api="CONCH", path_clip_model=str(args.weights_root))
    checkpoint = folder / "train_model-last.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("prompt_encoder.") for k in missing), (missing, unexpected)
    for parameter in model.prompt_encoder.parameters():
        parameter.requires_grad_(False)
    trainable = {k: p.numel() for k, p in model.named_parameters() if p.requires_grad}
    assert set(trainable) == set(state), (trainable.keys(), state.keys())
    model.cuda().eval()
    labels = pd.read_csv(ROOT / cfg["path_table"])
    ref = pd.read_csv(folder / "bag_clf_train_last_pred_test.csv").iloc[0]
    slides = labels[labels.patient_id == ref.patient_id].pathology_id
    tensor = torch.cat([torch.load(args.features_root / (sid + ".pt"),
                                  map_location="cpu", weights_only=True) for sid in slides]).unsqueeze(0).cuda()
    meta = build_meta(cfg)
    coordinates = meta.time_coordinates
    train_ref = pd.read_csv(folder / "bag_clf_train_last_pred_train.csv")
    train_ids = train_ref.patient_id.tolist()
    catroot = ROOT / "external/CAT-with-TPL-master/dataset_csv"
    table = pd.read_csv(catroot / "tcga_blca_all_clean.csv.zip", low_memory=False).drop_duplicates("case_id").set_index("case_id")
    features, _ = signature_features(table.columns, pd.read_csv(catroot / "signatures.csv"))
    train_raw = table.loc[train_ids, features].to_numpy(dtype=float)
    test_raw = table.loc[[ref.patient_id], features].to_numpy(dtype=float)
    train_time = table.loc[train_ids, "survival_months"].to_numpy(dtype=float)
    train_event = 1 - table.loc[train_ids, "censorship"].to_numpy(dtype=float)
    scaler = StandardScaler().fit(train_raw)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=42)
    train_x = pca.fit_transform(scaler.transform(train_raw))
    weight, bias, _ = fit_cox(train_x, train_time, train_event, 1200, .02, .1)
    omics_reference = -(train_x @ weight + bias)
    path_reference = predicted_event_times(train_ref[[f"surf_{i}" for i in range(1, 13)]].to_numpy(dtype=float), coordinates)
    train_q = .8 * train_empirical_cdf(path_reference, path_reference) + .2 * train_empirical_cdf(omics_reference, omics_reference)
    _, base_curve, scale, _ = monotone_breslow_projection(train_q, np.array([.5]), train_time, train_event, coordinates)
    sorted_path, sorted_omics = np.sort(path_reference), np.sort(omics_reference)
    def infer_path():
        output = model(tensor)
        output = output[0] if isinstance(output, tuple) else output
        curves = (1 - output.float().softmax(dim=-1).cumsum(dim=-1)).cpu().numpy().astype(float)
        return curves, predicted_event_times(curves, coordinates)
    def infer_qaclf():
        _, path_time = infer_path()
        x = pca.transform(scaler.transform(test_raw))
        omics_time = -(x @ weight + bias)
        q = (.8 * np.searchsorted(sorted_path, path_time, side="right")
             + .2 * np.searchsorted(sorted_omics, omics_time, side="right")) / len(train_ids)
        risk = np.exp(scale * score_to_risk_feature(q, len(train_ids)))
        curve = np.power(base_curve, risk[:, None])
        curve[:, 0], curve[:, -1] = 1, 0
        return curve, predicted_event_times(curve, coordinates)
    with torch.inference_mode():
        result = model(tensor)
        logits = result[0] if isinstance(result, tuple) else result
        survival = (1 - torch.softmax(logits.float(), dim=-1).cumsum(dim=-1)).cpu().numpy().ravel()
        expected = ref[[f"surf_{i}" for i in range(1, 13)]].to_numpy(dtype=float)
        assert np.max(np.abs(survival - expected)) < 2e-5, (survival, expected)
        uncached = logits.clone()
        model.register_buffer("pretrained_text_features", model.forward_text_only().detach().clone(), persistent=False)
        cached = model(tensor)
        cached_logits = cached[0] if isinstance(cached, tuple) else cached
        assert torch.allclose(uncached, cached_logits, atol=1e-5)
        for _ in range(20):
            model(tensor)
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            start = time.perf_counter()
            model(tensor)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - start))
        for _ in range(20):
            infer_path()
            infer_qaclf()
        full_times = {"VLSA": [], "QACLF": []}
        for repeat in range(100):
            order = [("VLSA", infer_path), ("QACLF", infer_qaclf)]
            if repeat % 2:
                order.reverse()
            for name, function in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                curve, predicted_time = function()
                torch.cuda.synchronize()
                full_times[name].append(1000 * (time.perf_counter() - start))
                assert np.isfinite(curve).all() and np.isfinite(predicted_time).all()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            model(tensor)
            torch.cuda.synchronize()
        flop_ops = [{"op": item.key, "flops": item.flops} for item in prof.key_averages() if item.flops]
        linear_flops = sum(item["flops"] for item in flop_ops if item["op"] in ["aten::mm", "aten::addmm", "aten::bmm"])
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "trainable_pathology_parameters": sum(trainable.values()), "parameter_breakdown": trainable,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
              "cpu": subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], text=True).strip(),
              "patient": str(ref.patient_id), "patches": tensor.shape[1], "feature_dim": tensor.shape[2],
              "reproduced_curve_max_absolute_error": float(np.max(np.abs(survival - expected))),
              "cached_prompt_outputs_verified": True, "warmup": 20, "repeats": 100,
              "pathology_inference_median_ms": float(np.median(times)),
              "pathology_inference_mean_ms": float(np.mean(times)),
              "pathology_inference_p95_ms": float(np.percentile(times, 95)),
              "omics_features": len(features), "incremental_cox_parameters": len(weight) + 1,
              "profiled_operations": flop_ops, "pathology_linear_macs": int(linear_flops / 2),
              "omics_linear_macs": len(features) * 64 + 64,
              "complete_times": {name: {"median_ms": float(np.median(values)), "mean_ms": float(np.mean(values)),
                                        "p95_ms": float(np.percentile(values, 95))} for name, values in full_times.items()},
              "scope": "BLCA fold-0 representative fixed-input benchmark; native retained checkpoint and training-fold-only fitted omics/calibration state. Patient-independent text cached and cache equivalence verified. Resident CONCH patch features to survival curve AND mean survival time. QACLF includes scaler, PCA, Cox, both ECDFs, fusion and Cox-Breslow curve. Excludes WSI encoding, disk I/O and input host-to-device transfer. This benchmark refit does not replace locked scientific predictions.",
              "mac_scope": "Dense matrix products/linear layers only; normalization, softmax, ECDF, log/exp and curve operations excluded. Multiply-accumulate=2 floating-point operations."}
    report["timing_warning_policy"] = "Known adaptive-quadrature IntegrationWarning output suppressed during timing; both returned curves and mean times checked finite. Primary evaluation code unchanged."
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
