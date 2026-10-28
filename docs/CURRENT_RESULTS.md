# Current QACLF Evidence

The 2026-10-28 update publishes QACLF code, retained predictions,
derived results and figures, not the current Chinese Word manuscript.

## Canonical Inputs and Outputs

- `result/qaclf-nested-v1-unified-evaluation-20261027/` is the current comparison.
- `figures/qaclf_current/` contains the current method, ranking, probability,
  ablation/sensitivity, omics robustness and OOF risk-stratification figures.
- `protocol/qaclf_nested_v1/` preserves the exact cloud runtime and 25 inner plans.
- `protocol/qaclf_nested_v1/results/` retains the frozen protocol, plans and 50
  completed-run summaries and pre-outer-evaluation epoch selections.
- `tools/qaclf_metrics.py` contains the unchanged `fixed_metrics` function from
  the retained `run_cat_mcat_nested.py`; its only module-level dependency is NumPy.
- `curve_inputs_manifest.json` links 225 compact probability inputs to their
  original source SHA256 and their public NPZ SHA256. These are predictions,
  not weights or feature embeddings. Native MCAT/CAT risk scores are included.

## Naming and Protocol Records

Distributed labels and paths use QACLF. Naming normalization does not alter
numerical experimental results. Earlier cohort-wide-bin MCAT/CAT adaptations and
old-window curve metrics remain scientific audit records, not current comparators.
Outdated manuscript drafts and superseded figure sets are removed from the current
tree; prior repository commits remain intact. `tools/verify_current_results.py` is a compatibility
entry point for the current verifier. The original October-20 verifier is retained
as `tools/verify_historical_results.py`.

## Reproducibility Boundary

The default verifier checks hashes, selection records, complete 25-fold coverage,
means, population SDs and equal-cohort aggregation. Optional probability
recomputation uses the native finite curve coordinates, canonical labels and
outer-training censoring distribution; no tests are used to fit new parameters.
It does not refit the same-projection control or load/recheck cloud checkpoints.
Checkpoint SHA256 values remain recorded claims from the cloud export, not
independently rehashed local model files. Full training requires the upstream
CAT-with-TPL implementation, cached CONCH/CLIP inputs and data documented in DATA.md.

The fixed-window comparison is not evidence that unrestricted predicted mean
survival times or tails were corrected. The additional mapping control and revised
baseline protocol are post-hoc and share the earlier outer tests. No model/version
is selected from the old and new results by its outer-test performance. No partial
MOTCat run is promoted to a complete five-cohort result.
