# Data setup

## Current QACLF release

The method is now named QACLF (formerly FICPF); historical data paths are retained.
Current summaries and 225 compact probability-curve inputs are under
`result/qaclf-nested-v1-unified-evaluation-20261027/`. The NPZ inputs contain
deidentified patient IDs, finite time coordinates, survival probabilities and,
for MCAT/CAT-with-TPL, native risk scores. They are not checkpoints or features.
`curve_inputs_manifest.json` records public-file and original-source SHA256 values.
Canonical outcomes and train/test membership come from `data_split/5foldcv/`.

`protocol/qaclf_nested_v1/` includes the exact cloud code, contracts and frozen
inner plans. The `results/` subdirectory includes 50 completed-run summaries and
epoch-selection records, not weights, scaler feature states or training logs.
Current metric verification needs no additional patient data. Re-training and
refitting still require the omitted upstream inputs below. Historical full-window
and old-comparator directories are retained for audit, not current tables.

## Included data

This repository includes fixed patient-level survival tables and outer five-fold
splits under `data_split/`, the CAT-with-TPL gene-signature definition, and the
locked patient-level predictions required to audit the reported results.

## Omitted large inputs

The preprocessed RNA-seq/CNV tables are omitted to keep the repository lightweight
and to avoid silently redistributing inherited TCGA derivatives. Obtain the
data under the applicable NCI Genomic Data Commons policy and reproduce the
CAT-with-TPL preprocessing. Place the complete tables here:

```text
external/CAT-with-TPL-master/dataset_csv/
  signatures.csv
  tcga_blca_all_clean.csv.zip
  tcga_brca_all_clean.csv.zip
  tcga_gbmlgg_all_clean.csv.zip
  tcga_luad_all_clean.csv.zip
  tcga_ucec_all_clean.csv.zip
```

Expected SHA256 values for the exact complete tables used in the locked analysis:

| File | SHA256 |
|---|---|
| `tcga_blca_all_clean.csv.zip` | `386e0adf343eb8b947385d175133a53b75e90a152ae2e0ceb1c180f82c8037e8` |
| `tcga_brca_all_clean.csv.zip` | `ebd2d31abb871ae1aec59606a9c4a5f7280fe238ce440ab376c70cac8bf6e974` |
| `tcga_gbmlgg_all_clean.csv.zip` | `6b4fc7a938dbf49bcf536f17e24eb52a8fc0721ac9243ab6fd671194d157d345` |
| `tcga_luad_all_clean.csv.zip` | `e90a1a28c79cab829737c56c2306cd7912bbf7b4485828a507d6b82033a174d1` |
| `tcga_ucec_all_clean.csv.zip` | `215bf025477bcf8ff6c7865f8c7a9a1a26ee0619eb36aba64618bd6f3e9d8e1b` |

Some unified external-baseline and ablation scripts also consume the
signature-filtered CAT-with-TPL tables under `dataset_csv_sig/`. Their hashes are
recorded in `manifest/omitted_data_sha256.csv`.

## Upstream pathology inputs

Raw WSIs, CONCH weights, WSI patch embeddings, and VLSA checkpoints are not
included. They are not needed to verify the frozen patient-level analysis, but
they are required to regenerate pathology predictions from the beginning.

## Additional retained results and figure assets

`result/retained-mil-inputs/` includes the test survival curves and original
configs for ABMIL, TransMIL, ILRA and DSMIL (25 held-out folds each), together
with the exact archived label-conversion source used to interpret those runs.
Configs retain original machine paths for provenance; verification uses the
bundled patient tables and splits instead. No checkpoint weights are included.

`result/ficpf-dataset-resource-audit/` contains per-slide and per-patient patch
counts, cohort statistics and the representative resource benchmark. Statistics
can be re-audited in a full workspace with the omitted patch features:

```bash
python tools/audit_ficpf_dataset_resources.py --features-root /path/to/datasets \
  --config-root result/retained-mil-inputs/dsmil \
  --output-dir result/reproduced_dataset_statistics
```

`tools/benchmark_ficpf_pathology_resources.py` requires the upstream VLSA model
code/environment, a retained BLCA fold-0 checkpoint, CONCH weights, patch
features and the omics table. It does not run in this lightweight analysis-only
repository by itself. Its CLI accepts `--baseline-dir`, `--weights-root`,
`--features-root` and `--output`; input files are not downloaded by the script.

`figures/ficpf_20260821/real_tcga_inputs/TCGA-3C-AALI/` contains only the small
deidentified TCGA example overview, three example regions, omics heatmap and
provenance needed to regenerate the method schematic, not the raw WSI. These
source examples are not quantitative experimental evidence for the plotted
schematic feature blocks or curves.

## Integrity check

After adding any omitted table, verify it with:

```bash
sha256sum external/CAT-with-TPL-master/dataset_csv/*.zip
```

On Windows PowerShell, use `Get-FileHash <path> -Algorithm SHA256`.
