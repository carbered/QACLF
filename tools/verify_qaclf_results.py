"""Verify current QACLF results; optionally recompute all fixed-window metrics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'result/qaclf-nested-v1-unified-evaluation-20261027'
COHORTS = ['BLCA', 'BRCA', 'GBMLGG', 'LUAD', 'UCEC']
EXPECTED = {'ABMIL': .6215081008075491, 'TransMIL': .6447394828654323,
            'ILRA': .6363855177059747, 'DSMIL': .6591261613970387,
            'VLSA': .695781808041706, 'MCAT': .6793235510689801,
            'CAT-with-TPL': .6696626840832384, 'QACLF': .7050168897889689,
            'VLSA same projection': .6958521144207692}
KEYS = ['c_index', 'ibs_12_60', 'brier_12', 'brier_36', 'brier_60']
LABELS = {'ABMIL-CONCH': 'ABMIL', 'TransMIL-CONCH': 'TransMIL', 'ILRA-CONCH': 'ILRA',
          'DSMIL-CONCH': 'DSMIL', 'Matched VLSA': 'VLSA', 'MCAT-CONCH': 'MCAT',
          'CAT-TPL-CONCH': 'CAT-with-TPL', 'FICPF': 'QACLF'}


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def near(actual, expected, tolerance=1e-9):
    assert math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0), (actual, expected)


def verify(recompute=False, recompute_probability=False):
    from verify_historical_results import verify as historical_verify
    historical = historical_verify(recompute)
    folds = rows(CURRENT / 'all_fold_metrics.csv')
    datasets = rows(CURRENT / 'dataset_metrics.csv')
    macros = rows(CURRENT / 'macro_metrics.csv')
    assert len(folds) == 225 and len(datasets) == 45 and len(macros) == 9
    assert {r['method'] for r in folds} == set(EXPECTED)
    assert len({(r['method'], r['dataset'], int(r['fold'])) for r in folds}) == 225
    for row in datasets:
        selected = [x for x in folds if x['method'] == row['method'] and x['dataset'] == row['dataset']]
        assert {int(x['fold']) for x in selected} == set(range(5)) and len(selected) == 5
        for key in KEYS:
            near(row[key], statistics.fmean(float(x[key]) for x in selected))
    for row in macros:
        selected = [x for x in datasets if x['method'] == row['method']]
        assert {x['dataset'] for x in selected} == set(COHORTS)
        for key in KEYS:
            near(row[key], statistics.fmean(float(x[key]) for x in selected))
        near(row['c_index'], EXPECTED[row['method']])
        count = sum(float(x['d_calibration_p_admin60']) >= .05 for x in folds if x['method'] == row['method'])
        assert count == int(row['d_cal_not_rejected_admin60'])
    comparison = rows(CURRENT / 'updated_eight_method_dataset_summary.csv')
    assert len(comparison) == 8
    for row in comparison:
        method = LABELS[row['method']]
        near(row['MACRO'], EXPECTED[method])
        for cohort in COHORTS:
            selected = [float(x['c_index']) for x in folds if x['method'] == method and x['dataset'] == cohort]
            near(row[cohort], statistics.fmean(selected))
            near(row[cohort + '_std'], statistics.pstdev(selected))
    code = ROOT / 'protocol/qaclf_nested_v1'
    contract = json.loads((code / 'nested_contract.json').read_text())
    for name, digest in contract['runtime_sha256'].items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest, name
    audit = json.loads((CURRENT / 'download_audit.json').read_text())
    assert audit['complete_jobs'] == 50 and audit['split_scaler_bin_and_epoch_checks']
    for method in ['mcat', 'cat_tpl']:
        for cohort in COHORTS:
            for fold in range(5):
                job = code / 'results' / method / cohort.lower() / f'fold_{fold}'
                summary = json.loads((job / 'summary.json').read_text())
                selection = json.loads((job / 'selection_frozen_before_outer_evaluation.json').read_text())
                losses = selection['inner_validation_nll']
                assert len(losses) == 20 and all(math.isfinite(v) for v in losses)
                chosen = min(range(20), key=losses.__getitem__) + 1
                assert chosen == selection['epochs'] == summary['selected_epochs']
                assert selection['outer_evaluations_so_far'] == 0 and summary['outer_evaluation_count'] == 1
    inputs = json.loads((CURRENT / 'curve_inputs_manifest.json').read_text())
    assert len(inputs) == 225
    assert {(x['method'], x['dataset'], int(x['fold'])) for x in inputs} == {
        (x['method'], x['dataset'], int(x['fold'])) for x in folds}
    for item in inputs:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256']
    recomputed = 0
    if recompute_probability:
        import sys
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from contextlib import redirect_stdout
        import io
        import warnings
        from qaclf_metrics import fixed_metrics
        from eval.cindex import concordance_index_censored
        lookup = {(x['method'], x['dataset'], int(x['fold'])): x for x in folds}
        for item in inputs:
            method, cohort, fold = item['method'], item['dataset'], int(item['fold'])
            directory = ROOT / 'data_split/5foldcv' / ('tcga_' + cohort.lower())
            labels = {x['patient_id']: x for x in rows(directory / ('mahmoodlab_tcga_' + cohort.lower() + '_survival.csv'))}
            split = rows(directory / f'splits_{fold}.csv')
            train_ids = [x['train'] for x in split if x['train']]
            test_ids = [x['val'] for x in split if x['val']]
            assert not set(train_ids) & set(test_ids)
            with np.load(ROOT / item['path'], allow_pickle=False) as z:
                assert z['patient_ids'].astype(str).tolist() == test_ids
                curves, times = z['survival_curves'], z['time_coordinates']
                assert np.isfinite(curves).all() and np.isfinite(times).all()
                assert np.diff(times).min() > 0 and curves.shape == (len(test_ids), len(times))
                assert curves.min() >= -1e-6 and curves.max() <= 1+1e-6
                assert np.diff(curves, axis=1).max() <= 1e-6
                actual = np.array([[float(labels[p]['t']), float(labels[p]['e'])] for p in test_ids])
                train = np.array([[float(labels[p]['t']), float(labels[p]['e'])] for p in train_ids])
                expected = lookup[method, cohort, fold]
                with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                    warnings.simplefilter('ignore')
                    values = fixed_metrics(curves, times, actual, train)
                for key, value in values.items():
                    if value is None:
                        assert expected[key] in ('', 'nan')
                    else:
                        near(value, expected[key], max(1e-9, abs(value)*1e-8))
                if 'native_risk' in z:
                    value = concordance_index_censored(actual[:,1].astype(bool), actual[:,0], z['native_risk'], tied_tol=1e-8)[0]
                    near(value, expected['c_index'])
            recomputed += 1
    return dict(model='QACLF', methods=8, mapping_controls=1, folds_verified=225,
                complete_nested_runs=50, curve_input_hashes_verified=225,
                probability_folds_recomputed=recomputed, training_executed=False,
                historical_snapshot_verified=historical,
                checkpoint_weights_rechecked=False,
                scope='Retained predictions and records, not a new training replication')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-c-index', action='store_true', help='Recheck 100 historical MIL test folds')
    parser.add_argument('--recompute-probability', action='store_true', help='Recompute all 225 current probability evaluations')
    args = parser.parse_args()
    print(json.dumps(verify(args.recompute_c_index, args.recompute_probability), indent=2))
