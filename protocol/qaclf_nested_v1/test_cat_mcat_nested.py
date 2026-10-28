"""Nested protocol safety tests; synthetic observations are not study results."""

import ast
import types
import unittest

import numpy as np
import pandas as pd

import run_cat_mcat_nested as run


def patients(n=80):
    return pd.DataFrame(dict(survival_months=np.arange(1, n + 1, dtype=float) * 2,
                             censorship=np.tile([0., 1.], n // 2)),
                        index=["p%d" % i for i in range(n)])


class NestedProtocolTests(unittest.TestCase):
    def test_bins_use_fitting_patients_only(self):
        frame = patients()
        fitting = frame.index[:60].tolist()
        expected = run.fit_bins(frame, fitting)
        frame.loc[frame.index[60:], "survival_months"] = 1e8
        frame.loc[frame.index[60:], "censorship"] = 0
        np.testing.assert_array_equal(run.fit_bins(frame, fitting), expected)

    def test_overflow_is_censored_only_in_encoded_copy(self):
        frame = patients(20)
        original = frame.copy(deep=True)
        bins = np.array([0., 5., 10., 15., 20.000001])
        time, c, labels, overflow = run.encode_outcomes(frame, bins)
        self.assertTrue(np.all(c[overflow] == 1))
        self.assertTrue(np.all(labels[overflow] == 3))
        self.assertLess(time.max(), bins[-1])
        pd.testing.assert_frame_equal(frame, original)

    def test_inner_partition_excludes_outer_test(self):
        frame = patients()
        plan = run.make_plan(frame, frame.index[:60], frame.index[60:], "blca", 0)
        self.assertFalse(set(plan["inner_fit"]) & set(plan["outer_test"]))
        self.assertFalse(set(plan["inner_validation"]) & set(plan["outer_test"]))
        self.assertEqual(set(plan["inner_fit"]) | set(plan["inner_validation"]), set(plan["outer_train"]))

    def test_outer_outcomes_do_not_change_selection_plan(self):
        frame = patients()
        first = run.make_plan(frame, frame.index[:60], frame.index[60:], "blca", 0)
        frame.loc[frame.index[60:], "survival_months"] = 1e6
        frame.loc[frame.index[60:], "censorship"] = 0
        second = run.make_plan(frame, frame.index[:60], frame.index[60:], "blca", 0)
        self.assertEqual(first, second)

    def test_overlap_rejected(self):
        frame = patients()
        with self.assertRaisesRegex(ValueError, "overlap"):
            run.make_plan(frame, frame.index[:60], frame.index[59:], "blca", 0)

    def test_missing_binary_censorship_rejected(self):
        raw = pd.DataFrame(dict(case_id=["a"], slide_id=["s"], survival_months=[10.], censorship=[2.]))
        with self.assertRaisesRegex(ValueError, "Invalid"):
            run.canonical_patients(raw)

    def test_conflicting_slide_outcomes_rejected(self):
        raw = pd.DataFrame(dict(case_id=["a", "a"], slide_id=["s", "t"],
                                survival_months=[10., 11.], censorship=[0., 0.]))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            run.canonical_patients(raw)

    def test_selection_earliest_tie(self):
        self.assertEqual(run.select_epoch([2., 1., 1., 3.]), 2)

    def test_selection_nonfinite_rejected(self):
        for values in [[], [1., np.nan], [np.inf]]:
            with self.assertRaisesRegex(ValueError, "Invalid"):
                run.select_epoch(values)

    def test_degenerate_bins_rejected(self):
        frame = patients()
        frame.survival_months = 10.
        with self.assertRaises(ValueError):
            run.fit_bins(frame, frame.index)

    def test_stage_filters_outer_rows_without_mutating_raw(self):
        frame = patients(20).reset_index().rename(columns={"index": "case_id"})
        frame["slide_id"] = frame.case_id + "-slide"
        original = frame.copy(deep=True)
        stage = run.prepare_stage_frame(frame, frame.case_id[:10], frame.case_id[10:14], [0., 5., 10., 15., 30.])
        self.assertEqual(set(stage.case_id), set(frame.case_id[:14]))
        pd.testing.assert_frame_equal(frame, original)

    def test_constructor_removes_global_fitting(self):
        source = '''class Generic_WSI_Survival_Dataset:
    def __init__(self):
        slide_data = pd.read_csv("must-not-be-read")
        disc_labels, q_bins = pd.qcut(slide_data["time"], q=4, retbins=True)
        q_bins[-1] = slide_data["time"].max()
        q_bins[0] = slide_data["time"].min()
        self.frame, self.bins = slide_data, q_bins
'''
        frame = pd.DataFrame(dict(time=[1., 2.]))
        init = run.patched_constructor(source, {}, frame, [0., 1., 2., 3., 4.])
        instance = types.SimpleNamespace()
        init(instance)
        np.testing.assert_array_equal(instance.bins, [0., 1., 2., 3., 4.])
        pd.testing.assert_frame_equal(instance.frame, frame)

    def test_unexpected_constructor_rejected(self):
        source = '''class Generic_WSI_Survival_Dataset:
    def __init__(self):
        pass
'''
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            run.patched_constructor(source, {}, pd.DataFrame(), [0., 1., 2., 3., 4.])

    def test_python38_syntax(self):
        ast.parse((run.HERE / "run_cat_mcat_nested.py").read_text(encoding="utf-8"), feature_version=(3, 8))

    def test_nonfinite_gradient_update_is_refused(self):
        import torch
        model = torch.nn.Linear(1, 1)
        opt = run.guarded_optimizer(torch, model, lambda value: None)
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite gradient"):
            opt.step()

    def test_empty_gradient_update_is_refused(self):
        import torch
        model = torch.nn.Linear(1, 1)
        opt = run.guarded_optimizer(torch, model, lambda value: None)
        with self.assertRaisesRegex(RuntimeError, "no gradients"):
            opt.step()

    def test_fixed_metrics_retain_unbounded_diagnostic_flag(self):
        import sys
        sys.path.insert(0, str(run.HERE.parent))
        curves = np.array([[1., 1., 1., 1., 1.], [1., .9, .6, .4, .1]])
        x = np.array([0., 12., 36., 60., 150.])
        actual = np.array([[20., 1.], [80., 0.]])
        train = np.array([[10., 1.], [40., 0.], [120., 1.], [150., 0.]])
        result = run.fixed_metrics(curves, x, actual, train)
        self.assertIsNone(result["diagnostic_full_mean_mae"])
        self.assertEqual(result["diagnostic_unbounded_means"], 1)
        self.assertTrue(np.isfinite(result["ibs_12_60"]))


if __name__ == "__main__":
    unittest.main()
