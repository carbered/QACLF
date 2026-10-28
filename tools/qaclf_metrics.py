"""Exact fixed-window metric function from the retained nested-v1 runtime."""
import numpy as np


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
