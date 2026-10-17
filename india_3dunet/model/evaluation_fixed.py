"""E_pre with its floor corrected, plus week-1..4 reporting helpers.

`model/evaluation.py`'s `epre` returns **1.3863 (= log 4) for a perfect
forecast**, not 0 as its docstring and plan.md S1.5 both state. Verified
directly by feeding the observations in as the prediction.

The formula as written is

    E = log[ (sigma_ratio^2 * 16) / (1 + r)^4 ]     sigma_ratio = s1/s2 + s2/s1

and with s1 == s2 and r == 1 that is log[(4 * 16) / 16] = log 4. For the floor
to sit at 0 the constant must be 4, not 16:

    E = log[ (sigma_ratio^2 * 4) / (1 + r)^4 ]  ->  log[(4 * 4) / 16] = log 1 = 0

Since the difference is a constant offset, **no ranking or comparison in any
existing results file changes** -- every reported E_pre is simply 1.3863 too
high. Fixing it only makes the number readable against its true floor.

The original `model/evaluation.py` is left untouched; import from here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from model.evaluation import _weighted_mean, rmses, scorr

# Forecast days reported as weeks 1-4. Apply these to arrays already passed
# through preprocess/alignment.py, where index t holds IMD date issue_date+1+t,
# so forecast "day 7" is index 6. On the raw cache, index t is IMD issue+t,
# one day early relative to ECMWF.
WEEK_DAYS = [7, 14, 21, 28]
WEEK_INDICES = [d - 1 for d in WEEK_DAYS]

EPRE_OFFSET = float(np.log(4.0))   # the bug: what a perfect forecast scores


def epre_fixed(pred: np.ndarray, obs: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    """Eq. 1 of the reference paper, normalised so a perfect forecast is 0."""
    r = scorr(pred, obs, lat, mask)
    if np.isnan(r):
        return np.nan

    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    pred_mean = _weighted_mean(pred, weights)
    obs_mean = _weighted_mean(obs, weights)
    sd_pred = np.sqrt(_weighted_mean((pred - pred_mean) ** 2, weights))
    sd_obs = np.sqrt(_weighted_mean((obs - obs_mean) ** 2, weights))

    if sd_pred == 0 or sd_obs == 0:
        return np.nan

    ratio = sd_obs / sd_pred + sd_pred / sd_obs
    return float(np.log((ratio ** 2) * 4 / (1 + r) ** 4))


def variance_ratio(pred: np.ndarray, obs: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    """SD(pred) / SD(obs) -- the shrinkage diagnostic behind E_pre.

    Below 1 means the forecast is under-dispersed, i.e. flattened toward the
    mean. The reference paper found exactly this for precipitation in
    coastal/mountain terrain (plan.md S1.6), and literature-review.md S3/S6.1
    flags it as the signature failure of a pure-MSE loss.
    """
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    sd_pred = np.sqrt(_weighted_mean((pred - _weighted_mean(pred, weights)) ** 2, weights))
    sd_obs = np.sqrt(_weighted_mean((obs - _weighted_mean(obs, weights)) ** 2, weights))
    return float(sd_pred / sd_obs) if sd_obs > 0 else np.nan


def scores_by_lead(pred: np.ndarray, obs: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> dict:
    """Per-lead means over samples, with the corrected E_pre and the variance
    ratio added. `pred`/`obs`: (n_samples, n_lead, lat, lon)."""
    n_samples, n_lead = pred.shape[:2]
    corr = np.full((n_samples, n_lead), np.nan)
    rmse = np.full((n_samples, n_lead), np.nan)
    epr = np.full((n_samples, n_lead), np.nan)
    vrat = np.full((n_samples, n_lead), np.nan)

    for i in range(n_samples):
        for t in range(n_lead):
            corr[i, t] = scorr(pred[i, t], obs[i, t], lat, mask)
            rmse[i, t] = rmses(pred[i, t], obs[i, t], lat, mask)
            epr[i, t] = epre_fixed(pred[i, t], obs[i, t], lat, mask)
            vrat[i, t] = variance_ratio(pred[i, t], obs[i, t], lat, mask)

    return {
        "pattern_corr": np.nanmean(corr, axis=0),
        "rmse": np.nanmean(rmse, axis=0),
        "epre": np.nanmean(epr, axis=0),
        "variance_ratio": np.nanmean(vrat, axis=0),
        "pattern_corr_per_sample": corr,
    }
