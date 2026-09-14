"""Skill-score metrics, ported from the reference paper's
`model/evaluation.py` (github.com/W-US-BCSR/IRnAF) with one change: every
function takes an explicit `mask` (True = valid IMD cell) instead of relying
on NaN-masking alone, since our domain has systematic ocean/no-gauge
coverage rather than occasional missing values.
"""
from __future__ import annotations

import numpy as np


def _weighted_mean(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean over the last two (lat, lon) axes.

    `x` may have NaN outside the valid mask (padding, no-gauge cells) --
    those entries carry zero weight, but `NaN * 0 == NaN` in IEEE floats,
    so they're zeroed explicitly rather than left to the weighting alone.
    """
    x = np.where(weights > 0, x, 0.0)
    w = weights / weights.sum()
    return np.tensordot(x, w, axes=([-2, -1], [0, 1]))


def scorr(pred: np.ndarray, obs: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    """Latitude-weighted spatial (pattern) correlation over one (lat, lon) map."""
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    if weights.sum() == 0:
        return np.nan

    pred_mean = _weighted_mean(pred, weights)
    obs_mean = _weighted_mean(obs, weights)
    pred_a = pred - pred_mean
    obs_a = obs - obs_mean

    cov = _weighted_mean(pred_a * obs_a, weights)
    sd_pred = np.sqrt(_weighted_mean(pred_a ** 2, weights))
    sd_obs = np.sqrt(_weighted_mean(obs_a ** 2, weights))

    if sd_pred == 0 or sd_obs == 0:
        return np.nan
    return float(cov / (sd_pred * sd_obs))


def rmses(pred: np.ndarray, obs: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    """Latitude-weighted RMSE over one (lat, lon) map."""
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    if weights.sum() == 0:
        return np.nan
    mse = _weighted_mean((pred - obs) ** 2, weights)
    return float(np.sqrt(mse))


def epre(pred: np.ndarray, obs: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    """Eq. 1 of the reference paper: a Taylor-diagram-style score combining
    variance-ratio mismatch and correlation. 0 for a perfect forecast,
    increasing with either variance mismatch or decorrelation."""
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
    return float(np.log((ratio ** 2) * 16 / (1 + r) ** 4))


def skill_scores_by_lead(
    pred: np.ndarray, obs: np.ndarray, lat: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    """`pred`/`obs`: (n_samples, n_lead, lat, lon). Returns per-lead arrays
    (mean and std across samples) for pattern correlation, RMSE, and E_pre."""
    n_samples, n_lead = pred.shape[:2]
    corr = np.full((n_samples, n_lead), np.nan)
    rmse = np.full((n_samples, n_lead), np.nan)
    epr = np.full((n_samples, n_lead), np.nan)

    for i in range(n_samples):
        for t in range(n_lead):
            corr[i, t] = scorr(pred[i, t], obs[i, t], lat, mask)
            rmse[i, t] = rmses(pred[i, t], obs[i, t], lat, mask)
            epr[i, t] = epre(pred[i, t], obs[i, t], lat, mask)

    return {
        "pattern_corr_mean": np.nanmean(corr, axis=0),
        "pattern_corr_std": np.nanstd(corr, axis=0),
        "rmse_mean": np.nanmean(rmse, axis=0),
        "rmse_std": np.nanstd(rmse, axis=0),
        "epre_mean": np.nanmean(epr, axis=0),
        "epre_std": np.nanstd(epr, axis=0),
    }
