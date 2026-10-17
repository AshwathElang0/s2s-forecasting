"""The departures-from-climatology reformulation (experiments/exp03_departure_from_normal/DESIGN.md S5.2).

Wraps the existing preprocess/dataset_loader.py read-only: `load_raw`,
`split_dates` and friends are imported unchanged, and everything here operates
on the arrays they return.

The two scalings are global scalars rather than per-cell variances. Dividing
by a local standard deviation blows up in cells and seasons that are almost
always dry, and a global scalar still puts the model anomaly and the observed
anomaly on unit variance -- which is exactly what makes a sigmoid gate the
right parameterisation (experiments/exp03_departure_from_normal/DESIGN.md S3).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from preprocess.climatology import (
    day_of_year,
    inverse_transform,
    model_clim_for_dates,
    obs_clim_for_dates,
    transform,
    valid_mask_native,
)

# preprocess/build_dataset.py's channel order: tp_mean, tp_spread, 12
# circulation means, lag_obs, doy_sin, doy_cos.
TP_MEAN_CHANNEL = 0
LAG_OBS_CHANNEL = -3


def _lag_clim(obs_clim: dict, dates: list[datetime], lead_days: int) -> np.ndarray:
    """(N, lead, H, W) climatology for the lag_obs channel, which holds the
    observation from issue_date - 1 broadcast across every lead day.

    Indexed by the sample's own year, so it uses the same causal climatology
    as the rest of that sample."""
    per_date = np.stack([
        obs_clim[d.year][day_of_year(d - timedelta(days=1)) - 1] for d in dates
    ])
    return np.repeat(per_date[:, None], lead_days, axis=1)


def anomaly_fields(features, target, target_var, dates, obs_clim, model_clim):
    """Unscaled anomalies: (target_anom, raw_anom, lag_anom), each
    (N, lead, H, W) on the transformed scale. Division by the global sigmas
    happens later so the sigmas can come from the train split only."""
    lead_days = features.shape[2]
    obs_batch = obs_clim_for_dates(obs_clim, dates, lead_days)
    mdl_batch = model_clim_for_dates(model_clim, dates)

    target_anom = transform(target[:, 0], target_var) - obs_batch
    raw_anom = np.log1p(np.clip(features[:, TP_MEAN_CHANNEL], 0, None)) - mdl_batch
    lag_anom = transform(features[:, LAG_OBS_CHANNEL], target_var) - _lag_clim(obs_clim, dates, lead_days)

    return target_anom, raw_anom, lag_anom


def compute_sigmas(target_anom, raw_anom, mask) -> tuple[float, float]:
    """(sigma_obs, sigma_mdl) over valid cells of the train split."""
    valid = valid_mask_native(mask)

    def _sigma(a):
        v = a[:, :, valid]
        v = v[np.isfinite(v)]
        return float(v.std() or 1.0)

    return _sigma(target_anom), _sigma(raw_anom)


def substitute_channels(features, raw_anom, lag_anom, sigma_mdl, sigma_obs, mask):
    """Replace the tp_mean and lag_obs channels with their standardized
    anomalies, in place -- `features` is several GB for a full split. The
    circulation and day-of-year channels are left alone: doy_sin/doy_cos still
    identify the regime, they just can no longer give away the seasonal cycle.
    """
    features[:, TP_MEAN_CHANNEL] = raw_anom / sigma_mdl

    valid = valid_mask_native(mask)
    lag = np.nan_to_num(lag_anom / sigma_obs, nan=0.0)
    features[:, LAG_OBS_CHANNEL] = np.where(valid, lag, 0.0)
    return features


def compute_feature_stats(features, mask, target_var, sigma_obs, sigma_mdl,
                          drop_anchor: bool = False) -> dict:
    """Per-channel stats over already-substituted features (train split only)."""
    valid = valid_mask_native(mask)
    n_channels = features.shape[1]
    feat_mean = np.zeros(n_channels)
    feat_std = np.ones(n_channels)
    for c in range(n_channels):
        vals = features[:, c][:, :, valid]
        feat_mean[c] = vals.mean()
        feat_std[c] = vals.std() or 1.0

    return {
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "sigma_obs": sigma_obs,
        "sigma_mdl": sigma_mdl,
        "target_var": target_var,
        "anomaly": True,
        "drop_anchor": drop_anchor,
    }


def model_features(features, stats):
    """The channels actually fed to the encoder.

    With `drop_anchor`, tp_mean is withheld from the input stack so the gated
    anchor is the *only* path carrying ECMWF's precipitation signal. The first
    run left it in both places, and the correction head simply reconstructed
    the signal through the conv stack, leaving the gate to decay to ~3-4x
    below its loss-optimal value (experiments/exp03_departure_from_normal/runs/gate_v1_first_try (train log)).

    tp_spread stays: ensemble disagreement is the natural driver of how much
    to trust the anchor, and it is not duplicated by it.

    Returns a strided view, not a copy -- `features` is several GB.
    """
    return features[:, 1:] if stats.get("drop_anchor") else features


def standardize_features(features, stats):
    """In-place per-channel standardization."""
    feat_mean = np.array(stats["feat_mean"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    feat_std = np.array(stats["feat_std"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    features -= feat_mean
    features /= feat_std
    return features


def scale_target_and_anchor(target_anom, raw_anom, stats):
    """(target_std, anchor), each (N, 1, lead, H, W).

    `anchor` is the standardized ECMWF anomaly the gated head blends against.
    It goes to the model separately from the feature stack even though it also
    appears as channel 0, because the head consumes it directly."""
    tgt = np.nan_to_num(target_anom / stats["sigma_obs"], nan=0.0)[:, None].astype(np.float32)
    anchor = (raw_anom / stats["sigma_mdl"])[:, None].astype(np.float32)
    return tgt, anchor


def inverse_anom(pred_anom, dates, obs_clim, stats):
    """(N, lead, H, W) standardized anomaly -> physical units (mm, or degC)."""
    obs_batch = obs_clim_for_dates(obs_clim, dates, pred_anom.shape[1])
    return inverse_transform(pred_anom * stats["sigma_obs"] + obs_batch, stats["target_var"])
