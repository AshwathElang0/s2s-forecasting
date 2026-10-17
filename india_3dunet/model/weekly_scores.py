"""Scores on aligned forecast days 1-28 (weeks 1-4), shared by Experiments 9 and 10.

Inputs are (N, >=28, PAD_LAT, PAD_LON) mm/day arrays already on aligned days:
index t = forecast day t+1 = IMD date issue+1+t (preprocess/alignment.py).

Daily scores (days 1-28, mean over days):
  corr        pattern correlation of the rainfall map (results.md "standard")
  anom_corr   pattern correlation after removing the causal log1p climatology
              (results.md "climatology removed")
  rmse        latitude-weighted RMSE, mm/day
  bias        latitude-weighted mean of prediction - observation, mm/day
  sd_ratio    SD of predicted / observed daily rain over all land cells and days

Weekly scores, on 7-day totals with a 1951-2020 day-of-year mm climatology
removed (the same normal as results.md S8.5):
  acc         anomaly correlation pooled over land cells and forecast dates
  ai_acc      correlation of the all-India (area-mean) anomaly across dates
  wk_rmse     RMSE of the weekly total, mm/week
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import config
from model.evaluation import rmses, scorr
from preprocess.build_dataset import _pad_spatial

N_DAYS = 28
N_WEEKS = 4
CLIM_YEARS = (1951, 2020)
CLIM_HALF_WINDOW = 15


def mm_climatology() -> np.ndarray:
    """(366, PAD_LAT, PAD_LON) mean daily rain (mm) per day of year over
    CLIM_YEARS, circular +/-15-day smoothing. Before every test year (2021+)."""
    path = config.CACHE_ROOT / "bias_correction" / f"obs_mm_clim_{CLIM_YEARS[0]}_{CLIM_YEARS[1]}.npy"
    if path.exists():
        return np.load(path)
    with xr.open_dataset(config.CACHE_ROOT / "imd_targets_1p5deg.nc") as ds:
        times = ds.time.values.astype("datetime64[D]")
        years = times.astype("datetime64[Y]").astype(int) + 1970
        sel = (years >= CLIM_YEARS[0]) & (years <= CLIM_YEARS[1])
        rain = ds["rain"].values[sel]
        doys = np.array([d.timetuple().tm_yday for d in times[sel].astype("O")])
    daily = np.stack([np.nanmean(rain[doys == k], axis=0) for k in range(1, 367)])
    idx = (np.arange(366)[:, None] + np.arange(-CLIM_HALF_WINDOW, CLIM_HALF_WINDOW + 1)[None]) % 366
    clim = _pad_spatial(daily[idx].mean(axis=1), np.nan).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, clim)
    return clim


def mm_clim_for_dates(clim: np.ndarray, dates: list[datetime], n_days: int = N_DAYS) -> np.ndarray:
    """(N, n_days, H, W) mm climatology on aligned days issue+1..issue+n_days."""
    return np.stack([
        clim[[(d + timedelta(days=t + 1)).timetuple().tm_yday - 1 for t in range(n_days)]]
        for d in dates
    ])


def _area_weights(lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    return w / w.sum()


def _weekly(x: np.ndarray) -> np.ndarray:
    return np.stack([x[:, 7 * w:7 * w + 7].sum(axis=1) for w in range(N_WEEKS)], axis=1)


def weekly_anomalies(pred, obs, clim_mm, mask):
    """(pred_anom, obs_anom), each (N, 4, n_cells) weekly-total anomalies."""
    m = mask.astype(bool)
    pa = (_weekly(pred[:, :N_DAYS]) - _weekly(clim_mm[:, :N_DAYS]))[..., m]
    oa = (_weekly(obs[:, :N_DAYS]) - _weekly(clim_mm[:, :N_DAYS]))[..., m]
    return pa, oa


def _pooled_corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def weekly_metrics(pa, oa, cell_w, idx=None) -> dict:
    """acc / ai_acc / wk_rmse per week from precomputed weekly anomalies.
    `cell_w` is the area weight of each land cell; `idx` resamples dates."""
    if idx is not None:
        pa, oa = pa[idx], oa[idx]
    out = {}
    for w in range(N_WEEKS):
        out[f"wk{w + 1}_acc"] = _pooled_corr(pa[:, w], oa[:, w])
        out[f"wk{w + 1}_ai_acc"] = float(np.corrcoef(pa[:, w] @ cell_w, oa[:, w] @ cell_w)[0, 1])
        out[f"wk{w + 1}_rmse"] = float(np.sqrt(np.mean((pa[:, w] - oa[:, w]) ** 2)))
    return out


def daily_metrics(pred, obs, clim_log, lat, mask) -> dict:
    """Mean over days 1-28 and dates of the per-map scores, plus bias and SD ratio."""
    pred, obs, clim_log = pred[:, :N_DAYS], obs[:, :N_DAYS], clim_log[:, :N_DAYS]
    pa = np.log1p(np.clip(pred, 0, None)) - clim_log
    oa = np.log1p(np.clip(obs, 0, None)) - clim_log
    n, L = pred.shape[:2]
    corr, acorr, rmse = (np.full((n, L), np.nan) for _ in range(3))
    for i in range(n):
        for t in range(L):
            corr[i, t] = scorr(pred[i, t], obs[i, t], lat, mask)
            acorr[i, t] = scorr(pa[i, t], oa[i, t], lat, mask)
            rmse[i, t] = rmses(pred[i, t], obs[i, t], lat, mask)
    m = mask.astype(bool)
    w = _area_weights(lat, mask)[m]
    p, o = pred[..., m], obs[..., m]
    return {
        "corr": float(np.nanmean(corr)),
        "anom_corr": float(np.nanmean(acorr)),
        "rmse": float(np.nanmean(rmse)),
        "bias": float(np.nanmean((p - o) @ w)),
        "sd_ratio": float(np.nanstd(p) / np.nanstd(o)),
        "anom_corr_by_day": np.nanmean(acorr, axis=0).tolist(),
    }


def score(pred, obs, clim_mm, clim_log, lat, mask) -> dict:
    pa, oa = weekly_anomalies(pred, obs, clim_mm, mask)
    cell_w = _area_weights(lat, mask)[mask.astype(bool)]
    return {**daily_metrics(pred, obs, clim_log, lat, mask), **weekly_metrics(pa, oa, cell_w)}


def bootstrap_weekly(preds_a, preds_b, obs, clim_mm, lat, mask, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired date-resampling bootstrap of (a - b) for every weekly metric:
    {metric: [difference, 2.5%, 97.5%]}. `preds_a`/`preds_b` are lists (one
    array per training seed); each side's metric is the mean over its seeds,
    so predictions are never averaged into an ensemble. Covers which test
    dates were drawn, not training randomness -- compare against the seed
    spread too."""
    cell_w = _area_weights(lat, mask)[mask.astype(bool)]
    _, oa = weekly_anomalies(preds_a[0], obs, clim_mm, mask)
    a_pa = [weekly_anomalies(p, obs, clim_mm, mask)[0] for p in preds_a]
    b_pa = [weekly_anomalies(p, obs, clim_mm, mask)[0] for p in preds_b]

    def side(pas, idx=None):
        ms = [weekly_metrics(pa, oa, cell_w, idx) for pa in pas]
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    full_a, full_b = side(a_pa), side(b_pa)
    rng = np.random.default_rng(seed)
    n = len(obs)
    draws = {k: [] for k in full_a}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma, mb = side(a_pa, idx), side(b_pa, idx)
        for k in draws:
            draws[k].append(ma[k] - mb[k])
    return {k: [full_a[k] - full_b[k], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items()}


def format_table(results: dict, columns: list[str]) -> str:
    """Markdown table, one row per method."""
    lines = ["| method | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for name, r in results.items():
        # nan = undefined, e.g. climatology's anomaly is zero so it has no correlation
        lines.append(f"| {name} | " + " | ".join("—" if np.isnan(r[c]) else f"{r[c]:.3f}" for c in columns) + " |")
    return "\n".join(lines)
