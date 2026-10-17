"""Dyn++ bias correction of the ECMWF rain forecast, on correctly aligned days.

Dyn++ is the "dynamical model, corrected" member of ABC (Mouatadid et al.
2023, Nat. Comms. 14, 3482). This is the simplified multiplicative version
scored against the U-Net and the friend's TCN in the head-to-head:

    factor[month, week, cell] = (mean observed weekly total + 1)
                                / (mean ECMWF weekly total + 1)

fitted over training issue dates whose month is within +/-1 of `month`,
clipped to [0.25, 4], and multiplied onto every daily ECMWF value of that
lead week. Forecast days 29-32 reuse week 4's factor. The paper's version
also averages recent runs and picks its window adaptively; neither is done
here, so this is a lower bound on Dyn++.

Two fits:

  full   all training years            -> applied to val and test dates
  loyo   leave-one-year-out, per year  -> applied to training dates, so a
         model trained on top of Dyn++ never sees a correction fitted on the
         year it is learning from

Everything runs on the aligned pair from `load_aligned`: ECMWF lead t
(sample cache channel 0, mm/day) against IMD date issue+1+t, read directly
from the regridded IMD file instead of the cache's one-day-early target
(preprocess/alignment.py).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

import config
from preprocess.build_dataset import IMDTargetStore, _pad_spatial

BC_ROOT = config.CACHE_ROOT / "bias_correction"
CLIP = (0.25, 4.0)
N_WEEKS = 4
MONTH_HALF_WINDOW = 1


def week_of_day(n_days: int = config.LEAD_DAYS) -> np.ndarray:
    """Lead-day index -> lead-week index (0..3); days 28+ stay in week 4."""
    return np.minimum(np.arange(n_days) // 7, N_WEEKS - 1)


def load_aligned(dates: list[datetime], target_var: str = "rain") -> tuple[np.ndarray, np.ndarray]:
    """(tp_mean, obs), each (N, LEAD_DAYS, PAD_LAT, PAD_LON) mm/day.

    tp_mean[:, t] is ECMWF lead t; obs[:, t] is IMD on issue+1+t (NaN outside
    the IMD mask). Cached under data_cache/bias_correction/ by date, since
    reading channel 0 of 1,000+ samples takes a few minutes."""
    BC_ROOT.mkdir(parents=True, exist_ok=True)
    path = BC_ROOT / f"aligned_{target_var}.npz"
    cached = {}
    if path.exists():
        with np.load(path) as z:
            cached = {str(d): (tp, ob) for d, tp, ob in zip(z["dates"], z["tp_mean"], z["obs"])}

    missing = [d for d in dates if f"{d:%Y-%m-%d}" not in cached]
    if missing:
        store = IMDTargetStore()
        sample_dir = config.CACHE_ROOT / "samples" / target_var
        for d in missing:
            with np.load(sample_dir / f"{d:%Y_%m_%d}.npz") as z:
                tp = z["features"][0]
            ob = store.slice_days(target_var, d + timedelta(days=1), config.LEAD_DAYS)
            cached[f"{d:%Y-%m-%d}"] = (tp.astype(np.float32), _pad_spatial(ob, np.nan).astype(np.float32))
        keys = sorted(cached)
        np.savez(path, dates=np.array(keys),
                 tp_mean=np.stack([cached[k][0] for k in keys]),
                 obs=np.stack([cached[k][1] for k in keys]))

    tp = np.stack([cached[f"{d:%Y-%m-%d}"][0] for d in dates])
    obs = np.stack([cached[f"{d:%Y-%m-%d}"][1] for d in dates])
    return tp, obs


def _weekly_totals(x: np.ndarray) -> np.ndarray:
    """(N, lead, H, W) daily -> (N, 4, H, W) totals over days 1-7, ..., 22-28."""
    return np.stack([x[:, 7 * w:7 * w + 7].sum(axis=1) for w in range(N_WEEKS)], axis=1)


def _month_in_window(months: np.ndarray, m: int) -> np.ndarray:
    return np.abs(((months - m + 6) % 12) - 6) <= MONTH_HALF_WINDOW


def _ratios(sum_f, sum_o, count) -> np.ndarray:
    """(12, 4, H, W) clipped factors from per-month sums; months are 1-indexed."""
    out = np.ones((12, N_WEEKS) + sum_f.shape[-2:], dtype=np.float32)
    months = np.arange(1, 13)
    for m in months:
        sel = _month_in_window(months, m)
        n = count[sel].sum()
        if n == 0:
            continue
        f = sum_f[sel].sum(axis=0) / n
        o = sum_o[sel].sum(axis=0) / n
        out[m - 1] = np.clip((o + 1) / (f + 1), *CLIP)
    return np.nan_to_num(out, nan=1.0)


def fit_dynpp(tp: np.ndarray, obs: np.ndarray, dates: list[datetime], mask: np.ndarray) -> dict:
    """Factors from training dates: {"full": (12,4,H,W), "loyo": {year: (12,4,H,W)}}.

    Cells outside the IMD `mask` keep a factor of 1, so ECMWF passes through
    unchanged over ocean and neighbouring countries."""
    wf, wo = _weekly_totals(tp), _weekly_totals(obs)
    years = np.array([d.year for d in dates])
    months = np.array([d.month for d in dates])

    shape = (12, N_WEEKS) + tp.shape[-2:]
    by_year = {}
    for y in np.unique(years):
        sf, so, c = np.zeros(shape), np.zeros(shape), np.zeros(12)
        for m in range(1, 13):
            sel = (years == y) & (months == m)
            sf[m - 1] = np.nansum(wf[sel], axis=0)
            so[m - 1] = np.nansum(wo[sel], axis=0)
            c[m - 1] = sel.sum()
        by_year[int(y)] = (sf, so, c)

    def ratios(sf, so, c):
        r = _ratios(sf, so, c)
        r[..., ~mask] = 1.0
        return r

    total = [sum(v[i] for v in by_year.values()) for i in range(3)]
    loyo = {y: ratios(*(total[i] - v[i] for i in range(3))) for y, v in by_year.items()}
    return {"full": ratios(*total), "loyo": loyo}


def apply_dynpp(tp: np.ndarray, dates: list[datetime], factors: dict, use_loyo: bool = False) -> np.ndarray:
    """Multiply each date's daily ECMWF rain by its (issue month, lead week) factor."""
    wk = week_of_day(tp.shape[1])
    out = np.empty_like(tp)
    for n, d in enumerate(dates):
        f = factors["loyo"][d.year] if use_loyo else factors["full"]
        out[n] = tp[n] * f[d.month - 1][wk]
    return out


def save_factors(factors: dict, path) -> None:
    np.savez(path, full=factors["full"], **{f"loyo_{y}": v for y, v in factors["loyo"].items()})


def load_factors(path) -> dict:
    with np.load(path) as z:
        return {"full": z["full"],
                "loyo": {int(k[5:]): z[k] for k in z.files if k.startswith("loyo_")}}
