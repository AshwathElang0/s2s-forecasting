"""Strictly causal observed and model climatologies (experiments/exp03_departure_from_normal/DESIGN.md S5.1).

For a sample issued in year Y, **every** number used to build its "normal"
comes from a date strictly before Y. No sample ever contributes to the
climatology it is scored against, and no future year leaks backwards into an
earlier training fold -- the exact failure literature-review.md S5 names:
"model climatology and observation climatology computed on the full record
leak future information into the training folds."

Two climatologies:

- **observed**, per (day-of-year, cell), from the cached IMD record.
  Rain uses an expanding window `[1951, Y-1]`; temperature uses a rolling
  30-year normal `[Y-30, Y-1]`, which both matches WMO practice and tracks
  the warming trend that a fixed window would suppress (review S2.1).
- **model**, per (issue-DOY bin, lead day, cell), from the `tp_mean` channel
  of the existing sample cache, using train-split issue dates from years
  `< Y` only.

The model side is the binding constraint. The reforecast archive starts
2004-01-04, so a sample in 2004 has *zero* prior forecast years and cannot be
given a causal model climatology at all -- `usable_dates()` drops those 52
training samples rather than quietly letting them look ahead.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xarray as xr

import config
from preprocess.build_dataset import padded_native_slice

CLIM_ROOT = config.CACHE_ROOT / "climatology"

OBS_FIRST_YEAR = 1951          # start of the cached IMD record
TEMP_NORMAL_YEARS = 30         # WMO-style rolling normal for tmax/tmin
OBS_HALF_WINDOW = 10           # days, circular
MODEL_N_BINS = 52              # weekly bins, matching the reforecast cadence
MODEL_HALF_BINS = 3            # ~ +/- 21 days, circular


def transform(x: np.ndarray, target_var: str) -> np.ndarray:
    if target_var == "rain":
        return np.log1p(np.clip(x, 0, None))
    return x


def inverse_transform(x: np.ndarray, target_var: str) -> np.ndarray:
    if target_var == "rain":
        return np.expm1(x)
    return x


def day_of_year(d: datetime) -> int:
    return d.timetuple().tm_yday


def issue_bin(d: datetime) -> int:
    return min((day_of_year(d) - 1) // 7, MODEL_N_BINS - 1)


def obs_window(target_var: str, as_of_year: int) -> tuple[int, int]:
    """Inclusive [lo, hi] observation years usable by a sample in `as_of_year`.
    Rain expands (more years = a steadier daily rainfall normal); temperature
    rolls, so the normal tracks the trend instead of averaging it away."""
    if target_var == "rain":
        return OBS_FIRST_YEAR, as_of_year - 1
    return as_of_year - TEMP_NORMAL_YEARS, as_of_year - 1


def _circular_smooth_sum(arr: np.ndarray, half: int) -> np.ndarray:
    """Sum `arr` over a circular window of +/-`half` along axis 0."""
    n = arr.shape[0]
    idx = (np.arange(n)[:, None] + np.arange(-half, half + 1)[None, :]) % n
    out = np.zeros_like(arr)
    for k in range(idx.shape[1]):
        out += arr[idx[:, k]]
    return out


def _pad(arr: np.ndarray, fill: float) -> np.ndarray:
    pad_bottom = config.PAD_LAT - config.NATIVE_LAT - config.PAD_TOP
    pad_right = config.PAD_LON - config.NATIVE_LON - config.PAD_LEFT
    width = [(0, 0)] * (arr.ndim - 2) + [
        (config.PAD_TOP, pad_bottom),
        (config.PAD_LEFT, pad_right),
    ]
    return np.pad(arr, width, mode="constant", constant_values=fill)


# --- observed climatology ------------------------------------------------

def build_obs_climatology_by_year(target_var: str, as_of_years: list[int]) -> dict[int, np.ndarray]:
    """{year: (366, PAD_LAT, PAD_LON)} day-of-year means on the transformed
    scale, each built only from observations strictly before that year."""
    with xr.open_dataset(config.CACHE_ROOT / "imd_targets_1p5deg.nc") as ds:
        times = ds.time.values.astype("datetime64[D]")
        years = times.astype("datetime64[Y]").astype(int) + 1970
        data = transform(ds[target_var].values.astype(np.float32), target_var)

    doys = np.array([d.timetuple().tm_yday for d in times.astype("O")])
    valid = np.isfinite(data)

    out = {}
    for as_of in as_of_years:
        lo, hi = obs_window(target_var, as_of)
        sel = np.flatnonzero((years >= lo) & (years <= hi))

        sums = np.zeros((366, config.NATIVE_LAT, config.NATIVE_LON), dtype=np.float64)
        counts = np.zeros_like(sums)
        for i in sel:
            sums[doys[i] - 1] += np.where(valid[i], data[i], 0.0)
            counts[doys[i] - 1] += valid[i]

        sums = _circular_smooth_sum(sums, OBS_HALF_WINDOW)
        counts = _circular_smooth_sum(counts, OBS_HALF_WINDOW)
        clim = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        out[as_of] = _pad(clim.astype(np.float32), np.nan)
        print(f"  obs climatology as-of {as_of}: years {lo}-{hi} ({len(sel)} days)")

    return out


# --- model climatology ---------------------------------------------------

def _finalize_model(sums: np.ndarray, counts: np.ndarray) -> np.ndarray | None:
    if counts.sum() == 0:
        return None
    s = _circular_smooth_sum(sums, MODEL_HALF_BINS)
    c = _circular_smooth_sum(counts.reshape(-1, 1, 1, 1), MODEL_HALF_BINS)
    return np.divide(s, c, out=np.zeros_like(s), where=c > 0).astype(np.float32)


def build_model_climatology_by_year(
    target_var: str, train_dates: list[datetime], as_of_years: list[int]
) -> dict[int, np.ndarray | None]:
    """{year: (MODEL_N_BINS, LEAD_DAYS, PAD_LAT, PAD_LON) or None}.

    One pass over the train cache in chronological order, snapshotting the
    running accumulation *before* each year is folded in -- so year Y's
    climatology contains years < Y and nothing else. Years after the train
    split (val/test) get the full train accumulation, which is still strictly
    prior to them.

    `None` means no prior forecast year existed (2004); those dates are not
    usable under a causal climatology.
    """
    sample_dir = config.CACHE_ROOT / "samples" / target_var
    by_year: dict[int, list[datetime]] = {}
    for d in train_dates:
        by_year.setdefault(d.year, []).append(d)

    shape = (MODEL_N_BINS, config.LEAD_DAYS, config.PAD_LAT, config.PAD_LON)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(MODEL_N_BINS, dtype=np.float64)

    out: dict[int, np.ndarray | None] = {}
    for year in sorted(by_year):
        out[year] = _finalize_model(sums, counts)
        n_prior = int(counts.sum())
        print(f"  model climatology as-of {year}: {n_prior} prior issue dates"
              + ("  [UNUSABLE -- dropped]" if n_prior == 0 else ""))
        for d in by_year[year]:
            with np.load(sample_dir / f"{d:%Y_%m_%d}.npz") as z:
                tp_mean = z["features"][0]
            sums[issue_bin(d)] += np.log1p(np.clip(tp_mean, 0, None))
            counts[issue_bin(d)] += 1

    full = _finalize_model(sums, counts)
    for year in as_of_years:
        if year not in out:
            out[year] = full
    return out


# --- cache + accessors ---------------------------------------------------

def _cache_path(target_var: str, kind: str) -> Path:
    return CLIM_ROOT / f"{kind}_{target_var}_causal.npz"


def _save(d: dict, path: Path) -> None:
    np.savez_compressed(path, **{str(k): v for k, v in d.items() if v is not None},
                        _missing=np.array([k for k, v in d.items() if v is None]))


def _load(path: Path) -> dict:
    with np.load(path) as z:
        out = {int(k): z[k] for k in z.files if k != "_missing"}
        for k in z["_missing"]:
            out[int(k)] = None
    return out


def ensure_climatology(target_var: str, train_dates: list[datetime],
                       all_dates: list[datetime], rebuild: bool = False):
    """Build (and cache) both causal climatologies. Returns (obs, model) dicts
    keyed by year."""
    CLIM_ROOT.mkdir(parents=True, exist_ok=True)
    as_of_years = sorted({d.year for d in all_dates})

    obs_p, mdl_p = _cache_path(target_var, "obs"), _cache_path(target_var, "model")

    if rebuild or not obs_p.exists():
        print(f"building causal observed climatology for {target_var}...")
        _save(build_obs_climatology_by_year(target_var, as_of_years), obs_p)
    if rebuild or not mdl_p.exists():
        print(f"building causal model climatology from {len(train_dates)} train issue dates...")
        _save(build_model_climatology_by_year(target_var, train_dates, as_of_years), mdl_p)

    return _load(obs_p), _load(mdl_p)


def usable_dates(dates: list[datetime], model_clim: dict) -> list[datetime]:
    """Drop dates with no causal model climatology (the first archive year)."""
    return [d for d in dates if model_clim.get(d.year) is not None]


def obs_clim_for_dates(obs_clim: dict, dates: list[datetime], lead_days: int,
                       offset: int = 0) -> np.ndarray:
    """(N, lead, PAD_LAT, PAD_LON), each row using its own year's causal
    climatology, indexed by the lead day's *valid* date (issue_date + offset + lead).

    offset=0 matches the sample cache's target (IMD issue+t). offset=1 matches
    ECMWF lead t, which verifies against IMD issue+1+t (preprocess/alignment.py)."""
    out = np.empty((len(dates), lead_days, config.PAD_LAT, config.PAD_LON), dtype=np.float32)
    for n, d in enumerate(dates):
        doys = np.array([day_of_year(d + timedelta(days=t + offset)) for t in range(lead_days)])
        out[n] = obs_clim[d.year][doys - 1]
    return out


def model_clim_for_dates(model_clim: dict, dates: list[datetime]) -> np.ndarray:
    """(N, lead, PAD_LAT, PAD_LON), each row using its own year's causal
    climatology, indexed by the issue date's weekly bin."""
    return np.stack([model_clim[d.year][issue_bin(d)] for d in dates]).astype(np.float32)


def valid_mask_native(mask: np.ndarray) -> np.ndarray:
    """The IMD mask restricted to the unpadded region (padding excluded)."""
    lat_sl, lon_sl = padded_native_slice()
    valid = np.zeros_like(mask)
    valid[lat_sl, lon_sl] = mask[lat_sl, lon_sl]
    return valid
