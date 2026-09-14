"""Assemble one (features, target) training sample per ECMWF issue date.

Feature channels (literature-review.md S7.2's feature list, minus the two
items this project's local data can't supply -- see plan.md's "honest
scope"):

  - tp ensemble mean + spread            (spread = the model's own
                                           per-forecast uncertainty signal,
                                           S7.2)
  - msl, gh(200,1000), q(200,500,700,850), u(200,1000), v(200,1000), wtmp
    ensemble mean                        (no spread -- keeps channel count
                                           sane; spread-of-aux-vars is a
                                           later ablation)
  - lagged IMD state: the target variable's own last-observed value
    (issue_date - 1 day), broadcast across the lead axis -- "the single
    biggest cheap win in Das et al." per the literature review
  - day-of-year sin/cos, varying along the lead axis

Not included (no local source found under ../s2s-forecasting/data/):
orography, MJO RMM / MISO index. Flagged in plan.md as follow-up work,
not silently dropped.

Every array here is on the padded (config.PAD_LAT, config.PAD_LON) grid,
with padded rows/cols treated as invalid (zero-filled features, NaN
target) so the loss/mask machinery in model/unet3d.py handles them for
free.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import config
from preprocess.ecmwf_features import ensemble_mean_spread, load_variable

CIRCULATION_CHANNELS = [
    ("msl", None),
    ("gh", 200),
    ("gh", 1000),
    ("q", 200),
    ("q", 500),
    ("q", 700),
    ("q", 850),
    ("u", 200),
    ("u", 850),
    ("v", 200),
    ("v", 850),
    ("wtmp", None),
]


def _pad_spatial(arr: np.ndarray, fill_value: float) -> np.ndarray:
    """Pad the last two axes from (config.NATIVE_LAT, config.NATIVE_LON) to
    (config.PAD_LAT, config.PAD_LON)."""
    pad_bottom = config.PAD_LAT - config.NATIVE_LAT - config.PAD_TOP
    pad_right = config.PAD_LON - config.NATIVE_LON - config.PAD_LEFT
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(config.PAD_TOP, pad_bottom), (config.PAD_LEFT, pad_right)]
    return np.pad(arr, pad_width, mode="constant", constant_values=fill_value)


def padded_native_slice() -> tuple[slice, slice]:
    """Slices that recover the native (unpadded) region from a padded array."""
    return (
        slice(config.PAD_TOP, config.PAD_TOP + config.NATIVE_LAT),
        slice(config.PAD_LEFT, config.PAD_LEFT + config.NATIVE_LON),
    )


class IMDTargetStore:
    """Loads the cached, already-regridded IMD netCDF
    (scripts/cache_imd_targets.py's output) once into memory and serves
    fast date-indexed slices."""

    def __init__(self, path=None):
        path = path or (config.CACHE_ROOT / "imd_targets_1p5deg.nc")
        ds = xr.open_dataset(path)
        self.times = ds.time.values.astype("datetime64[D]")
        self.data = {v: ds[v].values for v in config.TARGETS}
        self.mask = ds["mask"].values.astype(bool)
        self._index = {t: i for i, t in enumerate(self.times)}

    def slice_days(self, var: str, start: datetime, n_days: int) -> np.ndarray:
        """(n_days, lat, lon), NaN for any date outside the cached record."""
        out = np.full((n_days, config.NATIVE_LAT, config.NATIVE_LON), np.nan)
        for t in range(n_days):
            day = np.datetime64((start + timedelta(days=t)).date(), "D")
            idx = self._index.get(day)
            if idx is not None:
                out[t] = self.data[var][idx]
        return out

    def value_on(self, var: str, day: datetime) -> np.ndarray:
        idx = self._index.get(np.datetime64(day.date(), "D"))
        if idx is None:
            return np.full((config.NATIVE_LAT, config.NATIVE_LON), np.nan)
        return self.data[var][idx]


def _doy_sin_cos(issue_date: datetime, lead_days: int) -> tuple[np.ndarray, np.ndarray]:
    doy = np.array([(issue_date + timedelta(days=t)).timetuple().tm_yday for t in range(lead_days)])
    angle = 2 * np.pi * doy / 365.25
    return np.sin(angle), np.cos(angle)


def build_circulation_features(issue_date: datetime, lead_days: int = config.LEAD_DAYS):
    """Returns (channels (C, lead, native_lat, native_lon), channel_names)."""
    tp = load_variable("tp", issue_date, lead_days)
    tp_mean, tp_spread = ensemble_mean_spread(tp)

    channels = [tp_mean, tp_spread]
    names = ["tp_mean", "tp_spread"]

    for short_name, level in CIRCULATION_CHANNELS:
        members = load_variable(short_name, issue_date, lead_days)
        if level is not None:
            level_idx = config.ECMWF_VARS[short_name]["levels"].index(level)
            members = members[:, :, level_idx]
        mean, _ = ensemble_mean_spread(members)
        if short_name == "wtmp":
            # sst is only defined over ocean; land cells are legitimately
            # NaN, not missing data, and the model needs a finite input.
            mean = np.nan_to_num(mean, nan=0.0)
        channels.append(mean)
        names.append(f"{short_name}{level or ''}_mean")

    return np.stack(channels, axis=0), names


def build_sample(issue_date: datetime, target_var: str, imd_store: IMDTargetStore, lead_days: int = config.LEAD_DAYS):
    """One (features, target, mask) sample for `target_var` in {"rain",
    "tmax", "tmin"}, on the padded grid.

    features: (C, lead_days, PAD_LAT, PAD_LON) float32
    target:   (1, lead_days, PAD_LAT, PAD_LON) float32, NaN where invalid
    mask:     (PAD_LAT, PAD_LON) bool, True = valid IMD cell
    """
    circ, names = build_circulation_features(issue_date, lead_days)

    lag_value = imd_store.value_on(target_var, issue_date - timedelta(days=1))
    lag_channel = np.broadcast_to(lag_value, (lead_days, *lag_value.shape)).copy()
    lag_channel = np.nan_to_num(lag_channel, nan=0.0)

    sin_doy, cos_doy = _doy_sin_cos(issue_date, lead_days)
    sin_channel = np.broadcast_to(sin_doy[:, None, None], (lead_days, config.NATIVE_LAT, config.NATIVE_LON))
    cos_channel = np.broadcast_to(cos_doy[:, None, None], (lead_days, config.NATIVE_LAT, config.NATIVE_LON))

    all_channels = np.concatenate(
        [circ, lag_channel[None], sin_channel[None], cos_channel[None]], axis=0
    ).astype(np.float32)
    channel_names = names + ["lag_obs", "doy_sin", "doy_cos"]

    target = imd_store.slice_days(target_var, issue_date, lead_days)[None].astype(np.float32)

    features_padded = _pad_spatial(all_channels, fill_value=0.0)
    target_padded = _pad_spatial(target, fill_value=np.nan)

    mask = np.zeros((config.PAD_LAT, config.PAD_LON), dtype=bool)
    lat_sl, lon_sl = padded_native_slice()
    mask[lat_sl, lon_sl] = imd_store.mask

    return features_padded, target_padded, mask, channel_names
