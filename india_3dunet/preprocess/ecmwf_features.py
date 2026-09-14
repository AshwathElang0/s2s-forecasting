"""Ingest ECMWF S2S reforecast files and build model-ready feature arrays.

Fixes the two data bugs documented in ../s2s-forecasting/data/data.md and
literature-review.md S1.1, both of which silently corrupt results if
missed:

1. The `step` coordinate's reference *year* is wrong (grib_to_netcdf wrote
   whatever year the conversion ran in, not the file's real issue year).
   We never trust it -- valid dates are always `issue_date_from_filename +
   step_index days`, and we read `step` as a plain integer-hours
   coordinate (`decode_times=False`) specifically so nothing downstream can
   accidentally pick up the bogus decoded dates.
2. `tp` is a running accumulation from the forecast start, not a
   per-interval value -- day-D rainfall is `tp[D] - tp[D-1]`, not
   `tp[D]` itself. We difference it here, once, so every consumer
   downstream gets an actual daily rainfall field.

ECMWF is already on the project's native 1.5 grid (data.md), so unlike
IMD, no regridding happens here -- only the two fixes above, unit
handling, and folding `_cf_` (1 member) + `_pf_` (10 members) into one
11-member ensemble axis.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

import config

_FILENAME_RE = re.compile(r"(cf|pf)_reforecast_(\d{4})_(\d{2})_(\d{2})\.nc$")


def issue_date_from_filename(path: Path) -> datetime:
    m = _FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognised ECMWF filename: {path.name}")
    _, y, mo, d = m.groups()
    return datetime(int(y), int(mo), int(d))


def _variable_dir(short_name: str) -> Path:
    return config.ECMWF_ROOT / f"ECMWF_{short_name}_India_2004-2023"


def list_issue_dates(short_name: str = "tp") -> list[datetime]:
    """Issue dates available for a variable, from the `_cf_` filenames
    (every variable's cf/pf pair is keyed by the same issue date)."""
    files = sorted(_variable_dir(short_name).glob("*_cf_reforecast_*.nc"))
    return [issue_date_from_filename(f) for f in files]


def _open_member_file(path: Path, var_name: str, levels: list[int] | None, lead_days: int) -> np.ndarray:
    """Returns (member, day, [level,] lat, lon) for one cf or pf file,
    with `step` truncated to whole days (files are daily-step already,
    per data.md) and sliced to the first `lead_days` steps."""
    # Windows hits its open-file-handle limit fast if xarray Datasets are
    # left open (they lazily hold the netCDF4 file handle) -- this loop
    # opens tens of thousands of files across a full build, so every one
    # must be explicitly closed once its values are materialised.
    with xr.open_dataset(path, decode_times=False) as ds:
        da = ds[var_name]

        step_days = np.round(ds["step"].values / 24).astype(int) - 1  # 24h -> day index 0
        order = np.argsort(step_days)
        da = da.isel(step=order[:lead_days])

        if levels is not None:
            lev_dim = "level" if "level" in da.dims else "lev"
            da = da.sel({lev_dim: levels})

        if "number" not in da.dims:
            da = da.expand_dims("number")  # control member -> a length-1 member axis

        # order dims to (member, day, [level,] lat, lon)
        dim_order = ["number", "step"] + ([("level" if "level" in da.dims else "lev")] if levels is not None else []) + ["latitude", "longitude"]
        da = da.transpose(*dim_order)
        return da.values


def load_variable(short_name: str, issue_date: datetime, lead_days: int = config.LEAD_DAYS) -> np.ndarray:
    """Load one variable for one issue date as an 11-member ensemble:
    (member=11, day=lead_days, [level,] lat, lon). Member 0 is the control
    (`_cf_`), members 1..10 are the perturbed forecasts (`_pf_`), matching
    literature-review.md S1.1's "11 members (1 control + 10 perturbed)".
    """
    spec = config.ECMWF_VARS[short_name]
    var_name, levels = spec["var"], spec["levels"]
    tag = issue_date.strftime("%Y_%m_%d")
    d = _variable_dir(short_name)

    cf_path = d / f"ECMWF_{short_name}_India_cf_reforecast_{tag}.nc"
    pf_path = d / f"ECMWF_{short_name}_India_pf_reforecast_{tag}.nc"

    cf = _open_member_file(cf_path, var_name, levels, lead_days)
    pf = _open_member_file(pf_path, var_name, levels, lead_days)
    combined = np.concatenate([cf, pf], axis=0)  # (11, day, [level,] lat, lon)

    if short_name == "tp":
        # Running accumulation -> per-day rainfall. Day 0's value is
        # already the day-1 accumulation (no day -1 to subtract), every
        # later day is a first difference. Negative artefacts from the
        # packed-int16 storage are clipped to 0 (non-physical for rain).
        daily = np.empty_like(combined)
        daily[:, 0] = combined[:, 0]
        daily[:, 1:] = combined[:, 1:] - combined[:, :-1]
        combined = np.clip(daily, 0, None)

    return combined


def ensemble_mean_spread(members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`members`: (11, day, ..., lat, lon) -> (mean, spread) each without
    the member axis. Spread is the sample std across members -- the
    model's own per-forecast uncertainty estimate, literature-review.md
    S7.2's "ensemble mean and spread ... lets the network learn how much
    to shrink per forecast rather than a fixed per-lead shrinkage."""
    return members.mean(axis=0), members.std(axis=0)
