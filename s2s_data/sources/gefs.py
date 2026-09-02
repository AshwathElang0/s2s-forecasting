"""NOAA GEFSv12 reforecast, streamed from AWS Open Data (grib2 over S3).

Role in the project: the dynamical S2S forecast that gets corrected/enhanced
(the "raw forecast" input, analogous to what Zhang et al. did with GEOS5).

Key constraint discovered while probing this source: GEFSv12 reforecast has
two lead-time products per init date -- `Days:1-10` (0.25 deg grid, 3-hourly
accumulation) and, for Wednesday-initialized runs only, `Days:10-35` (0.5 deg
grid, 6-hourly accumulation). Non-Wednesday init dates stop at `Days:10-16`
and cannot cover a week-4 (day 22-28) lead. So the week 1-4 study is
necessarily restricted to Wednesday init dates.

The `tp` variable in both products is a per-interval accumulation (precip
that fell since the *previous* step), not a running total since init --
verified empirically (values don't grow monotonically with lead time).
Weekly totals are therefore a plain sum of all sub-daily values whose valid
time falls inside the target week.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr

from s2s_data.config import CACHE_DIR, GEFS_BUCKET, GEFS_EXTENDED_WEEKDAY
from s2s_data.regrid import crop_to_conus_bounds, to_target_grid

_FS = s3fs.S3FileSystem(anon=True)
_GRIB_CACHE_DIR = CACHE_DIR / "gefs_grib"


def is_extended_init_date(init_date: dt.date) -> bool:
    """Only Wednesday inits have the Days:10-35 product needed for week 4."""
    return init_date.weekday() == GEFS_EXTENDED_WEEKDAY


def _init_date_str(init_date: dt.date) -> str:
    return f"{init_date:%Y%m%d}00"


def _remote_path(init_date: dt.date, member: str, lead_range: str) -> str:
    date_str = _init_date_str(init_date)
    return (
        f"{GEFS_BUCKET}/GEFSv12/reforecast/{init_date.year}/{date_str}/"
        f"{member}/Days:{lead_range}/apcp_sfc_{date_str}_{member}.grib2"
    )


def _cached_local_path(init_date: dt.date, member: str, lead_range: str) -> Path:
    date_str = _init_date_str(init_date)
    local_dir = _GRIB_CACHE_DIR / date_str
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir / f"apcp_sfc_{date_str}_{member}_{lead_range.replace(':', '-')}.grib2"


def _download_if_missing(init_date: dt.date, member: str, lead_range: str) -> Path:
    local = _cached_local_path(init_date, member, lead_range)
    if not local.exists():
        remote = _remote_path(init_date, member, lead_range)
        _FS.get(remote, str(local))
    return local


def load_member_precip(init_date: dt.date, member: str) -> xr.DataArray:
    """Load, regrid, and concatenate one member's full lead-time precip record.

    Requires `init_date` to be a Wednesday (raises otherwise) since week 4
    needs the Days:10-35 product.
    """
    if not is_extended_init_date(init_date):
        raise ValueError(
            f"{init_date} is not a Wednesday init -- no Days:10-35 product "
            "available, so week-4 lead can't be computed."
        )

    early_path = _download_if_missing(init_date, member, "1-10")
    late_path = _download_if_missing(init_date, member, "10-35")

    # Perturbed-member files mix 'cf' (control) and 'pf' (perturbed) GRIB
    # messages; cfgrib needs an explicit dataType filter to disambiguate.
    data_type = "cf" if member == "c00" else "pf"
    backend_kwargs = {"filter_by_keys": {"dataType": data_type}}

    early = xr.open_dataset(early_path, engine="cfgrib", backend_kwargs=backend_kwargs)["tp"]
    late = xr.open_dataset(late_path, engine="cfgrib", backend_kwargs=backend_kwargs)["tp"]

    early = early.swap_dims({"step": "valid_time"})
    late = late.swap_dims({"step": "valid_time"})

    early_regridded = to_target_grid(
        crop_to_conus_bounds(early, lat_name="latitude", lon_name="longitude"),
        lat_name="latitude",
        lon_name="longitude",
    )
    late_regridded = to_target_grid(
        crop_to_conus_bounds(late, lat_name="latitude", lon_name="longitude"),
        lat_name="latitude",
        lon_name="longitude",
    )

    combined = xr.concat([early_regridded, late_regridded], dim="valid_time")
    combined = combined.sortby("valid_time")
    combined.name = "precip"
    combined.attrs["units"] = "kg m-2 (mm, per-interval accumulation)"
    return combined


def weekly_totals_by_lead(
    member_precip: xr.DataArray, init_date: dt.date, lead_weeks: tuple[int, ...]
) -> xr.DataArray:
    """Sum per-interval precip into non-overlapping weekly totals for each lead week.

    Week `w` covers valid times in [init_date + (w-1)*7 days, init_date + w*7 days).
    """
    init_ts = xr.DataArray(
        [init_date + dt.timedelta(days=7 * (w - 1)) for w in lead_weeks],
        dims="lead_week",
        coords={"lead_week": list(lead_weeks)},
    ).astype("datetime64[ns]")

    totals = []
    for w in lead_weeks:
        start = init_date + dt.timedelta(days=7 * (w - 1))
        end = init_date + dt.timedelta(days=7 * w)
        end_ts = np.datetime64(end)
        window = member_precip.sel(valid_time=slice(str(start), str(end)))
        # slice is inclusive on both ends; drop the trailing boundary sample
        window = window.where(window.valid_time < end_ts, drop=True)
        totals.append(window.sum(dim="valid_time", skipna=False))

    result = xr.concat(totals, dim="lead_week")
    result = result.assign_coords(lead_week=list(lead_weeks))
    result.name = "precip_weekly_total"
    return result
