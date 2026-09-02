"""CESM2 Large Ensemble (CESM2-LE), streamed from AWS Open Data (Zarr/S3).

Role in the project: large synthetic ensemble for pretraining the
regime/confidence signal in a "perfect model" framework, before it's
fine-tuned jointly with the correction model on real GEFS reforecasts (see
project plan). Also usable as an independent large-ensemble check on the
extreme-event climatology used to set thresholds.

CESM2-LE has no single "total precipitation" variable — it's split into
convective (PRECC) and large-scale/stable (PRECL) components, both in m/s.
"""

from __future__ import annotations

import xarray as xr

from s2s_data.regrid import crop_to_conus_bounds, to_target_grid

_STORE_TEMPLATE = "s3://ncar-cesm2-lens/atm/daily/cesm2LE-{experiment}-{forcing_variant}-{var}.zarr"

M_PER_S_TO_MM_PER_DAY = 1000.0 * 86400.0


def _open_var(experiment: str, forcing_variant: str, var: str) -> xr.DataArray:
    store = _STORE_TEMPLATE.format(experiment=experiment, forcing_variant=forcing_variant, var=var)
    ds = xr.open_zarr(store, storage_options={"anon": True}, consolidated=True)
    return ds[var]


def load_precip_mm_per_day(
    experiment: str = "historical",
    forcing_variant: str = "cmip6",
    members: list[str] | None = None,
) -> xr.DataArray:
    """Total precip (PRECC + PRECL), converted from m/s to mm/day, lazy (dask-backed)."""
    precc = _open_var(experiment, forcing_variant, "PRECC")
    precl = _open_var(experiment, forcing_variant, "PRECL")
    total = (precc + precl) * M_PER_S_TO_MM_PER_DAY
    total.name = "precip"

    if members is not None:
        total = total.sel(member_id=members)
    return total


def crop_and_regrid_to_conus(da: xr.DataArray) -> xr.DataArray:
    """Crop to just past CONUS bounds (cheap on the lazy dask array), then interpolate."""
    cropped = crop_to_conus_bounds(da, lat_name="lat", lon_name="lon")
    return to_target_grid(cropped, lat_name="lat", lon_name="lon")


def weekly_totals(daily_precip: xr.DataArray) -> xr.DataArray:
    return daily_precip.resample(time="7D").sum(skipna=False)
