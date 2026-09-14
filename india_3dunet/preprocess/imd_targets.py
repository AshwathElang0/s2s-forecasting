"""Build IMD rain/tmax/tmin targets on the ECMWF 1.5 degree grid.

Per literature-review.md S1.3:
  - rain (0.25 -> 1.5): conservative, area-weighted (IMD cell centres
    aren't aligned to the 1.5 grid, so a naive 6x6 block-mean is wrong).
  - tmax/tmin (1.0 -> 1.5): conservative is correct here too; we use
    conservative for both since our regridder (preprocess/regrid.py) makes
    it no more expensive than bilinear and it's the more defensible choice.
  - mask: derived from IMD's own no-data pattern (IMD declares NaN, but
    rain in particular embeds -999 sentinels -- data/data.md's caveat),
    not from a hand-drawn India outline.

Every regridded field is returned already masked (NaN outside the
IMD-observed footprint), on the exact ECMWF native grid
(config.NATIVE_LAT x config.NATIVE_LON).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import config
from preprocess.regrid import build_conservative_weights, conservative_regrid

_SENTINELS = {"rain": -999.0}


def _load_raw(path, var_name: str) -> xr.DataArray:
    ds = xr.open_dataset(path, decode_times=False)
    da = ds[var_name]
    dates = pd.date_range("1951-01-01", periods=da.sizes["time"], freq="D")
    da = da.assign_coords(time=dates)
    sentinel = _SENTINELS.get(var_name)
    if sentinel is not None:
        da = da.where(da != sentinel)
    return da


def _native_grid():
    lat = np.linspace(config.NATIVE_LAT_VALS[0], config.NATIVE_LAT_VALS[1], config.NATIVE_LAT)
    lon = np.linspace(config.NATIVE_LON_VALS[0], config.NATIVE_LON_VALS[1], config.NATIVE_LON)
    return lat, lon


def build_valid_mask(sample: xr.DataArray, min_valid_frac: float = 0.99) -> np.ndarray:
    """A source-grid cell counts as valid if it's non-NaN for at least
    `min_valid_frac` of the sampled time steps -- robust to the odd bad
    day without treating genuinely dry/no-gauge cells as valid.
    """
    valid_frac = sample.notnull().mean(dim="time").values
    return valid_frac >= min_valid_frac


def build_target(
    path,
    var_name: str,
    time_sample_stride: int = 90,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (regridded DataArray on the native ECMWF grid, tgt_lat,
    tgt_lon, native-grid valid mask used) -- the caller combines masks
    across rain/tmax/tmin to get the single project-wide mask.
    """
    da = _load_raw(path, var_name)
    sample = da.isel(time=slice(0, None, time_sample_stride))
    src_valid = build_valid_mask(sample)

    tgt_lat, tgt_lon = _native_grid()
    weights = build_conservative_weights(da.lat.values, da.lon.values, tgt_lat, tgt_lon, src_valid)

    values = conservative_regrid(da.values, da.lat.values, da.lon.values, tgt_lat, tgt_lon, weights=weights)
    out = xr.DataArray(
        values,
        dims=["time", "lat", "lon"],
        coords={"time": da.time.values, "lat": tgt_lat, "lon": tgt_lon},
        name=var_name,
    )
    return out, tgt_lat, tgt_lon, src_valid


def build_all_targets() -> dict[str, xr.DataArray]:
    """Regrid rain, tmax, tmin to the ECMWF grid and intersect their
    per-target masks (NaN in any of the three at a cell => masked
    everywhere, since the project scores all three on one shared mask,
    literature-review.md S1.3).
    """
    rain, lat, lon, _ = build_target(config.IMD_RAIN_FILE, "rain")
    tmax, _, _, _ = build_target(config.IMD_TMAX_FILE, "tmax")
    tmin, _, _, _ = build_target(config.IMD_TMIN_FILE, "tmin")

    shared_mask = np.isfinite(rain.isel(time=0).values)
    for da in (tmax, tmin):
        # align on lat/lon (both already on the same native grid)
        shared_mask &= np.isfinite(da.isel(time=0).values)

    rain = rain.where(shared_mask)
    tmax = tmax.where(shared_mask)
    tmin = tmin.where(shared_mask)

    return {"rain": rain, "tmax": tmax, "tmin": tmin, "mask": xr.DataArray(shared_mask, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})}
