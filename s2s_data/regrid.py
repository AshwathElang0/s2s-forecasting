"""Regridding to the common CONUS analysis grid.

Uses plain bilinear interpolation (xarray's built-in `.interp`) rather than a
conservative remapper (e.g. xESMF). xESMF needs the ESMF Fortran/MPI library,
which isn't worth installing at this prototyping stage. Bilinear is a known
simplification: it doesn't conserve grid-cell area-weighted totals the way a
conservative remap would, which matters most when going from coarse (CESM2,
~1 deg) to fine (CPC, 0.25 deg) grids for a quantity like precipitation. If
regridding artifacts show up in later validation, swap this for xESMF
conservative remapping (`brew install esmf`) without changing the calling
code's interface.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from s2s_data.config import TARGET_LAT, TARGET_LON


def to_target_grid(da: xr.DataArray, lat_name: str = "lat", lon_name: str = "lon") -> xr.DataArray:
    """Bilinearly interpolate a DataArray onto the shared CONUS analysis grid.

    Output dims are always renamed to ``lat``/``lon`` regardless of the
    input's native dimension names, so every source lands on identically
    named dimensions after regridding.
    """
    regridded = da.interp(
        {lat_name: TARGET_LAT, lon_name: TARGET_LON},
        method="linear",
        kwargs={"fill_value": None},
    )
    rename_map = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        regridded = regridded.rename(rename_map)
    return snap_to_target_coords(regridded)


def snap_to_target_coords(da: xr.DataArray, lat_name: str = "lat", lon_name: str = "lon") -> xr.DataArray:
    """Force lat/lon coordinates to be bit-identical to TARGET_LAT/TARGET_LON.

    Sources that are natively already on the target grid (e.g. CPC) can still
    carry float32 coordinate values that are numerically close but not
    bit-identical to the float64 TARGET_LAT/TARGET_LON array. Functions like
    `xr.corr` align on exact coordinate match, so a tiny dtype/rounding
    difference silently produces an empty intersection (all-NaN result)
    instead of an error. Call this on every source's final output so
    everything shares one coordinate array by construction.
    """
    if da.sizes[lat_name] != len(TARGET_LAT) or da.sizes[lon_name] != len(TARGET_LON):
        raise ValueError(
            f"shape mismatch: {da.sizes[lat_name]}x{da.sizes[lon_name]} vs "
            f"target {len(TARGET_LAT)}x{len(TARGET_LON)} -- regrid before snapping"
        )
    return da.assign_coords({lat_name: TARGET_LAT, lon_name: TARGET_LON})


def crop_to_conus_bounds(
    da: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
    pad_degrees: float = 2.0,
) -> xr.DataArray:
    """Cut a global field down to just past the CONUS grid bounds before interpolating.

    Avoids pulling full global fields (e.g. CESM2's 666GB dask arrays) through
    interpolation only to throw away everything outside CONUS.
    """
    lat_min, lat_max = TARGET_LAT.min() - pad_degrees, TARGET_LAT.max() + pad_degrees
    lon_min, lon_max = TARGET_LON.min() - pad_degrees, TARGET_LON.max() + pad_degrees

    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values

    lat_ascending = lat_vals[0] < lat_vals[-1]
    lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)

    return da.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)})
