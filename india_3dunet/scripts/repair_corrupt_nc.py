"""Repair 0-byte ECMWF .nc files by reconverting them from their .grib
sibling with cfgrib.

While building the sample cache, ~4% of the `q` (specific humidity) `.nc`
files under ../s2s-forecasting/data/Model data ecmwf/ turned out to be
0 bytes -- an interrupted/failed `grib_to_netcdf` run, not a bug in this
pipeline. The `.grib` originals are intact, so this reconverts just the
broken files in place rather than skipping those forecast dates entirely.

Usage:  python scripts/repair_corrupt_nc.py [--dry_run]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def find_corrupt_files() -> list[Path]:
    return [p for p in config.ECMWF_ROOT.rglob("*.nc") if p.stat().st_size == 0]


def repair_one(nc_path: Path) -> None:
    grib_path = nc_path.with_suffix(".grib")
    if not grib_path.exists() or grib_path.stat().st_size == 0:
        raise FileNotFoundError(f"no usable .grib sibling for {nc_path}")

    with xr.open_dataset(grib_path, engine="cfgrib") as ds:
        var_name = next(iter(ds.data_vars))
        da = ds[var_name]

        # step: timedelta64 -> plain integer hours, matching the healthy
        # .nc files' convention (and ecmwf_features.py's expectation).
        step_hours = (da["step"].values / np.timedelta64(1, "h")).astype("int32")
        da = da.assign_coords(step=step_hours)

        # isobaricInhPa -> level, ascending, matching the healthy files.
        if "isobaricInhPa" in da.dims:
            da = da.rename({"isobaricInhPa": "level"})
            da = da.sortby("level")
            da = da.assign_coords(level=da["level"].astype("int32"))

        da = da.astype("float64")
        for extra in ("valid_time", "time"):
            if extra in da.coords:
                da = da.drop_vars(extra)

        da.latitude.values.setflags(write=True)
        da = da.assign_coords(latitude=da.latitude.astype("float32"), longitude=da.longitude.astype("float32"))

        out_ds = da.to_dataset(name=var_name)
        tmp_path = nc_path.with_suffix(".nc.repair_tmp")
        out_ds.to_netcdf(tmp_path)

    tmp_path.replace(nc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    corrupt = find_corrupt_files()
    print(f"Found {len(corrupt)} zero-byte .nc files")

    for i, path in enumerate(corrupt):
        if args.dry_run:
            print(f"[{i + 1}/{len(corrupt)}] would repair {path}")
            continue
        try:
            repair_one(path)
            print(f"[{i + 1}/{len(corrupt)}] repaired {path}")
        except Exception as e:
            print(f"[{i + 1}/{len(corrupt)}] FAILED {path}: {e!r}")


if __name__ == "__main__":
    main()
