"""One-time step: regrid the full IMD rain/tmax/tmin record to the ECMWF
1.5 grid and cache it to NetCDF, so `preprocess/build_dataset.py` doesn't
re-run the (~15-20 min) conservative regrid for every issue date.

Run once:  python scripts/cache_imd_targets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from preprocess.imd_targets import build_all_targets


def main():
    config.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = config.CACHE_ROOT / "imd_targets_1p5deg.nc"

    print("Regridding IMD rain/tmax/tmin to the ECMWF 1.5deg grid (one-time, ~15-20 min)...")
    targets = build_all_targets()

    import xarray as xr
    ds = xr.Dataset({"rain": targets["rain"], "tmax": targets["tmax"], "tmin": targets["tmin"]})
    ds["mask"] = targets["mask"]
    ds.to_netcdf(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
