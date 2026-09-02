"""Probe availability of the open-source data sources for the S2S enhancement project.

Each check does the smallest possible read (metadata / a single timestep) rather than
downloading full datasets, so this can run quickly and repeatedly.
"""

from __future__ import annotations

import sys
import traceback


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_cpc_precip() -> None:
    """CPC Unified Gauge-Based Daily Precipitation over CONUS (NOAA PSL, OPeNDAP)."""
    section("1. CPC Unified Gauge-Based Daily Precipitation (verification/labels)")
    import xarray as xr

    # Historical archive runs through ~2006; recent years live under precip/RT/.
    url = "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_us_precip/RT/precip.V1.0.2020.nc"
    ds = xr.open_dataset(url)
    print(ds)
    print("\nSample variable stats (first timestep):")
    print(ds["precip"].isel(time=0).values[:3, :3])


def check_cesm2_le() -> None:
    """CESM2 Large Ensemble on AWS Open Data (Zarr, cloud-optimized)."""
    section("2. CESM2 Large Ensemble (AWS S3, regime-signal pretraining)")
    import xarray as xr

    catalog_url = "https://ncar-cesm2-lens.s3-us-west-2.amazonaws.com/catalogs/aws-cesm2-le.csv"
    import pandas as pd

    catalog = pd.read_csv(catalog_url)
    print(f"Catalog entries: {len(catalog)}")
    print(f"Available variables: {sorted(catalog['variable'].dropna().unique().tolist())}")

    # No single "total precip" variable — CESM splits it into convective (PRECC)
    # and large-scale (PRECL) components, summed for total precipitation.
    precip_rows = catalog[
        catalog["variable"].isin(["PRECC", "PRECL"]) & (catalog["frequency"] == "daily")
    ]
    print(f"\nDaily PRECC/PRECL rows: {len(precip_rows)}")
    print(precip_rows.to_string())
    if len(precip_rows):
        row = precip_rows.iloc[0]
        store = row["path"]
        ds = xr.open_zarr(store, storage_options={"anon": True}, consolidated=True)
        print(ds)


def check_s2s_reforecast() -> None:
    """NOAA GEFSv12 reforecast on AWS Open Data (dynamical forecast input).

    Note: the ECMWF S2S archive via the IRI Data Library now requires a free
    account login (302 redirect to a sign-in page) and is no longer fully
    anonymous. GEFSv12 reforecast on AWS is fully open (no auth), so it's the
    more practical dynamical-model source to build the pipeline against.
    """
    section("3. S2S Reforecast — NOAA GEFSv12 on AWS (dynamical forecast input)")
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    year_dirs = fs.ls("noaa-gefs-retrospective/GEFSv12/reforecast/2020/")
    print(f"Reforecast init dates available for 2020: {len(year_dirs)}")
    print(year_dirs[:5])

    sample_dir = fs.ls(year_dirs[0])
    print(f"\nContents of {year_dirs[0]}:")
    print(sample_dir[:10])


def check_climate_indices() -> None:
    """MJO RMM index and ONI (ENSO) index — plain text time series."""
    section("4. MJO / ONI climate indices (physical attribution covariates)")
    import pandas as pd

    oni_url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    oni = pd.read_csv(oni_url, sep=r"\s+")
    print("ONI (ENSO) index sample:")
    print(oni.tail(6).to_string())

    mjo_url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_mjo_index/proj_norm_order.ascii"
    mjo = pd.read_csv(mjo_url, sep=r"\s+", skiprows=1, header=None)
    print("\nMJO RMM index sample (last 3 rows):")
    print(mjo.tail(3).to_string())


def main() -> None:
    checks = [
        check_cpc_precip,
        check_cesm2_le,
        check_s2s_reforecast,
        check_climate_indices,
    ]
    results = {}
    for check in checks:
        name = check.__doc__.splitlines()[0]
        try:
            check()
            results[name] = "OK"
        except Exception as exc:  # noqa: BLE001 - want to report and continue
            results[name] = f"FAILED: {exc!r}"
            print(f"\n--- Traceback for {name} ---")
            traceback.print_exc()

    section("SUMMARY")
    for name, status in results.items():
        print(f"[{status.split(':')[0]:6s}] {name}")

    if any(v != "OK" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
