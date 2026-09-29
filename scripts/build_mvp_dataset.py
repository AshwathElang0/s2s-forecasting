"""Build a minimum-viable training dataset for the S2S enhancement idea.

Scope (deliberately small — this is for a first pass at testing the core
idea, not the full-scale study):
  - GEFSv12 reforecast, Wednesday init dates only, 2018-2019 (~104 dates,
    5 members each) -- the raw forecast to be corrected.
  - CPC verification + extreme-event labels, using a 20-year (2000-2019)
    climatology for the 99th-percentile threshold (vs. the 1-year
    placeholder used in the earlier smoke test).
  - ONI and MJO RMM index values at each init date, as covariates for later
    physical-attribution analysis.
  - CESM2-LE is intentionally NOT included yet. The MVP tests whether
    gating on a confidence signal helps at all, using each date's 5-member
    GEFS ensemble spread as a cheap, already-available confidence proxy
    instead of the full CESM2-LE-pretrained regime signal. That larger
    piece is worth building only once this MVP shows the core idea works.

Resumable: writes one NetCDF file per init date under
data_cache/mvp_dataset/per_init_date/, skips dates already built, and logs
(rather than crashes on) any per-date failure so a long run can be safely
interrupted and restarted.
"""

from __future__ import annotations

import datetime as dt
import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

from s2s_data.config import CACHE_DIR, EXTREME_PERCENTILE, GEFS_MEMBERS, LEAD_WEEKS
from s2s_data.sources import cpc, gefs, indices

MVP_START = dt.date(2018, 1, 1)
MVP_END = dt.date(2019, 12, 31)
CLIMATOLOGY_YEARS = list(range(2000, 2020))

OUTPUT_DIR = CACHE_DIR / "mvp_dataset"
PER_INIT_DIR = OUTPUT_DIR / "per_init_date"
CLIMATOLOGY_CACHE = OUTPUT_DIR / "climatology_threshold.nc"
LOG_PATH = OUTPUT_DIR / "build_log.txt"


def log(message: str) -> None:
    print(message, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(message + "\n")


def wednesdays_between(start: dt.date, end: dt.date):
    d = start
    while d.weekday() != 2:
        d += dt.timedelta(days=1)
    while d <= end:
        yield d
        d += dt.timedelta(days=7)


def build_climatology_threshold() -> xr.DataArray:
    if CLIMATOLOGY_CACHE.exists():
        log(f"Climatology threshold already cached at {CLIMATOLOGY_CACHE}")
        return xr.open_dataarray(CLIMATOLOGY_CACHE)

    log(f"Building {len(CLIMATOLOGY_YEARS)}-year climatological threshold "
        f"({CLIMATOLOGY_YEARS[0]}-{CLIMATOLOGY_YEARS[-1]})...")
    daily = cpc.load_years(CLIMATOLOGY_YEARS)
    weekly = cpc.weekly_totals(daily)
    threshold = cpc.climatological_threshold(weekly, EXTREME_PERCENTILE)
    threshold = threshold.compute()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold.to_netcdf(CLIMATOLOGY_CACHE)
    log("Climatology threshold built and cached.")
    return threshold


import threading

_SAVE_LOCK = threading.Lock()


def process_init_date(
    init_date: dt.date,
    threshold: xr.DataArray,
    cpc_weekly_all: xr.DataArray,
    oni,
    rmm,
) -> str:
    out_path = PER_INIT_DIR / f"{init_date}.nc"
    if out_path.exists():
        return "skipped (already built)"

    member_weekly = []
    for member in GEFS_MEMBERS:
        precip = gefs.load_member_precip(init_date, member)
        weekly = gefs.weekly_totals_by_lead(precip, init_date, LEAD_WEEKS)
        member_weekly.append(weekly)
    ensemble = xr.concat(member_weekly, dim="member").assign_coords(member=list(GEFS_MEMBERS))
    ensemble.name = "gefs_precip_weekly"
    # Drop leftover scalar GRIB coords (init "time", "number", "surface") --
    # they're redundant with ds.attrs["init_date"] below and otherwise
    # collide with CPC's per-lead-week "time" (the *verification* date) when
    # merged into one Dataset.
    ensemble = ensemble.drop_vars(["time", "number", "surface"], errors="ignore")

    verif_weeks = []
    for w in LEAD_WEEKS:
        start = init_date + dt.timedelta(days=7 * (w - 1))
        verif_weeks.append(cpc_weekly_all.sel(time=str(start), method="nearest"))
    verif = xr.concat(verif_weeks, dim="lead_week").assign_coords(lead_week=list(LEAD_WEEKS))
    verif = verif.rename({"time": "valid_date"})
    verif.name = "cpc_precip_weekly"

    labels = cpc.extreme_labels(verif, threshold)
    labels.name = "extreme_label"

    ds = xr.Dataset({da.name: da for da in [ensemble, verif, labels]})
    ds.attrs["init_date"] = str(init_date)

    oni_past = oni[oni["date"] <= str(init_date)]
    ds.attrs["oni"] = float(oni_past.iloc[-1]["oni"]) if len(oni_past) else float("nan")

    rmm_row = rmm[rmm["date"] == str(init_date)]
    if len(rmm_row):
        row = rmm_row.iloc[0]
        ds.attrs["rmm1"] = float(row["rmm1"])
        ds.attrs["rmm2"] = float(row["rmm2"])
        ds.attrs["mjo_phase"] = float(row["phase"])
        ds.attrs["mjo_amplitude"] = float(row["amplitude"])

    PER_INIT_DIR.mkdir(parents=True, exist_ok=True)
    with _SAVE_LOCK:
        ds.to_netcdf(out_path)
    return "OK"



import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MVP dataset for S2S forecasting.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument("--max-dates", type=int, default=None, help="Max init dates to process (default: all)")
    parser.add_argument("--start-date", type=str, default=str(MVP_START), help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=str(MVP_END), help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)

    log(f"\n{'=' * 70}\nMVP dataset build started {dt.datetime.now().isoformat()}\n{'=' * 70}")

    threshold = build_climatology_threshold()
    log("Loading ONI and RMM indices...")
    oni = indices.load_oni()
    rmm = indices.load_rmm()

    years_needed = list(range(start_date.year, end_date.year + 2))
    log(f"Preloading CPC verification data in memory for years {years_needed}...")
    cpc_daily = cpc.load_years(years_needed).load()
    cpc_weekly_all = cpc.weekly_totals(cpc_daily).compute()
    log("CPC verification data preloaded into memory.")

    init_dates = list(wednesdays_between(start_date, end_date))
    if args.max_dates is not None:
        init_dates = init_dates[:args.max_dates]

    log(f"Processing {len(init_dates)} Wednesday init dates from {start_date} to {end_date} (workers={args.workers})")

    n_ok = n_skip = n_fail = 0

    if args.workers <= 1:
        for i, init_date in enumerate(init_dates, 1):
            try:
                status = process_init_date(init_date, threshold, cpc_weekly_all, oni, rmm)
            except Exception as exc:  # noqa: BLE001 - log and keep going
                status = f"FAILED: {exc!r}"
                log(f"[{i}/{len(init_dates)}] {init_date}: {status}")
                log(traceback.format_exc())
            else:
                log(f"[{i}/{len(init_dates)}] {init_date}: {status}")

            if status == "OK":
                n_ok += 1
            elif status.startswith("skipped"):
                n_skip += 1
            else:
                n_fail += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_date = {
                executor.submit(process_init_date, d, threshold, cpc_weekly_all, oni, rmm): (i, d)
                for i, d in enumerate(init_dates, 1)
            }
            for future in as_completed(future_to_date):
                i, init_date = future_to_date[future]
                try:
                    status = future.result()
                except Exception as exc:  # noqa: BLE001
                    status = f"FAILED: {exc!r}"
                    log(f"[{i}/{len(init_dates)}] {init_date}: {status}")
                    log(traceback.format_exc())
                else:
                    log(f"[{i}/{len(init_dates)}] {init_date}: {status}")

                if status == "OK":
                    n_ok += 1
                elif status.startswith("skipped"):
                    n_skip += 1
                else:
                    n_fail += 1


    log(f"\n{'=' * 70}\nDONE: {n_ok} built, {n_skip} skipped (already cached), "
        f"{n_fail} failed, out of {len(init_dates)} total\n{'=' * 70}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

