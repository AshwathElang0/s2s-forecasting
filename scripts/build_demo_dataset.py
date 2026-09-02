"""End-to-end smoke test of the S2S data pipeline.

For one Wednesday GEFSv12 init date, builds:
  - the raw dynamical forecast's weekly precip totals for lead weeks 1-4
    (regridded to the CONUS analysis grid), across all 5 members
  - the matching CPC-verified weekly precip totals and extreme-event labels
  - MJO/ONI covariates for that date

This validates that every source can be pulled, regridded, and aligned onto
one common (lead_week, lat, lon) grid -- the shape the actual training
pipeline will consume. Not a full data build (that's a much larger, likely
background job); this is the "does the plumbing work" check.
"""

from __future__ import annotations

import datetime as dt

from s2s_data.config import CACHE_DIR, GEFS_MEMBERS, LEAD_WEEKS
from s2s_data.sources import cpc, gefs, indices


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    init_date = dt.date(2018, 7, 11)  # a Wednesday
    assert gefs.is_extended_init_date(init_date), f"{init_date} must be a Wednesday"

    section(f"GEFSv12 reforecast — init {init_date} ({len(GEFS_MEMBERS)} members)")
    member_weekly = []
    for member in GEFS_MEMBERS:
        precip = gefs.load_member_precip(init_date, member)
        weekly = gefs.weekly_totals_by_lead(precip, init_date, LEAD_WEEKS)
        member_weekly.append(weekly)
        print(f"  {member}: shape={weekly.shape}, "
              f"mean weekly total (mm)={float(weekly.mean()):.2f}")

    import xarray as xr

    ensemble = xr.concat(member_weekly, dim="member").assign_coords(member=list(GEFS_MEMBERS))
    print(f"\nFull ensemble array: dims={ensemble.dims}, shape={ensemble.shape}")
    print(f"Ensemble-mean weekly total by lead week (CONUS mean, mm):")
    print(ensemble.mean(dim=["member", "lat", "lon"]).to_series())

    section("CPC verification for the same weeks")
    years_needed = sorted({init_date.year, (init_date + dt.timedelta(days=28)).year})
    cpc_daily = cpc.load_years(years_needed)
    cpc_weekly_all = cpc.weekly_totals(cpc_daily)

    verif_weeks = []
    for w in LEAD_WEEKS:
        start = init_date + dt.timedelta(days=7 * (w - 1))
        week_slice = cpc_weekly_all.sel(time=str(start), method="nearest")
        verif_weeks.append(week_slice)
    verif = xr.concat(verif_weeks, dim="lead_week").assign_coords(lead_week=list(LEAD_WEEKS))
    print(f"CPC verification array: shape={verif.shape}")
    print(f"CPC-observed CONUS-mean weekly total by lead week (mm):")
    print(verif.mean(dim=["lat", "lon"]).to_series())

    section("Extreme-event threshold (single-year climatology, smoke test only)")
    threshold = cpc.climatological_threshold(cpc_weekly_all)
    labels = cpc.extreme_labels(verif, threshold)
    print(f"Extreme labels shape: {labels.shape}")
    print(f"Fraction of CONUS grid cells labeled extreme, by lead week:")
    print(labels.mean(dim=["lat", "lon"]).to_series())
    print(
        "\nNOTE: threshold fit on ~1 year of data as a plumbing check only -- "
        "the real pipeline needs a multi-decade climatology (per the "
        "validation plan) for the 99th-percentile threshold to be meaningful."
    )

    section("Forecast skill sanity check (ensemble mean vs. CPC, per lead week)")
    ens_mean = ensemble.mean(dim="member")
    for w in LEAD_WEEKS:
        fc = ens_mean.sel(lead_week=w)
        ob = verif.sel(lead_week=w)
        corr = xr.corr(fc, ob, dim=["lat", "lon"])
        print(f"  week {w}: spatial correlation(forecast, observed) = {float(corr):.3f}")

    section("Climate indices")
    oni = indices.load_oni()
    rmm = indices.load_rmm()
    print(f"ONI: {len(oni)} rows, latest: {oni.iloc[-1].to_dict()}")
    print(f"RMM: {len(rmm)} rows, latest: {rmm.iloc[-1].to_dict()}")
    oni_at_init = oni[oni["date"] <= str(init_date)].iloc[-1]
    rmm_at_init = rmm[rmm["date"] == str(init_date)]
    print(f"\nONI nearest to init date {init_date}: {oni_at_init.to_dict()}")
    print(f"RMM on init date {init_date}: "
          f"{rmm_at_init.iloc[0].to_dict() if len(rmm_at_init) else 'not found'}")

    section("DONE — pipeline runs end-to-end")


if __name__ == "__main__":
    main()
