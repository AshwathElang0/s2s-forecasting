# Data

Two datasets live in this project: observed rainfall from IMD, and ECMWF sub-seasonal-to-seasonal (S2S) reforecasts for several atmospheric/ocean variables over a domain centered on India. The likely intent is to use the ECMWF fields as predictors and IMD rainfall as the verification/target.

## 1. IMD Rainfall — `IMD Rainfal/merged_imd_rain_data_1951_2025.nc`

A single merged NetCDF4/HDF5 file of gridded daily observed rainfall over India from the India Meteorological Department (imdpune.gov.in).

- **Variable:** `rain` — Rainfall, `mm/day`, shape `(time=27394, lat=129, lon=135)`
- **Grid:** 0.25° resolution, lat 6.5°N–38.5°N (129 pts), lon 66.5°E–100.0°E (135 pts) — covers the Indian subcontinent
- **Time:** daily, 1951-01-01 to 2025-12-31 (27,394 consecutive days, no gaps), stored as days since 1901-01-01
- **CRS:** EPSG:4326

**Caveat:** the `rain` variable declares `_FillValue = NaN`, but at least the first time slice contains sentinel values of exactly `-999.0` for masked/no-data cells rather than actual `NaN`s. Downstream code should explicitly mask `rain == -999` (in addition to NaN) rather than relying on the stated fill value alone.

## 2. IMD Temperature — `IMD Temperature/merged_imd_tmax_data_1951_2025.nc`, `merged_imd_tmin_data_1951_2025.nc`

Gridded daily observed max/min temperature over India from IMD, downloaded via the [`imdlib`](https://imdlib.readthedocs.io/en/latest/) Python package (`imd.get_data('tmax'/'tmin', 1951, 2025, fn_format='yearwise', ...)`, then `.to_netcdf()`), matching the same 1951–2025 time range as the rainfall dataset above. Raw yearly `.GRD` files are cached under `IMD Temperature/raw/tmax/` and `raw/tmin/`.

- **Variables:** `tmax` (Maximum Temperature) and `tmin` (Minimum Temperature), one file each — `°C`, shape `(time=27394, lat=31, lon=31)`
- **Grid:** coarser than rainfall — 1° resolution, lat 7.5°N–37.5°N (31 pts), lon 67.5°E–97.5°E (31 pts). This is `imdlib`'s native temperature grid, not a downsampling artifact.
- **Time:** daily, 1951-01-01 to 2025-12-31 (27,394 days), matching the IMD rainfall record exactly
- **CRS:** EPSG:4326
- Fill/sentinel values are masked to `NaN` already by `imdlib`'s `get_xarray()` (it detects the sentinel from the first grid cell, typically `99.9`), so no extra `-999`-style masking is needed as with rainfall.

## 3. ECMWF S2S Reforecasts — `Model data ecmwf/`

Seven variable-specific subfolders, each holding one ECMWF sub-seasonal reforecast per weekly issue date from **2004-01-04 to 2023-12-27** (20 years), as matching `.grib` + `.nc` pairs (the `.nc` was produced from the `.grib` via `grib_to_netcdf`). `.idx` files are just grib index caches and can be ignored/regenerated.

| Folder | Variable in file | Long name | Units | Levels |
|---|---|---|---|---|
| `ECMWF_gh_India_2004-2023` | `gh` | Geopotential height | gpm | 200, 1000 mb |
| `ECMWF_msl_India_2004-2023` | `msl` | Mean sea level pressure | Pa | surface |
| `ECMWF_q_India_2004-2023` | `q` | Specific humidity | kg/kg | 200, 500, 700, 850 mb |
| `ECMWF_tp_India_2004-2023` | `tp` | Total precipitation | kg m⁻² (≈ mm) | surface |
| `ECMWF_u_India_2004-2023` | `u` | U-wind (eastward) | m/s | 200, 1000 mb |
| `ECMWF_v_India_2004-2023` | `v` | V-wind (northward) | m/s | 200, 1000 mb |
| `ECMWF_wtmp_India_2004-2023` | `sst` (⚠ variable is `sst`, not `wtmp`) | Sea surface temperature | K | surface |

**Grid (same for all variables):** 1.5° resolution, lon 21.0°E–109.5°E (60 pts), lat -19.5°S–39.0°N (40 pts). This is a much larger domain than "India" alone — it spans the Arabian Sea, Bay of Bengal, and much of the Indian Ocean/monsoon region, presumably to capture large-scale monsoon drivers.

**Forecast structure — each file is one issue date, containing:**
- **Lead time (`step`):** 46 daily steps, 24h to 1104h (i.e. day 1 through day 46 ahead of the issue date)
- **Ensemble type**, encoded in the filename:
  - `..._cf_reforecast_YYYY_MM_DD.nc` — **control forecast**, single deterministic member
  - `..._pf_reforecast_YYYY_MM_DD.nc` — **perturbed forecast**, 10-member ensemble (`number` dim = 1–10)
- Values are packed as scaled `int16` (`scale_factor`/`add_offset`), standard for `grib_to_netcdf` output — most NetCDF readers (xarray, netCDF4) unpack this automatically.

**File counts** (cf / pf, out of ~1040 weekly issue dates each over the 20-year span):

| Variable | cf | pf |
|---|---|---|
| gh | 1038 | 1039 |
| msl | 1040 | 1040 |
| q | 1039 | 1039 |
| tp | 1039 | 1039 |
| u | 1040 | 1040 |
| v | 1040 | 1040 |
| wtmp | 1040 | 1040 |

A few variables (gh, q, tp) are missing one file relative to the full 1040/1040 set — worth checking for gaps before assuming complete coverage in a specific week.

**⚠ Reference-date caveat:** the `step` variable's units (`hours since <date>`) are **not reliable as the true issue date** — every file inspected shows the reference date's year pinned to whatever year the conversion pipeline used (e.g. a file named `..._2004_01_04.nc` reports `hours since 2024-01-04`, and one named `..._2010_06_10.nc` also reports year 2024). The month/day matches the filename but the year does not. **Always derive the actual reforecast issue date from the filename** (`YYYY_MM_DD`), and compute valid dates as `issue_date + step (in the step variable, hours)`, not from the `step` variable's stated reference date directly.

## Suggested use

- IMD `rain` is the observational truth grid (0.25°, India only, 1951–2025).
- IMD `tmax`/`tmin` are observational truth for temperature, same 1951–2025 time range but coarser (1°, India only).
- ECMWF fields are coarser (1.5°), wider-domain S2S reforecasts, useful as predictors for statistical/ML downscaling or skill evaluation against IMD rainfall/temperature, particularly `tp` (precipitation) directly, or `gh`/`msl`/`u`/`v`/`q`/`sst` as large-scale circulation/moisture/SST predictors of monsoon rainfall and temperature.
- Regridding will be needed to compare the datasets (0.25° rain grid vs. 1° temperature grid vs. 1.5° wider-domain ECMWF grid), and the reforecast issue dates only cover a weekly cadence, not daily.
