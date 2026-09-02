"""CPC Unified Gauge-Based Daily Precipitation over CONUS.

Role in the project: ground truth for both (a) verifying weekly precipitation
totals against forecasts, and (b) defining the extreme-event label (>=99th
percentile of the local climatological weekly-total distribution).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from s2s_data.config import CACHE_DIR, EXTREME_PERCENTILE
from s2s_data.regrid import snap_to_target_coords

_MAX_DOWNLOAD_ATTEMPTS = 8
_RETRY_BACKOFF_SECONDS = 3

# Plain HTTP file download (THREDDS `fileServer` service) instead of OPeNDAP
# (`dodsC`). The dodsC endpoint was found empirically to be unreliable in
# this pipeline: identical requests would sometimes come back silently
# zero-filled instead of erroring, and retries within the same process kept
# failing together (consistent with a bad cached connection/session rather
# than a one-off blip). Downloading the actual file sidesteps the DAP
# protocol layer entirely and lets us cache the file locally so we don't
# repeatedly hit the free server anyway.
#
# The archive is split across two subdirectories with a hard cutover: years
# <=2006 only exist under the historical path, years >=2007 only under RT
# (confirmed by listing both THREDDS catalogs directly) -- there's no single
# path that works for an arbitrary year.
_HISTORICAL_URL_TEMPLATE = (
    "https://psl.noaa.gov/thredds/fileServer/Datasets/cpc_us_precip/precip.V1.0.{year}.nc"
)
_RT_URL_TEMPLATE = (
    "https://psl.noaa.gov/thredds/fileServer/Datasets/cpc_us_precip/RT/precip.V1.0.{year}.nc"
)
_RT_START_YEAR = 2007
_CPC_CACHE_DIR = CACHE_DIR / "cpc"


def _url_for_year(year: int) -> str:
    template = _RT_URL_TEMPLATE if year >= _RT_START_YEAR else _HISTORICAL_URL_TEMPLATE
    return template.format(year=year)


def _looks_corrupted(da: xr.DataArray) -> bool:
    """A real annual CONUS precipitation field always has rain somewhere;
    an all-zero read indicates a corrupted/truncated download."""
    sample = da.isel(time=slice(0, min(10, da.sizes.get("time", 1))))
    return bool((sample.fillna(0) == 0).all())


def _local_path(year: int) -> Path:
    _CPC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CPC_CACHE_DIR / f"precip.V1.0.{year}.nc"


def _content_length(url: str) -> int | None:
    response = requests.head(url, timeout=30, allow_redirects=True)
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    return int(length) if length is not None else None


def _download_or_resume(year: int, dest: Path) -> None:
    """Download with resume-from-offset on failure.

    The observed truncation pattern (connection reliably drops a few hundred
    KB to a few MB before the *end* of an ~9-12MB file, on an otherwise
    working connection) looks like a proxy/middlebox timing out a long-lived
    stream rather than a real server outage. Restarting from scratch on every
    retry wastes the megabytes that *did* come through; resuming via an HTTP
    Range request from the current partial size is both faster and more
    likely to eventually complete.
    """
    url = _url_for_year(year)
    tmp_dest = dest.with_suffix(".nc.partial")
    total_size = _content_length(url)

    for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
        existing = tmp_dest.stat().st_size if tmp_dest.exists() else 0
        if total_size is not None and existing >= total_size:
            break

        headers = {"Range": f"bytes={existing}-"} if existing else {}
        mode = "ab" if existing else "wb"
        try:
            with requests.get(url, timeout=120, stream=True, headers=headers) as response:
                if existing and response.status_code == 200:
                    # Server ignored the Range request -- it's sending the
                    # whole file again from byte 0, so start the local file over.
                    existing = 0
                    mode = "wb"
                elif response.status_code not in (200, 206):
                    response.raise_for_status()
                with open(tmp_dest, mode) as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except (requests.exceptions.RequestException, OSError) as exc:
            print(
                f"[cpc] year {year}: download attempt {attempt}/{_MAX_DOWNLOAD_ATTEMPTS} "
                f"failed at {tmp_dest.stat().st_size if tmp_dest.exists() else 0} bytes "
                f"({exc!r}), will resume..."
            )
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue

        if total_size is None or tmp_dest.stat().st_size >= total_size:
            break

    if total_size is not None and (
        not tmp_dest.exists() or tmp_dest.stat().st_size < total_size
    ):
        got = tmp_dest.stat().st_size if tmp_dest.exists() else 0
        raise RuntimeError(
            f"CPC precip download for {year} incomplete after "
            f"{_MAX_DOWNLOAD_ATTEMPTS} attempts ({got}/{total_size} bytes)."
        )
    tmp_dest.rename(dest)


def load_year(year: int) -> xr.DataArray:
    """Load one year of CPC daily precipitation (mm/day) at native 0.25 deg CONUS grid.

    Downloads and caches the file locally on first use (see module docstring
    for why this is preferred over OPeNDAP). Re-downloads and retries if the
    cached file is ever found to be corrupted (all-zero).
    """
    dest = _local_path(year)

    if not dest.exists():
        _download_or_resume(year, dest)

    da = xr.open_dataset(dest)["precip"].load()
    if _looks_corrupted(da):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"CPC precip for {year} downloaded successfully but looks all-zero "
            "-- likely a genuine data issue, not a transport failure."
        )
    return snap_to_target_coords(da)


def load_years(years: list[int]) -> xr.DataArray:
    """Load and concatenate multiple years along time."""
    return xr.concat([load_year(y) for y in years], dim="time")


def weekly_totals(daily_precip: xr.DataArray) -> xr.DataArray:
    """Non-overlapping 7-day sums, labeled by the start date of each window."""
    return daily_precip.resample(time="7D").sum(skipna=False)


def climatological_threshold(
    weekly: xr.DataArray, percentile: float = EXTREME_PERCENTILE
) -> xr.DataArray:
    """Per-gridpoint weekly-total threshold above which a week counts as extreme.

    Computed across all available weeks/years passed in — the caller controls
    the climatology period (e.g. a multi-decade record for a real fit, a
    single year only as a smoke test).
    """
    return weekly.quantile(percentile / 100.0, dim="time", skipna=True).drop_vars(
        "quantile", errors="ignore"
    )


def extreme_labels(weekly: xr.DataArray, threshold: xr.DataArray) -> xr.DataArray:
    """Boolean/0-1 label: did this week's total exceed the local extreme threshold."""
    return (weekly > threshold).astype("int8")
