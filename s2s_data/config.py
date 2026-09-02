"""Shared constants for the S2S data pipeline.

Grid, lead-time, and threshold choices here are the ones referenced throughout
the paper plan: weekly aggregation, lead weeks 1-4, CONUS domain on the CPC
verification grid, 99th-percentile extreme-event definition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"

# Target verification/analysis grid: CPC Unified 0.25 deg CONUS grid.
# Longitudes are in 0-360 convention to match CESM2/GEFS (no dateline issues
# for a CONUS-only domain).
TARGET_LAT = np.round(np.arange(20.125, 49.875 + 1e-6, 0.25), 3)
TARGET_LON = np.round(np.arange(230.125, 304.875 + 1e-6, 0.25), 3)

LEAD_WEEKS = (1, 2, 3, 4)
EXTREME_PERCENTILE = 99

# GEFSv12 reforecast: 5 members total (1 control + 4 perturbed).
GEFS_MEMBERS = ("c00", "p01", "p02", "p03", "p04")
GEFS_BUCKET = "noaa-gefs-retrospective"

# Only Wednesday-initialized GEFSv12 reforecasts extend to 35 days
# (Days:10-35); all other days stop at Days:10-16, which cannot cover a
# week-4 (day 22-28) lead. This is a hard constraint on which init dates are
# usable for the full week 1-4 study.
GEFS_EXTENDED_WEEKDAY = 2  # Python weekday(): Monday=0 ... Wednesday=2

CESM2LE_BUCKET_CATALOG = (
    "https://ncar-cesm2-lens.s3-us-west-2.amazonaws.com/catalogs/aws-cesm2-le.csv"
)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
RMM_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
# BoM blocks non-browser user agents (returns 403 without one).
BROWSER_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
