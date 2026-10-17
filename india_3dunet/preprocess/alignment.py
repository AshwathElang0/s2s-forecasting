"""Line up ECMWF lead days with the IMD dates they actually forecast.

IMD labels date D with the 24 h of rain ending 08:30 IST (03 UTC) on D. The
ECMWF reforecast starts at 00 UTC, so its first lead day (0-24 h, cached lead
index 0) overlaps 21 of 24 hours with IMD date issue+1, not issue+0. Checked
empirically over 208 issue dates (2008-2019): ECMWF lead index 0 correlates
0.525 with IMD issue+1 against 0.438 with IMD issue+0, and the same holds at
every lead.

The sample cache (build_dataset.py) stores target index t = IMD date issue+t,
which puts every observation-side array one day early relative to ECMWF. Until
the cache is rebuilt, score on these slices instead of the raw arrays:

  observation side  (cached target, U-Net predictions, valid-date climatology)
                    -> obs_side(x)   = x[:, OBS_OFFSET : OBS_OFFSET + n]
  ECMWF side        (tp_mean input channel, model climatology by lead)
                    -> ecmwf_side(x) = x[:, :n]

After both slices, index k means forecast day k+1: ECMWF lead k scored against
IMD date issue+1+k. U-Net output 0 (mostly rain that fell before the forecast
started) is dropped.

Once the cache is rebuilt with targets starting at issue+1, set OBS_OFFSET to 0.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

import config

OBS_OFFSET = 1
ALIGNED_DAYS = config.LEAD_DAYS - OBS_OFFSET   # 31 scoreable forecast days


def obs_side(x: np.ndarray, n: int = ALIGNED_DAYS) -> np.ndarray:
    """Observation-side array (N, lead, ...) -> aligned (N, n, ...)."""
    return x[:, OBS_OFFSET:OBS_OFFSET + n]


def ecmwf_side(x: np.ndarray, n: int = ALIGNED_DAYS) -> np.ndarray:
    """ECMWF-side array (N, lead, ...) -> aligned (N, n, ...)."""
    return x[:, :n]


def valid_date(issue_date: datetime, k: int) -> datetime:
    """IMD date verified by aligned index k (forecast day k+1)."""
    return issue_date + timedelta(days=k + 1)
