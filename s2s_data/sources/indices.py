"""Climate indices used for physical attribution, not as forecast inputs.

Role in the project: after the regime/confidence signal is trained, its
values are correlated against MJO phase/amplitude and ENSO state to check
that it tracks known teleconnections rather than being a statistical
artifact (mirroring the AI2ES interpretability analyses).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from s2s_data.config import BROWSER_USER_AGENT, ONI_URL, RMM_URL

_ONI_SEASON_CENTER_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def load_oni() -> pd.DataFrame:
    """NOAA CPC Oceanic Nino Index, one row per overlapping 3-month season."""
    df = pd.read_csv(ONI_URL, sep=r"\s+")
    df["month"] = df["SEAS"].map(_ONI_SEASON_CENTER_MONTH)
    df["date"] = pd.to_datetime(dict(year=df["YR"], month=df["month"], day=1))
    return df[["date", "ANOM"]].rename(columns={"ANOM": "oni"}).sort_values("date")


def load_rmm() -> pd.DataFrame:
    """BoM Wheeler-Hendon RMM MJO index (daily RMM1, RMM2, phase, amplitude).

    Requires a browser User-Agent header -- BoM returns 403 without one.
    Missing values are coded as 1e36 or 999 in the source file.
    """
    response = requests.get(RMM_URL, headers={"User-Agent": BROWSER_USER_AGENT}, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(
        io.StringIO(response.text),
        sep=r"\s+",
        skiprows=2,
        header=None,
        usecols=range(7),
        names=["year", "month", "day", "rmm1", "rmm2", "phase", "amplitude"],
    )
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    for col in ["rmm1", "rmm2", "amplitude"]:
        df[col] = df[col].replace([1e36, 999], np.nan)
    df["phase"] = df["phase"].replace(999, np.nan)
    return df[["date", "rmm1", "rmm2", "phase", "amplitude"]]
