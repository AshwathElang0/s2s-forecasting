from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ecmwf_abc_experiment.py"
spec = importlib.util.spec_from_file_location("run_ecmwf_abc_experiment", module_path)
mod = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(mod)
load_truth_weekly = mod.load_truth_weekly


def test_load_truth_weekly_aggregates_daily_imd_to_four_week_targets(tmp_path: Path):
    time_index = pd.date_range("2019-12-25", periods=36, freq="D")
    lat = np.array([0.0, 10.0])
    lon = np.array([0.0, 10.0])
    rain = np.arange(len(time_index) * len(lat) * len(lon), dtype=float).reshape(len(time_index), len(lat), len(lon))
    ds = xr.Dataset({"rain": (("time", "lat", "lon"), rain)}, coords={"time": time_index, "lat": lat, "lon": lon})
    obs_path = tmp_path / "imd_rain.nc"
    ds.to_netcdf(obs_path)

    weekly = load_truth_weekly(obs_path, dt.date(2020, 1, 1), lat, lon)

    assert weekly.shape == (4, 2, 2)
    expected_week_1 = rain[8:15].sum(axis=0)
    expected_week_2 = rain[15:22].sum(axis=0)
    expected_week_3 = rain[22:29].sum(axis=0)
    expected_week_4 = rain[29:36].sum(axis=0)
    np.testing.assert_allclose(weekly[0], expected_week_1)
    np.testing.assert_allclose(weekly[1], expected_week_2)
    np.testing.assert_allclose(weekly[2], expected_week_3)
    np.testing.assert_allclose(weekly[3], expected_week_4)


def test_load_truth_weekly_masks_negative_imd_missing_values(tmp_path: Path):
    time_index = pd.date_range("2019-12-25", periods=36, freq="D")
    lat = np.array([0.0])
    lon = np.array([0.0])
    rain = np.ones((len(time_index), 1, 1), dtype=float)
    rain[8, 0, 0] = -999.0
    obs_path = tmp_path / "imd_rain_missing.nc"
    xr.Dataset({"rain": (("time", "lat", "lon"), rain)}, coords={"time": time_index, "lat": lat, "lon": lon}).to_netcdf(obs_path)

    weekly = load_truth_weekly(obs_path, dt.date(2020, 1, 1), lat, lon)

    assert np.isnan(weekly[0, 0, 0])
