"""Run the ECMWF-only 28-day daily precipitation post-processing experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from s2s_models.daily_postprocessing import ECMWFDailyPostprocessor

DEFAULT_DATA_DIR = Path("/home/ash/weather-data/ECMWF_tp_India_2004-2023")
DEFAULT_IMD_DATA_PATH = Path("/home/ash/Downloads/IMD Rainfall/merged_imd_rain_data_1951_2025.nc")
DEFAULT_OUTPUT_DIR = Path("data_cache/eval_results/ecmwf_daily_postprocessing")
N_DAYS = 28


def available_dates(data_dir: Path) -> list[dt.date]:
    dates = []
    for path in data_dir.glob("*_cf_reforecast_*.grib"):
        try:
            date_text = path.stem.rsplit("_", 3)[-3:]
            dates.append(dt.date.fromisoformat("-".join(date_text)))
        except ValueError:
            continue
    return sorted(set(dates))


def path_for(data_dir: Path, kind: str, init_date: dt.date) -> Path:
    return data_dir / f"ECMWF_tp_India_{kind}_reforecast_{init_date:%Y_%m_%d}.grib"


def load_member_daily(path: Path, data_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        values = ds["tp"].load().values
        lat = ds["latitude"].values
        lon = ds["longitude"].values
    finally:
        ds.close()
    if data_type == "cf":
        values = values[None, ...]
    increments = np.diff(values, axis=1, prepend=np.zeros_like(values[:, :1]))
    return np.maximum(increments[:, :N_DAYS], 0.0), lat, lon


def load_truth_daily(
    obs_path: Path,
    init_date: dt.date,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    ds = xr.open_dataset(obs_path)
    try:
        rain = ds["rain"].load()
        obs_lat = np.asarray(rain["lat"].values, dtype=float)
        obs_lon = np.asarray(rain["lon"].values, dtype=float)
        time_values = pd.DatetimeIndex(rain["time"].values)
        start_idx = int(np.searchsorted(time_values, pd.Timestamp(init_date), side="left")) + 1
        window = rain.isel(time=slice(start_idx, start_idx + N_DAYS)).values.astype(float)
        if window.shape[0] != N_DAYS:
            raise ValueError(f"Not enough IMD data for 28 daily targets at {init_date}")
        window[window < 0] = np.nan
        lat_idx = np.asarray([int(np.argmin(np.abs(obs_lat - value))) for value in lat], dtype=int)
        lon_idx = np.asarray([int(np.argmin(np.abs(obs_lon - value))) for value in lon], dtype=int)
        return window[:, lat_idx][:, :, lon_idx]
    finally:
        ds.close()


def load_dates(
    data_dir: Path,
    dates: list[dt.date],
    truth_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    members = []
    targets = []
    lat = lon = None
    for init_date in dates:
        member_daily, lat, lon = load_member_daily(path_for(data_dir, "pf", init_date), "pf")
        target_daily = load_truth_daily(truth_path, init_date, lat, lon)
        members.append(member_daily)
        targets.append(target_daily)
    return np.stack(members), np.stack(targets), lat, lon


def flatten_daily_data(member_daily: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert date/grid arrays into independent grid-cell trajectories."""
    if member_daily.ndim == 5:
        n_dates, n_members, n_days, n_lat, n_lon = member_daily.shape
        features_input = member_daily.transpose(0, 3, 4, 1, 2).reshape(n_dates * n_lat * n_lon, n_members, n_days)
        target_output = targets.transpose(0, 2, 3, 1).reshape(n_dates * n_lat * n_lon, n_days)
    elif member_daily.ndim == 4:
        n_dates, n_members, n_days, n_cells = member_daily.shape
        features_input = member_daily.transpose(0, 3, 1, 2).reshape(n_dates * n_cells, n_members, n_days)
        target_output = targets.transpose(0, 2, 1).reshape(n_dates * n_cells, n_days)
    else:
        raise ValueError("member_daily must have shape (dates, members, days, lat, lon) or (dates, members, days, cells)")
    return ECMWFDailyPostprocessor.make_features(features_input), target_output


def valid_training_cells(train_targets: np.ndarray, test_targets: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(train_targets), axis=(0, 1)) & np.all(np.isfinite(test_targets), axis=(0, 1))


def scores(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residuals = predictions - targets
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history["epoch"], history["train_loss"], label="training loss", linewidth=2)
    if "validation_loss" in history:
        axis.plot(history["epoch"][:len(history["validation_loss"])], history["validation_loss"], label="validation loss", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Log-rainfall loss plus smoothness penalty")
    axis.set_title("ECMWF-only daily post-processor training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "daily_postprocessing_loss.png", dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    dates = available_dates(args.data_dir)
    train_dates = [date for date in dates if args.train_start_year <= date.year <= args.train_end_year]
    test_dates = [date for date in dates if args.test_start_year <= date.year <= args.test_end_year]
    if args.max_dates_per_year > 0:
        train_dates = [date for date in train_dates if date.year == args.train_start_year][:args.max_dates_per_year]
        test_dates = [date for date in test_dates if date.year == args.test_start_year][:args.max_dates_per_year]
    if len(train_dates) < 4 or len(test_dates) < 1:
        raise ValueError("Need at least four training dates and one test date")

    print(f"Train: {len(train_dates)} dates ({train_dates[0]} to {train_dates[-1]})")
    print(f"Test: {len(test_dates)} dates ({test_dates[0]} to {test_dates[-1]})")
    train_members, train_targets, lat, lon = load_dates(args.data_dir, train_dates, args.truth_data_path)
    test_members, test_targets, test_lat, test_lon = load_dates(args.data_dir, test_dates, args.truth_data_path)
    if not (np.array_equal(lat, test_lat) and np.array_equal(lon, test_lon)):
        raise ValueError("Train/test grids differ")

    valid_cells = valid_training_cells(train_targets, test_targets)
    if not np.any(valid_cells):
        raise ValueError("No grid cells have complete valid targets")
    train_members = train_members[:, :, :, valid_cells]
    test_members = test_members[:, :, :, valid_cells]
    train_targets = np.nan_to_num(train_targets[:, :, valid_cells], nan=0.0)
    test_targets = np.nan_to_num(test_targets[:, :, valid_cells], nan=0.0)

    train_features, train_y = flatten_daily_data(train_members, train_targets)
    test_features, test_y = flatten_daily_data(test_members, test_targets)
    validation_count = max(1, len(train_dates) // 5)
    fit_date_count = len(train_dates) - validation_count
    cells = int(np.sum(valid_cells))
    fit_rows = fit_date_count * cells
    fit_features, fit_y = train_features[:fit_rows], train_y[:fit_rows]
    validation_features, validation_y = train_features[fit_rows:], train_y[fit_rows:]

    model = ECMWFDailyPostprocessor(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        smoothness_weight=args.smoothness_weight,
        extreme_weight=args.extreme_weight,
        extreme_threshold=args.extreme_threshold,
    )
    model.fit(fit_features, fit_y, validation_data=(validation_features, validation_y))
    predictions = model.predict(test_features)
    raw_predictions = test_features[:, :, 0]

    result = {
        "experiment": "ECMWF-only daily 28-day precipitation post-processing",
        "forecast_time_inputs": ["ECMWF perturbed members", "calendar encoded by forecast order"],
        "observations_used_at_forecast_time": False,
        "train_dates": [str(date) for date in train_dates],
        "test_dates": [str(date) for date in test_dates],
        "valid_grid_cells": cells,
        "daily_scores": {
            "raw_ensemble_mean": scores(test_y, raw_predictions),
            "temporal_residual_model": scores(test_y, predictions),
        },
        "weekly_scores": {},
        "training_history": model.history_,
    }
    for week in range(4):
        start, end = week * 7, (week + 1) * 7
        result["weekly_scores"][str(week + 1)] = {
            "raw_ensemble_mean": scores(test_y[:, start:end].sum(axis=1), raw_predictions[:, start:end].sum(axis=1)),
            "temporal_residual_model": scores(test_y[:, start:end].sum(axis=1), predictions[:, start:end].sum(axis=1)),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "daily_postprocessing_results.json").write_text(json.dumps(result, indent=2))
    plot_history(model.history_, args.output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--truth-data-path", type=Path, default=DEFAULT_IMD_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-start-year", type=int, default=2017)
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--test-start-year", type=int, default=2019)
    parser.add_argument("--test-end-year", type=int, default=2020)
    parser.add_argument("--max-dates-per-year", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--extreme-weight", type=float, default=1.5)
    parser.add_argument("--extreme-threshold", type=float, default=10.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
