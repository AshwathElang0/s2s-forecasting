"""Run Standard ABC and Gated-ABC on a local ECMWF India proxy subset.

The local ECMWF collection contains forecasts but no independent Indian
observations. This experiment therefore uses the control forecast as a
pseudo-observation and the ten perturbed forecasts as the ensemble input.
Results are an internal consistency/proof-of-plumbing check, not verified
forecast skill.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from s2s_models.abc import GatedABC, StandardABC
from s2s_models.metrics import (
    anomaly_correlation_coefficient,
    brier_score,
    brier_skill_score,
    mean_absolute_error,
    root_mean_squared_error,
)


DEFAULT_DATA_DIR = Path("/home/ash/weather-data/ECMWF_tp_India_2004-2023")
DEFAULT_OUTPUT_DIR = Path("data_cache/eval_results/ecmwf_india_proxy_subset")
LEAD_WEEKS = (1, 2, 3, 4)


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


def load_member_weekly(path: Path, data_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        values = ds["tp"].load().values
        lat = ds["latitude"].values
        lon = ds["longitude"].values
    finally:
        ds.close()

    if data_type == "cf":
        values = values[None, ...]
    # ECMWF tp is accumulated from initialization; convert to daily increments.
    increments = np.diff(values, axis=1, prepend=np.zeros_like(values[:, :1]))
    weekly = np.stack(
        [increments[:, 7 * (week - 1): 7 * week].sum(axis=1) for week in LEAD_WEEKS],
        axis=1,
    )
    return weekly, lat, lon


def load_dates(data_dir: Path, dates: list[dt.date]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = []
    members = []
    all_lat = all_lon = None
    for init_date in dates:
        cf_path = path_for(data_dir, "cf", init_date)
        pf_path = path_for(data_dir, "pf", init_date)
        target, lat, lon = load_member_weekly(cf_path, "cf")
        ensemble, pf_lat, pf_lon = load_member_weekly(pf_path, "pf")
        if not (np.array_equal(lat, pf_lat) and np.array_equal(lon, pf_lon)):
            raise ValueError(f"Grid mismatch for {init_date}")
        targets.append(target[0])
        members.append(ensemble)
        all_lat, all_lon = lat, lon

    return np.stack(members), np.stack(targets), all_lat, all_lon


def make_features(
    members: np.ndarray,
    dates: list[dt.date],
    lat: np.ndarray,
    lon: np.ndarray,
    threshold: np.ndarray,
) -> pd.DataFrame:
    # members: date, member, lead_week, lat, lon
    n_dates, n_members, n_leads, n_lat, n_lon = members.shape
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    rows = []
    for date_idx, init_date in enumerate(dates):
        doy = init_date.timetuple().tm_yday
        for lead_idx, lead_week in enumerate(LEAD_WEEKS):
            weekly = members[date_idx, :, lead_idx]
            flat = weekly.reshape(n_members, -1)
            n_points = flat.shape[1]
            rows.append(pd.DataFrame({
                "ens_mean": flat.mean(axis=0),
                "ens_std": flat.std(axis=0, ddof=1),
                "ens_min": flat.min(axis=0),
                "ens_max": flat.max(axis=0),
                **{f"f_{i:02d}": flat[i] for i in range(n_members)},
                "threshold": threshold[lead_idx].reshape(-1),
                "mean_to_threshold": flat.mean(axis=0) / (threshold[lead_idx].reshape(-1) + 1e-3),
                "spread_to_threshold": flat.std(axis=0, ddof=1) / (threshold[lead_idx].reshape(-1) + 1e-3),
                "lat": np.tile(lat_grid.reshape(-1), 1),
                "lon": np.tile(lon_grid.reshape(-1), 1),
                "doy_sin": np.full(n_points, np.sin(2 * np.pi * doy / 365.25)),
                "doy_cos": np.full(n_points, np.cos(2 * np.pi * doy / 365.25)),
                "oni": np.zeros(n_points),
                "mjo_amplitude": np.zeros(n_points),
                "mjo_phase": np.zeros(n_points),
                "lead_week": np.full(n_points, lead_week),
            }))
    return pd.concat(rows, ignore_index=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, p_pred: np.ndarray, clim: np.ndarray, labels: np.ndarray, p_ref: float) -> dict[str, float]:
    return {
        "acc": anomaly_correlation_coefficient(y_true, y_pred, clim, uncentered=True),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "brier_score": brier_score(labels, p_pred),
        "brier_skill_score": brier_skill_score(labels, p_pred, y_ref_prob=p_ref),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dates = available_dates(args.data_dir)
    train_dates = [d for d in dates if d.year == args.train_year][:args.max_dates_per_year]
    test_dates = [d for d in dates if d.year == args.test_year][:args.max_dates_per_year]
    if len(train_dates) < 2 or len(test_dates) < 2:
        raise ValueError("Not enough ECMWF dates for the requested years/subset")

    print(f"Train: {len(train_dates)} dates ({train_dates[0]} to {train_dates[-1]})")
    print(f"Test:  {len(test_dates)} dates ({test_dates[0]} to {test_dates[-1]})")
    train_members, train_target, lat, lon = load_dates(args.data_dir, train_dates)
    test_members, test_target, test_lat, test_lon = load_dates(args.data_dir, test_dates)
    if not (np.array_equal(lat, test_lat) and np.array_equal(lon, test_lon)):
        raise ValueError("Train/test ECMWF grids differ")

    results = {}
    for lead_idx, lead_week in enumerate(LEAD_WEEKS):
        train_y = train_target[:, lead_idx].reshape(len(train_dates), -1)
        test_y = test_target[:, lead_idx].reshape(len(test_dates), -1)
        train_threshold = np.quantile(train_y, 0.99, axis=0)
        threshold = np.stack([train_threshold] * len(LEAD_WEEKS)).reshape(4, len(lat), len(lon))
        train_x = make_features(train_members, train_dates, lat, lon, threshold)
        test_x = make_features(test_members, test_dates, lat, lon, threshold)
        train_x = train_x[train_x["lead_week"] == lead_week].reset_index(drop=True)
        test_x = test_x[test_x["lead_week"] == lead_week].reset_index(drop=True)
        train_y_flat = train_y.reshape(-1)
        test_y_flat = test_y.reshape(-1)
        train_labels = (train_y_flat > train_x["threshold"].to_numpy()).astype(int)
        test_labels = (test_y_flat > test_x["threshold"].to_numpy()).astype(int)
        train_clim = np.full_like(train_y_flat, np.mean(train_y_flat), dtype=float)
        test_clim = np.full_like(test_y_flat, np.mean(train_y_flat), dtype=float)
        p_ref = float(np.mean(train_labels))

        standard = StandardABC(lead_week=lead_week).fit(train_x, train_y_flat)
        standard_pred = standard.predict(test_x)
        standard_p = standard.predict_extreme_proba(test_x, test_x["threshold"].to_numpy())

        gated = GatedABC(
            lead_week=lead_week,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=2048,
            random_state=42,
        ).fit(train_x, train_y_flat, train_labels)
        gated_pred, weights = gated.predict_with_weights(test_x)
        gated_p = gated.predict_extreme_proba(test_x, test_x["threshold"].to_numpy())

        results[str(lead_week)] = {
            "standard_abc": evaluate(test_y_flat, standard_pred, standard_p, test_clim, test_labels, p_ref),
            "gated_abc": evaluate(test_y_flat, gated_pred, gated_p, test_clim, test_labels, p_ref),
            "mean_test_gate_weights": weights.mean(axis=0).tolist(),
            "train_extreme_rate": p_ref,
            "test_extreme_rate": float(np.mean(test_labels)),
        }
        print(lead_week, results[str(lead_week)])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "ECMWF India proxy subset",
        "warning": "The ECMWF control forecast is used as pseudo-observation; no independent Indian observations are present.",
        "data_dir": str(args.data_dir),
        "train_year": args.train_year,
        "test_year": args.test_year,
        "train_dates": [str(d) for d in train_dates],
        "test_dates": [str(d) for d in test_dates],
        "grid": {"n_lat": len(lat), "n_lon": len(lon), "lat_min": float(lat.min()), "lat_max": float(lat.max()), "lon_min": float(lon.min()), "lon_max": float(lon.max())},
        "results_by_lead": results,
    }
    (args.output_dir / "ecmwf_proxy_results.json").write_text(json.dumps(payload, indent=2))
    lines = [
        "# ECMWF India ABC Proxy Experiment",
        "",
        "> This is an internal consistency test. The ECMWF control forecast is used as a pseudo-observation because no independent Indian observations are present in the local data directory. It is not a verified forecast-skill comparison.",
        "",
        f"- Train: {len(train_dates)} dates from {train_dates[0]} to {train_dates[-1]}",
        f"- Test: {len(test_dates)} dates from {test_dates[0]} to {test_dates[-1]}",
        f"- Grid: {len(lat)} x {len(lon)} at 1.5 degrees",
        "- Ensemble: 10 ECMWF perturbed members; control used as pseudo-observation",
        "",
        "| Lead | Standard ABC ACC | Gated ABC ACC | Standard ABC BSS | Gated ABC BSS |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for lead_week in LEAD_WEEKS:
        row = results[str(lead_week)]
        lines.append(
            f"| {lead_week} | {row['standard_abc']['acc']:.4f} | {row['gated_abc']['acc']:.4f} | "
            f"{row['standard_abc']['brier_skill_score']:+.4f} | {row['gated_abc']['brier_skill_score']:+.4f} |"
        )
    lines.extend([
        "",
        "Weekly ECMWF precipitation was reconstructed from cumulative GRIB precipitation by differencing consecutive lead accumulations, then summing seven daily increments.",
    ])
    (args.output_dir / "ecmwf_proxy_report.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-dates-per-year", type=int, default=30)
    parser.add_argument("--train-year", type=int, default=2018)
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    run(parser.parse_args())


if __name__ == "__main__":
    main()