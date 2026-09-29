"""Feature extraction and dataset preparation for S2S extreme precipitation models."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from s2s_data.config import CACHE_DIR, TARGET_LAT, TARGET_LON


class FeatureDataset(NamedTuple):
    X: pd.DataFrame
    y: np.ndarray
    p_raw: np.ndarray
    spread: np.ndarray
    tau: np.ndarray
    init_dates: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    lead_week: int
    y_precip: np.ndarray | None = None


class DatasetLoader:
    """Loads and formats cached per-init-date NetCDF files into tabular ML features."""

    def __init__(self, data_dir: Path | None = None, threshold_path: Path | None = None):
        self.data_dir = data_dir or (CACHE_DIR / "mvp_dataset" / "per_init_date")
        self.threshold_path = threshold_path or (CACHE_DIR / "mvp_dataset" / "climatology_threshold.nc")
        
        if not self.threshold_path.exists():
            raise FileNotFoundError(
                f"Climatology threshold not found at {self.threshold_path}. "
                "Run scripts/build_mvp_dataset.py first."
            )
        
        self.threshold_da = xr.open_dataarray(self.threshold_path).load()
        # Land mask: True where threshold is non-null
        self.land_mask = ~np.isnan(self.threshold_da.values)
        self.n_land_cells = int(self.land_mask.sum())
        
        # Precompute 2D lat/lon mesh for land points
        lon_grid, lat_grid = np.meshgrid(TARGET_LON, TARGET_LAT)
        self.land_lat = lat_grid[self.land_mask]
        self.land_lon = lon_grid[self.land_mask]
        self.land_tau = self.threshold_da.values[self.land_mask]

    def available_init_dates(self) -> list[dt.date]:
        """List all available cached init dates sorted chronologically."""
        files = sorted(self.data_dir.glob("*.nc"))
        dates = []
        for f in files:
            try:
                date = dt.date.fromisoformat(f.stem)
                dates.append(date)
            except ValueError:
                continue
        return sorted(dates)

    def load_date_dataset(self, init_date: dt.date) -> xr.Dataset:
        file_path = self.data_dir / f"{init_date}.nc"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset for {init_date} not found at {file_path}")
        return xr.open_dataset(file_path).load()

    def extract_lead_week(
        self,
        lead_week: int,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
    ) -> FeatureDataset:
        """Extract ML features, raw forecast, spread, and label for a specific lead week across all dates."""
        all_dates = self.available_init_dates()
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        if not all_dates:
            raise ValueError(f"No cached datasets found between {start_date} and {end_date}")

        records = []
        lead_idx = lead_week - 1

        for d in all_dates:
            ds = self.load_date_dataset(d)
            # gefs: (member: 5, lead_week: 4, lat: 120, lon: 300)
            gefs_data = ds["gefs_precip_weekly"].isel(lead_week=lead_idx).values  # (5, 120, 300)
            labels = ds["extreme_label"].isel(lead_week=lead_idx).values          # (120, 300)
            
            # Subselect CONUS land cells
            # Shape: (5, N_land)
            members = gefs_data[:, self.land_mask]
            y_land = labels[self.land_mask].astype(int)
            cpc_data = ds["cpc_precip_weekly"].isel(lead_week=lead_idx).values
            y_precip_land = cpc_data[self.land_mask].astype(float)

            ens_mean = np.mean(members, axis=0)
            ens_std = np.std(members, axis=0, ddof=1) if members.shape[0] > 1 else np.zeros_like(ens_mean)
            ens_min = np.min(members, axis=0)
            ens_max = np.max(members, axis=0)
            
            # Fraction of members exceeding local threshold
            exceed_bool = members > self.land_tau[None, :]
            p_raw = np.mean(exceed_bool, axis=0)
            exceed_count = np.sum(exceed_bool, axis=0)

            # Date / seasonal features
            doy = d.timetuple().tm_yday
            doy_sin = np.sin(2 * np.pi * doy / 365.25)
            doy_cos = np.cos(2 * np.pi * doy / 365.25)
            
            # Climate indices attached as attributes
            oni = float(ds.attrs.get("oni", np.nan))
            mjo_amp = float(ds.attrs.get("mjo_amplitude", np.nan))
            mjo_phase = float(ds.attrs.get("mjo_phase", np.nan))

            n_pts = len(y_land)
            batch_df = pd.DataFrame({
                "ens_mean": ens_mean,
                "ens_std": ens_std,
                "ens_min": ens_min,
                "ens_max": ens_max,
                "f_c00": members[0],
                "f_p01": members[1],
                "f_p02": members[2],
                "f_p03": members[3],
                "f_p04": members[4],
                "raw_exceed_frac": p_raw,
                "raw_exceed_count": exceed_count,
                "threshold": self.land_tau,
                "mean_to_threshold": ens_mean / (self.land_tau + 1e-3),
                "spread_to_threshold": ens_std / (self.land_tau + 1e-3),
                "lat": self.land_lat,
                "lon": self.land_lon,
                "doy_sin": np.full(n_pts, doy_sin),
                "doy_cos": np.full(n_pts, doy_cos),
                "oni": np.full(n_pts, oni if not np.isnan(oni) else 0.0),
                "mjo_amplitude": np.full(n_pts, mjo_amp if not np.isnan(mjo_amp) else 0.0),
                "mjo_phase": np.full(n_pts, mjo_phase if not np.isnan(mjo_phase) else 0.0),
            })
            
            records.append({
                "df": batch_df,
                "y": y_land,
                "y_precip": y_precip_land,
                "p_raw": p_raw,
                "spread": ens_std,
                "init_date": np.full(n_pts, d),
                "lat": self.land_lat,
                "lon": self.land_lon,
                "tau": self.land_tau,
            })

        X_full = pd.concat([r["df"] for r in records], ignore_index=True)
        y_full = np.concatenate([r["y"] for r in records])
        y_precip_full = np.concatenate([r["y_precip"] for r in records])
        p_raw_full = np.concatenate([r["p_raw"] for r in records])
        spread_full = np.concatenate([r["spread"] for r in records])
        tau_full = np.concatenate([r["tau"] for r in records])
        dates_full = np.concatenate([r["init_date"] for r in records])
        lat_full = np.concatenate([r["lat"] for r in records])
        lon_full = np.concatenate([r["lon"] for r in records])

        return FeatureDataset(
            X=X_full,
            y=y_full,
            p_raw=p_raw_full,
            spread=spread_full,
            tau=tau_full,
            init_dates=dates_full,
            lat=lat_full,
            lon=lon_full,
            lead_week=lead_week,
            y_precip=y_precip_full,
        )


def extract_features_for_lead_week(
    lead_week: int,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> FeatureDataset:
    loader = DatasetLoader()
    return loader.extract_lead_week(lead_week, start_date=start_date, end_date=end_date)

