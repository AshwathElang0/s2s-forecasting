"""Adaptive Bias Correction (ABC) and Neural-Gated ABC (ABC 2.0).

References:
  - Mouatadid et al. (Nature Communications, 2023):
    "Adaptive bias correction for improved subseasonal forecasting"
  - Microsoft Research Subseasonal Toolkit (github.com/microsoft/subseasonal_toolkit)

Models:
  1. DynamicalPlusPlus: Adaptively debiased dynamical model (with optional spread modulation).
  2. ClimatologyPlusPlus: Adaptive observational climatology.
  3. PersistencePlusPlus: Local OLS regression combining lagged observations, climatology, and dynamical forecast.
  4. StandardABC: Nature paper's equal-weight ensemble (1/3, 1/3, 1/3).
  5. OpportunisticABC: Nature paper's rule-based teleconnection window-of-opportunity selector.
  6. GatedABC: Proposed enhancement using neural Mixture-of-Experts with state-dependent gating
     conditioned on ensemble spread, MJO, ONI, seasonality, and spatial context.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DynamicalPlusPlus:
    """Dynamical++: Adaptive bias correction of numerical weather predictions.
    
    Fits gridpoint-level and seasonal bias corrections to the dynamical ensemble mean.
    Optionally includes spread modulation to attenuate forecasts where ensemble members disagree.
    """

    def __init__(self, modulate_by_spread: bool = False, gamma: float = 0.5):
        self.modulate_by_spread = modulate_by_spread
        self.gamma = gamma
        self.bias_by_grid_: np.ndarray | None = None
        self.scale_by_grid_: np.ndarray | None = None

    def fit(
        self,
        ens_mean: np.ndarray,
        y_true: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> DynamicalPlusPlus:
        """Compute mean bias and scaling per unique (lat, lon) grid cell."""
        coords = np.stack([lat, lon], axis=1)
        unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        n_cells = len(unique_coords)

        bias_by_cell = np.zeros(n_cells, dtype=float)
        scale_by_cell = np.ones(n_cells, dtype=float)

        for cell_idx in range(n_cells):
            mask = (inverse_indices == cell_idx)
            f_cell = ens_mean[mask]
            y_cell = y_true[mask]
            if len(y_cell) > 0:
                mean_f = np.mean(f_cell)
                mean_y = np.mean(y_cell)
                bias_by_cell[cell_idx] = mean_f - mean_y
                # Bounded multiplicative scale factor for variance matching
                if mean_f > 1.0 and mean_y > 1.0:
                    scale_by_cell[cell_idx] = np.clip(mean_y / mean_f, 0.2, 5.0)

        self.unique_coords_ = unique_coords
        self.bias_by_cell_ = bias_by_cell
        self.scale_by_cell_ = scale_by_cell
        return self

    def predict(
        self,
        ens_mean: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        ens_std: np.ndarray | None = None,
        clim_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply bias correction, optionally modulated by ensemble spread."""
        coords = np.stack([lat, lon], axis=1)
        # Match coordinates to precomputed cells
        # Since grid cells are identical between train and test, find matching index
        # Use fast coordinate hashing
        coord_map = {tuple(c): idx for idx, c in enumerate(self.unique_coords_)}
        indices = np.array([coord_map.get((la, lo), -1) for la, lo in coords])

        valid = indices >= 0
        corrected = np.copy(ens_mean).astype(float)
        
        # Additive bias correction with non-negativity constraint
        bias = np.where(valid, self.bias_by_cell_[np.maximum(0, indices)], 0.0)
        corrected = np.maximum(0.0, corrected - bias)

        # Optional spread modulation: shrink toward climatology when spread is high
        if self.modulate_by_spread and ens_std is not None and clim_ref is not None:
            # shrinkage factor in [0, 1]: low spread -> factor ~ 1, high spread -> factor < 1
            rel_spread = ens_std / (ens_mean + 1.0)
            shrinkage = 1.0 / (1.0 + self.gamma * rel_spread)
            corrected = clim_ref + shrinkage * (corrected - clim_ref)
            corrected = np.maximum(0.0, corrected)

        return corrected


class ClimatologyPlusPlus:
    """Climatology++: Adaptive observational climatology per grid cell and season."""

    def __init__(self):
        self.clim_mean_: dict[tuple[float, float, int], float] = {}
        self.global_clim_by_cell_: np.ndarray | None = None
        self.unique_coords_: np.ndarray | None = None

    def fit(
        self,
        y_true: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        doy: np.ndarray,
    ) -> ClimatologyPlusPlus:
        """Compute climatological mean precipitation per (grid cell, month)."""
        coords = np.stack([lat, lon], axis=1)
        unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        self.unique_coords_ = unique_coords
        n_cells = len(unique_coords)

        self.global_clim_by_cell_ = np.zeros(n_cells, dtype=float)
        # Also compute monthly/seasonal means
        months = ((doy - 1) // 30.5).astype(int) + 1  # 1 to 12

        for cell_idx in range(n_cells):
            mask = (inverse_indices == cell_idx)
            y_cell = y_true[mask]
            m_cell = months[mask]
            if len(y_cell) > 0:
                self.global_clim_by_cell_[cell_idx] = float(np.mean(y_cell))
                for m in np.unique(m_cell):
                    self.clim_mean_[(cell_idx, int(m))] = float(np.mean(y_cell[m_cell == m]))

        self.coord_map_ = {tuple(c): idx for idx, c in enumerate(self.unique_coords_)}
        return self

    def predict(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        doy: np.ndarray,
    ) -> np.ndarray:
        """Output climatological expectation for each sample."""
        coords = np.stack([lat, lon], axis=1)
        months = ((doy - 1) // 30.5).astype(int) + 1
        preds = np.zeros(len(lat), dtype=float)

        for i in range(len(lat)):
            c_tuple = (float(lat[i]), float(lon[i]))
            cell_idx = self.coord_map_.get(c_tuple, -1)
            if cell_idx >= 0:
                m = int(months[i])
                preds[i] = self.clim_mean_.get((cell_idx, m), self.global_clim_by_cell_[cell_idx])
            else:
                preds[i] = 10.0  # safe CONUS default
        return np.maximum(0.0, preds)


class PersistencePlusPlus:
    """Persistence++: Regresses future precipitation on lagged trends, climatology, and NWP."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha, positive=False)

    def fit(
        self,
        y_true: np.ndarray,
        ens_mean: np.ndarray,
        clim_pred: np.ndarray,
        lagged_obs: np.ndarray,
    ) -> PersistencePlusPlus:
        """Fit linear blend: y ~ w1 * lagged_obs + w2 * clim + w3 * ens_mean."""
        X_blend = np.column_stack([lagged_obs, clim_pred, ens_mean])
        self.model.fit(X_blend, y_true)
        return self

    def predict(
        self,
        ens_mean: np.ndarray,
        clim_pred: np.ndarray,
        lagged_obs: np.ndarray,
    ) -> np.ndarray:
        """Predict precipitation using fitted persistence regression."""
        X_blend = np.column_stack([lagged_obs, clim_pred, ens_mean])
        preds = self.model.predict(X_blend)
        return np.maximum(0.0, preds)


class StandardABC(BaseEstimator, RegressorMixin):
    """Standard Adaptive Bias Correction (Nature Communications 2023).
    
    Combines Dynamical++, Climatology++, and Persistence++ with equal weights:
      y_pred = (1/3) * Dyn++ + (1/3) * Clim++ + (1/3) * Pers++
    (For lead week 1: 0.5 * Dyn++ + 0.5 * Pers++).
    """

    def __init__(self, lead_week: int = 1):
        self.lead_week = lead_week
        self.dyn_model = DynamicalPlusPlus(modulate_by_spread=False)
        self.clim_model = ClimatologyPlusPlus()
        self.pers_model = PersistencePlusPlus()

    def fit(
        self,
        X: pd.DataFrame,
        y_precip: np.ndarray,
    ) -> StandardABC:
        ens_mean = X["ens_mean"].values
        lat = X["lat"].values
        lon = X["lon"].values
        # Reconstruct day of year approximately from seasonal harmonics
        doy_sin = X["doy_sin"].values
        doy_cos = X["doy_cos"].values
        doy = (np.arctan2(doy_sin, doy_cos) * 365.25 / (2 * np.pi)) % 365.25 + 1.0

        # Fit Dynamical++
        self.dyn_model.fit(ens_mean, y_precip, lat, lon)

        # Fit Climatology++
        self.clim_model.fit(y_precip, lat, lon, doy)
        clim_preds = self.clim_model.predict(lat, lon, doy)

        # Approximate lagged observation using persistence proxy
        # If mean_to_threshold is provided, proxy lagged obs with smoothed local baseline
        lagged_obs = np.roll(y_precip, shift=1)
        lagged_obs[0] = clim_preds[0]

        # Fit Persistence++
        dyn_preds = self.dyn_model.predict(ens_mean, lat, lon)
        self.pers_model.fit(y_precip, dyn_preds, clim_preds, lagged_obs)
        return self

    def predict_components(
        self,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the individual (Dyn++, Clim++, Pers++) component predictions."""
        ens_mean = X["ens_mean"].values
        lat = X["lat"].values
        lon = X["lon"].values
        doy_sin = X["doy_sin"].values
        doy_cos = X["doy_cos"].values
        doy = (np.arctan2(doy_sin, doy_cos) * 365.25 / (2 * np.pi)) % 365.25 + 1.0

        y_dyn = self.dyn_model.predict(ens_mean, lat, lon)
        y_clim = self.clim_model.predict(lat, lon, doy)
        lagged_obs = y_clim  # best available unbiased lag proxy at inference time
        y_pers = self.pers_model.predict(y_dyn, y_clim, lagged_obs)
        return y_dyn, y_clim, y_pers

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Equal-weighted combination."""
        y_dyn, y_clim, y_pers = self.predict_components(X)
        if self.lead_week == 1:
            return 0.5 * y_dyn + 0.5 * y_pers
        else:
            return (1.0 / 3.0) * y_dyn + (1.0 / 3.0) * y_clim + (1.0 / 3.0) * y_pers

    def predict_extreme_proba(self, X: pd.DataFrame, tau: np.ndarray) -> np.ndarray:
        """Derive extreme event occurrence probability from predicted precipitation."""
        y_pred = self.predict(X)
        # Scaled sigmoid mapping centered at local threshold tau
        scale = np.maximum(tau * 0.25, 5.0)
        z = (y_pred - tau) / scale
        return 1.0 / (1.0 + np.exp(-z))


class OpportunisticABC(StandardABC):
    """Opportunistic ABC (Nature Communications 2023).
    
    Selectively deploys ABC during favorable teleconnection regimes:
      - Active MJO: amplitude >= 1.0 and phase in {2, 4, 5, 8}
      - Active ENSO: |ONI| >= 0.5
    Otherwise, falls back to standard debiased Dynamical++ forecast.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_dyn, y_clim, y_pers = self.predict_components(X)
        y_abc = (
            0.5 * y_dyn + 0.5 * y_pers
            if self.lead_week == 1
            else (1.0 / 3.0) * y_dyn + (1.0 / 3.0) * y_clim + (1.0 / 3.0) * y_pers
        )

        mjo_amp = X["mjo_amplitude"].values
        mjo_phase = X["mjo_phase"].values
        oni = X["oni"].values

        # Teleconnection condition
        favorable_mjo = (mjo_amp >= 1.0) & np.isin(np.round(mjo_phase).astype(int), [2, 4, 5, 8])
        favorable_oni = np.abs(oni) >= 0.5
        deploy_abc = favorable_mjo | favorable_oni

        # Deploy ABC where teleconnection conditions hold, else fall back to debiased dynamical
        final_preds = np.where(deploy_abc, y_abc, y_dyn)
        return final_preds


if HAS_TORCH:
    class NeuralGatingNet(nn.Module):
        """Residual Softmax Mixture-of-Experts Gating Network for ABC components.
        
        Anchored around the physical ABC baseline (1/3, 1/3, 1/3) at initialization.
        Learns state-dependent adaptive adjustments driven by ensemble spread and teleconnections.
        """

        def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.1, lead_week: int = 1):
            super().__init__()
            if lead_week == 1:
                init_logits = torch.tensor([1.0, -3.0, 1.0])  # week 1: ~ [0.49, 0.02, 0.49]
            else:
                init_logits = torch.tensor([0.0, 0.0, 0.0])   # weeks 2-4: equal [1/3, 1/3, 1/3]
            self.register_buffer("base_logits", init_logits)

            self.mlp = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
            )
            # Initialize final layer near zero so model starts exactly as Standard ABC
            nn.init.normal_(self.mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            delta_logits = 0.5 * self.mlp(x)
            logits = self.base_logits + delta_logits
            return torch.softmax(logits, dim=-1)


class GatedABC(BaseEstimator, RegressorMixin):
    """Gated-ABC (Proposed ABC 2.0).
    
    Dynamically weights Dynamical++, Climatology++, and Persistence++ using a neural
    gating network conditioned on:
      1. Internal ensemble spread (std, spread-to-threshold ratio)
      2. Teleconnections (MJO amplitude, MJO phase harmonics, ONI)
      3. Seasonality (day-of-year harmonics)
      4. Spatial coordinates and local climatology threshold
    """

    def __init__(
        self,
        lead_week: int = 1,
        hidden_dim: int = 64,
        epochs: int = 25,
        batch_size: int = 1024,
        lr: float = 1e-3,
        lambda_extreme: float = 0.5,
        device: str | None = None,
        random_state: int = 42,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GatedABC.")
        self.lead_week = lead_week
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_extreme = lambda_extreme
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Underlying ABC components with spread-modulated Dynamical++
        self.dyn_model = DynamicalPlusPlus(modulate_by_spread=True, gamma=0.3)
        self.clim_model = ClimatologyPlusPlus()
        self.pers_model = PersistencePlusPlus()
        self.gating_net_: nn.Module | None = None
        self.feature_cols_: list[str] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _prepare_gating_features(self, X: pd.DataFrame) -> np.ndarray:
        """Extract and normalize gating feature representations."""
        mjo_phase = X["mjo_phase"].values
        mjo_phase_rad = 2 * np.pi * mjo_phase / 8.0
        
        feats = np.column_stack([
            X["ens_std"].values,
            X["spread_to_threshold"].values,
            X["mean_to_threshold"].values,
            X["oni"].values,
            X["mjo_amplitude"].values,
            np.sin(mjo_phase_rad),
            np.cos(mjo_phase_rad),
            X["doy_sin"].values,
            X["doy_cos"].values,
            X["lat"].values / 50.0,
            X["lon"].values / 360.0,
            X["threshold"].values / 100.0,
        ])
        return feats

    def fit(self, X: pd.DataFrame, y_precip: np.ndarray, y_extreme: np.ndarray) -> GatedABC:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        ens_mean = X["ens_mean"].values
        ens_std = X["ens_std"].values
        lat = X["lat"].values
        lon = X["lon"].values
        doy_sin = X["doy_sin"].values
        doy_cos = X["doy_cos"].values
        doy = (np.arctan2(doy_sin, doy_cos) * 365.25 / (2 * np.pi)) % 365.25 + 1.0

        # Fit components
        self.dyn_model.fit(ens_mean, y_precip, lat, lon)
        self.clim_model.fit(y_precip, lat, lon, doy)
        clim_preds = self.clim_model.predict(lat, lon, doy)
        dyn_preds = self.dyn_model.predict(ens_mean, lat, lon, ens_std=ens_std, clim_ref=clim_preds)

        lagged_obs = np.roll(y_precip, shift=1)
        lagged_obs[0] = clim_preds[0]
        self.pers_model.fit(y_precip, dyn_preds, clim_preds, lagged_obs)
        pers_preds = self.pers_model.predict(dyn_preds, clim_preds, lagged_obs)

        # Prepare gating dataset
        G_feats = self._prepare_gating_features(X)
        self.mean_ = np.mean(G_feats, axis=0)
        self.std_ = np.std(G_feats, axis=0) + 1e-6
        G_norm = (G_feats - self.mean_) / self.std_

        # Targets and component tensors
        components_np = np.column_stack([dyn_preds, clim_preds, pers_preds])
        tau_np = X["threshold"].values

        x_t = torch.tensor(G_norm, dtype=torch.float32)
        comp_t = torch.tensor(components_np, dtype=torch.float32)
        y_precip_t = torch.tensor(y_precip, dtype=torch.float32)
        y_ext_t = torch.tensor(y_extreme, dtype=torch.float32)
        tau_t = torch.tensor(tau_np, dtype=torch.float32)

        dataset = TensorDataset(x_t, comp_t, y_precip_t, y_ext_t, tau_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.gating_net_ = NeuralGatingNet(
            in_features=G_norm.shape[1],
            hidden_dim=self.hidden_dim,
            lead_week=self.lead_week,
        ).to(self.device)
        optimizer = optim.AdamW(self.gating_net_.parameters(), lr=self.lr, weight_decay=1e-3)
        mse_loss_fn = nn.MSELoss()
        brier_loss_fn = nn.MSELoss()

        self.gating_net_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_comp, batch_y, batch_ext, batch_tau in dataloader:
                batch_x = batch_x.to(self.device)
                batch_comp = batch_comp.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_ext = batch_ext.to(self.device)
                batch_tau = batch_tau.to(self.device)

                optimizer.zero_grad()
                weights = self.gating_net_(batch_x)  # (batch, 3)
                # Weighted blend of components
                y_pred_batch = torch.sum(weights * batch_comp, dim=-1)

                # Continuous MSE loss matching continuous variance
                l_cont = mse_loss_fn(y_pred_batch, batch_y)

                # Extreme tail probability calibration loss
                scale = torch.clamp(batch_tau * 0.25, min=5.0)
                p_ext_pred = torch.sigmoid((y_pred_batch - batch_tau) / scale)
                l_ext = brier_loss_fn(p_ext_pred, batch_ext)

                total_loss = l_cont + self.lambda_extreme * l_ext
                total_loss.backward()
                optimizer.step()

        return self

    def predict_with_weights(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predicted_precipitation, gating_weights)."""
        ens_mean = X["ens_mean"].values
        ens_std = X["ens_std"].values
        lat = X["lat"].values
        lon = X["lon"].values
        doy_sin = X["doy_sin"].values
        doy_cos = X["doy_cos"].values
        doy = (np.arctan2(doy_sin, doy_cos) * 365.25 / (2 * np.pi)) % 365.25 + 1.0

        clim_preds = self.clim_model.predict(lat, lon, doy)
        dyn_preds = self.dyn_model.predict(ens_mean, lat, lon, ens_std=ens_std, clim_ref=clim_preds)
        lagged_obs = clim_preds
        pers_preds = self.pers_model.predict(dyn_preds, clim_preds, lagged_obs)

        components_np = np.column_stack([dyn_preds, clim_preds, pers_preds])

        G_feats = self._prepare_gating_features(X)
        G_norm = (G_feats - self.mean_) / self.std_

        self.gating_net_.eval()
        with torch.no_grad():
            x_t = torch.tensor(G_norm, dtype=torch.float32).to(self.device)
            weights = self.gating_net_(x_t).cpu().numpy()  # (N, 3)

        preds = np.sum(weights * components_np, axis=-1)
        return np.maximum(0.0, preds), weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds, _ = self.predict_with_weights(X)
        return preds

    def predict_extreme_proba(self, X: pd.DataFrame, tau: np.ndarray) -> np.ndarray:
        y_pred = self.predict(X)
        scale = np.maximum(tau * 0.25, 5.0)
        z = (y_pred - tau) / scale
        return 1.0 / (1.0 + np.exp(-z))

