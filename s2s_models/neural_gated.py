"""PyTorch-based Joint Confidence-Gated Neural Network for S2S extreme precipitation.

Implements the proper learnable weighting factor architecture described in APPROACH.md §5.3:
  p_learned = c_theta(X_conf) * p_phi(X) + (1 - c_theta(X_conf)) * p_clim

Features:
- Differentiable dual-head architecture:
    1. Event predictor branch (p_phi in [0, 1])
    2. Confidence / gating branch (c_theta in [0, 1])
- Joint end-to-end training minimizing direct Brier Score Loss.
- Automatic GPU / CUDA acceleration.
- Scikit-learn estimator compatible wrapper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class JointConfidenceGatedNet(nn.Module):
        """Dual-head neural network with learnable confidence gating."""

        def __init__(
            self,
            in_features: int,
            p_clim: float = 0.012,
            hidden_dim: int = 64,
            dropout_rate: float = 0.1,
        ):
            super().__init__()
            self.p_clim = float(p_clim)

            # Shared feature representation
            self.shared = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

            # Head 1: Event Probability Branch (p_event in [0, 1])
            self.event_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Head 2: Learnable Confidence Gating Branch (c in [0, 1])
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Returns (p_gated, p_event, confidence)."""
            h = self.shared(x)
            # Sigmoid activation ensures strictly valid probabilities [0, 1]
            p_event = torch.sigmoid(self.event_head(h)).squeeze(-1)
            confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)

            # Gated combination: p = c * p_event + (1 - c) * p_clim
            p_gated = confidence * p_event + (1.0 - confidence) * self.p_clim
            return p_gated, p_event, confidence


class NeuralGatedBaseline(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible estimator wrapping the JointConfidenceGatedNet."""

    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 25,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
        random_state: int = 42,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for NeuralGatedBaseline. Install torch first.")
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device_str = device
        self.random_state = random_state

        self.model_: nn.Module | None = None
        self.scaler_: StandardScaler | None = None
        self.p_clim_: float = 0.012
        self.feature_names_: list[str] | None = None

    def _get_device(self) -> torch.device:
        if self.device_str:
            return torch.device(self.device_str)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        val_data: tuple[pd.DataFrame | np.ndarray, np.ndarray] | None = None,
    ) -> NeuralGatedBaseline:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = self._get_device()

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y, dtype=np.float32)
        self.p_clim_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.012

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr).astype(np.float32)

        in_features = X_scaled.shape[1]
        self.model_ = JointConfidenceGatedNet(
            in_features=in_features,
            p_clim=self.p_clim_,
            hidden_dim=self.hidden_dim,
        ).to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_scaled),
            torch.from_numpy(y_arr),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Direct Brier Score loss (mean squared error of probability vs 0/1 label)
        criterion = nn.MSELoss()

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                p_gated, _, _ = self.model_(batch_x)
                loss = criterion(p_gated, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
        return_confidence: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Model is not fitted.")

        device = self._get_device()
        self.model_.eval()

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        X_scaled = self.scaler_.transform(X_arr).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(X_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        all_p = []
        all_c = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                p_gated, _, conf = self.model_(batch_x)
                all_p.append(p_gated.cpu().numpy())
                all_c.append(conf.cpu().numpy())

        p_out = np.concatenate(all_p)
        c_out = np.concatenate(all_c)
        p_out = np.clip(p_out, 0.0, 1.0)

        if return_confidence:
            return p_out, c_out
        return p_out

