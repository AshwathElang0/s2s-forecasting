"""ECMWF-only daily precipitation post-processing.

This module contains the first research direction for the daily 28-day problem:
a shared temporal residual network that corrects the ECMWF ensemble trajectory
without using observations at prediction time.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class TemporalResidualNet(nn.Module):
        """Shared temporal correction model initialized at the ECMWF baseline."""

        def __init__(self, n_features: int, hidden_dim: int = 32, dropout: float = 0.1):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, hidden_dim),
                nn.GELU(),
            )
            self.correction = nn.Conv1d(hidden_dim, 1, kernel_size=1)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            encoded = self.encoder(features.transpose(1, 2))
            return self.correction(encoded).squeeze(1)


class ECMWFDailyPostprocessor:
    """Post-process 28 daily ECMWF ensemble forecasts without observed inputs.

    Parameters
    ----------
    hidden_dim:
        Width of the shared temporal correction network.
    epochs:
        Maximum optimization epochs.
    smoothness_weight:
        Penalty on changes in the learned log-rainfall correction between adjacent
        forecast days. This encourages temporal coherence without smoothing the
        ECMWF baseline itself.
    extreme_weight:
        Additional weighting given to days whose observed rainfall exceeds
        ``extreme_threshold``. This specifically targets the heavy-rain tail that
        otherwise drifts toward the mean under a pure MAE-like objective.
    extreme_threshold:
        Rainfall threshold in mm/day used to define the extreme tail for the loss.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        smoothness_weight: float = 0.05,
        extreme_weight: float = 0.0,
        extreme_threshold: float = 10.0,
        patience: int = 5,
        random_state: int = 42,
        device: str | None = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for ECMWFDailyPostprocessor.")
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.smoothness_weight = smoothness_weight
        self.extreme_weight = extreme_weight
        self.extreme_threshold = extreme_threshold
        self.patience = patience
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: nn.Module | None = None
        self.history_: dict[str, Any] = {}
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None

    @staticmethod
    def _validate_arrays(features: np.ndarray, targets: np.ndarray | None = None) -> None:
        if features.ndim != 3:
            raise ValueError("features must have shape (samples, days, features)")
        if features.shape[1] != 28:
            raise ValueError("features must contain exactly 28 forecast days")
        if not np.all(np.isfinite(features)):
            raise ValueError("features contain non-finite values")
        if targets is not None:
            if targets.shape != features.shape[:2]:
                raise ValueError("targets must have shape (samples, 28)")
            if not np.all(np.isfinite(targets)) or np.any(targets < 0):
                raise ValueError("targets must be finite and non-negative")

    @staticmethod
    def make_features(member_daily: np.ndarray) -> np.ndarray:
        """Create daily ECMWF-only features from member forecasts.

        Parameters
        ----------
        member_daily: array with shape (samples, members, 28).

        Returns six daily channels: ensemble mean, standard deviation, q10, median,
        q90, and the fraction of members with positive rainfall.
        """
        member_daily = np.asarray(member_daily, dtype=float)
        if member_daily.ndim != 3 or member_daily.shape[2] != 28:
            raise ValueError("member_daily must have shape (samples, members, 28)")
        if not np.all(np.isfinite(member_daily)) or np.any(member_daily < 0):
            raise ValueError("member_daily must be finite and non-negative")
        return np.stack([
            member_daily.mean(axis=1),
            member_daily.std(axis=1),
            np.quantile(member_daily, 0.10, axis=1),
            np.quantile(member_daily, 0.50, axis=1),
            np.quantile(member_daily, 0.90, axis=1),
            np.mean(member_daily > 0, axis=1),
        ], axis=-1)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean_) / self.feature_std_

    def _predict_tensor(self, features: torch.Tensor, baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        correction = self.model_(features)
        prediction_log = torch.log1p(torch.clamp(baseline, min=0.0)) + correction
        prediction = torch.expm1(prediction_log).clamp(min=0.0)
        return prediction, correction

    def _weighted_log_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_log = torch.log1p(targets.clamp(min=0.0))
        prediction_log = torch.log1p(predictions.clamp(min=0.0))
        elementwise_loss = loss_fn(prediction_log, target_log)
        extreme_mask = (targets > self.extreme_threshold).float()
        if self.extreme_weight > 0.0:
            weights = 1.0 + self.extreme_weight * extreme_mask
            weighted_loss = (elementwise_loss * weights).mean()
        else:
            weighted_loss = elementwise_loss.mean()
        if torch.any(extreme_mask > 0):
            extreme_loss = elementwise_loss[extreme_mask > 0].mean()
        else:
            extreme_loss = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        return weighted_loss, extreme_loss

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "ECMWFDailyPostprocessor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        self._validate_arrays(features, targets)
        if validation_data is not None:
            validation_features, validation_targets = map(np.asarray, validation_data)
            self._validate_arrays(validation_features, validation_targets)
        else:
            validation_features = validation_targets = None

        self.feature_mean_ = features.mean(axis=(0, 1), keepdims=True)
        self.feature_std_ = features.std(axis=(0, 1), keepdims=True) + 1e-6
        normalized = self._normalize(features).astype(np.float32)
        baseline = features[:, :, 0].astype(np.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(
            torch.tensor(normalized, dtype=torch.float32),
            torch.tensor(baseline, dtype=torch.float32),
            target_tensor,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = TemporalResidualNet(features.shape[-1], self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss(reduction="none")
        self.history_ = {"epoch": [], "train_loss": [], "train_rmse": [], "train_extreme_loss": []}
        if validation_data is not None:
            self.history_.update({"validation_loss": [], "validation_rmse": [], "validation_extreme_loss": []})

        validation_normalized = validation_baseline = validation_target_tensor = None
        if validation_data is not None:
            validation_normalized = torch.tensor(self._normalize(validation_features), dtype=torch.float32).to(self.device)
            validation_baseline = torch.tensor(validation_features[:, :, 0], dtype=torch.float32).to(self.device)
            validation_target_tensor = torch.tensor(validation_targets, dtype=torch.float32).to(self.device)

        best_state = None
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        self.model_.train()
        for epoch in range(self.epochs):
            self.model_.train()
            for batch_features, batch_baseline, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_baseline = batch_baseline.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimizer.zero_grad()
                predictions, correction = self._predict_tensor(batch_features, batch_baseline)
                data_loss, extreme_loss = self._weighted_log_loss(predictions, batch_targets, loss_fn)
                smoothness_loss = torch.mean((correction[:, 1:] - correction[:, :-1]) ** 2)
                loss = data_loss + self.smoothness_weight * smoothness_loss
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                train_features_tensor = torch.tensor(normalized, dtype=torch.float32).to(self.device)
                train_baseline_tensor = torch.tensor(baseline, dtype=torch.float32).to(self.device)
                train_predictions, _ = self._predict_tensor(train_features_tensor, train_baseline_tensor)
                train_loss, train_extreme_loss = self._weighted_log_loss(train_predictions, target_tensor.to(self.device), loss_fn)
                train_rmse = torch.sqrt(torch.mean((train_predictions - target_tensor.to(self.device)) ** 2))
            self.history_["epoch"].append(epoch + 1)
            self.history_["train_loss"].append(float(train_loss.cpu()))
            self.history_["train_rmse"].append(float(train_rmse.cpu()))
            self.history_["train_extreme_loss"].append(float(train_extreme_loss.cpu()))

            if validation_data is not None:
                with torch.no_grad():
                    validation_predictions, validation_correction = self._predict_tensor(
                        validation_normalized,
                        validation_baseline,
                    )
                    validation_data_loss, validation_extreme_loss = self._weighted_log_loss(
                        validation_predictions,
                        validation_target_tensor,
                        loss_fn,
                    )
                    validation_smoothness = torch.mean(
                        (validation_correction[:, 1:] - validation_correction[:, :-1]) ** 2
                    )
                    validation_loss = validation_data_loss + self.smoothness_weight * validation_smoothness
                    validation_rmse = torch.sqrt(torch.mean((validation_predictions - validation_target_tensor) ** 2))
                self.history_["validation_loss"].append(float(validation_loss.cpu()))
                self.history_["validation_rmse"].append(float(validation_rmse.cpu()))
                self.history_["validation_extreme_loss"].append(float(validation_extreme_loss.cpu()))
                if float(validation_loss) < best_validation_loss:
                    best_validation_loss = float(validation_loss)
                    best_state = copy.deepcopy(self.model_.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if self.patience > 0 and epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.history_["best_epoch"] = (
            int(np.argmin(self.history_["validation_loss"])) + 1
            if validation_data is not None else len(self.history_["epoch"])
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict 28 daily rainfall values from ECMWF-derived features only."""
        features = np.asarray(features, dtype=float)
        self._validate_arrays(features)
        if self.model_ is None or self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("fit must be called before predict")
        self.model_.eval()
        with torch.no_grad():
            feature_tensor = torch.tensor(self._normalize(features), dtype=torch.float32).to(self.device)
            baseline_tensor = torch.tensor(features[:, :, 0], dtype=torch.float32).to(self.device)
            predictions, _ = self._predict_tensor(feature_tensor, baseline_tensor)
        return predictions.cpu().numpy()
