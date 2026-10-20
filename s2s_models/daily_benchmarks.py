"""ECMWF-only daily post-processing benchmarks.

The neural benchmark is a compact DeepSets-style model: it encodes each
ensemble member with shared weights, pools members symmetrically, and then
models the pooled 28-day trajectory. It is a paper-inspired adaptation, not
an exact reproduction of any published training recipe.
"""

from __future__ import annotations

import copy

import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LeadClimatology:
    """Training-period mean rainfall for each forecast lead day."""

    def fit(self, targets: np.ndarray) -> "LeadClimatology":
        targets = np.asarray(targets, dtype=float)
        self.mean_ = targets.mean(axis=0)
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        return np.broadcast_to(self.mean_, (n_samples, self.mean_.shape[0])).copy()


class LeadBiasCorrection:
    """Additive lead-dependent correction of the raw ensemble mean."""

    def fit(self, baseline: np.ndarray, targets: np.ndarray) -> "LeadBiasCorrection":
        self.bias_ = np.asarray(targets, dtype=float).mean(axis=0) - np.asarray(baseline, dtype=float).mean(axis=0)
        return self

    def predict(self, baseline: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.asarray(baseline, dtype=float) + self.bias_)


class EmpiricalQuantileMapping:
    """Lead-wise empirical quantile mapping from ECMWF mean to observed rain."""

    def __init__(self, n_quantiles: int = 101):
        self.n_quantiles = n_quantiles

    def fit(self, baseline: np.ndarray, targets: np.ndarray) -> "EmpiricalQuantileMapping":
        baseline = np.asarray(baseline, dtype=float)
        targets = np.asarray(targets, dtype=float)
        self.quantile_levels_ = np.linspace(0.0, 1.0, self.n_quantiles)
        self.baseline_quantiles_ = np.quantile(baseline, self.quantile_levels_, axis=0)
        self.target_quantiles_ = np.quantile(targets, self.quantile_levels_, axis=0)
        return self

    def predict(self, baseline: np.ndarray) -> np.ndarray:
        baseline = np.asarray(baseline, dtype=float)
        output = np.empty_like(baseline)
        for day in range(baseline.shape[1]):
            output[:, day] = np.interp(
                baseline[:, day],
                self.baseline_quantiles_[:, day],
                self.target_quantiles_[:, day],
            )
        return np.maximum(0.0, output)


if HAS_TORCH:
    class DeepSetTrajectoryNet(nn.Module):
        """Permutation-invariant member encoder followed by temporal convolutions."""

        def __init__(self, hidden_dim: int = 16):
            super().__init__()
            self.member_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.temporal = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, 1, kernel_size=1),
            )
            nn.init.zeros_(self.temporal[-1].weight)
            nn.init.zeros_(self.temporal[-1].bias)

        def forward(self, member_daily: torch.Tensor) -> torch.Tensor:
            encoded = self.member_encoder(member_daily.unsqueeze(-1)).mean(dim=1)
            return self.temporal(encoded.transpose(1, 2)).squeeze(1)


class DeepSetDailyPostprocessor:
    """ECMWF-only permutation-invariant neural residual benchmark."""

    def __init__(self, hidden_dim: int = 16, epochs: int = 20, batch_size: int = 256, learning_rate: float = 1e-3, patience: int = 5, device: str | None = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DeepSetDailyPostprocessor.")
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, member_daily: np.ndarray, targets: np.ndarray, validation_data: tuple[np.ndarray, np.ndarray] | None = None) -> "DeepSetDailyPostprocessor":
        torch.manual_seed(42)
        member_daily = np.asarray(member_daily, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        scale = member_daily.std(axis=(0, 1, 2), keepdims=True) + 1e-6
        self.member_scale_ = scale
        normalized = member_daily / scale
        self.model_ = DeepSetTrajectoryNet(self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        train_x = torch.tensor(normalized, device=self.device)
        train_y = torch.tensor(targets, device=self.device)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True)
        validation_x = validation_y = None
        if validation_data is not None:
            validation_x = torch.tensor(np.asarray(validation_data[0], dtype=np.float32) / scale, device=self.device)
            validation_y = torch.tensor(np.asarray(validation_data[1], dtype=np.float32), device=self.device)
        self.history_ = {"epoch": [], "train_loss": [], "validation_loss": []}
        best_state = None
        best_loss = float("inf")
        stale = 0
        for epoch in range(self.epochs):
            self.model_.train()
            for batch_x, batch_y in loader:
                baseline = batch_x.mean(dim=1) * float(scale.squeeze())
                correction = self.model_(batch_x)
                prediction = torch.expm1(torch.log1p(baseline.clamp(min=0.0)) + correction).clamp(min=0.0)
                loss = loss_fn(torch.log1p(prediction), torch.log1p(batch_y))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            self.model_.eval()
            with torch.no_grad():
                baseline = train_x.mean(dim=1) * float(scale.squeeze())
                correction = self.model_(train_x)
                prediction = torch.expm1(torch.log1p(baseline.clamp(min=0.0)) + correction).clamp(min=0.0)
                train_loss = loss_fn(torch.log1p(prediction), torch.log1p(train_y))
                validation_loss = train_loss
                if validation_x is not None:
                    validation_baseline = validation_x.mean(dim=1) * float(scale.squeeze())
                    validation_correction = self.model_(validation_x)
                    validation_prediction = torch.expm1(torch.log1p(validation_baseline.clamp(min=0.0)) + validation_correction).clamp(min=0.0)
                    validation_loss = loss_fn(torch.log1p(validation_prediction), torch.log1p(validation_y))
            self.history_["epoch"].append(epoch + 1)
            self.history_["train_loss"].append(float(train_loss.cpu()))
            self.history_["validation_loss"].append(float(validation_loss.cpu()))
            if float(validation_loss) < best_loss:
                best_loss = float(validation_loss)
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
            if self.patience > 0 and stale >= self.patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, member_daily: np.ndarray) -> np.ndarray:
        member_daily = np.asarray(member_daily, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            features = torch.tensor(member_daily / self.member_scale_, device=self.device)
            baseline = features.mean(dim=1) * float(self.member_scale_.squeeze())
            correction = self.model_(features)
            prediction = torch.expm1(torch.log1p(baseline.clamp(min=0.0)) + correction).clamp(min=0.0)
        return prediction.cpu().numpy()