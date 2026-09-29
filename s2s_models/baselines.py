"""Implementation of the four S2S prediction baselines.

Baselines:
1. Raw Ensemble: fraction of members exceeding the local 99th percentile threshold.
2. Climatology: constant unconditional base rate (~0.01).
3. Fixed-Strength Correction (Zhang-style): ML classifier predicting P(y=1|X) with fixed strength.
4. Spread-Gated Correction (Proposed MVP): p_gated = c(spread) * p_fixed + (1 - c(spread)) * p_clim.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


class RawEnsembleBaseline(BaseEstimator, ClassifierMixin):
    """Baseline 1: Empirical probability from raw ensemble member exceedances."""

    def __init__(self):
        self.fitted_ = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> RawEnsembleBaseline:
        self.fitted_ = True
        return self

    def predict_proba(self, p_raw: np.ndarray) -> np.ndarray:
        """Return raw ensemble exceedance probabilities."""
        p_raw = np.clip(np.asarray(p_raw, dtype=float), 0.0, 1.0)
        return p_raw


class ClimatologyBaseline(BaseEstimator, ClassifierMixin):
    """Baseline 2: Unconditional climatological probability of extreme occurrence."""

    def __init__(self, default_rate: float = 0.01):
        self.default_rate = default_rate
        self.rate_ = default_rate

    def fit(self, X: pd.DataFrame | np.ndarray | None, y: np.ndarray) -> ClimatologyBaseline:
        y = np.asarray(y, dtype=float)
        self.rate_ = float(np.mean(y)) if len(y) > 0 else self.default_rate
        return self

    def predict_proba(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, self.rate_, dtype=float)


class FixedCorrectionBaseline(BaseEstimator, ClassifierMixin):
    """Baseline 3: Fixed-strength ML post-processing model (Zhang et al. style)."""

    def __init__(
        self,
        model_type: Literal["hist_gb", "random_forest"] = "hist_gb",
        random_state: int = 42,
        **model_kwargs,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self.model_: BaseEstimator | None = None

    def _init_estimator(self) -> BaseEstimator:
        if self.model_type == "hist_gb":
            kwargs = {
                "max_iter": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_samples_leaf": 50,
                "random_state": self.random_state,
                "class_weight": None,
            }
            kwargs.update(self.model_kwargs)
            return HistGradientBoostingClassifier(**kwargs)
        elif self.model_type == "random_forest":
            kwargs = {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_leaf": 50,
                "n_jobs": -1,
                "random_state": self.random_state,
                "class_weight": None,
            }
            kwargs.update(self.model_kwargs)
            return RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> FixedCorrectionBaseline:
        self.model_ = self._init_estimator()
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        probs = self.model_.predict_proba(X)
        if hasattr(self.model_, "classes_") and 1 in self.model_.classes_:
            idx = list(self.model_.classes_).index(1)
            return probs[:, idx]
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]



class SpreadGatedBaseline(BaseEstimator, ClassifierMixin):
    """Baseline 4: Adaptive confidence-gated correction using ensemble spread.
    
    p_gated = c(spread) * p_fixed + (1 - c(spread)) * p_clim
    where c in [0, 1] is a decreasing function of spread.
    """

    def __init__(
        self,
        base_model: FixedCorrectionBaseline | None = None,
        gate_method: Literal["percentile", "exponential"] = "percentile",
        decay_factor: float = 1.0,
    ):
        self.base_model = base_model or FixedCorrectionBaseline()
        self.gate_method = gate_method
        self.decay_factor = decay_factor
        self.climatology_ = ClimatologyBaseline()
        self.spread_quantiles_: np.ndarray | None = None
        self.spread_median_: float = 1.0

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        spread: np.ndarray,
    ) -> SpreadGatedBaseline:
        spread = np.asarray(spread, dtype=float)
        self.climatology_.fit(X, y)
        self.base_model.fit(X, y)

        valid_spread = spread[~np.isnan(spread)]
        if len(valid_spread) > 0:
            self.spread_quantiles_ = np.percentile(valid_spread, np.linspace(0, 100, 101))
            self.spread_median_ = float(np.median(valid_spread))
        return self

    def compute_confidence(self, spread: np.ndarray) -> np.ndarray:
        """Compute confidence c in [0, 1] as an inverse function of ensemble spread."""
        spread = np.asarray(spread, dtype=float)
        if self.gate_method == "percentile" and self.spread_quantiles_ is not None:
            # High spread -> high percentile -> low confidence
            ranks = np.searchsorted(self.spread_quantiles_, spread) / 100.0
            ranks = np.clip(ranks, 0.0, 1.0)
            c = 1.0 - ranks
        elif self.gate_method == "exponential":
            scale = max(self.spread_median_ * self.decay_factor, 1e-4)
            c = np.exp(-spread / scale)
            c = np.clip(c, 0.0, 1.0)
        else:
            # Fallback linear scaling
            p5, p95 = np.percentile(spread, [5, 95])
            norm = (spread - p5) / max(p95 - p5, 1e-4)
            c = 1.0 - np.clip(norm, 0.0, 1.0)
        return c

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
        spread: np.ndarray,
        return_confidence: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        p_fixed = self.base_model.predict_proba(X)
        p_clim = self.climatology_.predict_proba(len(p_fixed))
        c = self.compute_confidence(spread)

        p_gated = c * p_fixed + (1.0 - c) * p_clim
        p_gated = np.clip(p_gated, 0.0, 1.0)

        if return_confidence:
            return p_gated, c
        return p_gated


class LearnedParametricGateBaseline(BaseEstimator, ClassifierMixin):
    """Learned parametric confidence weighting gate optimized on Brier Score loss.
    
    p_learned = c_theta(spread, tau, ...) * p_fixed + (1 - c_theta) * p_clim
    where c_theta is a calibrated logistic gate whose weights are directly optimized
    to minimize Brier loss on the training validation data.
    """

    def __init__(
        self,
        base_model: FixedCorrectionBaseline | None = None,
        max_iter: int = 100,
    ):
        self.base_model = base_model or FixedCorrectionBaseline()
        self.max_iter = max_iter
        self.climatology_ = ClimatologyBaseline()
        self.weights_: np.ndarray | None = None
        self.spread_mean_: float = 0.0
        self.spread_std_: float = 1.0

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        spread: np.ndarray,
    ) -> LearnedParametricGateBaseline:
        from scipy.optimize import minimize

        y = np.asarray(y, dtype=float)
        spread = np.asarray(spread, dtype=float)

        self.climatology_.fit(X, y)
        self.base_model.fit(X, y)

        p_fixed = self.base_model.predict_proba(X)
        p_clim = self.climatology_.predict_proba(len(y))

        # Standardize spread for numerical stability
        self.spread_mean_ = float(np.nanmean(spread))
        self.spread_std_ = float(np.nanstd(spread)) or 1.0
        z_spread = (spread - self.spread_mean_) / self.spread_std_

        # Objective function: Brier Score as a function of [w_spread, bias]
        def loss_fn(params: np.ndarray) -> float:
            w, b = params[0], params[1]
            logit = w * z_spread + b
            c = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
            p_pred = c * p_fixed + (1.0 - c) * p_clim
            return float(np.mean((p_pred - y) ** 2))

        # Initialize with negative weight on spread (higher spread -> lower confidence)
        init_params = np.array([-1.0, 0.0])
        res = minimize(loss_fn, init_params, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.weights_ = res.x
        return self

    def compute_confidence(self, spread: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model not fitted.")
        spread = np.asarray(spread, dtype=float)
        z_spread = (spread - self.spread_mean_) / self.spread_std_
        w, b = self.weights_[0], self.weights_[1]
        logit = w * z_spread + b
        c = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
        return np.clip(c, 0.0, 1.0)

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
        spread: np.ndarray,
        return_confidence: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        p_fixed = self.base_model.predict_proba(X)
        p_clim = self.climatology_.predict_proba(len(p_fixed))
        c = self.compute_confidence(spread)

        p_gated = c * p_fixed + (1.0 - c) * p_clim
        p_gated = np.clip(p_gated, 0.0, 1.0)

        if return_confidence:
            return p_gated, c
        return p_gated


