"""Evaluation metrics for S2S extreme precipitation probabilistic forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score (mean squared error of probability vs 0/1 outcome)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_ref_prob: np.ndarray | float | None = None,
) -> float:
    """Compute Brier Skill Score relative to reference forecast (default: climatology).
    
    BSS = 1 - (BS_model / BS_reference)
    BSS = 0 means no skill over reference; BSS > 0 indicates skill.
    """
    bs_model = brier_score(y_true, y_prob)
    if y_ref_prob is None:
        p_clim = float(np.mean(y_true))
        bs_ref = brier_score(y_true, np.full_like(y_true, p_clim))
    elif np.isscalar(y_ref_prob):
        bs_ref = brier_score(y_true, np.full_like(y_true, float(y_ref_prob)))
    else:
        bs_ref = brier_score(y_true, y_ref_prob)

    if bs_ref <= 0.0:
        return 0.0
    return float(1.0 - (bs_model / bs_ref))


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_ref_prob: np.ndarray | float | None = None,
) -> dict[str, float]:
    """Calculate comprehensive suite of probabilistic and discrimination metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    bs = brier_score(y_true, y_prob)
    bss = brier_skill_score(y_true, y_prob, y_ref_prob=y_ref_prob)

    # Classification discrimination metrics (only valid if at least one positive and one negative)
    n_pos = int(np.sum(y_true))
    n_total = len(y_true)
    if 0 < n_pos < n_total:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float("nan")
    else:
        auc = float("nan")
        pr_auc = float("nan")

    return {
        "brier_score": bs,
        "brier_skill_score": bss,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "mean_prob": float(np.mean(y_prob)),
        "base_rate": float(np.mean(y_true)),
        "n_samples": n_total,
        "n_positive": n_pos,
    }


def reliability_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute calibration curve data (binned forecast prob vs observed frequency)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_predicted = np.full(n_bins, np.nan)
    fraction_positives = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        bin_counts[i] = count
        if count > 0:
            mean_predicted[i] = float(np.mean(y_prob[mask]))
            fraction_positives[i] = float(np.mean(y_true[mask]))

    return {
        "bin_centers": bin_centers,
        "mean_predicted": mean_predicted,
        "fraction_positives": fraction_positives,
        "bin_counts": bin_counts,
    }


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def anomaly_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_clim: np.ndarray | float,
    uncentered: bool = True,
) -> float:
    """Compute spatial or spatiotemporal Anomaly Correlation Coefficient (ACC).
    
    Parameters:
        y_true: Ground truth observed values.
        y_pred: Predicted values.
        y_clim: Climatological baseline values.
        uncentered: If True, compute uncentered anomaly correlation as in
                    Mouatadid et al. (Nature Comms 2023). If False, compute
                    centered (Pearson) correlation of anomalies.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_clim = np.asarray(y_clim, dtype=float)

    # Compute anomalies relative to climatology
    anom_true = y_true - y_clim
    anom_pred = y_pred - y_clim

    if not uncentered:
        anom_true = anom_true - np.mean(anom_true)
        anom_pred = anom_pred - np.mean(anom_pred)

    norm_true = np.sqrt(np.sum(anom_true ** 2))
    norm_pred = np.sqrt(np.sum(anom_pred ** 2))

    if norm_true == 0.0 or norm_pred == 0.0:
        return 0.0

    acc = float(np.sum(anom_true * anom_pred) / (norm_true * norm_pred))
    return float(np.clip(acc, -1.0, 1.0))


