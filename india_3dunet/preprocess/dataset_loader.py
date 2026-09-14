"""Load cached samples (scripts/build_all_samples.py's output), split them
into fixed calendar blocks (config.TRAIN_YEARS/VAL_YEARS/TEST_YEARS), and
standardize -- fold-internal, i.e. every mean/std comes from the training
block only and is then applied unchanged to val/test, per
literature-review.md S7.1 ("computing them once ... and evaluating on a
held-out year is a leak").

Rain gets `log1p` before standardizing (literature-review.md S6.2); tmax/
tmin are standardized directly. Feature channels are standardized
per-channel over the valid (non-padded, IMD-masked) region only, so the
padding border and outside-India cells don't drag the statistics around.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from preprocess.build_dataset import padded_native_slice


def _date_from_filename(path: Path) -> datetime:
    return datetime.strptime(path.stem, "%Y_%m_%d")


def list_cached_dates(target: str) -> list[datetime]:
    d = config.CACHE_ROOT / "samples" / target
    return sorted(_date_from_filename(p) for p in d.glob("*.npz"))


def split_dates(dates: list[datetime]) -> dict[str, list[datetime]]:
    return {
        "train": [d for d in dates if d.year in config.TRAIN_YEARS],
        "val": [d for d in dates if d.year in config.VAL_YEARS],
        "test": [d for d in dates if d.year in config.TEST_YEARS],
    }


def load_raw(target: str, dates: list[datetime]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (features (N,C,lead,H,W), target (N,1,lead,H,W), mask (H,W))."""
    d = config.CACHE_ROOT / "samples" / target
    feats, tgts, mask = [], [], None
    for date in dates:
        z = np.load(d / f"{date:%Y_%m_%d}.npz")
        feats.append(z["features"])
        tgts.append(z["target"])
        if mask is None:
            mask = z["mask"]
    return np.stack(feats), np.stack(tgts), mask


def _target_transform(x: np.ndarray, target_var: str) -> np.ndarray:
    if target_var == "rain":
        return np.log1p(np.clip(x, 0, None))
    return x


def _target_inverse_transform(x: np.ndarray, target_var: str) -> np.ndarray:
    if target_var == "rain":
        return np.expm1(x)
    return x


def compute_stats(features: np.ndarray, target: np.ndarray, target_var: str, mask: np.ndarray) -> dict:
    """Per-channel feature stats + a single target mean/std, computed only
    over valid (masked-in) grid cells."""
    lat_sl, lon_sl = padded_native_slice()
    valid = np.zeros_like(mask)
    valid[lat_sl, lon_sl] = mask[lat_sl, lon_sl]

    n_channels = features.shape[1]
    feat_mean = np.zeros(n_channels)
    feat_std = np.ones(n_channels)
    for c in range(n_channels):
        vals = features[:, c][:, :, valid]
        feat_mean[c] = vals.mean()
        feat_std[c] = vals.std() or 1.0

    tgt_transformed = _target_transform(target, target_var)
    tgt_vals = tgt_transformed[:, 0][:, :, valid]
    tgt_vals = tgt_vals[np.isfinite(tgt_vals)]
    tgt_mean = float(tgt_vals.mean())
    tgt_std = float(tgt_vals.std() or 1.0)

    return {
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "target_mean": tgt_mean,
        "target_std": tgt_std,
        "target_var": target_var,
    }


def save_stats(stats: dict, path: Path) -> None:
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: Path) -> dict:
    return json.loads(path.read_text())


def apply_stats(features: np.ndarray, target: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    feat_mean = np.array(stats["feat_mean"]).reshape(1, -1, 1, 1, 1)
    feat_std = np.array(stats["feat_std"]).reshape(1, -1, 1, 1, 1)
    features_std = (features - feat_mean) / feat_std

    tgt_transformed = _target_transform(target, stats["target_var"])
    target_std = (tgt_transformed - stats["target_mean"]) / stats["target_std"]
    target_std = np.nan_to_num(target_std, nan=0.0)
    return features_std.astype(np.float32), target_std.astype(np.float32)


def inverse_target(target_std: np.ndarray, stats: dict) -> np.ndarray:
    tgt_transformed = target_std * stats["target_std"] + stats["target_mean"]
    return _target_inverse_transform(tgt_transformed, stats["target_var"])
