"""Inference + skill scoring for the gated model.

Scores in mm using the unmodified model/evaluation.py, so the numbers line up
with the existing runs. Also dumps the learned gate -- the "where and when do
we trust ECMWF" diagnostic (README.md S3).

Usage:  python experiments/exp03_departure_from_normal/test_departure.py --target rain --model_path experiments/exp03_departure_from_normal/runs/gate_v1_first_try --unit 64
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from preprocess.anomaly import (
    anomaly_fields,
    inverse_anom,
    model_features,
    scale_target_and_anchor,
    standardize_features,
    substitute_channels,
)
from preprocess.climatology import ensure_climatology, usable_dates
from model.gated_unet import GatedDataset, GatedUNet3D, PlainAnomalyUNet3D
from model.evaluation import skill_scores_by_lead
from preprocess.dataset_loader import list_cached_dates, load_raw, load_stats, split_dates
from train import padded_lat_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=config.TARGETS, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--unit", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    best_epoch = int((model_path / "best_epoch").read_text().strip())
    stats = load_stats(model_path / "stats.json")

    all_dates = list_cached_dates(args.target)
    splits = split_dates(all_dates)

    obs_clim, model_clim = ensure_climatology(args.target, splits["train"], all_dates)
    test_dates = usable_dates(splits["test"], model_clim)

    test_feat, test_tgt_raw, mask_np = load_raw(args.target, test_dates)
    te_tgt_a, te_raw_a, te_lag_a = anomaly_fields(
        test_feat, test_tgt_raw, args.target, test_dates, obs_clim, model_clim
    )
    substitute_channels(test_feat, te_raw_a, te_lag_a, stats["sigma_mdl"], stats["sigma_obs"], mask_np)
    standardize_features(test_feat, stats)
    test_tgt_std, test_anchor = scale_target_and_anchor(te_tgt_a, te_raw_a, stats)

    test_x = model_features(test_feat, stats)
    loader = DataLoader(
        GatedDataset(test_x, test_tgt_std, test_anchor),
        batch_size=args.batch_size, shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plain = bool(stats.get("plain_head"))
    model_cls = PlainAnomalyUNet3D if plain else GatedUNet3D
    model = model_cls(in_channels=test_x.shape[1], unit=args.unit).to(device)
    model.load_state_dict(torch.load(model_path / f"model_epoch_{best_epoch + 1}.pt", map_location=device))
    model.eval()

    preds_std, gates = [], []
    with torch.no_grad():
        for batch in loader:
            pred, gate = model(batch["imgs"].to(device), batch["anchor"].to(device), return_gate=True)
            preds_std.append(pred.cpu().numpy())
            if gate is not None:
                gates.append(gate.cpu().numpy())

    preds_std = np.concatenate(preds_std, axis=0)[:, 0]

    preds = inverse_anom(preds_std, test_dates, obs_clim, stats)
    obs = test_tgt_raw[:, 0]

    lat = padded_lat_array()
    scores = skill_scores_by_lead(preds, obs, lat, mask_np)

    out_prefix = model_path / f"test_{args.target}"
    np.savez(f"{out_prefix}_predictions.npz", preds=preds, obs=obs, mask=mask_np)
    np.savez(f"{out_prefix}_scores.npz", **scores)

    print(f"Test complete. {len(test_dates)} samples.")
    print("Pattern corr by lead day (first 7):", np.round(scores["pattern_corr_mean"][:7], 3))
    print("RMSE by lead day (first 7):", np.round(scores["rmse_mean"][:7], 3))

    if gates:
        gate = np.concatenate(gates, axis=0)[:, 0]
        np.savez(f"{out_prefix}_gate.npz", gate=gate.astype(np.float32))
        valid = mask_np.astype(bool)
        print("Mean gate by lead day:", np.round(gate[:, :, valid].mean(axis=(0, 2)), 3))
    print("Saved under", model_path)


if __name__ == "__main__":
    main()
