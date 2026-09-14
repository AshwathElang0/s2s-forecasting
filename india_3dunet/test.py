"""Inference + skill scoring on the test split, mirroring the reference
paper's code/model_test.py.

Usage:  python test.py --target rain --model_path runs/rain
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from model.evaluation import skill_scores_by_lead
from model.unet3d import CustomDataset, UNet3D
from preprocess.dataset_loader import apply_stats, inverse_target, list_cached_dates, load_raw, load_stats, split_dates
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
    test_dates = split_dates(all_dates)["test"]

    test_feat, test_tgt_raw, mask_np = load_raw(args.target, test_dates)
    test_feat_std, test_tgt_std = apply_stats(test_feat, test_tgt_raw, stats)

    test_ds = CustomDataset(test_feat_std, test_tgt_std)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=test_feat.shape[1], unit=args.unit).to(device)
    model.load_state_dict(torch.load(model_path / f"model_epoch_{best_epoch + 1}.pt", map_location=device))
    model.eval()

    preds_std = []
    with torch.no_grad():
        for batch in test_loader:
            preds_std.append(model(batch["imgs"].to(device)).cpu().numpy())
    preds_std = np.concatenate(preds_std, axis=0)

    preds = inverse_target(preds_std, stats)[:, 0]  # (N, lead, H, W)
    obs = test_tgt_raw[:, 0]

    lat = padded_lat_array()
    scores = skill_scores_by_lead(preds, obs, lat, mask_np)

    out_prefix = model_path / f"test_{args.target}"
    np.savez(f"{out_prefix}_predictions.npz", preds=preds, obs=obs, mask=mask_np)
    np.savez(f"{out_prefix}_scores.npz", **scores)

    print(f"Test complete. {len(test_dates)} samples.")
    print("Pattern corr by lead day (first 7):", np.round(scores["pattern_corr_mean"][:7], 3))
    print("RMSE by lead day (first 7):", np.round(scores["rmse_mean"][:7], 3))
    print("Saved predictions/scores under", model_path)


if __name__ == "__main__":
    main()
