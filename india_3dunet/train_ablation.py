"""Variable-count ablation, the reference paper's V1/V2/V4/V8 test (plan.md S1.6).

Their finding on the western US: past the target variable plus one or two
circulation fields, **more auxiliary variables gave no significant extra
skill**. We carry 17 channels and have never checked whether they earn their
keep. This runs the same test on our data.

Plain absolute rainfall, single-channel head -- no anomaly reformulation, no
gate. Identical to the original train.py path apart from (a) the channel
subset, (b) a fixed seed, and (c) scoring with model/evaluation_fixed.py
so E_pre reads against its true floor and the shrinkage ratio is reported.

Used by Experiments 5-7. Each set writes to the runs/ folder of the
experiment it belongs to (SET_EXPERIMENT), as <set>_s<seed>.

Usage:
  python train_ablation.py --set rain_only
  python train_ablation.py --set no_obs --seed 1
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader

import config
from model.unet3d import CustomDataset, UNet3D
from model.evaluation_fixed import WEEK_DAYS, WEEK_INDICES, scores_by_lead
from preprocess.dataset_loader import (
    apply_stats,
    compute_stats,
    inverse_target,
    list_cached_dates,
    load_raw,
    load_stats,
    save_stats,
    split_dates,
)
from train import padded_lat_array, run_epoch

# preprocess/build_dataset.py's channel order.
CHANNELS = [
    "tp_mean", "tp_spread",                                    # 0, 1
    "msl", "gh200", "gh1000",                                  # 2, 3, 4
    "q200", "q500", "q700", "q850",                            # 5-8
    "u200", "u850", "v200", "v850",                            # 9-12
    "wtmp",                                                    # 13
    "lag_obs", "doy_sin", "doy_cos",                           # 14, 15, 16
]
BASE = ["tp_mean", "tp_spread", "lag_obs", "doy_sin", "doy_cos"]
MOISTURE = ["q200", "q500", "q700", "q850"]
WIND = ["u200", "u850", "v200", "v850"]
REST = ["msl", "gh200", "gh1000", "wtmp"]

# Nested, so each step isolates one group's contribution.
SETS = {
    "rain_only": BASE,                            # 5 channels, no auxiliary fields
    "humidity": BASE + MOISTURE,                  # 9
    "winds": BASE + MOISTURE + WIND,              # 13
    "all_inputs": BASE + MOISTURE + WIND + REST,  # 17 -- the original configuration
    # 16: all_inputs minus yesterday's observed IMD rain, which would not exist
    # at forecast time. The headline model.
    "no_obs": [c for c in BASE + MOISTURE + WIND + REST if c != "lag_obs"],
    # 14: no_obs minus the day-of-year channels, to measure what the calendar adds
    "ecmwf_only": [c for c in BASE + MOISTURE + WIND + REST
                   if c not in ("lag_obs", "doy_sin", "doy_cos")],
}
# Which experiment's runs/ folder each set writes to. Runs trained before the
# reorganisation used the old names tp/moisture/wind/all/no_lag/no_lag_no_doy
# (stored as "ablation_set" in their stats.json).
SET_EXPERIMENT = {
    "rain_only": "exp05_input_variables", "humidity": "exp05_input_variables",
    "winds": "exp05_input_variables", "all_inputs": "exp05_input_variables",
    "no_obs": "exp06_yesterday_rain", "ecmwf_only": "exp07_calendar_input",
}


def channel_indices(names: list[str]) -> list[int]:
    return [CHANNELS.index(n) for n in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--target", default="rain", choices=config.TARGETS)
    parser.add_argument("--unit", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=11)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out_root", default=None,
                        help="defaults to experiments/<SET_EXPERIMENT[set]>/runs")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    names = SETS[args.set]
    idx = channel_indices(names)
    out_root = Path(args.out_root) if args.out_root else config.EXPERIMENTS_ROOT / SET_EXPERIMENT[args.set] / "runs"
    model_path = out_root / f"{args.set}_s{args.seed}"
    model_path.mkdir(parents=True, exist_ok=True)
    print(f"ablation '{args.set}': {len(idx)} channels -> {names}")

    all_dates = list_cached_dates(args.target)
    splits = split_dates(all_dates)
    print({k: len(v) for k, v in splits.items()})

    def prepare(dates, stats=None):
        """Subset channels *before* standardizing, and standardize in float32
        in place.

        preprocess/dataset_loader.py's apply_stats builds its constants with
        np.array(...) (float64), which upcasts the whole feature block: 9 GiB
        for (724, 17, 32, 48, 64) instead of 4.5, and that OOMs here. Per-channel
        statistics are independent, so computing them on the subset is
        numerically identical to computing on all 17 and slicing afterwards.
        """
        feat, tgt, mask = load_raw(args.target, dates)
        if len(idx) < feat.shape[1]:
            feat = feat[:, idx]
        if stats is None:
            stats = compute_stats(feat, tgt, args.target, mask)

        fm = np.array(stats["feat_mean"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
        fs = np.array(stats["feat_std"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
        feat -= fm
        feat /= fs

        tgt = np.log1p(np.clip(tgt, 0, None)) if args.target == "rain" else tgt
        tgt = (tgt - stats["target_mean"]) / stats["target_std"]
        return feat, np.nan_to_num(tgt, nan=0.0).astype(np.float32), mask, stats

    train_feat, train_tgt, mask_np, stats = prepare(splits["train"])
    stats["ablation_set"] = args.set
    stats["ablation_channels"] = names
    save_stats(stats, model_path / "stats.json")
    val_feat, val_tgt, _, _ = prepare(splits["val"], stats)

    train_loader = DataLoader(CustomDataset(train_feat, train_tgt), batch_size=args.batch_size,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(CustomDataset(val_feat, val_tgt), batch_size=args.batch_size,
                            shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
    lat_t = torch.tensor(padded_lat_array(), dtype=torch.float32, device=device)

    model = UNet3D(in_channels=len(idx), unit=args.unit).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(device.type, enabled=use_amp)

    train_losses, val_losses = [], []
    best_val, best_epoch = float("inf"), 0

    for epoch in range(args.n_epochs):
        tr = run_epoch(model, train_loader, mask, lat_t, device, optimizer, scaler, use_amp)
        va = run_epoch(model, val_loader, mask, lat_t, device, use_amp=use_amp)
        train_losses.append(tr)
        val_losses.append(va)

        if va[0] < best_val:
            best_val, best_epoch = va[0], epoch
            torch.save(model.state_dict(), model_path / "best.pt")
        print(f"epoch [{epoch + 1}/{args.n_epochs}] train={tr[0]:.4f} val={va[0]:.4f}")

        if len(val_losses) >= args.patience + 1:
            recent = [v[0] for v in val_losses[-args.patience:]]
            if min(recent) > val_losses[-args.patience - 1][0] + 0.001:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    with open(model_path / "train_history.pkl", "wb") as f:
        pickle.dump({"train_losses": train_losses, "val_losses": val_losses}, f)
    (model_path / "best_epoch").write_text(str(best_epoch))
    print(f"Best epoch: {best_epoch + 1} (val={best_val:.4f})")

    del train_feat, train_tgt, val_feat, val_tgt

    # --- test ---
    _, test_tgt_raw, _ = load_raw(args.target, splits["test"])
    test_feat_s, test_tgt_s, _, _ = prepare(splits["test"], stats)

    model.load_state_dict(torch.load(model_path / "best.pt", map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in DataLoader(CustomDataset(test_feat_s, test_tgt_s), batch_size=4):
            preds.append(model(batch["imgs"].to(device)).cpu().numpy())
    preds = inverse_target(np.concatenate(preds, axis=0), stats)[:, 0]
    obs = test_tgt_raw[:, 0]

    s = scores_by_lead(preds, obs, padded_lat_array(), mask_np)
    np.savez(model_path / f"test_{args.target}_predictions.npz", preds=preds, obs=obs, mask=mask_np)
    np.savez(model_path / "scores_fixed.npz",
             **{k: v for k, v in s.items() if k != "pattern_corr_per_sample"})

    # Summary column averages weeks 1-4 only. A mean over all 32 leads is
    # misleading for SD ratio: near-zero-variance days send it to huge values.
    print(f"\n{args.set} ({len(idx)} channels), {len(splits['test'])} test samples")
    print(f"{'':18s}" + "".join(f"{'day ' + str(d):>10}" for d in WEEK_DAYS) + f"{'mean(wk1-4)':>13}")
    for k, label in [("pattern_corr", "pattern corr"), ("rmse", "RMSE (mm)"),
                     ("epre", "E_pre"), ("variance_ratio", "SD ratio")]:
        row = "".join(f"{s[k][i]:>10.3f}" for i in WEEK_INDICES)
        print(f"{label:18s}{row}{np.nanmean(s[k][WEEK_INDICES]):>13.3f}")


if __name__ == "__main__":
    main()
