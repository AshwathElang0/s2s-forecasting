"""Train the gated residual 3D U-Net (experiments/exp03_departure_from_normal/DESIGN.md).

Same optimiser, schedule, batch size, early stopping and loss as the original
train.py -- masked MSE + masked pattern correlation, unchanged -- so an A/B
against exp01_training_size/runs/n724 isolates the architecture change.

Usage:
  python experiments/exp03_departure_from_normal/train_departure.py --target rain --model_path experiments/exp03_departure_from_normal/runs/gate_v1_first_try \
      --unit 64 --batch_size 11 --n_epochs 100 --patience 10
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import config
from preprocess.anomaly import (
    anomaly_fields,
    compute_feature_stats,
    compute_sigmas,
    model_features,
    scale_target_and_anchor,
    standardize_features,
    substitute_channels,
)
from preprocess.climatology import ensure_climatology, usable_dates
from model.gated_unet import GatedDataset, GatedUNet3D, PlainAnomalyUNet3D
from model.unet3d import masked_mse_loss, masked_pattern_correlation_loss
from preprocess.dataset_loader import list_cached_dates, load_raw, save_stats, split_dates
from train import padded_lat_array


def run_epoch(model, loader, mask, lat, device, optimizer=None, scaler=None, use_amp=False):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = total_mse = total_corr = 0.0
    n_batches = 0
    for batch in loader:
        imgs = batch["imgs"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        anchor = batch["anchor"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs, anchor)
        preds = preds.float()

        mse = masked_mse_loss(preds, labels, mask)
        corr = masked_pattern_correlation_loss(preds, labels, lat, mask)
        loss = mse + corr

        if train_mode:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_corr += corr.item()
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches, total_corr / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=config.TARGETS, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--unit", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=11)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gate_bias_init", type=float, default=0.0,
                        help="sigmoid(bias) is the initial trust gate; 0.0 -> 0.5. See model/gated_unet.py")
    parser.add_argument("--rebuild_clim", action="store_true")
    parser.add_argument("--drop_anchor", action="store_true", default=True,
                        help="withhold tp_mean from the input stack so the gate is the only "
                             "path for ECMWF precipitation (see preprocess/anomaly.py)")
    parser.add_argument("--keep_anchor", dest="drop_anchor", action="store_false")
    parser.add_argument("--plain_head", action="store_true",
                        help="single-channel head predicting the anomaly directly, no gate")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    all_dates = list_cached_dates(args.target)
    splits = split_dates(all_dates)
    print("calendar split:", {k: len(v) for k, v in splits.items()})

    obs_clim, model_clim = ensure_climatology(
        args.target, splits["train"], all_dates, rebuild=args.rebuild_clim
    )

    # The first archive year has no prior forecast year, so it gets no causal
    # model climatology and is dropped rather than allowed to look ahead.
    train_dates = usable_dates(splits["train"], model_clim)
    val_dates = usable_dates(splits["val"], model_clim)
    dropped = len(splits["train"]) - len(train_dates)
    print(f"usable: train={len(train_dates)} (dropped {dropped}) val={len(val_dates)}")

    train_feat, train_tgt, mask_np = load_raw(args.target, train_dates)
    tr_tgt_a, tr_raw_a, tr_lag_a = anomaly_fields(
        train_feat, train_tgt, args.target, train_dates, obs_clim, model_clim
    )
    del train_tgt

    sigma_obs, sigma_mdl = compute_sigmas(tr_tgt_a, tr_raw_a, mask_np)
    print(f"sigma_obs={sigma_obs:.4f}  sigma_mdl={sigma_mdl:.4f}")

    substitute_channels(train_feat, tr_raw_a, tr_lag_a, sigma_mdl, sigma_obs, mask_np)
    stats = compute_feature_stats(train_feat, mask_np, args.target, sigma_obs, sigma_mdl,
                                  drop_anchor=args.drop_anchor)
    stats["plain_head"] = args.plain_head
    save_stats(stats, model_path / "stats.json")

    standardize_features(train_feat, stats)
    train_tgt, train_anchor = scale_target_and_anchor(tr_tgt_a, tr_raw_a, stats)
    del tr_tgt_a, tr_raw_a, tr_lag_a

    val_feat, val_tgt, _ = load_raw(args.target, val_dates)
    va_tgt_a, va_raw_a, va_lag_a = anomaly_fields(
        val_feat, val_tgt, args.target, val_dates, obs_clim, model_clim
    )
    del val_tgt
    substitute_channels(val_feat, va_raw_a, va_lag_a, sigma_mdl, sigma_obs, mask_np)
    standardize_features(val_feat, stats)
    val_tgt, val_anchor = scale_target_and_anchor(va_tgt_a, va_raw_a, stats)
    del va_tgt_a, va_raw_a, va_lag_a

    train_x, val_x = model_features(train_feat, stats), model_features(val_feat, stats)
    print(f"input channels: {train_x.shape[1]} of {train_feat.shape[1]} "
          f"(drop_anchor={args.drop_anchor})")

    train_loader = DataLoader(
        GatedDataset(train_x, train_tgt, train_anchor), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        GatedDataset(val_x, val_tgt, val_anchor), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print("device:", device, "amp:", use_amp)

    mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
    lat = torch.tensor(padded_lat_array(), dtype=torch.float32, device=device)

    if args.plain_head:
        model = PlainAnomalyUNet3D(in_channels=train_x.shape[1], unit=args.unit).to(device)
    else:
        model = GatedUNet3D(
            in_channels=train_x.shape[1], unit=args.unit, gate_bias_init=args.gate_bias_init
        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(device.type, enabled=use_amp)

    train_losses, val_losses = [], []
    best_val_loss, best_epoch = float("inf"), 0

    for epoch in range(args.n_epochs):
        train_loss = run_epoch(model, train_loader, mask, lat, device, optimizer, scaler, use_amp)
        val_loss = run_epoch(model, val_loader, mask, lat, device, use_amp=use_amp)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        torch.save(model.state_dict(), model_path / f"model_epoch_{epoch + 1}.pt")
        with open(model_path / "train_history.pkl", "wb") as f:
            pickle.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

        print(
            f"epoch [{epoch + 1}/{args.n_epochs}] "
            f"train(loss/mse/corr)={train_loss} val(loss/mse/corr)={val_loss}"
        )

        if val_loss[0] < best_val_loss:
            best_val_loss, best_epoch = val_loss[0], epoch

        if len(val_losses) >= args.patience + 1:
            recent = [v[0] for v in val_losses[-args.patience:]]
            if min(recent) > val_losses[-args.patience - 1][0] + 0.001:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    with open(model_path / "best_epoch", "w") as f:
        f.write(str(best_epoch))
    print(f"Best epoch: {best_epoch + 1} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
