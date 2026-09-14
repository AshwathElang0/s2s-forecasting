"""Train the 3D U-Net for one target variable (rain, tmax, or tmin).

Mirrors the reference paper's code/model_train.py structure (Adam, fixed
LR, checkpoint-every-epoch + early stopping on val loss), adapted to:
masked loss (model/unet3d.py), fold-internal standardization
(preprocess/dataset_loader.py), and this project's channel layout
(preprocess/build_dataset.py).

Usage:  python train.py --target rain --model_path runs/rain
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import config
from model.unet3d import CustomDataset, UNet3D, masked_mse_loss, masked_pattern_correlation_loss
from preprocess.build_dataset import padded_native_slice
from preprocess.dataset_loader import (
    apply_stats,
    compute_stats,
    list_cached_dates,
    load_raw,
    save_stats,
    split_dates,
)


def native_lat_array() -> np.ndarray:
    return np.linspace(config.NATIVE_LAT_VALS[0], config.NATIVE_LAT_VALS[1], config.NATIVE_LAT)


def padded_lat_array() -> np.ndarray:
    lat = native_lat_array()
    step = lat[1] - lat[0]
    pad_bottom = config.PAD_LAT - config.NATIVE_LAT - config.PAD_TOP
    top = lat[0] - step * np.arange(config.PAD_TOP, 0, -1)
    bottom = lat[-1] + step * np.arange(1, pad_bottom + 1)
    return np.concatenate([top, lat, bottom])


def run_epoch(model, loader, mask, lat, device, optimizer=None, scaler=None, use_amp=False):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = total_mse = total_corr = 0.0
    n_batches = 0
    for batch in loader:
        imgs = batch["imgs"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        # autocast covers only the conv stack (fp16 on tensor cores is the
        # actual speedup); loss is computed in fp32 since the masked
        # weighted-mean/correlation math isn't autocast-audited the way
        # torch's built-in losses are.
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs)
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
    parser.add_argument("--batch_size", type=int, default=11)  # reference's GPU-memory-bound default
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--limit_train", type=int, default=None,
        help="cap the number of train samples loaded (each is ~7MB decompressed; "
        "load_raw() reads the whole split into memory up front, so this is the "
        "knob for a quick run instead of the full split)",
    )
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True, help="mixed-precision training (default on; RTX-class GPUs get a real speedup and it frees GPU memory)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; data already lives fully in host RAM as numpy, so 0 (no subprocess pickling overhead) is usually fastest")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True  # fixed input shape every batch -- let cudnn pick the fastest kernel

    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    all_dates = list_cached_dates(args.target)
    splits = split_dates(all_dates)
    if args.limit_train:
        splits["train"] = splits["train"][: args.limit_train]
    if args.limit_val:
        splits["val"] = splits["val"][: args.limit_val]
    print({k: len(v) for k, v in splits.items()})

    train_feat, train_tgt, mask_np = load_raw(args.target, splits["train"])
    val_feat, val_tgt, _ = load_raw(args.target, splits["val"])

    stats = compute_stats(train_feat, train_tgt, args.target, mask_np)
    save_stats(stats, model_path / "stats.json")

    train_feat, train_tgt = apply_stats(train_feat, train_tgt, stats)
    val_feat, val_tgt = apply_stats(val_feat, val_tgt, stats)

    train_ds = CustomDataset(train_feat, train_tgt)
    val_ds = CustomDataset(val_feat, val_tgt)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print("device:", device, "amp:", use_amp)

    mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
    lat = torch.tensor(padded_lat_array(), dtype=torch.float32, device=device)

    n_channels = train_feat.shape[1]
    model = UNet3D(in_channels=n_channels, unit=args.unit).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(device.type, enabled=use_amp)

    train_losses, val_losses = [], []
    best_val_loss, best_epoch = float("inf"), 0

    for epoch in range(args.n_epochs):
        train_loss = run_epoch(model, train_loader, mask, lat, device, optimizer, scaler, use_amp)
        val_loss = run_epoch(model, val_loader, mask, lat, device, optimizer=None, scaler=None, use_amp=use_amp)

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
