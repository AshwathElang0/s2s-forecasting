"""Experiment 10: train the U-Net on top of Dyn++ instead of from scratch.

Three variants, same 16 inputs as the headline model (no observed IMD rain),
same optimiser, batch size, early stopping and 724 training forecasts:

  plain_aligned  the headline U-Net, but trained on correctly aligned targets
                 (IMD issue+1+t). The control: separates the alignment fix
                 from the Dyn++ ideas. log1p scale, loss MSE + (1 - pattern corr).
  residual       ECMWF rain input replaced by Dyn++ rain; output = Dyn++ + a
                 learned correction on the log1p scale (model/residual_unet.py).
                 Same loss as plain_aligned.
  residual_mm    output = Dyn++ x exp(learned log-factor), trained on rain in
                 mm/day. Aimed at two weaknesses found in Exp 8:
                   MSE in mm         targets the mean amount, not the mean of
                                     log(1 + rain), which reads back too dry
                   anomaly pattern   1 - pattern corr of the departure from
                                     normal, so geography earns nothing
                 An optional all-India term (squared error of the area-mean
                 rain / its training variance) is off by default: it is an
                 untested idea of ours, not from the literature. Turn it on
                 with --area_weight 1.0.
                 A 1-epoch trial of the log-space version of this loss went
                 ~50% too dry on val, which is why this one works in mm.

Training dates get the leave-one-year-out Dyn++ (fit without their own year);
val and test get the full training fit. Run Experiment 9 first: it writes the
factors to data_cache/bias_correction/dynpp_factors.npz.

After training, predicts val and test. A per-lead-week mean correction
(observed / predicted rain, summed over val) is stored in stats.json as
`val_mean_fix` and applied optionally at scoring time, to test whether any
remaining dry bias is just a constant.

Usage:
  python experiments/exp10_unet_on_bias_correction/train_residual.py --variant residual_mm --seed 0
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import config
from model.residual_unet import (MultiplicativeResidualUNet3D, PlainUNet3D, ResidualDataset,
                                 ResidualUNet3D, area_mean)
from model.unet3d import masked_mse_loss, masked_pattern_correlation_loss
from preprocess.bias_correction import BC_ROOT, apply_dynpp, load_aligned, load_factors, week_of_day
from preprocess.climatology import ensure_climatology, obs_clim_for_dates, valid_mask_native
from preprocess.dataset_loader import list_cached_dates, load_raw, split_dates
from train import padded_lat_array
from train_ablation import SETS, channel_indices

# variant -> (model class, target scale)
VARIANTS = {
    "plain_aligned": (PlainUNet3D, "log"),
    "residual": (ResidualUNet3D, "log"),
    "residual_mm": (MultiplicativeResidualUNet3D, "mm"),
}


def loss_terms(pred, y, clim, lat, mask, variant, am_var, area_weight):
    """(total, mse, pattern, area). area is reported for residual_mm but only
    enters the total with a nonzero area_weight."""
    mse = masked_mse_loss(pred, y, mask)
    if variant == "residual_mm":
        pattern = masked_pattern_correlation_loss(pred - clim, y - clim, lat, mask)
        area = (area_mean(pred - y, lat, mask) ** 2).mean() / am_var
        return mse + pattern + area_weight * area, mse, pattern, area
    pattern = masked_pattern_correlation_loss(pred, y, lat, mask)
    return mse + pattern, mse, pattern, torch.zeros((), device=pred.device)


def run_epoch(model, loader, lat, mask, device, variant, am_var, area_weight,
              optimizer=None, scaler=None, use_amp=False):
    train_mode = optimizer is not None
    model.train(train_mode)
    sums, n = np.zeros(4), 0
    for b in loader:
        x, y, a, c = (b[k].to(device, non_blocking=True) for k in ("imgs", "labels", "anchor", "clim"))
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            with autocast(device_type=device.type, enabled=use_amp):
                pred = model(x, a)
            terms = loss_terms(pred.float(), y, c, lat, mask, variant, am_var, area_weight)
            if train_mode:
                scaler.scale(terms[0]).backward()
                scaler.step(optimizer)
                scaler.update()
        sums += [t.item() for t in terms]
        n += 1
    return (sums / n).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--unit", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=11)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--area_weight", type=float, default=0.0,
                        help="weight of the all-India term; off by default (untested idea, see docstring)")
    parser.add_argument("--run_name", default=None, help="defaults to <variant>_s<seed>")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    run_dir = HERE / "runs" / (args.run_name or f"{args.variant}_s{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cls, scale = VARIANTS[args.variant]
    residual = args.variant != "plain_aligned"
    names = SETS["no_obs"]
    idx = channel_indices(names)
    assert names[0] == "tp_mean"
    all_dates = list_cached_dates("rain")
    splits = split_dates(all_dates)
    factors = load_factors(BC_ROOT / "dynpp_factors.npz")
    obs_clim, _ = ensure_climatology("rain", splits["train"], all_dates)
    print(f"{args.variant} seed {args.seed}: {len(idx)} inputs, Dyn++ anchor={residual}, scale={scale}")

    def load(split):
        """Features plus target / anchor / climatology in mm/day, on aligned days."""
        dates = splits[split]
        feat, _, mask = load_raw("rain", dates)
        feat = feat[:, idx]
        tp, obs = load_aligned(dates)
        anchor = apply_dynpp(tp, dates, factors, use_loyo=(split == "train")) if residual else np.zeros_like(tp)
        if residual:
            feat[:, 0] = np.log1p(np.clip(anchor, 0, None))   # Dyn++ in place of raw ECMWF rain
        clim = np.expm1(obs_clim_for_dates(obs_clim, dates, config.LEAD_DAYS, offset=1))
        return dates, feat, obs, anchor, clim, mask

    tr_dates, tr_feat, tr_obs, tr_anchor, tr_clim, mask_np = load("train")
    valid = valid_mask_native(mask_np)
    feat_mean = np.array([tr_feat[:, c][:, :, valid].mean() for c in range(tr_feat.shape[1])], np.float32)
    feat_std = np.array([tr_feat[:, c][:, :, valid].std() or 1.0 for c in range(tr_feat.shape[1])], np.float32)
    if scale == "log":
        lv = np.log1p(np.clip(tr_obs[:, :, valid], 0, None))
        t_mean, t_std = float(np.nanmean(lv)), float(np.nanstd(lv))
    else:   # mm: scale only, so zero rain stays zero and the anchor stays a multiplier
        t_mean, t_std = 0.0, float(np.nanstd(tr_obs[:, :, valid]))

    def to_scale(mm):
        v = (np.log1p(np.clip(mm, 0, None)) - t_mean) / t_std if scale == "log" else mm / t_std
        return np.nan_to_num(v, nan=0.0).astype(np.float32)[:, None]

    def from_scale(v):
        return np.clip(np.expm1(v * t_std + t_mean) if scale == "log" else v * t_std, 0, None)

    def to_model(feat, obs, anchor, clim):
        feat -= feat_mean.reshape(1, -1, 1, 1, 1)
        feat /= feat_std.reshape(1, -1, 1, 1, 1)
        return feat, to_scale(obs), (to_scale(anchor) if residual else np.zeros_like(to_scale(obs))), to_scale(clim)

    tr = to_model(tr_feat, tr_obs, tr_anchor, tr_clim)
    del tr_obs, tr_anchor, tr_clim
    lat_cpu = torch.tensor(padded_lat_array(), dtype=torch.float32)
    am_var = float(area_mean(torch.from_numpy(tr[1] - tr[3]), lat_cpu, torch.from_numpy(mask_np)).var())

    stats = {"variant": args.variant, "seed": args.seed, "scale": scale, "channels": names,
             "feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist(), "target_mean": t_mean,
             "target_std": t_std, "area_mean_var": am_var, "area_weight": args.area_weight}
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"target scale: mean {t_mean:.3f} sd {t_std:.3f}; area-mean anomaly variance {am_var:.4f}")

    va_dates, *va_raw, _ = load("val")
    va = to_model(*va_raw)
    va_obs = va_raw[1]
    del va_raw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
    lat = lat_cpu.to(device)
    loader = lambda d, shuffle: DataLoader(ResidualDataset(*d), batch_size=args.batch_size,
                                           shuffle=shuffle, pin_memory=True)
    tr_loader, va_loader = loader(tr, True), loader(va, False)

    model = model_cls(in_channels=len(idx), unit=args.unit).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(device.type, enabled=use_amp)
    common = dict(lat=lat, mask=mask, device=device, variant=args.variant, am_var=am_var,
                  area_weight=args.area_weight)

    start = run_epoch(model, va_loader, **common, use_amp=use_amp)
    print(f"before training: val loss/mse/pattern/area = {np.round(start, 4).tolist()}")
    history = {"val_before_training": start, "train_losses": [], "val_losses": []}
    best_val, best_epoch = float("inf"), 0
    for epoch in range(args.n_epochs):
        trl = run_epoch(model, tr_loader, **common, optimizer=optimizer, scaler=scaler, use_amp=use_amp)
        val = run_epoch(model, va_loader, **common, use_amp=use_amp)
        history["train_losses"].append(trl)
        history["val_losses"].append(val)
        if val[0] < best_val:
            best_val, best_epoch = val[0], epoch
            torch.save(model.state_dict(), run_dir / "best.pt")
        print(f"epoch [{epoch + 1}/{args.n_epochs}] train={trl[0]:.4f} val={val[0]:.4f} "
              f"(mse {val[1]:.4f} pattern {val[2]:.4f} area {val[3]:.4f})", flush=True)
        vl = [v[0] for v in history["val_losses"]]
        if len(vl) >= args.patience + 1 and min(vl[-args.patience:]) > vl[-args.patience - 1] + 0.001:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    with open(run_dir / "train_history.pkl", "wb") as f:
        pickle.dump(history, f)
    (run_dir / "best_epoch").write_text(str(best_epoch))
    print(f"Best epoch: {best_epoch + 1} (val={best_val:.4f})")
    del tr, tr_loader

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()

    def predict(data):
        out = []
        with torch.no_grad():
            for b in DataLoader(ResidualDataset(*data), batch_size=4):
                with autocast(device_type=device.type, enabled=use_amp):
                    p = model(b["imgs"].to(device), b["anchor"].to(device))
                out.append(p.float().cpu().numpy()[:, 0])
        return from_scale(np.concatenate(out)).astype(np.float32)

    va_pred = predict(va)
    m = mask_np.astype(bool)
    wk = week_of_day(config.LEAD_DAYS)
    fix = [float(np.nansum(va_obs[:, wk == w][..., m]) / np.nansum(va_pred[:, wk == w][..., m])) for w in range(4)]
    stats["val_mean_fix"] = fix
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("val mean fix (obs/pred) by week:", np.round(fix, 3).tolist())
    del va, va_pred

    te_dates, *te_raw, _ = load("test")
    te_obs = te_raw[1]
    te = to_model(*te_raw)
    np.savez(run_dir / "test_rain_predictions.npz", preds=predict(te), obs=te_obs, mask=mask_np,
             dates=np.array([f"{d:%Y-%m-%d}" for d in te_dates]))
    print(f"saved {len(te_dates)} test predictions under {run_dir}")


if __name__ == "__main__":
    main()
