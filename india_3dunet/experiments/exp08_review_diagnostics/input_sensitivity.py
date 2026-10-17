"""Which inputs does one predicted value actually depend on?

For one output voxel (a forecast day at one grid cell) we back-propagate to the
input and measure |d output / d input| over the whole (channel, lead, lat, lon)
block. Non-zero gradient = that input can influence the prediction (the
receptive field); the size of the gradient = how strongly it does.

Model: exp06_yesterday_rain/runs/no_obs_s0 (16 inputs, no observed IMD rain). Averaged over all
153 test forecasts, so it describes the trained network, not one sample.

Writes figures/review/10_input_sensitivity.png and figures/review/sensitivity.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from model.unet3d import UNet3D
from train_ablation import SETS
from preprocess.dataset_loader import list_cached_dates, load_raw, load_stats, split_dates

RUN = config.find_run("no_obs_s0")
OUT = Path(__file__).resolve().parent / "figures" / "review"
CH_ALL = ["tp_mean", "tp_spread", "msl", "gh200", "gh1000", "q200", "q500", "q700", "q850",
          "u200", "u850", "v200", "v850", "wtmp", "lag_obs", "doy_sin", "doy_cos"]
TARGET_DAY = 14                       # forecast day whose output cell we probe
CELLS = {"Nagpur (central India)": (21.0, 79.5), "Mumbai (west coast)": (19.5, 72.0)}


def main():
    names = SETS["no_obs"]; idx = [CH_ALL.index(n) for n in names]
    stats = load_stats(RUN / "stats.json")
    dates = split_dates(list_cached_dates("rain"))["test"]
    feat, _, mask = load_raw("rain", dates)
    feat = feat[:, idx]
    fm = np.array(stats["feat_mean"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    fs = np.array(stats["feat_std"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    feat = ((feat - fm) / fs).astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=len(idx)).to(dev)
    model.load_state_dict(torch.load(RUN / "best.pt", map_location=dev)); model.eval()

    lat = np.linspace(39, -19.5, 40); lat = np.concatenate([39 + 1.5 * np.arange(config.PAD_TOP, 0, -1), lat,
                      -19.5 - 1.5 * np.arange(1, config.PAD_LAT - 40 - config.PAD_TOP + 1)])
    lon = np.linspace(21, 109.5, 60); lon = np.concatenate([21 - 1.5 * np.arange(config.PAD_LEFT, 0, -1), lon,
                      109.5 + 1.5 * np.arange(1, config.PAD_LON - 60 - config.PAD_LEFT + 1)])
    ext = [lon[0] - .75, lon[-1] + .75, lat[-1] - .75, lat[0] + .75]
    t = TARGET_DAY - 1

    res, fig_data = {}, {}
    for cname, (clat, clon) in CELLS.items():
        i, j = int(np.argmin(abs(lat - clat))), int(np.argmin(abs(lon - clon)))
        acc = np.zeros(feat.shape[1:], dtype=np.float64)
        for n in range(len(feat)):
            x = torch.tensor(feat[n:n + 1], device=dev, requires_grad=True)
            model(x)[0, 0, t, i, j].backward()
            acc += x.grad[0].abs().cpu().numpy()
        acc /= len(feat)                                   # (C, lead, lat, lon)
        total = acc.sum()
        sup = acc > 1e-9 * acc.max()                       # structural support (receptive field)
        li, lo = np.where(sup.any(axis=(0, 1)))
        by_ch = acc.sum(axis=(1, 2, 3)) / total
        by_lead = acc.sum(axis=(0, 2, 3)) / total
        by_space = acc.sum(axis=(0, 1)) / total
        # distance (in cells) from the target cell, share of total sensitivity
        r, c = np.meshgrid(np.arange(acc.shape[2]), np.arange(acc.shape[3]), indexing="ij")
        dist = np.maximum(abs(r - i), abs(c - j))
        within = {f"{d}_cells_{d * 1.5:g}deg": float(by_space[dist <= d].sum()) for d in (1, 2, 4, 8)}
        res[cname] = dict(
            cell=[float(lat[i]), float(lon[j])],
            support_lat_deg=[float(lat[li.max()]), float(lat[li.min()])],
            support_lon_deg=[float(lon[lo.min()]), float(lon[lo.max()])],
            support_lead_days=[int(np.where(sup.any(axis=(0, 2, 3)))[0].min() + 1), int(np.where(sup.any(axis=(0, 2, 3)))[0].max() + 1)],
            share_by_channel={n: float(v) for n, v in zip(names, by_ch)},
            share_within_distance=within,
            share_same_day=float(by_lead[t]), share_days_within_3=float(by_lead[max(0, t - 3):t + 4].sum()))
        fig_data[cname] = (by_space, by_lead, by_ch, (i, j))

    fig, ax = plt.subplots(len(CELLS), 3, figsize=(15, 4.6 * len(CELLS)), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    for r, (cname, (bs, bl, bc, (i, j))) in enumerate(fig_data.items()):
        a = ax[r, 0]; a.grid(False)
        im = a.imshow(bs / bs.max(), extent=ext, cmap="magma_r", norm=matplotlib.colors.PowerNorm(0.35), origin="upper")
        a.plot(lon[j], lat[i], "c*", ms=14, mec="k"); a.set_title(f"{cname}: where the input comes from\n(day {TARGET_DAY} output at the star, all 16 channels, all 32 days)", fontsize=9.5)
        a.set_xlabel("longitude"); a.set_ylabel("latitude")
        a.add_patch(plt.Rectangle((21 - .75, -19.5 - .75), 88.5 + 1.5, 58.5 + 1.5, fill=False, ec="b", lw=1.2, ls="--"))
        fig.colorbar(im, ax=a, shrink=0.8, label="relative sensitivity (power-scaled)")
        a = ax[r, 1]; a.bar(np.arange(1, 33), bl, color="#4c72b0"); a.axvline(TARGET_DAY, color="k", ls=":", label=f"output day {TARGET_DAY}")
        a.set_title("which forecast days (lead axis) it uses", fontsize=9.5); a.set_xlabel("input forecast day"); a.set_ylabel("share of total sensitivity"); a.legend()
        a = ax[r, 2]; o = np.argsort(bc)[::-1]; a.barh(np.array(names)[o][::-1], bc[o][::-1], color="#55a868")
        a.set_title("which input variables it uses", fontsize=9.5); a.set_xlabel("share of total sensitivity")
    fig.suptitle("What one predicted value depends on (3D U-Net, mean |gradient| over the 153 test forecasts)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "10_input_sensitivity.png", bbox_inches="tight", dpi=150)
    (OUT / "sensitivity.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
