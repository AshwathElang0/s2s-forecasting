"""Figures for the professor's review comments (results.md, "Review response").

Model shown: exp06_yesterday_rain/runs/no_obs_s0 -- the 3D U-Net with 16 inputs, i.e. **without
yesterday's observed IMD rain**. IMD is therefore used only as the training
target, never as an input at forecast time.

Everything here is read-only on the existing run directory: no retraining, the
test predictions are the ones already saved by train_ablation.py.

Writes PNGs to figures/review_aligned/ and the numbers behind them to
figures/review_aligned/summary.json. Forecast day d is ECMWF lead index d-1 scored
against IMD date issue+d (preprocess/alignment.py); the original, one-day-early
figures in figures/review/ are left as they were.

Usage:  python scripts/make_review_figures.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import config
from model.unet3d import masked_mse_loss, masked_pattern_correlation_loss
from preprocess.alignment import ALIGNED_DAYS, ecmwf_side, obs_side, valid_date
from preprocess.dataset_loader import list_cached_dates, load_stats, split_dates
from train import padded_lat_array

RUN = config.find_run("no_obs_s0")
OUT = Path(__file__).resolve().parent / "figures" / "review_aligned"
LEADS = [7, 14, 21, 28]            # forecast days, as in results.md
C_OBS, C_MODEL, C_RAW, C_CLIM = "#222222", "#d95f02", "#1b9e77", "#9a9a9a"
MODEL_LABEL = "3D U-Net (no IMD input)"
RAW_LABEL = "Raw ECMWF (ens. mean)"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False,
})

CITIES = {  # nearest 1.5 deg cell is used
    "Delhi": (28.6, 77.2), "Mumbai": (19.1, 72.9),
    "Kolkata": (22.6, 88.4), "Chennai": (13.1, 80.3),
}


# --------------------------------------------------------------------- data
def load_everything():
    z = np.load(RUN / "test_rain_predictions.npz")
    preds, obs, mask = z["preds"], z["obs"], z["mask"]
    dates = split_dates(list_cached_dates("rain"))["test"]
    assert len(dates) == len(preds)

    # Raw ECMWF ensemble-mean rain = channel 0 of the cached input sample.
    d = config.CACHE_ROOT / "samples" / "rain"
    raw = np.stack([np.load(d / f"{x:%Y_%m_%d}.npz")["features"][0] for x in dates])

    lat = padded_lat_array()
    lon = np.linspace(config.NATIVE_LON_VALS[0], config.NATIVE_LON_VALS[1], config.NATIVE_LON)
    pad_lon = np.concatenate([
        lon[0] - 1.5 * np.arange(config.PAD_LEFT, 0, -1), lon,
        lon[-1] + 1.5 * np.arange(1, config.PAD_LON - config.NATIVE_LON - config.PAD_LEFT + 1)])
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    # preds_full/obs_full keep the layout the model was trained on (learning-curve test loss);
    # everything else is scored on aligned days.
    return dict(preds=obs_side(preds), obs=obs_side(obs), raw=ecmwf_side(raw), preds_full=preds, obs_full=obs,
                mask=mask, dates=dates, lat=lat, lon=pad_lon, w=w)


def area_mean(x, w):
    """x (..., lat, lon) -> (...), area-weighted over valid cells."""
    x = np.nan_to_num(x, nan=0.0)
    return (x * w).sum(axis=(-2, -1)) / w.sum()


def clim_by_doy():
    """Arithmetic day-of-year rainfall mean per cell from 1951-2020 only (nothing
    from the 2021-23 test years), smoothed +/-10 days. (366, 48, 64)."""
    with xr.open_dataset(config.CACHE_ROOT / "imd_targets_1p5deg.nc") as ds:
        t = ds.time.values.astype("datetime64[D]")
        yr = t.astype("datetime64[Y]").astype(int) + 1970
        sel = np.flatnonzero(yr <= 2020)
        rain = ds["rain"].values[sel].astype(np.float32)
        doy = ds.time.dt.dayofyear.values[sel]
    sums = np.zeros((366,) + rain.shape[1:]); cnt = np.zeros_like(sums)
    ok = np.isfinite(rain)
    np.add.at(sums, doy - 1, np.where(ok, rain, 0)); np.add.at(cnt, doy - 1, ok)
    idx = (np.arange(366)[:, None] + np.arange(-10, 11)[None]) % 366
    s = sums[idx].sum(1); c = cnt[idx].sum(1)
    clim = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
    pad_b = config.PAD_LAT - config.NATIVE_LAT - config.PAD_TOP
    pad_r = config.PAD_LON - config.NATIVE_LON - config.PAD_LEFT
    return np.pad(clim, [(0, 0), (config.PAD_TOP, pad_b), (config.PAD_LEFT, pad_r)], constant_values=np.nan)


def clim_for(D, clim, day_idx):
    """(N, len(day_idx), 48, 64) climatology for each sample's valid dates."""
    out = np.empty((len(D["dates"]), len(day_idx)) + clim.shape[1:], dtype=np.float32)
    for n, d in enumerate(D["dates"]):
        for k, t in enumerate(day_idx):
            out[n, k] = clim[valid_date(d, int(t)).timetuple().tm_yday - 1]
    return out


def stats(p, o):
    p, o = np.asarray(p, float), np.asarray(o, float)
    r = np.corrcoef(p, o)[0, 1]
    slope = np.polyfit(o, p, 1)[0]
    return dict(r=float(r), rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                bias=float(np.mean(p - o)), slope=float(slope), n=int(len(p)))


# ------------------------------------------------------- 1. learning curves
def fig_learning_curves(D, summary):
    h = pickle.load(open(RUN / "train_history.pkl", "rb"))
    tr, va = np.array(h["train_losses"]), np.array(h["val_losses"])     # (epochs, 3): total, mse, 1-corr
    best = int(Path(RUN / "best_epoch").read_text())                     # 0-based
    st = load_stats(RUN / "stats.json")

    # Test loss of the best checkpoint, same loss as training (MSE + 1-corr on
    # the standardized log1p scale), batches of 11 like training/validation.
    def std(x):
        return np.nan_to_num((np.log1p(np.clip(x, 0, None)) - st["target_mean"]) / st["target_std"], nan=0.0)
    p = torch.tensor((np.log1p(np.clip(D["preds_full"], -0.999, None)) - st["target_mean"]) / st["target_std"])[:, None]
    y = torch.tensor(std(D["obs_full"]), dtype=torch.float32)[:, None]
    m = torch.tensor(D["mask"]); lat = torch.tensor(D["lat"], dtype=torch.float32)
    vals = []
    for i in range(0, len(p), 11):
        mse = masked_mse_loss(p[i:i + 11], y[i:i + 11], m).item()
        cl = masked_pattern_correlation_loss(p[i:i + 11], y[i:i + 11], lat, m).item()
        vals.append((mse + cl, mse, cl))
    test = np.mean(vals, axis=0)

    ep = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, k, title in zip(ax, range(3), ["Total loss  (MSE + 1 - corr)", "MSE part", "1 - pattern-correlation part"]):
        a.plot(ep, tr[:, k], "-o", ms=3, color="#1f77b4", label="train")
        a.plot(ep, va[:, k], "-o", ms=3, color=C_MODEL, label="validation (2018-20)")
        a.scatter([best + 1], [test[k]], marker="*", s=170, color="k", zorder=5,
                  label="test (2021-23), best checkpoint")
        a.axvline(best + 1, color="k", ls=":", lw=1)
        a.set_title(title); a.set_xlabel("epoch")
    ax[0].set_ylabel("loss (standardized log-rain units)"); ax[0].legend(fontsize=8)
    fig.suptitle(f"Learning curves, {MODEL_LABEL}: best epoch {best + 1} chosen on validation loss "
                 f"(dotted); test is never used for model selection", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "01_learning_curves.png", bbox_inches="tight"); plt.close(fig)
    summary["learning_curves"] = dict(best_epoch=best + 1, n_epochs=len(tr),
                                      train_at_best=tr[best].tolist(), val_at_best=va[best].tolist(),
                                      test_at_best=test.tolist(), train_last=tr[-1].tolist())


# ------------------------------------------------------------- 2. scatters
def binned_mean(o, p, edges):
    c = 0.5 * (edges[1:] + edges[:-1]); idx = np.digitize(o, edges) - 1
    mu = np.array([p[idx == i].mean() if (idx == i).sum() >= 30 else np.nan for i in range(len(c))])
    return c, mu


def fig_gridcell_scatter(D, summary, log_axes):
    valid = D["mask"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.4), sharex=True, sharey=True)
    summary.setdefault("gridcell_scatter", {})
    for j, L in enumerate(LEADS):
        o = D["obs"][:, L - 1][:, valid]; ok = np.isfinite(o)
        for i, (name, arr, col) in enumerate([(MODEL_LABEL, D["preds"], C_MODEL), (RAW_LABEL, D["raw"], C_RAW)]):
            p = np.clip(arr[:, L - 1][:, valid][ok], 0, None); ob = o[ok]
            s = stats(p, ob); a = ax[i, j]
            if log_axes:
                x, yv, lim = np.log1p(ob), np.log1p(p), np.log1p(100)
                edges = np.linspace(0, lim, 25)
                a.hexbin(x, yv, gridsize=45, extent=(0, lim, 0, lim), bins="log", mincnt=1, cmap="viridis")
                a.plot([0, lim], [0, lim], "k--", lw=1)
                c, mu = binned_mean(x, yv, edges); a.plot(c, mu, color="r", lw=2)
                a.set_xlim(0, lim); a.set_ylim(0, lim)
                ticks = np.log1p([0, 1, 5, 20, 50, 100]); a.set_xticks(ticks); a.set_yticks(ticks)
                a.set_xticklabels([0, 1, 5, 20, 50, 100]); a.set_yticklabels([0, 1, 5, 20, 50, 100])
            else:
                lim = 80
                a.hexbin(ob, p, gridsize=50, extent=(0, lim, 0, lim), bins="log", mincnt=1, cmap="viridis")
                a.plot([0, lim], [0, lim], "k--", lw=1)
                c, mu = binned_mean(ob, p, np.linspace(0, lim, 21)); a.plot(c, mu, color="r", lw=2)
                a.set_xlim(0, lim); a.set_ylim(0, lim)
            a.set_aspect("equal")
            a.text(0.04, 0.96, f"r = {s['r']:.2f}\nRMSE = {s['rmse']:.1f}\nbias = {s['bias']:+.2f}\nslope = {s['slope']:.2f}",
                   transform=a.transAxes, va="top", fontsize=9,
                   bbox=dict(fc="white", ec="none", alpha=0.8))
            if i == 0: a.set_title(f"forecast day {L}")
            if j == 0: a.set_ylabel(f"{name}\npredicted (mm/day)")
            if i == 1: a.set_xlabel("observed IMD (mm/day)")
            if not log_axes: summary["gridcell_scatter"][f"{'model' if i == 0 else 'raw'}_day{L}"] = s
    fig.suptitle("Observed vs predicted daily rainfall, every Indian land cell x every test forecast (2021-23)\n"
                 "dashed = perfect 1:1, red = mean prediction for each observed-rain bin, colour = point count (log)"
                 + ("   [axes: log(1+rain), tick labels in mm/day]" if log_axes else "   [axes clipped at 80 mm/day]"), y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / ("03_scatter_gridcell_logaxes.png" if log_axes else "02_scatter_gridcell.png"), bbox_inches="tight")
    plt.close(fig)


def fig_areamean_scatter(D, clim, summary):
    """All-India mean rain, 7-day means for weeks 1-4 (the standard S2S unit)."""
    summary["areamean_weekly"] = {}
    months = np.array([d.month for d in D["dates"]])
    monsoon = (months >= 6) & (months <= 9)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True, sharey=True)
    for j in range(4):
        days = np.arange(7 * j, 7 * j + 7)
        o = area_mean(D["obs"][:, days], D["w"]).mean(1)
        for i, (name, arr, col) in enumerate([(MODEL_LABEL, D["preds"], C_MODEL), (RAW_LABEL, D["raw"], C_RAW)]):
            p = area_mean(arr[:, days], D["w"]).mean(1); a = ax[i, j]
            a.scatter(o[~monsoon], p[~monsoon], s=18, c="#7f7f7f", alpha=0.8, label="Oct-May issue dates")
            a.scatter(o[monsoon], p[monsoon], s=22, c=col, alpha=0.9, label="Jun-Sep issue dates")
            a.plot([0, 16], [0, 16], "k--", lw=1); a.set_xlim(0, 16); a.set_ylim(0, 16); a.set_aspect("equal")
            s = stats(p, o); s["monsoon_r"] = float(np.corrcoef(p[monsoon], o[monsoon])[0, 1])
            summary["areamean_weekly"][f"{'model' if i == 0 else 'raw'}_week{j + 1}"] = s
            a.text(0.04, 0.96, f"r = {s['r']:.2f}  (Jun-Sep only: {s['monsoon_r']:.2f})\nRMSE = {s['rmse']:.2f}\nbias = {s['bias']:+.2f}",
                   transform=a.transAxes, va="top", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.8))
            if i == 0: a.set_title(f"week {j + 1} (days {days[0] + 1}-{days[-1] + 1})")
            if j == 0: a.set_ylabel(f"{name}\npredicted (mm/day)")
            if i == 1: a.set_xlabel("observed IMD (mm/day)")
            if i == 0 and j == 0: a.legend(loc="lower right", fontsize=8)
    fig.suptitle("All-India mean rainfall, weekly average: observed vs predicted, one dot per test forecast (153)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "04_scatter_allindia_weekly.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------- 3. case studies
def pick_cases(D):
    o = area_mean(D["obs"][:, :28], D["w"]).mean(1)
    months = np.array([d.month for d in D["dates"]])
    mons = np.flatnonzero((months >= 6) & (months <= 8))
    cases = {"wettest 28-day period (Jun-Aug issue)": int(mons[np.argmax(o[mons])]),
             "driest Jun-Aug 28-day period": int(mons[np.argmin(o[mons])]),
             "median Jun-Aug 28-day period": int(mons[np.argsort(o[mons])[len(mons) // 2]])}
    pre = np.flatnonzero((months >= 3) & (months <= 5))
    cases["driest-vs-wettest pre-monsoon: wettest Mar-May"] = int(pre[np.argmax(o[pre])])
    cases["post-monsoon / NE monsoon: wettest Oct-Dec"] = int(np.flatnonzero(months >= 10)[np.argmax(o[months >= 10])])
    cases["winter: wettest Dec-Feb"] = int(np.flatnonzero((months == 12) | (months <= 2))[
        np.argmax(o[(months == 12) | (months <= 2)])])
    return cases


def crop(D):
    r, c = np.where(D["mask"])
    return slice(r.min() - 1, r.max() + 2), slice(c.min() - 1, c.max() + 2)


def fig_case_maps(D, n, title, fname, weekly):
    rs, cs = crop(D)
    ext = [D["lon"][cs][0] - .75, D["lon"][cs][-1] + .75, D["lat"][rs][-1] - .75, D["lat"][rs][0] + .75]
    mask = D["mask"][rs, cs]
    fig, ax = plt.subplots(4, 4, figsize=(13.5, 14), constrained_layout=True)
    for i, L in enumerate(LEADS):
        sl = np.arange(7 * i, 7 * i + 7) if weekly else np.array([L - 1])
        get = lambda arr: np.where(mask, np.nanmean(arr[n, sl][:, rs, cs], axis=0), np.nan)
        o, p, r = get(D["obs"]), get(D["preds"]), get(D["raw"])
        vmax = max(np.nanpercentile(np.concatenate([o[mask], p[mask], r[mask]]), 99), 5)
        panels = [(o, "IMD observed", "YlGnBu", 0, vmax), (r, RAW_LABEL, "YlGnBu", 0, vmax),
                  (p, MODEL_LABEL, "YlGnBu", 0, vmax), (p - o, "U-Net minus observed", "RdBu", -vmax, vmax)]
        for j, (f, t, cmap, lo, hi) in enumerate(panels):
            a = ax[i, j]; a.grid(False)
            im = a.imshow(f, extent=ext, cmap=cmap, vmin=lo, vmax=hi, origin="upper")
            if i == 0: a.set_title(t)
            if j == 0:
                lab = f"week {i + 1} mean\n(days {sl[0] + 1}-{sl[-1] + 1})" if weekly else f"day {L}"
                a.set_ylabel(lab, fontsize=11)
            a.set_xticks([]); a.set_yticks([])
            if j in (2, 3): fig.colorbar(im, ax=a, shrink=0.75, pad=0.02, label="mm/day")
            if j == 0:
                s = f"r = {np.corrcoef(p[mask], o[mask])[0, 1]:.2f} (U-Net)\nr = {np.corrcoef(r[mask], o[mask])[0, 1]:.2f} (ECMWF)"
                a.text(0.02, 0.02, s, transform=a.transAxes, fontsize=8, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8))
    fig.suptitle(title, fontsize=12)
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def fig_case_series(D, cases):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharey=False)
    for a, (name, n) in zip(axs.flat, cases.items()):
        days = np.arange(1, ALIGNED_DAYS + 1)
        for arr, lab, col, ls in [(D["obs"], "IMD observed", C_OBS, "-"), (D["raw"], RAW_LABEL, C_RAW, "-"),
                                  (D["preds"], MODEL_LABEL, C_MODEL, "-")]:
            a.plot(days, area_mean(arr[n], D["w"]), color=col, ls=ls, lw=2 if col == C_OBS else 1.6, label=lab)
        for L in LEADS: a.axvline(L, color="k", alpha=0.12)
        a.set_title(f"{name}\nissued {D['dates'][n]:%Y-%m-%d}", fontsize=9.5)
        a.set_xlabel("forecast day"); a.set_ylabel("all-India mean rain (mm/day)")
    axs.flat[0].legend(fontsize=8)
    fig.suptitle(f"Daily all-India mean rainfall across the {ALIGNED_DAYS} forecast days, same test cases as the maps (grey bars = days 7/14/21/28)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "07_case_timeseries_allindia.png", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------ 4. time series, fixed lead
def valid_dates(D, days):
    return np.array([[valid_date(d, int(t)) for t in days] for d in D["dates"]])


def fig_series_allindia(D, clim, summary):
    summary["series_allindia"] = {}
    fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for j, a in enumerate(axs):
        days = np.arange(7 * j, 7 * j + 7)
        vd = [valid_date(d, int(days.mean())) for d in D["dates"]]
        o = area_mean(D["obs"][:, days], D["w"]).mean(1)
        p = area_mean(D["preds"][:, days], D["w"]).mean(1)
        r = area_mean(D["raw"][:, days], D["w"]).mean(1)
        c = area_mean(clim_for(D, clim, days), D["w"]).mean(1)
        a.plot(vd, c, color=C_CLIM, ls="--", lw=1.3, label="climatology (1951-2020)")
        a.plot(vd, r, color=C_RAW, lw=1.2, label=RAW_LABEL)
        a.plot(vd, p, color=C_MODEL, lw=1.6, label=MODEL_LABEL)
        a.plot(vd, o, color=C_OBS, lw=2, label="IMD observed")
        # also skill relative to climatology: correlate departures
        sa = dict(r_model=float(np.corrcoef(p, o)[0, 1]), r_raw=float(np.corrcoef(r, o)[0, 1]),
                  r_clim=float(np.corrcoef(c, o)[0, 1]),
                  r_anom_model=float(np.corrcoef(p - c, o - c)[0, 1]), r_anom_raw=float(np.corrcoef(r - c, o - c)[0, 1]))
        summary["series_allindia"][f"week{j + 1}"] = sa
        a.set_title(f"Week {j + 1} (days {days[0] + 1}-{days[-1] + 1}):  r = {sa['r_model']:.2f} U-Net, {sa['r_raw']:.2f} ECMWF, "
                    f"{sa['r_clim']:.2f} climatology.   Departures from climatology: r = {sa['r_anom_model']:.2f} U-Net, "
                    f"{sa['r_anom_raw']:.2f} ECMWF", fontsize=9.5, loc="left")
        a.set_ylabel("mm/day")
        if j == 0: a.legend(ncol=4, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.0))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("All-India mean rainfall, weekly average at each lead, test years 2021-23 (x-axis = date the week is valid)", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "05_timeseries_allindia_by_week.png", bbox_inches="tight"); plt.close(fig)


def fig_series_cities(D, clim, summary):
    summary["series_cities"] = {}
    la, lo = D["lat"], D["lon"]
    fig, axs = plt.subplots(len(CITIES), 2, figsize=(15, 3 * len(CITIES)), sharex=True)
    for ci, (city, (clat, clon)) in enumerate(CITIES.items()):
        r_i, c_i = np.argmin(abs(la - clat)), np.argmin(abs(lo - clon))
        assert D["mask"][r_i, c_i], city
        for k, wk in enumerate([0, 2]):
            a = axs[ci, k]; days = np.arange(7 * wk, 7 * wk + 7)
            vd = [valid_date(d, int(days.mean())) for d in D["dates"]]
            ser = lambda arr: np.nanmean(arr[:, days, r_i, c_i], axis=1)
            o, p, r = ser(D["obs"]), ser(D["preds"]), ser(D["raw"])
            c = clim_for(D, clim, days)[:, :, r_i, c_i].mean(1)
            a.plot(vd, c, color=C_CLIM, ls="--", lw=1.2, label="climatology")
            a.plot(vd, r, color=C_RAW, lw=1.1, label=RAW_LABEL)
            a.plot(vd, p, color=C_MODEL, lw=1.5, label=MODEL_LABEL)
            a.plot(vd, o, color=C_OBS, lw=1.8, label="IMD observed")
            s = dict(r_model=float(np.corrcoef(p, o)[0, 1]), r_raw=float(np.corrcoef(r, o)[0, 1]),
                     r_clim=float(np.corrcoef(c, o)[0, 1]))
            summary["series_cities"][f"{city}_week{wk + 1}"] = s
            a.set_title(f"{city} ({la[r_i]:.1f}N, {lo[c_i]:.1f}E), week {wk + 1}:  r = {s['r_model']:.2f} U-Net, "
                        f"{s['r_raw']:.2f} ECMWF, {s['r_clim']:.2f} clim.", fontsize=9.5, loc="left")
            a.set_ylabel("mm/day")
            if ci == 0 and k == 0: a.legend(ncol=2, fontsize=8)
    for a in axs[-1]: a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Single 1.5-degree grid cell, weekly-mean rainfall, test years 2021-23", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "06_timeseries_cities.png", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_everything(); summary = {"run": RUN.name, "n_test": len(D["dates"])}
    print("loaded", D["preds"].shape, "valid cells", int(D["mask"].sum()))
    print("sanity: mean obs / U-Net / raw rain over valid cells (mm/day):",
          float(np.nanmean(D["obs"][:, :, D["mask"]])), float(D["preds"][:, :, D["mask"]].mean()),
          float(D["raw"][:, :, D["mask"]].mean()))
    clim = clim_by_doy()

    fig_learning_curves(D, summary)
    fig_gridcell_scatter(D, summary, log_axes=False)
    fig_gridcell_scatter(D, summary, log_axes=True)
    fig_areamean_scatter(D, clim, summary)
    fig_series_allindia(D, clim, summary)
    fig_series_cities(D, clim, summary)

    cases = pick_cases(D)
    summary["cases"] = {k: f"{D['dates'][n]:%Y-%m-%d}" for k, n in cases.items()}
    fig_case_series(D, cases)
    for k, (name, n) in enumerate(cases.items()):
        slug = f"case{k + 1}_{D['dates'][n]:%Y%m%d}"
        t = f"{name}: forecast issued {D['dates'][n]:%Y-%m-%d}"
        fig_case_maps(D, n, t + "  (daily maps)", f"08_{slug}_daily_maps.png", weekly=False)
        fig_case_maps(D, n, t + "  (weekly-mean maps)", f"09_{slug}_weekly_maps.png", weekly=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
