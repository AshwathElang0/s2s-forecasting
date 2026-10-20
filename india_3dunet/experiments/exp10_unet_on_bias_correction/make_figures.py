"""Predicted-vs-observed figures for Experiments 9-10 (progress report, page 2).

Compares, on aligned days (forecast day d = IMD date issue+d), the forecasts
that matter after Experiment 10:

  Dyn++          Exp 9 bias correction
  residual       U-Net on top of Dyn++, trained in log space (seed 0)
  residual_mm    U-Net on top of Dyn++, trained in mm/day (seed 0)

with IMD observed rain and the 1951-2020 climatology. Same layout and helpers
as Experiment 8's review figures (exp08_review_diagnostics/make_review_figures.py),
so the plots read the same way.

Writes PNGs and summary.json to figures/ next to this script. Experiment 11's
make_figures.py reuses this with its final residual_mm (past-only factors).

Usage:  python experiments/exp10_unet_on_bias_correction/make_figures.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "exp08_review_diagnostics"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import config
from make_review_figures import CITIES, area_mean, clim_by_doy, clim_for, crop, pick_cases
from preprocess.alignment import valid_date
from preprocess.dataset_loader import list_cached_dates, split_dates
from train import padded_lat_array

OUT = HERE / "figures"
RUNS = {  # label: run folder (seed 0)
    "residual (log space)": HERE / "runs" / "residual_s0",
    "residual_mm (mm/day)": HERE / "runs" / "residual_mm_s0",
}
N_DAYS = 28
C_OBS, C_CLIM = "#222222", "#9a9a9a"
MODELS = {  # label: (colour, line width)
    "Dyn++ (bias correction)": ("#1b9e77", 1.3),
    "residual (log space)": ("#7570b3", 1.3),
    "residual_mm (mm/day)": ("#d95f02", 1.8),
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False,
})


def load():
    dates = split_dates(list_cached_dates("rain"))["test"]
    with np.load(config.EXPERIMENTS_ROOT / "exp09_bias_correction" / "results" / "dynpp_test_predictions.npz") as z:
        dyn, obs, mask = z["preds"][:, :N_DAYS], z["obs"][:, :N_DAYS], z["mask"]
    preds = {"Dyn++ (bias correction)": dyn}
    for lab, run in RUNS.items():
        with np.load(run / "test_rain_predictions.npz") as z:
            preds[lab] = z["preds"][:, :N_DAYS]
    lat = padded_lat_array()
    lon = np.linspace(config.NATIVE_LON_VALS[0], config.NATIVE_LON_VALS[1], config.NATIVE_LON)
    pad_lon = np.concatenate([
        lon[0] - 1.5 * np.arange(config.PAD_LEFT, 0, -1), lon,
        lon[-1] + 1.5 * np.arange(1, config.PAD_LON - config.NATIVE_LON - config.PAD_LEFT + 1)])
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    return dict(dates=dates, obs=obs, preds=preds, mask=mask, lat=lat, lon=pad_lon, w=w)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def fig_allindia_by_week(D, clim, summary):
    summary["allindia_anomaly_r"] = {}
    fig, axs = plt.subplots(4, 1, figsize=(14, 11.5), sharex=True)
    for j, a in enumerate(axs):
        days = np.arange(7 * j, 7 * j + 7)
        vd = [valid_date(d, int(days.mean())) for d in D["dates"]]
        o = area_mean(D["obs"][:, days], D["w"]).mean(1)
        c = area_mean(clim_for(D, clim, days), D["w"]).mean(1)
        a.plot(vd, c, color=C_CLIM, ls="--", lw=1.2, label="climatology (1951-2020)")
        rs = {}
        for lab, p in D["preds"].items():
            col, lw = MODELS[lab]
            s = area_mean(p[:, days], D["w"]).mean(1)
            a.plot(vd, s, color=col, lw=lw, label=lab)
            rs[lab] = corr(s - c, o - c)
        a.plot(vd, o, color=C_OBS, lw=2, label="IMD observed")
        summary["allindia_anomaly_r"][f"week{j + 1}"] = rs
        a.set_title(f"Week {j + 1} (days {days[0] + 1}-{days[-1] + 1}).  Departure-from-normal correlation:  "
                    + ",  ".join(f"{k.split(' ')[0]} {v:.2f}" for k, v in rs.items()), fontsize=9.5, loc="left")
        a.set_ylabel("mm/day")
        if j == 0: a.legend(ncol=5, fontsize=8.5, loc="upper left")
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("All-India mean rainfall, weekly average at each lead, test years 2021-23 "
                 "(x-axis = date the week is valid)", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "01_allindia_timeseries_by_week.png", bbox_inches="tight"); plt.close(fig)


def fig_cities(D, clim, summary):
    summary["cities_r"] = {}
    la, lo = D["lat"], D["lon"]
    fig, axs = plt.subplots(len(CITIES), 2, figsize=(15, 3 * len(CITIES)), sharex=True)
    for ci, (city, (clat, clon)) in enumerate(CITIES.items()):
        r_i, c_i = np.argmin(abs(la - clat)), np.argmin(abs(lo - clon))
        for k, wk in enumerate([0, 2]):
            a = axs[ci, k]; days = np.arange(7 * wk, 7 * wk + 7)
            vd = [valid_date(d, int(days.mean())) for d in D["dates"]]
            o = np.nanmean(D["obs"][:, days, r_i, c_i], axis=1)
            c = clim_for(D, clim, days)[:, :, r_i, c_i].mean(1)
            a.plot(vd, c, color=C_CLIM, ls="--", lw=1.1, label="climatology")
            rs = {}
            for lab, p in D["preds"].items():
                col, lw = MODELS[lab]
                s = p[:, days, r_i, c_i].mean(1)
                a.plot(vd, s, color=col, lw=lw * 0.9, label=lab)
                rs[lab] = corr(s - c, o - c)
            a.plot(vd, o, color=C_OBS, lw=1.7, label="IMD observed")
            summary["cities_r"][f"{city}_week{wk + 1}"] = rs
            a.set_title(f"{city}, week {wk + 1}.  Departure corr: "
                        + ", ".join(f"{k.split(' ')[0]} {v:.2f}" for k, v in rs.items()), fontsize=9.5, loc="left")
            a.set_ylabel("mm/day")
            if ci == 0 and k == 0: a.legend(ncol=2, fontsize=8)
    for a in axs[-1]: a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Single 1.5-degree grid cell, weekly-mean rainfall, test years 2021-23", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "02_city_timeseries.png", bbox_inches="tight"); plt.close(fig)


def fig_case_series(D, cases):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    days = np.arange(1, N_DAYS + 1)
    for a, (name, n) in zip(axs.flat, cases.items()):
        for lab, p in D["preds"].items():
            col, lw = MODELS[lab]
            a.plot(days, area_mean(p[n], D["w"]), color=col, lw=lw, label=lab)
        a.plot(days, area_mean(D["obs"][n], D["w"]), color=C_OBS, lw=2, label="IMD observed")
        for L in (7, 14, 21, 28): a.axvline(L, color="k", alpha=0.12)
        a.set_title(f"{name}\nissued {D['dates'][n]:%Y-%m-%d}", fontsize=9.5)
        a.set_xlabel("forecast day"); a.set_ylabel("all-India mean rain (mm/day)")
    axs.flat[0].legend(fontsize=8)
    fig.suptitle("Daily all-India mean rainfall over 28 forecast days, the six Experiment 8 test cases", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "03_case_timeseries.png", bbox_inches="tight"); plt.close(fig)


def fig_case_maps(D, n, title):
    rs, cs = crop(D)
    ext = [D["lon"][cs][0] - .75, D["lon"][cs][-1] + .75, D["lat"][rs][-1] - .75, D["lat"][rs][0] + .75]
    mask = D["mask"][rs, cs]
    cols = [("IMD observed", D["obs"])] + list(D["preds"].items())
    fig, ax = plt.subplots(4, 4, figsize=(13.5, 14), constrained_layout=True)
    for i in range(4):
        sl = np.arange(7 * i, 7 * i + 7)
        fields = [np.where(mask, np.nanmean(arr[n, sl][:, rs, cs], axis=0), np.nan) for _, arr in cols]
        vmax = max(np.nanpercentile(np.concatenate([f[mask] for f in fields]), 99), 5)
        for j, ((lab, _), f) in enumerate(zip(cols, fields)):
            a = ax[i, j]; a.grid(False)
            im = a.imshow(f, extent=ext, cmap="YlGnBu", vmin=0, vmax=vmax, origin="upper")
            if i == 0: a.set_title(lab)
            if j == 0: a.set_ylabel(f"week {i + 1} mean\n(days {sl[0] + 1}-{sl[-1] + 1})", fontsize=11)
            if j > 0:
                a.text(0.02, 0.02, f"r = {corr(f[mask], fields[0][mask]):.2f}", transform=a.transAxes,
                       fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8))
            a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=ax[i, -1], shrink=0.75, pad=0.02, label="mm/day")
    fig.suptitle(title, fontsize=12)
    fig.savefig(OUT / "04_case1_weekly_maps.png", dpi=120); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    D = load(); summary = {}
    clim = clim_by_doy()
    fig_allindia_by_week(D, clim, summary)
    fig_cities(D, clim, summary)
    cases = pick_cases(D)
    summary["cases"] = {k: f"{D['dates'][n]:%Y-%m-%d}" for k, n in cases.items()}
    fig_case_series(D, cases)
    name, n = next(iter(cases.items()))
    fig_case_maps(D, n, f"{name}: forecast issued {D['dates'][n]:%Y-%m-%d} (weekly-mean maps; r = map correlation with observed)")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
