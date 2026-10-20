"""Experiment 12: does ECMWF's rain bias change smoothly through the year?

No training. Dyn++ assumes a forecast's bias in July looks like its bias in
June and August, which is why its factors pool issue months within +/-1 month.
This checks that assumption three ways, on the 2004-2017 training forecasts:

1. Seasonal cycle. Factors (observed / ECMWF, as in Dyn++) fitted on one issue
   month at a time; the land-median factor per month and lead week.
2. Pattern or noise? Single-month factor maps fitted separately on even and
   odd years, so every comparison uses independent forecasts. Map correlation
   of month m (even years) with month m+k (odd years), for k = 0..6. k = 0 is
   the ceiling: how alike the same month looks in two independent samples.
   If the bias is seasonal, similarity falls slowly with k; if it were noise,
   k = 1 would already be as low as k = 6.
3. Window width. Dyn++ with windows of 0, +/-1, +/-2, +/-3 months and all
   year, scored on validation (2018-20) and test (2021-23). The window should
   be picked on validation.

Writes results/summary.json, results/table.md and results/seasonal_bias.png.

Usage:  python experiments/exp12_seasonal_bias_check/seasonal_bias.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from model.weekly_scores import mm_clim_for_dates, mm_climatology, score
from preprocess.bias_correction import apply_dynpp, fit_dynpp, load_aligned
from preprocess.climatology import ensure_climatology, obs_clim_for_dates
from preprocess.dataset_loader import list_cached_dates, split_dates
from train import padded_lat_array

OUT = HERE / "results"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
WINDOWS = {"same month only": 0, "+/-1 month (used)": 1, "+/-2 months": 2, "+/-3 months": 3, "all year": 6}
WEEKLY = ["wk1_acc", "wk2_acc", "wk3_acc", "wk4_acc", "wk1_ai_acc", "wk2_ai_acc", "wk3_ai_acc", "wk4_ai_acc"]


def single_month(tp, obs, dates, mask, keep):
    """Single-month factors fitted on the dates where `keep` is true."""
    idx = np.flatnonzero(keep)
    return fit_dynpp(tp[idx], obs[idx], [dates[i] for i in idx], mask, half_window=0)["full"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_dates = list_cached_dates("rain")
    splits = split_dates(all_dates)
    tr = splits["train"]
    tp, obs = load_aligned(tr)
    with np.load(config.CACHE_ROOT / "samples" / "rain" / f"{tr[0]:%Y_%m_%d}.npz") as z:
        mask = z["mask"].astype(bool)
    lat = padded_lat_array()
    summary, lines = {}, []

    # 1. seasonal cycle of the land-median factor
    single = single_month(tp, obs, tr, mask, np.ones(len(tr), bool))
    med = np.median(single[..., mask], axis=-1)                       # (12, 4)
    summary["median_factor"] = {MONTHS[m]: med[m].tolist() for m in range(12)}
    lines += ["## 1. ECMWF's bias through the year",
              "Factor = (observed + 1) / (ECMWF + 1) on weekly totals, fitted on one issue month at a time "
              "(2004-2017). Median over Indian land cells; above 1 means ECMWF is too dry.", "",
              "| issue month | wk1 | wk2 | wk3 | wk4 |", "|---|---|---|---|---|"]
    lines += [f"| {MONTHS[m]} | " + " | ".join(f"{v:.2f}" for v in med[m]) + " |" for m in range(12)]

    # 2. map similarity vs month gap, even years vs odd years
    years = np.array([d.year for d in tr])
    ev = np.log(single_month(tp, obs, tr, mask, years % 2 == 0)[..., mask])
    od = np.log(single_month(tp, obs, tr, mask, years % 2 == 1)[..., mask])
    r = np.array([[[np.corrcoef(ev[m, w], od[(m + k) % 12, w])[0, 1] for m in range(12)]
                   for w in range(4)] for k in range(7)])                # (gap, week, month)
    summary["map_corr_by_gap"] = {k: r[k].mean(axis=1).tolist() for k in range(7)}
    summary["map_corr_by_gap_monsoon"] = {k: r[k][:, 5:9].mean(axis=1).tolist() for k in range(7)}
    lines += ["", "## 2. Is the bias a seasonal pattern or noise?",
              "Correlation over land of log factor maps: month m fitted on even years vs month m+k fitted "
              "on odd years (independent forecasts). Gap 0 = how alike the same month is in two "
              "independent samples (the ceiling). Mean over all 12 months; June-September in brackets.", "",
              "| months apart | wk1 | wk2 | wk3 | wk4 |", "|---|---|---|---|---|"]
    for k in range(7):
        lines.append(f"| {k} | " + " | ".join(f"{r[k, w].mean():.2f} ({r[k, w, 5:9].mean():.2f})"
                                               for w in range(4)) + " |")

    # 3. window width, scored on validation and test
    lines += ["", "## 3. Dyn++ with different month windows",
              "Weekly anomaly correlation (cell, then all-India). Choose on validation; test for reference.", ""]
    fits = {name: fit_dynpp(tp, obs, tr, mask, half_window=hw) for name, hw in WINDOWS.items()}
    mm_clim = mm_climatology()
    obs_clim, _ = ensure_climatology("rain", tr, all_dates)
    summary["windows"] = {}
    for split in ("val", "test"):
        dates = splits[split]
        tps, obss = load_aligned(dates)
        clim_mm = mm_clim_for_dates(mm_clim, dates, config.LEAD_DAYS)
        clim_log = obs_clim_for_dates(obs_clim, dates, config.LEAD_DAYS, offset=1)
        lines += [f"**{split}** ({len(dates)} forecasts, {dates[0].year}-{dates[-1].year})", "",
                  "| window | cell wk1 | wk2 | wk3 | wk4 | all-India wk1 | wk2 | wk3 | wk4 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        summary["windows"][split] = {}
        for name, f in fits.items():
            s = score(apply_dynpp(tps, dates, f), obss, clim_mm, clim_log, lat, mask)
            summary["windows"][split][name] = {k: float(s[k]) for k in WEEKLY}
            lines.append(f"| {name} | " + " | ".join(f"{s[k]:.3f}" for k in WEEKLY) + " |")
            print(f"scored {split} {name}")
        lines.append("")

    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for w in range(4):
        ax[0].plot(MONTHS, med[:, w], marker="o", label=f"week {w + 1}")
        ax[1].plot(range(7), r[:, w].mean(axis=1), marker="o", label=f"week {w + 1}")
    ax[0].axhline(1, color="k", lw=0.8, alpha=0.5)
    ax[0].set(title="ECMWF bias by issue month (land median)", ylabel="factor = observed / ECMWF")
    ax[1].set(title="Factor-map similarity, even vs odd years", xlabel="months apart",
              ylabel="map correlation (log factor)")
    for a in ax:
        a.grid(alpha=0.3); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "seasonal_bias.png", dpi=140)


if __name__ == "__main__":
    main()
