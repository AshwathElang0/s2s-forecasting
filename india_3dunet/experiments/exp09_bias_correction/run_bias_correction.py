"""Experiment 9: how far does plain bias correction get, with no neural network?

Fits Dyn++ (preprocess/bias_correction.py) on the training years and scores
it on the 153 test forecasts against the reference forecasts and the
existing U-Nets, all on aligned days (ECMWF lead t vs IMD issue+1+t):

  raw ECMWF           the 11-member mean, unchanged
  climatology         1951-2020 day-of-year normal, no forecast
  debiased (exp02)    obs normal + (log1p(ECMWF) - ECMWF's usual forecast), log space
  Dyn++               ECMWF x (observed / forecast) factor per cell, lead week, month
  ABC-lite            1/2 Dyn++ + 1/2 climatology. Its anomaly is half Dyn++'s,
                      so every correlation equals Dyn++'s; only RMSE and bias differ
  U-Net no_obs        the headline 16-input U-Net (Exp 6), 3 seeds
  U-Net departure     the departure-from-normal U-Net (Exp 3), 3 seeds

The U-Nets were trained on the one-day-early cache target, so their output
t+1 is the one scored against forecast day t+1 (preprocess/alignment.py).

Also writes the Dyn++ factors (full fit + leave-one-year-out) to
data_cache/bias_correction/dynpp_factors.npz for Experiment 10.

Usage:  python experiments/exp09_bias_correction/run_bias_correction.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np

import config
from model.weekly_scores import bootstrap_weekly, format_table, mm_clim_for_dates, mm_climatology, score
from preprocess.alignment import obs_side
from preprocess.bias_correction import BC_ROOT, apply_dynpp, fit_dynpp, load_aligned, save_factors
from preprocess.climatology import ensure_climatology, model_clim_for_dates, obs_clim_for_dates
from preprocess.dataset_loader import list_cached_dates, split_dates
from train import padded_lat_array

OUT = HERE / "results"
UNETS = {"U-Net no_obs (Exp 6)": "no_obs_s{}", "U-Net departure (Exp 3)": "departure_s{}"}
COLUMNS = ["wk1_acc", "wk2_acc", "wk3_acc", "wk4_acc", "wk1_ai_acc", "wk2_ai_acc", "wk3_ai_acc",
           "wk4_ai_acc", "anom_corr", "corr", "rmse", "bias", "sd_ratio"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_dates = list_cached_dates("rain")
    splits = split_dates(all_dates)
    train_dates, test_dates = splits["train"], splits["test"]

    print("loading aligned ECMWF / IMD pairs...")
    tp_tr, obs_tr = load_aligned(train_dates)
    tp_te, obs_te = load_aligned(test_dates)
    with np.load(config.CACHE_ROOT / "samples" / "rain" / f"{test_dates[0]:%Y_%m_%d}.npz") as z:
        mask = z["mask"]
    lat = padded_lat_array()

    factors = fit_dynpp(tp_tr, obs_tr, train_dates, mask)
    save_factors(factors, BC_ROOT / "dynpp_factors.npz")
    m = mask.astype(bool)
    f = factors["full"][..., m]
    print(f"Dyn++ factors over land: median {np.median(f):.2f}, "
          f"{(f <= 0.25).mean():.1%} at the 0.25 floor, {(f >= 4).mean():.1%} at the 4.0 cap")

    clim_mm = mm_clim_for_dates(mm_climatology(), test_dates, config.LEAD_DAYS)
    obs_clim, model_clim = ensure_climatology("rain", train_dates, all_dates)
    clim_log = obs_clim_for_dates(obs_clim, test_dates, config.LEAD_DAYS, offset=1)

    dynpp = apply_dynpp(tp_te, test_dates, factors)
    np.savez(OUT / "dynpp_test_predictions.npz", preds=dynpp.astype(np.float32), obs=obs_te, mask=mask,
             dates=np.array([f"{d:%Y-%m-%d}" for d in test_dates]))

    methods = {
        "raw ECMWF": [tp_te],
        "climatology": [clim_mm],
        "debiased (exp02, log space)": [np.expm1(clim_log + np.log1p(np.clip(tp_te, 0, None))
                                                 - model_clim_for_dates(model_clim, test_dates))],
        "Dyn++": [dynpp],
        "ABC-lite (1/2 Dyn++ + 1/2 clim)": [0.5 * dynpp + 0.5 * clim_mm],
    }
    for name, pattern in UNETS.items():
        seeds = []
        for s in range(3):
            with np.load(config.find_run(pattern.format(s)) / "test_rain_predictions.npz") as z:
                p, o = obs_side(z["preds"]), obs_side(z["obs"])
            gap = np.nanmax(np.abs(o[:, :28] - obs_te[:, :28]))
            assert len(p) == len(test_dates) and gap < 1e-3, (pattern, len(p), gap)
            seeds.append(p[:, :28])
        methods[name] = seeds

    results, spread = {}, {}
    for name, preds in methods.items():
        per_seed = [score(p, obs_te, clim_mm, clim_log, lat, mask) for p in preds]
        results[name] = {k: float(np.mean([r[k] for r in per_seed])) for k in COLUMNS}
        if len(per_seed) > 1:
            spread[name] = {k: float(np.ptp([r[k] for r in per_seed])) for k in COLUMNS}
        print(f"scored {name}")

    boot = {
        "Dyn++ - raw ECMWF": bootstrap_weekly(methods["Dyn++"], methods["raw ECMWF"], obs_te, clim_mm, lat, mask),
        "U-Net no_obs - Dyn++": bootstrap_weekly(methods["U-Net no_obs (Exp 6)"], methods["Dyn++"], obs_te, clim_mm, lat, mask),
        "U-Net departure - Dyn++": bootstrap_weekly(methods["U-Net departure (Exp 3)"], methods["Dyn++"], obs_te, clim_mm, lat, mask),
        "Dyn++ - debiased (exp02)": bootstrap_weekly(methods["Dyn++"], methods["debiased (exp02, log space)"], obs_te, clim_mm, lat, mask),
    }

    json.dump({"n_test": len(test_dates), "metrics": results, "seed_spread": spread, "bootstrap": boot},
              open(OUT / "summary.json", "w"), indent=1)
    table = format_table(results, COLUMNS)
    lines = [table, "", "Paired date bootstrap, difference [95% range]:"]
    for name, b in boot.items():
        lines.append(f"  {name}: " + "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]"
                                               for k, v in b.items() if "rmse" not in k))
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n{len(test_dates)} test forecasts. Saved under {OUT}")


if __name__ == "__main__":
    main()
