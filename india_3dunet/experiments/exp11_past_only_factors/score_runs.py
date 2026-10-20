"""Experiment 11, step 3: score residual_mm trained with past-only factors.

Compares, on the 153 test forecasts (aligned days 1-28, same metrics as
Experiments 9-10):

  Dyn++                          Exp 9
  residual_mm, leave-one-year-out   Exp 10 runs (3 seeds)
  residual_mm, past only            this experiment's runs
  residual_mm, leave-one-year-out restricted to the seeds this experiment has,
                                    so the comparison is like for like

Usage:  python experiments/exp11_past_only_factors/score_runs.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np

import config
from model.weekly_scores import bootstrap_weekly, format_table, mm_clim_for_dates, mm_climatology, score
from preprocess.climatology import ensure_climatology, obs_clim_for_dates
from preprocess.dataset_loader import list_cached_dates, split_dates
from train import padded_lat_array

EXP10 = config.EXPERIMENTS_ROOT / "exp10_unet_on_bias_correction" / "runs"
RUNS = {"residual_mm, leave-one-year-out (Exp 10)": EXP10 / "residual_mm_s{}",
        "residual_mm, past only (Exp 11)": HERE / "runs" / "residual_mm_past_s{}"}
SEEDS = [0, 1, 2]
REF = "residual_mm, leave-one-year-out (Exp 10)"
OUT = HERE / "results"
COLUMNS = ["wk1_acc", "wk2_acc", "wk3_acc", "wk4_acc", "wk1_ai_acc", "wk2_ai_acc", "wk3_ai_acc",
           "wk4_ai_acc", "wk1_rmse", "wk2_rmse", "wk3_rmse", "wk4_rmse", "anom_corr", "rmse", "bias", "sd_ratio"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_dates = list_cached_dates("rain")
    splits = split_dates(all_dates)
    test_dates = splits["test"]
    lat = padded_lat_array()

    with np.load(config.EXPERIMENTS_ROOT / "exp09_bias_correction" / "results" / "dynpp_test_predictions.npz") as z:
        dynpp, obs, mask = z["preds"], z["obs"], z["mask"]
    clim_mm = mm_clim_for_dates(mm_climatology(), test_dates, config.LEAD_DAYS)
    obs_clim, _ = ensure_climatology("rain", splits["train"], all_dates)
    clim_log = obs_clim_for_dates(obs_clim, test_dates, config.LEAD_DAYS, offset=1)

    methods, epochs = {"Dyn++ (Exp 9)": [dynpp]}, {}
    for name, pattern in RUNS.items():
        preds = []
        for s in SEEDS:
            run = Path(str(pattern).format(s))
            if not (run / "test_rain_predictions.npz").exists():
                continue
            with np.load(run / "test_rain_predictions.npz") as z:
                assert np.nanmax(np.abs(z["obs"] - obs)) < 1e-3
                preds.append(z["preds"])
            epochs.setdefault(name, []).append((run / "best_epoch").read_text().strip())
        if preds:
            methods[name] = preds
    # like-for-like: Exp 10 restricted to the seeds Exp 11 has
    n_past = len(methods.get("residual_mm, past only (Exp 11)", []))
    if 0 < n_past < len(methods[REF]):
        methods[f"{REF}, same {n_past} seed(s)"] = methods[REF][:n_past]

    results, spread = {}, {}
    for name, preds in methods.items():
        per_seed = [score(p, obs, clim_mm, clim_log, lat, mask) for p in preds]
        cols = [c for c in COLUMNS if c in per_seed[0]]
        results[name] = {k: float(np.mean([r[k] for r in per_seed])) for k in cols}
        if len(per_seed) > 1:
            spread[name] = {k: float(np.ptp([r[k] for r in per_seed])) for k in cols}
        print(f"scored {name} ({len(preds)} seeds)")
    cols = list(next(iter(results.values())))

    past = "residual_mm, past only (Exp 11)"
    boot = {f"{n} - Dyn++": bootstrap_weekly(methods[n], methods["Dyn++ (Exp 9)"], obs, clim_mm, lat, mask)
            for n in methods if n != "Dyn++ (Exp 9)"}
    if past in methods:
        ref = [n for n in methods if n.startswith(REF)][-1]   # same seeds when restricted
        boot[f"past only - {ref}"] = bootstrap_weekly(methods[past], methods[ref], obs, clim_mm, lat, mask)

    json.dump({"n_test": len(test_dates), "best_epochs": epochs, "metrics": results, "seed_spread": spread,
               "bootstrap": boot}, open(OUT / "summary.json", "w"), indent=1)
    lines = [format_table(results, cols), "", "Seed spread (max - min over seeds):", format_table(spread, cols),
             "", f"Best epochs: {epochs}", "", "Paired date bootstrap, difference [95% range]:"]
    for name, b in boot.items():
        lines.append(f"  {name}: " + "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]" for k, v in b.items()))
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
