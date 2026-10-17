"""Score Experiment 10's runs against Dyn++ and the headline U-Net.

Same metrics and aligned days 1-28 as Experiment 9 (model/weekly_scores.py),
3 seeds per variant, each also scored with its validation mean fix applied
(stats.json `val_mean_fix`: one obs/pred factor per lead week).

Usage:  python experiments/exp10_unet_on_bias_correction/score_runs.py
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
from preprocess.bias_correction import week_of_day
from preprocess.climatology import ensure_climatology, obs_clim_for_dates
from preprocess.dataset_loader import list_cached_dates, split_dates
from train import padded_lat_array

VARIANTS = ["plain_aligned", "residual", "residual_mm"]
SEEDS = [0, 1, 2]
OUT = HERE / "results"
COLUMNS = ["wk1_acc", "wk2_acc", "wk3_acc", "wk4_acc", "wk1_ai_acc", "wk2_ai_acc", "wk3_ai_acc",
           "wk4_ai_acc", "anom_corr", "corr", "rmse", "bias", "sd_ratio"]


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

    methods = {"Dyn++ (Exp 9)": [dynpp]}
    headline = []
    for s in SEEDS:
        with np.load(config.find_run(f"no_obs_s{s}") / "test_rain_predictions.npz") as z:
            headline.append(obs_side(z["preds"])[:, :28])
    methods["U-Net no_obs (Exp 6, trained misaligned)"] = headline

    wk = week_of_day(config.LEAD_DAYS)
    for v in VARIANTS:
        raw, fixed = [], []
        for s in SEEDS:
            run = HERE / "runs" / f"{v}_s{s}"
            if not (run / "test_rain_predictions.npz").exists():
                continue
            with np.load(run / "test_rain_predictions.npz") as z:
                p = z["preds"]
                assert np.nanmax(np.abs(z["obs"] - obs)) < 1e-3
            fix = np.array(json.loads((run / "stats.json").read_text())["val_mean_fix"], dtype=np.float32)
            raw.append(p)
            fixed.append(p * fix[wk][None, :, None, None])
        if raw:
            methods[f"{v} ({len(raw)} seeds)"] = raw
            methods[f"{v} + val mean fix"] = fixed

    results, spread = {}, {}
    for name, preds in methods.items():
        per_seed = [score(p, obs, clim_mm, clim_log, lat, mask) for p in preds]
        results[name] = {k: float(np.mean([r[k] for r in per_seed])) for k in COLUMNS}
        if len(per_seed) > 1:
            spread[name] = {k: float(np.ptp([r[k] for r in per_seed])) for k in COLUMNS}
        print(f"scored {name}")

    boot = {}
    names = list(methods)
    for name in names[2:]:
        if "mean fix" in name:
            continue
        boot[f"{name} - Dyn++"] = bootstrap_weekly(methods[name], methods["Dyn++ (Exp 9)"], obs, clim_mm, lat, mask)
        plain = [n for n in names if n.startswith("plain_aligned (")]
        if plain and not name.startswith("plain_aligned"):
            boot[f"{name} - plain_aligned"] = bootstrap_weekly(methods[name], methods[plain[0]], obs, clim_mm, lat, mask)

    json.dump({"n_test": len(test_dates), "metrics": results, "seed_spread": spread, "bootstrap": boot},
              open(OUT / "summary.json", "w"), indent=1)
    lines = [format_table(results, COLUMNS), "", "Seed spread (max - min over seeds):",
             format_table(spread, COLUMNS), "", "Paired date bootstrap, difference [95% range]:"]
    for name, b in boot.items():
        lines.append(f"  {name}: " + "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]"
                                               for k, v in b.items() if "rmse" not in k))
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
