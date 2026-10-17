"""Rescore every U-Net run from Experiments 2-7 on aligned days.

Experiments 1-8 scored ECMWF lead t against IMD date issue+t, one day early
(preprocess/alignment.py). This rescores the saved test predictions with
forecast day d = IMD date issue+d, and rebuilds each experiment's comparison:

  daily    pattern corr, climatology-removed corr (log space), RMSE, E_pre,
           SD ratio, as in the original tables; 31 aligned days
  weekly   Experiment 9's weekly-total scores (model/weekly_scores.py),
           days 1-28, so every model sits next to Dyn++

The reference forecasts (raw, climatology, log-debiased ECMWF) were already
rescored aligned by Exp 2's compute_baselines.py: results/aligned/baselines.log.

Per-run scores are cached in data_cache/aligned_rescore/; delete it to recompute.

Usage:  python experiments/rescore_aligned.py
Writes: experiments/aligned_rescore.md
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np

import config

sys.stdout.reconfigure(encoding="utf-8")
from model.evaluation import rmses, scorr
from model.evaluation_fixed import epre_fixed, variance_ratio
from model.weekly_scores import bootstrap_weekly, mm_clim_for_dates, mm_climatology, score
from preprocess.alignment import obs_side
from preprocess.climatology import ensure_climatology, obs_clim_for_dates, transform
from preprocess.dataset_loader import list_cached_dates, split_dates
from train import padded_lat_array

CACHE = config.REPO_ROOT / "data_cache" / "aligned_rescore"
OUT = HERE / "aligned_rescore.md"
DAYS = [7, 14, 21, 28]
DAILY = ("corr", "corr_anom", "rmse", "epre", "sd_ratio")
WEEKLY = ["wk1_acc", "wk2_acc", "wk3_acc", "wk4_acc", "wk1_ai_acc", "wk2_ai_acc", "wk3_ai_acc", "wk4_ai_acc"]


def seeds(name):
    return [f"{name}_s{s}" for s in range(3)]


SETUPS = {
    "Baseline (Exp 1, n724)": ["n724"],
    "rain only (5)": ["rain_only_s0"],
    "+ humidity (9)": ["humidity_s0"],
    "+ winds (13)": ["winds_s0"],
    "all inputs (17)": seeds("all_inputs"),
    "departure from normal": seeds("departure"),
    "gate": seeds("gate"),
    "no yesterday's rain (16, headline)": seeds("no_obs"),
    "ECMWF only (14)": seeds("ecmwf_only"),
}

all_dates = list_cached_dates("rain")
splits = split_dates(all_dates)
test_dates = splits["test"]
lat = padded_lat_array()
obs_clim, _ = ensure_climatology("rain", splits["train"], all_dates)
clim_log_31 = obs_side(obs_clim_for_dates(obs_clim, test_dates, config.LEAD_DAYS))
clim_log_28 = obs_clim_for_dates(obs_clim, test_dates, config.LEAD_DAYS, offset=1)
clim_mm = mm_clim_for_dates(mm_climatology(), test_dates, config.LEAD_DAYS)
with np.load(config.EXPERIMENTS_ROOT / "exp09_bias_correction" / "results" / "dynpp_test_predictions.npz") as z:
    dynpp, obs_w, mask_w = z["preds"], z["obs"], z["mask"]


def load_preds(run):
    with np.load(config.find_run(run) / "test_rain_predictions.npz") as z:
        p, o, m = obs_side(z["preds"]), obs_side(z["obs"]), z["mask"]
    assert len(p) == len(test_dates), run
    return p, o, m


def daily_per_sample(run):
    """{metric: (n_samples, 31)} on aligned days, cached."""
    path = CACHE / f"{run}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in DAILY}
    p, o, m = load_preds(run)
    pa, oa = transform(p, "rain") - clim_log_31, transform(o, "rain") - clim_log_31
    n, L = p.shape[:2]
    out = {k: np.full((n, L), np.nan) for k in DAILY}
    for i in range(n):
        for t in range(L):
            out["corr"][i, t] = scorr(p[i, t], o[i, t], lat, m)
            out["corr_anom"][i, t] = scorr(pa[i, t], oa[i, t], lat, m)
            out["rmse"][i, t] = rmses(p[i, t], o[i, t], lat, m)
            out["epre"][i, t] = epre_fixed(p[i, t], o[i, t], lat, m)
            out["sd_ratio"][i, t] = variance_ratio(p[i, t], o[i, t], lat, m)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def weekly_preds(run):
    return load_preds(run)[0][:, :28]


daily = {s: [daily_per_sample(r) for r in runs] for s, runs in SETUPS.items()}
print("daily scores done", flush=True)


def mean(setup, metric, days=None):
    """Seed-mean of the date-and-day mean; days are 1-based, inclusive (a, b)."""
    sl = slice(None) if days is None else slice(days[0] - 1, days[1])
    return float(np.mean([np.nanmean(r[metric][:, sl]) for r in daily[setup]]))


def at_day(setup, metric, d):
    # SD ratio: median over forecasts; the mean is dominated by near-dry maps with tiny observed spread
    f = np.nanmedian if metric == "sd_ratio" else np.nanmean
    return float(np.mean([f(r[metric][:, d - 1]) for r in daily[setup]]))


def seed_range(setup, metric):
    v = [np.nanmean(r[metric]) for r in daily[setup]]
    return f" ({min(v):.3f}–{max(v):.3f})" if len(v) > 1 else ""


def boot(a, b, metric, days=None, n_boot=10000):
    """Paired date bootstrap of seed-averaged per-sample scores: (diff, lo, hi)."""
    sl = slice(None) if days is None else slice(days[0] - 1, days[1])
    xa = np.mean([np.nanmean(r[metric][:, sl], axis=1) for r in daily[a]], axis=0)
    xb = np.mean([np.nanmean(r[metric][:, sl], axis=1) for r in daily[b]], axis=0)
    idx = np.random.default_rng(0).integers(0, len(xa), (n_boot, len(xa)))
    d = (xa - xb)[idx].mean(axis=1)
    return float(np.mean(xa - xb)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt_boot(t, nd=3):
    return f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


md = ["# Experiments 2–7 rescored on aligned days", "",
      "Generated by `experiments/rescore_aligned.py`. Forecast day d = ECMWF lead d−1 against IMD date issue+d.",
      "Daily scores: 31 aligned days, 153 test forecasts. Weekly scores: days 1–28, as in Experiments 9–10.",
      "Seeded setups: mean of 3 seeds, brackets = lowest–highest seed.", ""]

# --- Exp 2 / Exp 4: the Baseline model ---------------------------------------
B = "Baseline (Exp 1, n724)"
md += ["## Experiment 2 — Baseline model (references: exp02 results/aligned/baselines.log)", "",
       table(["", "corr, normal", "corr, clim. removed", "RMSE, mm/day"],
             [[B, f"{mean(B, 'corr'):.3f}", f"{mean(B, 'corr_anom'):.3f}", f"{mean(B, 'rmse'):.2f}"]]), ""]
md += ["## Experiment 4 — E_pre and SD ratio, Baseline model", "",
       table(["", *[f"day {d}" for d in DAYS]],
             [[lab, *[f"{at_day(B, k, d):.{nd}f}" for d in DAYS]]
              for lab, k, nd in [("pattern corr", "corr", 3), ("RMSE (mm)", "rmse", 2),
                                 ("E_pre", "epre", 2), ("SD ratio (median)", "sd_ratio", 2)]]), ""]

# --- Exp 3 -------------------------------------------------------------------
A17, DEP, GATE = "all inputs (17)", "departure from normal", "gate"
md += ["## Experiment 3 — departure from normal", "",
       table(["run", "corr, clim. removed", *[f"day {d}" for d in DAYS], "corr, standard", "RMSE"],
             [[s, f"{mean(s, 'corr_anom'):.3f}{seed_range(s, 'corr_anom')}",
               *[f"{at_day(s, 'corr_anom', d):.3f}" for d in DAYS],
               f"{mean(s, 'corr'):.3f}", f"{mean(s, 'rmse'):.2f}"] for s in (A17, DEP, GATE)]), "",
       f"- departure − Baseline (17), clim. removed: {fmt_boot(boot(DEP, A17, 'corr_anom'))}",
       f"- gate − departure, clim. removed: {fmt_boot(boot(GATE, DEP, 'corr_anom'))}",
       f"- departure − Baseline (17), standard corr: {fmt_boot(boot(DEP, A17, 'corr'))}", ""]

# --- Exp 5 -------------------------------------------------------------------
E5 = ["rain only (5)", "+ humidity (9)", "+ winds (13)", A17]
md += ["## Experiment 5 — input variables (seed 0 for each, to match the original)", ""]
for metric, nd, lab in [("corr", 3, "pattern corr, normal scoring"), ("corr_anom", 3, "corr, clim. removed"),
                        ("rmse", 2, "RMSE, mm/day")]:
    rows = []
    for s in E5:
        r = daily[s][0]
        rows.append([s, *[f"{np.nanmean(r[metric][:, d - 1]):.{nd}f}" for d in DAYS], f"{np.nanmean(r[metric]):.{nd}f}"])
    md += [f"{lab}:", "", table(["run", *[f"day {d}" for d in DAYS], "avg (31 days)"], rows), ""]

# --- Exp 6 (now 3 seeds each) ------------------------------------------------
NO = "no yesterday's rain (16, headline)"
rows = []
for lab, metric, days, nd in [("corr, standard, all days", "corr", None, 3),
                              ("corr, clim. removed, days 1–3", "corr_anom", (1, 3), 3),
                              ("corr, clim. removed, days 8–14", "corr_anom", (8, 14), 3),
                              ("corr, clim. removed, days 15–31", "corr_anom", (15, 31), 3),
                              ("corr, clim. removed, all days", "corr_anom", None, 3),
                              ("RMSE, mm/day, all days", "rmse", None, 2)]:
    rows.append([lab, f"{mean(A17, metric, days):.{nd}f}", f"{mean(NO, metric, days):.{nd}f}",
                 fmt_boot(boot(A17, NO, metric, days), nd)])
md += ["## Experiment 6 — yesterday's observed rain (3 seeds each)", "",
       table(["score", "with (17)", "without (16)", "with − without [95%]"], rows), "",
       f"Seed range, clim. removed: with{seed_range(A17, 'corr_anom')}, without{seed_range(NO, 'corr_anom')}", ""]

# --- Exp 7 -------------------------------------------------------------------
EC = "ECMWF only (14)"
md += ["## Experiment 7 — day-of-year input (3 seeds each)", "",
       table(["score", "with calendar (16)", "without (14)", "with − without [95%]"],
             [[lab, f"{mean(NO, m):.{nd}f}{seed_range(NO, m)}", f"{mean(EC, m):.{nd}f}{seed_range(EC, m)}",
               fmt_boot(boot(NO, EC, m), nd)]
              for lab, m, nd in [("corr, standard", "corr", 3), ("corr, clim. removed", "corr_anom", 3),
                                 ("RMSE, mm/day", "rmse", 2)]]), ""]

# --- weekly scores for every setup --------------------------------------------
print("weekly scores...", flush=True)
weekly, wpreds = {"Dyn++ (Exp 9)": score(dynpp, obs_w, clim_mm, clim_log_28, lat, mask_w)}, {}
for s, runs in SETUPS.items():
    wpreds[s] = [weekly_preds(r) for r in runs]
    per = [score(p, obs_w, clim_mm, clim_log_28, lat, mask_w) for p in wpreds[s]]
    weekly[s] = {k: float(np.mean([x[k] for x in per])) for k in WEEKLY + ["bias"]}
md += ["## Weekly totals, every setup (Experiment 9 scores)", "",
       table(["setup", "cell wk1", "wk2", "wk3", "wk4", "all-India wk1", "wk2", "wk3", "wk4", "bias mm/day"],
             [[s, *[f"{r[k]:.3f}" for k in WEEKLY], f"{r['bias']:+.2f}"] for s, r in weekly.items()]), ""]

lines = []
for a, b in [(DEP, A17), (A17, NO), (NO, EC)]:
    bw = bootstrap_weekly(wpreds[a], wpreds[b], obs_w, clim_mm, lat, mask_w)
    lines.append(f"- {a} − {b}: " + ", ".join(f"{k} {fmt_boot(bw[k])}" for k in WEEKLY))
md += ["Weekly paired bootstrap (seed-averaged):", "", *lines, ""]

OUT.write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
