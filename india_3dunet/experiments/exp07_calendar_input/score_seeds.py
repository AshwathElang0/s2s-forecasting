"""Score the calendar ablation: 16 inputs (with doy) vs 14 (no doy), 3 seeds each."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

import config
from preprocess.climatology import ensure_climatology, obs_clim_for_dates, transform
from model.evaluation import rmses, scorr
from preprocess.alignment import obs_side
from preprocess.dataset_loader import list_cached_dates, split_dates
from train import padded_lat_array

all_dates = list_cached_dates("rain")
splits = split_dates(all_dates)
test_dates = splits["test"]
obs_clim, _ = ensure_climatology("rain", splits["train"], all_dates)
# aligned: forecast day d = IMD issue+d (preprocess/alignment.py)
clim = obs_side(obs_clim_for_dates(obs_clim, test_dates, config.LEAD_DAYS))
lat = padded_lat_array()

SETUPS = {"with calendar (16)": "no_obs_s{}", "no calendar (14)": "ecmwf_only_s{}"}
DAYS = [7, 14, 21, 28]


def per_sample(run):
    with np.load(config.find_run(run) / "test_rain_predictions.npz") as z:
        p, o, m = obs_side(z["preds"]), obs_side(z["obs"]), z["mask"]
    assert len(p) == len(test_dates)
    pa, oa = transform(p, "rain") - clim, transform(o, "rain") - clim
    n, L = p.shape[:2]
    out = {k: np.empty((n, L)) for k in ("corr", "corr_anom", "rmse")}
    for i in range(n):
        for t in range(L):
            out["corr"][i, t] = scorr(p[i, t], o[i, t], lat, m)
            out["corr_anom"][i, t] = scorr(pa[i, t], oa[i, t], lat, m)
            out["rmse"][i, t] = rmses(p[i, t], o[i, t], lat, m)
    return out


res = {s: [per_sample(pat.format(k)) for k in range(3)] for s, pat in SETUPS.items()}

for metric in ("corr", "corr_anom", "rmse"):
    print(f"\n== {metric} ==")
    for s, runs in res.items():
        avg32 = [np.nanmean(r[metric]) for r in runs]
        days = [[np.nanmean(r[metric][:, d - 1]) for r in runs] for d in DAYS]
        print(f"{s:20s} 31d seeds {np.round(avg32, 3)} mean {np.mean(avg32):.3f} "
              + " ".join(f"d{d}={np.mean(v):.3f}" for d, v in zip(DAYS, days)))
    # paired bootstrap on seed-averaged per-forecast scores
    a = np.mean([np.nanmean(r[metric], axis=1) for r in res["with calendar (16)"]], axis=0)
    b = np.mean([np.nanmean(r[metric], axis=1) for r in res["no calendar (14)"]], axis=0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), (10000, len(a)))
    diffs = (a - b)[idx].mean(axis=1)
    print(f"with - without: {np.mean(a - b):+.4f}  95% [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]")
    # seed-level: all 9 cross pairs
    sa = [np.nanmean(r[metric]) for r in res["with calendar (16)"]]
    sb = [np.nanmean(r[metric]) for r in res["no calendar (14)"]]
    print(f"seed spread: with {max(sa) - min(sa):.4f}, without {max(sb) - min(sb):.4f}")
