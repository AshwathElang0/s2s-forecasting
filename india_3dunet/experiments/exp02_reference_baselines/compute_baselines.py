"""Score the baseline ladder, and re-score any run's saved predictions in
anomaly space.

`experiments/exp01_training_size/runs/n724/results.md` compares only against *raw* ECMWF.
literature-review.md S5 warns that "many reported gains are gains over a weak
reference", and S3.3 puts climatology first on the baseline ladder ("at weeks
4-6, climatology is genuinely hard to beat"). Having a climatology makes two
stronger references cheap:

  raw        tp_mean straight from the cache -- what results.md already has
  clim       the day-of-year observed climatology; no forecast at all
  debiased   obs_clim + (log1p(tp_mean) - model_clim), i.e. lead-and-calendar
             dependent anomaly correction (literature-review.md S2.1)

Every method is scored twice:

  mm-space       pattern correlation / RMSE on the physical field, directly
                 comparable to the numbers already in results.md
  anomaly-space  the same metrics after both sides have their day-of-year
                 climatology removed -- the honest measure, because
                 absolute-field correlation over India is inflated by the
                 seasonal cycle that both fields share

Usage:
  python experiments/exp02_reference_baselines/compute_baselines.py --target rain --out experiments/exp02_reference_baselines/results \
      --rescore experiments/exp01_training_size/runs/n724 experiments/exp03_departure_from_normal/runs/gate_v1_first_try
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import config
from preprocess.climatology import (
    ensure_climatology,
    inverse_transform,
    model_clim_for_dates,
    obs_clim_for_dates,
    transform,
    usable_dates,
)
from model.evaluation import skill_scores_by_lead
from preprocess.alignment import ALIGNED_DAYS, ecmwf_side, obs_side
from preprocess.dataset_loader import list_cached_dates, load_raw, split_dates
from train import padded_lat_array

# Forecast days, after alignment (preprocess/alignment.py): day d = ECMWF lead
# index d-1 scored against IMD date issue+d.
LEADS_TO_REPORT = [1, 4, 7, 14, 21, 28, ALIGNED_DAYS]


def score_both_spaces(preds_mm, obs_mm, obs_clim_batch, target_var, lat, mask):
    """(mm-space scores, anomaly-space scores)."""
    mm = skill_scores_by_lead(preds_mm, obs_mm, lat, mask)
    pred_a = transform(preds_mm, target_var) - obs_clim_batch
    obs_a = transform(obs_mm, target_var) - obs_clim_batch
    return mm, skill_scores_by_lead(pred_a, obs_a, lat, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=config.TARGETS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rescore", nargs="*", default=[],
                        help="run directories whose test_<target>_predictions.npz to re-score")
    args = parser.parse_args()

    all_dates = list_cached_dates(args.target)
    splits = split_dates(all_dates)

    obs_clim, model_clim = ensure_climatology(args.target, splits["train"], all_dates)
    test_dates = usable_dates(splits["test"], model_clim)

    features, target, mask = load_raw(args.target, test_dates)
    obs_mm = obs_side(target[:, 0])
    tp_mean = ecmwf_side(features[:, 0]).copy()
    del features, target

    obs_batch = obs_side(obs_clim_for_dates(obs_clim, test_dates, config.LEAD_DAYS))
    mdl_batch = ecmwf_side(model_clim_for_dates(model_clim, test_dates))

    methods = {
        "raw": tp_mean,
        "clim": inverse_transform(obs_batch, args.target),
        "debiased": inverse_transform(
            obs_batch + (np.log1p(np.clip(tp_mean, 0, None)) - mdl_batch), args.target
        ),
    }

    for run in args.rescore:
        path = Path(run) / f"test_{args.target}_predictions.npz"
        if not path.exists():
            print(f"skipping {path} (not found)")
            continue
        with np.load(path) as z:
            preds = z["preds"]
        if len(preds) != len(test_dates):
            print(f"skipping {path}: {len(preds)} predictions vs {len(test_dates)} usable test dates")
            continue
        methods[Path(run).name] = obs_side(preds)

    lat = padded_lat_array()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, preds in methods.items():
        mm, anom = score_both_spaces(preds, obs_mm, obs_batch, args.target, lat, mask)
        results[name] = {"mm": mm, "anom": anom}
        np.savez(out_dir / f"baseline_{name}_scores.npz",
                 **{f"mm_{k}": v for k, v in mm.items()},
                 **{f"anom_{k}": v for k, v in anom.items()})

    idx = [d - 1 for d in LEADS_TO_REPORT]
    for space in ("mm", "anom"):
        label = "mm-space (absolute field)" if space == "mm" else "anomaly-space (climatology removed)"
        for metric, title in (("pattern_corr_mean", "pattern correlation"), ("rmse_mean", "RMSE")):
            print(f"\n=== {title}, {label} ===")
            print(f"{'lead':<12}" + "".join(f"{d:>8}" for d in LEADS_TO_REPORT) + f"{'mean':>9}")
            for name, r in results.items():
                v = r[space][metric]
                print(f"{name:<12}" + "".join(f"{v[i]:>8.3f}" for i in idx) + f"{np.nanmean(v):>9.3f}")

    print(f"\n{len(test_dates)} test samples. Saved per-method scores under {out_dir}")


if __name__ == "__main__":
    main()
