"""Score the raw ECMWF ensemble-mean forecast against IMD, with the same
metrics/masking as the trained model -- the actual comparison the paper's
headline numbers are about (skill *relative to raw NWP*, not absolute
RMSE). See plan.md S6.

Channel 0 of every cached rain sample is `tp_mean`, the raw (unstandardized,
un-log-transformed) ECMWF ensemble-mean precipitation forecast in mm/day --
see preprocess/build_dataset.py's channel ordering. This script scores that
channel directly against the cached raw target, no model involved.

Usage:  python experiments/exp01_training_size/compute_raw_baseline.py --target rain --out experiments/exp01_training_size/runs/n60
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from model.evaluation import skill_scores_by_lead
from preprocess.alignment import ecmwf_side, obs_side
from preprocess.dataset_loader import list_cached_dates, load_raw, split_dates
from train import padded_lat_array

RAW_FORECAST_CHANNEL = {"rain": 0}  # tp_mean; add tmax/tmin's channel if/when scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=config.TARGETS, required=True)
    parser.add_argument("--out", required=True, help="directory to write raw_ecmwf_scores.npz into")
    parser.add_argument("--limit_test", type=int, default=None)
    args = parser.parse_args()

    if args.target not in RAW_FORECAST_CHANNEL:
        raise SystemExit(f"no raw-forecast channel mapping for target={args.target!r} yet")

    all_dates = list_cached_dates(args.target)
    test_dates = split_dates(all_dates)["test"]
    if args.limit_test:
        test_dates = test_dates[: args.limit_test]

    features, target, mask = load_raw(args.target, test_dates)
    # ECMWF lead k verifies against IMD issue+1+k (preprocess/alignment.py)
    raw_forecast = ecmwf_side(features[:, RAW_FORECAST_CHANNEL[args.target]])
    obs = obs_side(target[:, 0])

    lat = padded_lat_array()
    scores = skill_scores_by_lead(raw_forecast, obs, lat, mask)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "raw_ecmwf_scores.npz", **scores)

    print(f"{len(test_dates)} test samples, raw ECMWF ensemble-mean baseline for {args.target}")
    print("pattern_corr_mean by lead day:")
    print(np.round(scores["pattern_corr_mean"], 3))
    print("rmse_mean by lead day:")
    print(np.round(scores["rmse_mean"], 3))
    print(f"Saved to {out_dir / 'raw_ecmwf_scores.npz'}")


if __name__ == "__main__":
    main()
