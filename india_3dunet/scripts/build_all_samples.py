"""Build and cache one (features, target, mask) sample per ECMWF issue
date x target variable, resumable (skips files that already exist).

Requires scripts/cache_imd_targets.py to have been run first.

Usage:  python scripts/build_all_samples.py [--limit N] [--targets rain tmax tmin]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from preprocess.build_dataset import IMDTargetStore, build_sample
from preprocess.ecmwf_features import list_issue_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only build the first N issue dates")
    parser.add_argument("--stride", type=int, default=1, help="build every Nth issue date (for a quick, calendar-spread smoke test)")
    parser.add_argument("--targets", nargs="+", default=config.TARGETS)
    args = parser.parse_args()

    issue_dates = list_issue_dates("tp")[:: args.stride]
    if args.limit:
        issue_dates = issue_dates[: args.limit]

    imd_store = IMDTargetStore()

    for target in args.targets:
        out_dir = config.CACHE_ROOT / "samples" / target
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, issue_date in enumerate(issue_dates):
            out_path = out_dir / f"{issue_date:%Y_%m_%d}.npz"
            if out_path.exists():
                continue

            t0 = time.time()
            try:
                features, tgt, mask, channel_names = build_sample(issue_date, target, imd_store)
            except Exception as e:
                # ~4% of the raw ECMWF `q` files are 0-byte / corrupt on
                # disk (confirmed while building this pipeline -- not a
                # bug here). Skip the date rather than aborting the whole
                # cache build; this is exactly the kind of gap
                # scripts/check_data_sources.py-style tooling should flag.
                print(f"[{target}] SKIP {issue_date:%Y-%m-%d}: {e!r}")
                continue

            np.savez_compressed(out_path, features=features, target=tgt, mask=mask)

            print(
                f"[{target}] {i + 1}/{len(issue_dates)} {issue_date:%Y-%m-%d} "
                f"({time.time() - t0:.1f}s)"
            )

        with open(out_dir / "channel_names.txt", "w") as f:
            f.write("\n".join(channel_names))


if __name__ == "__main__":
    main()
