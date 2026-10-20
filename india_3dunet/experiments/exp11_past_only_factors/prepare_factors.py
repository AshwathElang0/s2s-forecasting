"""Experiment 11, step 1: add the past-years-only Dyn++ fit.

Refits the Dyn++ factors on 2004-2017, checks that the full and
leave-one-year-out fits match the ones Experiments 9-10 used, and saves them
back with the past-only fit added (year y fitted on the years before y; 2004
has no history and gets factors of 1). Also reports how far each year's
past-only factors are from the full fit, i.e. how noisy they are.

Writes results/factor_noise.md and adds past_<year> arrays to
data_cache/bias_correction/dynpp_factors.npz.

Usage:  python experiments/exp11_past_only_factors/prepare_factors.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np

import config
from preprocess.bias_correction import BC_ROOT, fit_dynpp, load_aligned, load_factors, save_factors
from preprocess.dataset_loader import list_cached_dates, split_dates

OUT = HERE / "results"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = split_dates(list_cached_dates("rain"))["train"]
    tp, obs = load_aligned(tr)
    with np.load(config.CACHE_ROOT / "samples" / "rain" / f"{tr[0]:%Y_%m_%d}.npz") as z:
        mask = z["mask"].astype(bool)

    path = BC_ROOT / "dynpp_factors.npz"
    old = load_factors(path)
    new = fit_dynpp(tp, obs, tr, mask)
    gap = max(np.abs(new["full"] - old["full"]).max(),
              max(np.abs(new["loyo"][y] - old["loyo"][y]).max() for y in old["loyo"]))
    assert gap < 1e-5, f"refit differs from the factors Exp 9-10 used: {gap}"
    save_factors(new, path)

    lines = [f"Refit matches the saved full and leave-one-year-out factors (max diff {gap:.1e}).",
             f"Past-only fit added to {path.name}.", "",
             "Distance from the full 2004-2017 fit: median over land cells, months and lead weeks",
             "of |log(factor / full factor)|. 0.10 is roughly a 10% difference.", "",
             "| training year | years of history | past-only | leave-one-year-out |", "|---|---|---|---|"]
    lf = np.log(new["full"][..., mask])
    for y in sorted(new["past"]):
        dp = np.median(np.abs(np.log(new["past"][y][..., mask]) - lf))
        dl = np.median(np.abs(np.log(new["loyo"][y][..., mask]) - lf))
        lines.append(f"| {y} | {y - 2004} | {dp:.3f} | {dl:.3f} |")
    (OUT / "factor_noise.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
