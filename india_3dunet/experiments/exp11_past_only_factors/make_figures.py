"""Predicted-vs-observed figures for the final residual_mm (past-only factors).

The same four figures as Experiment 10 (exp10_unet_on_bias_correction/make_figures.py):
observed IMD rain against Dyn++, residual (Exp 10, log space) and the final
residual_mm (this experiment, seed 0), test years 2021-2023.

Writes PNGs and summary.json to figures/ next to this script.

Usage:  python experiments/exp11_past_only_factors/make_figures.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp10_unet_on_bias_correction"))

import make_figures as exp10_figures

exp10_figures.OUT = HERE / "figures"
exp10_figures.RUNS["residual_mm (mm/day)"] = HERE / "runs" / "residual_mm_past_s0"

if __name__ == "__main__":
    exp10_figures.main()
