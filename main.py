"""Main entry point for S2S extreme precipitation prediction and baseline comparison."""

from __future__ import annotations

import sys
from scripts.run_baseline_comparison import main as run_baseline_main


def main():
    print("=" * 70)
    print("S2S Prediction Enhancement — Extreme Precipitation Occurrence")
    print("=" * 70)
    run_baseline_main()


if __name__ == "__main__":
    main()
