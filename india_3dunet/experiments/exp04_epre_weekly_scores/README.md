# Experiment 4 — the paper's E_pre metric at weeks 1–4

No runs of its own. Found that our port of E_pre gave 1.386 (log 4) for a
perfect forecast instead of 0. The fix and the SD-ratio diagnostic are in
`model/evaluation_fixed.py`, which `train_ablation.py` uses to write each
run's `scores_fixed.npz`; the numbers in
[results.md, Experiment 4](../../results.md#experiment-4--the-papers-e_pre-metric-reported-at-weeks-14)
come from those files.

**Answer:** the error was a constant offset, so no ranking changes. The
predictions carry about half the real variability (SD ratio 0.44–0.86).
