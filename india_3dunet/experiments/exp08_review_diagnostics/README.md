# Experiment 8 — review diagnostics (professor's comments)

Figures and numbers for the review response
([results.md, Section 8](../../results.md#8-review-response-professors-comments)),
all for the headline model `../exp06_yesterday_rain/runs/no_obs_s0`. No retraining.

- `make_review_figures.py`: learning curves, scatter plots, time series, case maps.
- `input_sensitivity.py`: which inputs influence one prediction (gradients).
- `figures/review/`: the original figures, scored one day off (see
  `preprocess/alignment.py`), with `summary.json` and `sensitivity.json`.
- `figures/review_aligned/`: the same figures re-made on aligned days.

```
python experiments/exp08_review_diagnostics/make_review_figures.py
python experiments/exp08_review_diagnostics/input_sensitivity.py
```

**Answer:** the U-Net gets the spatial pattern right in weeks 1–2, but it is
too dry (−1.3 to −1.7 mm/day), squashes heavy rain, and loses most of ECMWF's
all-India wet/dry signal (week 3: 0.22 vs 0.56).
