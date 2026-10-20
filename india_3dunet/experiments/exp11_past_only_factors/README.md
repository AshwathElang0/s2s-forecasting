# Experiment 11 — `residual_mm` with past-years-only Dyn++ factors

**Question.** In Exp 10, the Dyn++ input for a training forecast used factors
fitted on every training year except its own (leave-one-year-out). A July
2010 forecast was therefore corrected with IMD rain from 2004–2009 **and
2011–2017**, i.e. partly from its future. Does `residual_mm` still beat Dyn++
when the training factors may only use the past?

## Which IMD rain is used where

| use | Exp 10 | Exp 11 |
|---|---|---|
| U-Net inputs | none (16 ECMWF fields + calendar) | none |
| Dyn++ factors, training forecast from year Y | all training years except Y | **2004 … Y−1 only** |
| Dyn++ factors, validation (2018–20) and test (2021–23) | 2004–2017 | 2004–2017 (unchanged) |
| Training target | IMD rain on the forecast's dates | same |
| Rain normal in the loss | earlier years only | same |

- **No IMD from the forecast's own year, or from the days just before it,
  enters the Dyn++ correction in either experiment.** The only change is
  that Exp 11 also removes future years from the training-year factors.
- **The first year has no history.** 2004 gets factors of 1, i.e. raw
  ECMWF rain.
- **Validation and test use the same factors as before** (2004–2017, all in
  the past), so the Dyn++ baseline and its test scores are unchanged.

## How noisy are past-only factors?

A factor needs years of data, so early years' factors are rough. Below is
the median distance from the full 2004–2017 fit, over land cells, months
and lead weeks, as |log ratio|; 0.10 is about 10%.

| training year | years of history | past only | leave-one-year-out |
|---|---|---|---|
| 2004 | 0 | 0.236 | 0.017 |
| 2005 | 1 | 0.249 | 0.017 |
| 2006 | 2 | 0.171 | 0.019 |
| 2008 | 4 | 0.118 | 0.018 |
| 2010 | 6 | 0.096 | 0.019 |
| 2012 | 8 | 0.063 | 0.018 |
| 2014 | 10 | 0.045 | 0.015 |
| 2017 | 13 | 0.016 | 0.016 |

The full table is in `results/factor_noise.md`. In the first years, the
Dyn++ input the U-Net trains on is off by about 25%; by the last years it
matches the full fit. At test time the U-Net always gets the full 14-year
factors.

## Runs

| run | best epoch | val loss | val mean fix, wk1–4 |
|---|---|---|---|
| `residual_mm_past_s0` | 15 | 1.4250 | 1.14, 1.17, 1.20, 1.20 |
| `residual_mm_past_s1` | 22 | 1.4241 | 1.03, 1.11, 1.15, 1.15 |
| `residual_mm_past_s2` | 21 | 1.4185 | 1.08, 1.13, 1.19, 1.21 |
| `residual_mm_s0` / `s1` / `s2` (Exp 10) | 21 / 23 / 23 | | 1.06–1.18 |

Three seeds, the same as Exp 10. The first attempt at seed 0 crashed the
laptop (NVIDIA driver failure, `VIDEO_TDR_FAILURE`); the rerun and seeds
1–2 completed.

This is the **final version** of `residual_mm`: the Exp 10 figures
(`exp10_unet_on_bias_correction/figures/`) and the progress report's graphs
use it.

```
bash experiments/exp11_past_only_factors/run_all.sh     # factors, 3 seeds, scoring
```

## Results

Test, 2021–2023, 153 forecasts, aligned days 1–28. Both U-Nets: mean of 3 seeds.

**Weekly anomaly correlation:**

| method | cell wk1 | wk2 | wk3 | wk4 | all-India wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|---|---|---|---|
| Dyn++ | 0.668 | 0.512 | 0.330 | 0.239 | 0.910 | 0.837 | 0.590 | 0.389 |
| `residual_mm`, leave-one-year-out (Exp 10) | 0.730 | 0.558 | 0.404 | 0.299 | 0.901 | 0.838 | 0.624 | 0.420 |
| **`residual_mm`, past only (final)** | **0.725** | **0.544** | **0.399** | **0.287** | **0.906** | **0.838** | **0.629** | **0.421** |

Seed spread of the past-only runs: 0.011–0.015 per cell, 0.003–0.025 all-India.

**Weekly RMSE (mm/week) and daily scores:**

| method | cell wk1 | wk2 | wk3 | wk4 | daily RMSE | bias (mm/day) | SD ratio |
|---|---|---|---|---|---|---|---|
| Dyn++ | 26.8 | 30.8 | 34.0 | 35.0 | 6.57 | −0.06 | 0.63 |
| `residual_mm`, leave-one-year-out (Exp 10) | 24.6 | 29.8 | 32.7 | 34.2 | 6.23 | −0.37 | 0.60 |
| **`residual_mm`, past only (final)** | **24.8** | **30.1** | **32.8** | **34.4** | **6.23** | **−0.41** | **0.60** |

**Paired date bootstrap** (difference [95% range], weekly correlation per cell):

| comparison | wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|
| past only − Dyn++ | +0.058 [+0.040, +0.075] | +0.033 [+0.004, +0.061] | +0.069 [+0.018, +0.116] | +0.048 [+0.003, +0.092] |
| past only − leave-one-year-out | −0.005 [−0.012, +0.002] | −0.014 [−0.031, +0.001] | −0.005 [−0.028, +0.017] | −0.012 [−0.035, +0.011] |

## What it means

1. **Past-only factors cost nothing measurable.** The past-only model is
   0.005–0.014 below the leave-one-year-out one per cell, every 95% range
   includes 0, and the gaps are about the size of the seed spread. All-India
   scores are identical.
2. **The final `residual_mm` still beats Dyn++ in every week**, by
   +0.03 to +0.07 per cell, with every range above 0 (week 4 only just:
   +0.003). It ties Dyn++ on all-India and is 0.5–2.1 mm/week more accurate
   per cell.
3. **So Exp 10's result was not bought with future IMD years.** With
   factors from earlier years only, the U-Net learns an equally useful
   correction. That fits: at test time it always receives the full
   2004–2017 factors, and the rougher early-year training factors act as
   mild noise in one input.
4. **The gain over Dyn++ is a little smaller and less certain than in
   Exp 10** (week 2: +0.033 vs +0.046), because the past-only scores are
   slightly lower everywhere. We report the past-only version as final
   because it is the stricter, cleaner setup.

## Caveats

- **Validation and test factors are frozen at 2004–2017.** An operational
  system could also use 2018 onward, which would, if anything, help.
- Same limits as Exp 10: 3 test years; the bootstrap resamples dates, not
  years.
