# Experiment 9 — how far does plain bias correction get?

**Question.** Dyn++ multiplies ECMWF's rain by a correction factor per grid
cell, lead week and calendar month: (observed weekly rain + 1) / (ECMWF weekly
rain + 1), fitted on 2004–2017 and capped at ×0.25–×4. There is no neural
network. How does it compare with raw ECMWF and with our U-Nets?

Code: `run_bias_correction.py`; the method itself is in
`preprocess/bias_correction.py`, the scores in `model/weekly_scores.py`.
Everything is on aligned days (ECMWF lead t vs IMD issue+1+t), days 1–28,
for the same 153 test forecasts (2021–2023).

```
python experiments/exp09_bias_correction/run_bias_correction.py
```

Writes `results/table.md`, `results/summary.json` and the Dyn++ test
predictions, plus the Dyn++ factors that Experiment 10 needs
(`data_cache/bias_correction/dynpp_factors.npz`).

## Results

Weekly scores use 7-day totals with the 1951–2020 normal removed. "Cell"
pools every land cell and date; "all-India" correlates the area-mean anomaly
across dates (results.md §8.5). U-Net rows are the mean of 3 seeds.

| method | cell wk1 | wk2 | wk3 | wk4 | all-India wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|---|---|---|---|
| raw ECMWF | 0.556 | 0.382 | 0.210 | 0.130 | 0.890 | 0.790 | 0.567 | 0.396 |
| debiased (Exp 2, log space) | 0.612 | 0.406 | 0.212 | 0.141 | 0.753 | 0.522 | 0.269 | 0.145 |
| **Dyn++** | **0.668** | **0.512** | **0.330** | **0.239** | **0.910** | **0.837** | **0.590** | 0.389 |
| U-Net, 16 inputs (Exp 6) | 0.591 | 0.386 | 0.224 | 0.142 | 0.535 | 0.398 | 0.203 | 0.095 |
| U-Net, departure (Exp 3) | 0.603 | 0.374 | 0.181 | 0.106 | 0.607 | 0.432 | 0.183 | 0.079 |

Daily scores, days 1–28:

| method | corr, clim. removed | corr, standard | RMSE (mm/day) | bias (mm/day) | SD ratio |
|---|---|---|---|---|---|
| raw ECMWF | 0.223 | 0.356 | 6.82 | −0.03 | 0.60 |
| climatology | — | 0.369 | 6.75 | −0.17 | 0.47 |
| debiased (Exp 2, log space) | 0.266 | 0.382 | 6.50 | −1.20 | 0.48 |
| Dyn++ | 0.238 | 0.406 | 6.57 | −0.06 | 0.63 |
| ABC-lite (½ Dyn++ + ½ clim.) | 0.130 | 0.430 | 6.41 | −0.11 | 0.51 |
| U-Net, 16 inputs (Exp 6) | 0.342 | 0.448 | 6.39 | −1.55 | 0.39 |
| U-Net, departure (Exp 3) | **0.360** | 0.445 | 6.40 | −1.53 | 0.40 |

ABC-lite's anomaly is exactly half of Dyn++'s, so its weekly correlations
equal Dyn++'s; it only lowers RMSE.

Paired date bootstrap, 95% range:

| comparison | cell wk1 | cell wk3 | all-India wk1 | all-India wk3 |
|---|---|---|---|---|
| Dyn++ − raw ECMWF | +0.112 [+0.086, +0.141] | +0.120 [+0.076, +0.165] | +0.019 [+0.008, +0.034] | +0.023 [+0.001, +0.045] |
| U-Net (16) − Dyn++ | −0.077 [−0.101, −0.052] | −0.106 [−0.167, −0.047] | −0.374 [−0.522, −0.259] | −0.387 [−0.588, −0.198] |
| U-Net (departure) − Dyn++ | −0.065 [−0.090, −0.042] | −0.149 [−0.224, −0.085] | −0.303 [−0.418, −0.217] | −0.407 [−0.611, −0.217] |

## What it means

1. **Dyn++ is the best forecast we have on weekly totals.** It beats raw
   ECMWF by about +0.11 at every week, keeps ECMWF's all-India signal (and
   slightly improves it), and has almost no bias. It is a table of
   12 × 4 × 187 numbers.
2. **Both U-Nets are clearly worse than Dyn++ on weekly totals**, by 0.07–0.15
   per cell and by 0.3–0.4 on the all-India anomaly.
3. **The U-Nets still win on the metric they were tuned for**: daily
   climatology-removed pattern correlation in log space (0.34–0.36 vs 0.24).
   The two metrics disagree. The daily log-space score rewards getting the
   wet/dry pattern of individual days right, including light rain; the weekly
   mm score rewards getting the amounts right where it rains. Weekly totals
   are the usual unit for S2S, so the U-Net's headline gain does not carry
   over to the way S2S forecasts are normally judged.
4. **Log-space correction runs dry.** Both the Exp 2 debiasing (−1.2 mm/day)
   and the U-Nets (−1.5) work in log(1 + rain) and come out too dry; Dyn++
   works in mm and does not.

This makes Dyn++ the reference any post-processor has to beat, and the
starting point for Experiment 10.
