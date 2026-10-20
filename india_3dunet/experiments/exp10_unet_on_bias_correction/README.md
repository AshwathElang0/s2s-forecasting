# Experiment 10 — a U-Net on top of Dyn++

**Question.** Dyn++ (Exp 9) is the best forecast on weekly totals and no
U-Net beat it. Can a U-Net that starts from Dyn++ and learns a correction to
it do better?

All variants use the 16 inputs of the Exp 6 headline model. They train on
correctly aligned days (ECMWF lead t vs IMD issue+1+t) and are scored exactly
as in Exp 9: 153 test forecasts (2021–2023), days 1–28.

## The three variants

| variant | starts from | output | loss |
|---|---|---|---|
| `plain_aligned` | nothing (control) | log(1 + rain) directly | MSE + (1 − pattern corr), log space |
| `residual` | Dyn++ | Dyn++ + correction, log space | same as above |
| `residual_mm` | Dyn++ | Dyn++ × exp(correction), mm/day | MSE in mm + (1 − pattern corr of the departure from normal) |

### `residual` vs `residual_mm`, in plain words

Both start from the same Dyn++ forecast and learn a correction to it. They
differ in **how the correction is applied** and in **what scale the training
error is measured on**.

| | `residual` | `residual_mm` |
|---|---|---|
| Correction | **adds** a number in log(1 + rain) | **multiplies** Dyn++ by a factor, exp(c) (×0.5, ×1.2, …) |
| Error measured in | log(1 + rain) | mm/day, i.e. real rain amounts |
| Pattern score | whole map (geography included) | departure from normal only |
| Result | ~50% too dry, below Dyn++ | nearly unbiased, beats Dyn++ |

Why the scale matters: on a log scale, under-predicting a 40 mm/day downpour
as 20 mm/day costs about the same as predicting 2 mm/day instead of 1. Most
grid-days are dry or light, so a log-scale error rewards getting light rain
right and barely penalises missing heavy rain. Converting a log-scale "best
guess" back to mm also lands below the true average (averaging logs
underestimates the average amount). Together these make every log-space model
too dry: `residual` starts as an unbiased Dyn++ and training pulls it back to
roughly half the observed rain.

`residual_mm` measures error in mm/day, so missing a heavy-rain week costs
what it should. The multiplicative form keeps rain non-negative and makes "no
change" (factor 1) the starting point. It keeps Dyn++'s correct amounts and
adds skill on top. Scoring the pattern on departures from normal stops the
network earning credit for geography it already gets for free (the west
coast is always wetter).

- In the two residual variants, Dyn++ rain replaces raw ECMWF rain as input
  channel 0. Training dates use leave-one-year-out Dyn++ factors, so the
  network never sees a correction fitted on its own target year. Those
  factors still include later years; Exp 11 reruns `residual_mm` with
  past-years-only factors and gets the same scores.
- The correction head starts at zero, so before training each residual model
  *is* Dyn++.
- `residual_mm` has an optional all-India loss term (`--area_weight`). It is
  off by default and was off in every run here; the all-India error is only
  logged.
- Every run also stores a "val mean fix" in `stats.json`: one obs/pred factor
  per lead week, measured on the validation years. The scores are reported
  both with and without it.

Code: `train_residual.py` (training), `score_runs.py` (scoring),
`run_all.sh` (the full batch), `make_figures.py` (predicted-vs-observed time
series and case maps in `figures/`, used by the progress report). The same
figures for the final model are in Exp 11. The model classes live in
`model/residual_unet.py`.

```
python experiments/exp10_unet_on_bias_correction/train_residual.py --variant residual_mm --seed 0
python experiments/exp10_unet_on_bias_correction/score_runs.py
```

Needs `data_cache/bias_correction/dynpp_factors.npz` and the Dyn++ test
predictions, both written by Exp 9.

## Runs

| run | best epoch | val mean fix, wk1–4 |
|---|---|---|
| `plain_aligned_s0` | 5 | 1.73, 1.87, 1.98, 2.12 |
| `residual_s0` | 12 | 1.54, 1.83, 2.06, 2.13 |
| `residual_mm_s0` | 21 | 1.11, 1.11, 1.15, 1.15 |
| `residual_mm_s1` | 23 | 1.10, 1.13, 1.16, 1.16 |
| `residual_mm_s2` | 23 | 1.06, 1.09, 1.16, 1.18 |

`plain_aligned` and `residual` have one seed each. After seed 0,
`residual_mm` was clearly ahead, so only it got seeds 1 and 2. The first
`residual_mm_s1` attempt died at epoch 22 with a GPU driver fault (`CUDA
error: unknown error`; log kept as `train_crashed_cuda.log`) and was rerun.
An earlier `plain_aligned_s1` hung and was stopped (`runs/_hung_plain_aligned_s1`,
safe to delete).

## Results

### Weekly anomaly correlation

Weekly totals, with the 1951–2020 normal removed. `residual_mm` is the mean
of 3 seeds; the other two U-Nets are 1 seed.

| method | cell wk1 | wk2 | wk3 | wk4 | all-India wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|---|---|---|---|
| Dyn++ (Exp 9) | 0.668 | 0.512 | 0.330 | 0.239 | **0.910** | 0.837 | 0.590 | 0.389 |
| U-Net, 16 inputs (Exp 6) | 0.591 | 0.386 | 0.224 | 0.142 | 0.535 | 0.398 | 0.203 | 0.095 |
| `plain_aligned` | 0.551 | 0.358 | 0.207 | 0.136 | 0.459 | 0.331 | 0.170 | 0.073 |
| `residual` | 0.628 | 0.389 | 0.196 | 0.141 | 0.613 | 0.400 | 0.189 | 0.104 |
| **`residual_mm`** | **0.730** | **0.558** | **0.404** | **0.299** | 0.901 | **0.838** | **0.624** | **0.420** |
| `residual_mm` + val mean fix | 0.736 | 0.554 | 0.392 | 0.278 | 0.918 | 0.794 | 0.529 | 0.311 |

Seed spread for `residual_mm` (max − min over 3 seeds) is 0.007–0.019 per
cell and 0.02–0.07 all-India.

### Weekly RMSE

RMSE of the weekly total, in mm/week; lower is better.

| method | cell wk1 | wk2 | wk3 | wk4 | all-India wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|---|---|---|---|
| Dyn++ | 26.8 | 30.8 | 34.0 | 35.0 | **4.6** | **5.9** | **8.6** | **9.8** |
| U-Net, 16 inputs (Exp 6, s0) | 29.0 | 34.7 | 37.1 | 38.0 | 11.6 | 14.3 | 15.9 | 16.6 |
| `plain_aligned` | 31.7 | 35.6 | 37.7 | 38.4 | 15.2 | 16.1 | 17.2 | 17.7 |
| `residual` | 29.2 | 34.8 | 37.9 | 38.4 | 12.7 | 15.1 | 17.3 | 17.7 |
| **`residual_mm`** (3 seeds) | **24.6** | **29.8** | **32.7** | **34.2** | 5.4 | 6.3 | 8.8 | 10.0 |

### Daily scores, days 1–28

| method | corr, clim. removed | corr, standard | RMSE (mm/day) | bias (mm/day) | SD ratio |
|---|---|---|---|---|---|
| Dyn++ | 0.238 | 0.406 | 6.57 | −0.06 | 0.63 |
| `plain_aligned` | 0.353 | 0.447 | 6.44 | −1.65 | 0.37 |
| `residual` | 0.327 | 0.443 | 6.38 | −1.58 | 0.40 |
| `residual_mm` | 0.337 | 0.443 | **6.23** | −0.37 | 0.60 |

### Paired date bootstrap

Differences with 95% ranges; `residual_mm` is the mean of 3 seeds.

| comparison | cell wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|
| `residual_mm` − Dyn++ | +0.063 [+0.044, +0.082] | +0.046 [+0.020, +0.071] | +0.074 [+0.027, +0.118] | +0.060 [+0.014, +0.105] |
| `residual_mm` − Dyn++, all-India | −0.009 [−0.024, +0.005] | +0.001 [−0.028, +0.027] | +0.034 [−0.052, +0.109] | +0.031 [−0.066, +0.133] |
| `residual` − Dyn++ | −0.039 [−0.060, −0.020] | −0.123 [−0.168, −0.078] | −0.134 [−0.210, −0.066] | −0.099 [−0.182, −0.016] |
| `plain_aligned` − Dyn++ | −0.117 [−0.145, −0.088] | −0.154 [−0.205, −0.105] | −0.123 [−0.182, −0.065] | −0.103 [−0.170, −0.033] |

Weekly RMSE, `residual_mm` seed 0 − Dyn++ (mm/week): wk1 −2.1 [−2.8, −1.4],
wk2 −1.1 [−1.7, −0.4], wk3 −1.4 [−2.3, −0.5], wk4 −0.9 [−1.7, −0.2].

## What it means

1. **`residual_mm` is the first U-Net to beat Dyn++ on weekly totals.** It
   gains +0.05 to +0.07 per cell at every week; every 95% range excludes 0, and
   all 3 seeds agree within 0.02. It is also 1–2 mm/week more accurate per cell.
2. **It ties Dyn++ on the all-India score.** Every all-India range includes 0.
   This recovered without any all-India loss term. Its all-India RMSE is
   slightly higher in week 1 (5.4 vs 4.6 mm/week) and equal later.
3. **What makes it work is training in mm, not starting from Dyn++.**
   `residual` also starts from Dyn++ but trains in log space, and it ends up
   below Dyn++ and about 50% too dry, like every earlier U-Net. The
   multiplicative mm-scale version keeps Dyn++'s amounts (bias −0.37 mm/day)
   and adds skill on top.
4. **Fixing the one-day misalignment did not help the old design.**
   `plain_aligned` scores about the same as the Exp 6 model, or slightly
   worse. The old U-Net's weekly weakness came from its loss, not its dates.
5. **The val mean fix doesn't add skill to `residual_mm`.** It removes the
   remaining bias but costs correlation at weeks 3–4. Report `residual_mm`
   without it.

## Caveats

- **Only 3 test years (2021–2023).** The bootstrap resamples forecast dates,
  not years, so it understates year-to-year uncertainty.
- **`plain_aligned` and `residual` have one seed each.** Their gaps to Dyn++
  are much larger than the seed spread we have seen (≤0.07), so the
  conclusion holds, but the exact numbers are single-run.
- **Mild selection.** `residual_mm` got its extra seeds because its seed 0 test
  score was best. Seeds 1 and 2 reproduced it, so this does not change the
  result. A blocked cross-validation over years would remove the concern.
- Daily climatology-removed correlation in log space (the Exp 2–7 metric) still
  ranks the old departure U-Net (0.360) above `residual_mm` (0.337). As Exp 9
  showed, that metric disagrees with the weekly scores.
