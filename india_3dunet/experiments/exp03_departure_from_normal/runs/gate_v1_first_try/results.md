# Rain, climatology-anchored gated residual head

The architecture change described in `../../gated/README.md`, run against the
same calendar split as `rain_full`. Two things came out of it: a modest but
consistent skill gain, and a much more important discovery about what the
`rain_full` numbers were actually measuring.

## Config

```
python gated/train_gated.py --target rain --model_path runs/rain_gated \
  --unit 64 --batch_size 11 --n_epochs 100 --patience 10
python gated/test_gated.py --target rain --model_path runs/rain_gated --unit 64
python gated/compute_baselines.py --target rain --out runs/rain_gated \
  --rescore runs/rain_full runs/rain_gated
```

- Same architecture, capacity (`unit=64`), optimiser, batch size, loss
  (masked MSE + masked pattern correlation) and early-stopping rule as
  `rain_full`. The only change is the head and the target reformulation.
- **672 train / 155 val / 153 test.** The test set is identical to
  `rain_full`'s. Train is 52 samples smaller because the 2004 reforecast year
  has no prior year from which to build a *causal* model climatology, so those
  samples are dropped rather than allowed to look ahead (gated/README.md S5.1).
  The gated model therefore trains on ~7% less data — the comparison is
  conservative.
- All climatologies are strictly causal: a sample in year Y uses only
  observations and reforecasts from before Y.

## Training curve

**Training was cut short at epoch 14 by a transient `CUDA error: unknown
error`**, not by the early-stopping rule. Best epoch was 11 (val 1.5423) and
val loss had been flat for ~8 epochs, so the best checkpoint is very likely
near-converged — but this is not a clean early stop and should be re-run
before the number is quoted anywhere.

| epoch | train loss | val loss |
|---|---|---|
| 1 | 1.711 | 1.654 |
| 7 | 1.532 | 1.553 |
| 11 (best) | 1.487 | **1.542** |
| 14 (crash) | 1.408 | 1.558 |

Loss values are **not** comparable to `rain_full`'s (~0.96) because the target
is a standardized anomaly rather than a standardized absolute field.

What *is* comparable is the shape. `rain_full` overfit hard — val climbed
0.961 -> 1.103 (+15%) between epochs 7 and 17 while train fell 0.941 -> 0.563.
Here val moves 1.553 -> 1.558 (+0.3%) over the same stretch. The
overfitting-reduction part of the hypothesis held clearly.

## Results: the baseline ladder, both scorings

153 test samples, 2021-2023. `clim` is the causal day-of-year climatology with
no forecast information at all; `debiased` is lead-and-calendar-dependent
anomaly correction of ECMWF (literature-review.md S2.1).

**Pattern correlation, mm-space (absolute field — what results.md reports)**

| lead (day) | 1 | 4 | 7 | 14 | 21 | 28 | 32 | mean |
|---|---|---|---|---|---|---|---|---|
| raw ECMWF | 0.432 | 0.444 | 0.412 | 0.319 | 0.296 | 0.266 | 0.258 | 0.329 |
| climatology | 0.371 | 0.363 | 0.346 | 0.358 | 0.359 | 0.368 | 0.362 | **0.361** |
| debiased ECMWF | 0.464 | 0.472 | 0.438 | 0.341 | 0.307 | 0.286 | 0.270 | 0.353 |
| rain_full | 0.553 | 0.598 | 0.538 | 0.419 | 0.388 | 0.378 | 0.350 | 0.440 |
| **rain_gated** | **0.570** | **0.600** | 0.538 | **0.424** | **0.392** | **0.389** | **0.370** | **0.445** |

**Pattern correlation, anomaly-space (climatology removed — the honest measure)**

| lead (day) | 1 | 4 | 7 | 14 | 21 | 28 | 32 | mean |
|---|---|---|---|---|---|---|---|---|
| raw ECMWF | 0.436 | 0.431 | 0.375 | 0.173 | 0.091 | 0.032 | 0.029 | 0.183 |
| climatology | -0.007 | -0.010 | 0.004 | 0.003 | 0.007 | 0.006 | -0.014 | -0.002 |
| debiased ECMWF | 0.466 | 0.457 | 0.415 | 0.210 | 0.120 | 0.060 | 0.057 | 0.222 |
| rain_full | 0.556 | 0.622 | 0.545 | 0.318 | 0.216 | 0.148 | 0.145 | 0.332 |
| **rain_gated** | **0.594** | **0.641** | **0.553** | **0.332** | **0.225** | **0.180** | **0.162** | **0.350** |

**RMSE, anomaly-space** (mean over leads): clim 0.747, raw 0.875,
debiased 0.766, rain_full 0.679, **rain_gated 0.677**.

## The finding that matters most

**In mm-space, plain climatology (mean corr 0.361) beats raw ECMWF (0.329).**
A forecast containing no forecast information outscores the dynamical model on
the metric `rain_full/results.md` leads with.

That is the literature review's S5 warning landing exactly: "many reported
gains are gains over a weak reference." Absolute-field pattern correlation over
India is dominated by the shared seasonal and orographic pattern — both fields
are wet over the Ghats in July — so the metric largely measures *"does this
field look like Indian rainfall in this season"*, which climatology answers
perfectly and which `doy_sin`/`doy_cos` hand the network for free.

Removing climatology from both sides collapses the inflation:

| | mm-space mean corr | anomaly-space mean corr |
|---|---|---|
| rain_full | 0.440 | 0.332 |
| rain_full @ day 32 | 0.350 | **0.145** |
| raw ECMWF @ day 32 | 0.258 | **0.029** |

So `results.md`'s "32/32 lead days beat raw ECMWF, mean corr gain +0.111"
stands, but understates the reference problem: raw ECMWF at day 32 has
essentially *no* anomaly skill (0.029), so beating it there is a low bar. The
honest statement is the anomaly-space one.

**The good news:** both U-Nets clearly beat the strong baselines in anomaly
space — 0.350 and 0.332 against debiased ECMWF's 0.222 and climatology's
0.000. The post-processing is adding real information, not just repainting the
seasonal cycle. That conclusion survives the stricter scoring.

## Did the gated head work?

**Partly. The reformulation worked; the gate did not.**

The gain is real and consistent: +0.018 mean anomaly-space correlation over
`rain_full` (0.350 vs 0.332), with the largest gains at long lead
(+0.032 at day 28, +0.017 at day 32) — and achieved with 7% less training
data. Overfitting is dramatically reduced.

But the learned gate collapsed:

| lead (day) | 1 | 4 | 7 | 14 | 21 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| mean learned gate | 0.148 | 0.078 | 0.078 | 0.047 | 0.019 | 0.014 | 0.014 |
| loss-optimal (train `r`) | 0.534 | 0.485 | 0.470 | 0.242 | 0.111 | 0.071 | 0.044 |

The gate should have converged toward the correlation between the model and
observed anomaly. It sits roughly 3-4x below that at every lead. It retains the
right *shape* (monotone decay with lead) but not the right magnitude, so it
cannot be read as "how much the post-processor trusts ECMWF."

**The cause is a design error, and it is fixable.** `tp_mean`'s anomaly is fed
to the network *twice*: once as the anchor the head blends against, and once as
feature channel 0 of the input stack. The correction head can therefore
reconstruct the ECMWF signal through the conv stack on its own, making the
anchor pathway redundant — so the network routed the signal through the
correction and let the gate decay. The skill gain is coming from the anomaly
reformulation, not from the gating mechanism.

The clean next experiment is one line: **drop `tp_mean` from the feature stack**
so the anchor is the only path carrying ECMWF's precipitation signal. That
forces the gate to do the job it was designed for, and makes it interpretable.
Until that runs, the "windows of opportunity" diagnostic in gated/README.md S3
is not supported by this run.

## Honest scope

- Training was truncated by a CUDA fault at epoch 14, not early-stopped.
  Re-run before quoting.
- Still one calendar split, no blocked-CV averaging (literature-review.md S7.1).
- Still no orography/MJO features, no quantile-mapping baseline (the review's
  S4 bar), no distributional head or CRPS.
- The gate is not yet the interpretable object it was designed to be — see
  above.
- `rain_full` remains the better-converged run; the comparison above is
  best-checkpoint against best-checkpoint, which is fair, but the gated run had
  fewer epochs to work with.

## Artifacts

`model_epoch_1..14.pt`, `best_epoch` (11), `stats.json`, `train_history.pkl`,
`test_rain_predictions.npz` (153, 32, 48, 64), `test_rain_scores.npz`,
`test_rain_gate.npz` (the learned gate), and `baseline_*_scores.npz` for each
of raw / clim / debiased / rain_full / rain_gated in both scorings.
