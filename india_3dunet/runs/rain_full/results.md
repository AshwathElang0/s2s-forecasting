# Rain, full-dataset training run

Third and largest real training run — the full available cache, real
`TRAIN_YEARS`/`VAL_YEARS`/`TEST_YEARS` calendar split (not a hand-picked
subset), with the training-loop speedups from `rain_medium` already in
place. Still a working-pipeline demo in the sense that plan.md S6's gaps
(train-only standardization, no fold-internal climatology, no
orography/MJO, single calendar split rather than blocked CV averaging) are
still open — but this is the closest thing to a real number produced so
far, and the biggest, cleanest gain over raw ECMWF of the three runs.

## Config

```
python scripts/build_all_samples.py --targets rain      # filled the cache to 1032/1039 dates
python train.py --target rain --model_path runs/rain_full --unit 64 \
  --batch_size 11 --n_epochs 100 --patience 10
python test.py --target rain --model_path runs/rain_full --unit 64
python scripts/compute_raw_baseline.py --target rain --out runs/rain_full
```

- Model: same architecture/capacity as the other two runs (`unit=64`),
  same AMP + `cudnn.benchmark` speedups as `rain_medium`.
- Data: **724 train / 155 val / 153 test** — the full real calendar split
  (`config.TRAIN_YEARS` 2004-2017, `VAL_YEARS` 2018-2020, `TEST_YEARS`
  2021-2023) over the full cache (1032 of 1039 possible dates; 7 dates
  have genuinely missing source `.nc`/`.grib` files, see `plan.md` S4).
- Wall time: ~11.5s/epoch (same speedup as `rain_medium`), early-stopped
  at epoch 17.

## Training curve

Early stopped at epoch 17 (patience=10); best epoch 7 (val_loss=0.961).
Same overfitting shape as `rain_medium` (train loss keeps dropping past
the best epoch, val loss climbs) but at more data the overfit is gentler
initially before accelerating late (epoch 12 onward).

| epoch | train loss | val loss |
|---|---|---|
| 1 | 1.127 | 1.018 |
| 4 | 0.970 | 0.961 |
| 7 (best) | 0.941 | **0.961** |
| 10 | 0.904 | 0.972 |
| 14 | 0.768 | 1.033 |
| 17 (stop) | 0.563 | 1.103 |

## Test-set skill (153 real 2021-2023 test samples -- the full test split)

```
pattern_corr_mean, day 1-32:
[0.553 0.620 0.617 0.598 0.582 0.573 0.538 0.508 0.497 0.476 0.452 0.444
 0.426 0.419 0.423 0.415 0.402 0.384 0.387 0.384 0.388 0.374 0.382 0.365
 0.345 0.338 0.359 0.378 0.374 0.374 0.351 0.350]

rmse_mean (mm), day 1-32:
[5.553 5.420 5.017 5.282 5.967 6.250 6.105 6.112 6.305 5.999 6.014 6.578
 6.849 6.628 6.503 6.729 6.243 6.253 6.786 7.140 6.804 6.655 6.685 6.347
 6.233 6.973 7.029 6.853 6.716 6.876 6.475 6.511]
```

## Skill relative to raw ECMWF

| lead (day) | model corr | raw corr | gain | model RMSE | raw RMSE | MSSS |
|---|---|---|---|---|---|---|
| 1 | 0.553 | 0.432 | +0.121 | 5.553 | 6.796 | +0.332 |
| 4 | 0.598 | 0.444 | +0.154 | 5.282 | 6.398 | +0.319 |
| 7 | 0.538 | 0.412 | +0.126 | 6.105 | 6.602 | +0.145 |
| 14 | 0.419 | 0.319 | +0.100 | 6.628 | 6.999 | +0.103 |
| 21 | 0.388 | 0.296 | +0.091 | 6.804 | 7.172 | +0.100 |
| 28 | 0.378 | 0.266 | +0.112 | 6.853 | 7.236 | +0.103 |
| 32 | 0.350 | 0.258 | +0.092 | 6.511 | 6.880 | +0.104 |

**32/32 lead days beat raw ECMWF on both metrics**, and by a wider margin
than either smaller run: mean correlation gain **+0.111** (vs. +0.074 at
250 samples, +0.061 at 60), mean MSSS **+0.158** (vs. +0.088, +0.029).
The gain is also front-loaded — day 1-4 MSSS is +0.32-0.33, more than
double the day-14+ plateau of ~+0.10 — which is a sensible shape: short
lead is where ECMWF's own dynamical skill is highest, so there's more
signal for the network to sharpen rather than just fall back to
climatology-like shrinkage.

## All three runs, side by side

| | rain_small (60 train) | rain_medium (250 train) | rain_full (724 train) |
|---|---|---|---|
| day-1 pattern corr | 0.416 | 0.470 | **0.553** |
| day-1 MSSS vs raw | -0.022 | +0.185 | **+0.332** |
| mean corr gain over raw ECMWF | +0.061 | +0.074 | **+0.111** |
| mean MSSS over raw ECMWF | +0.029 | +0.088 | **+0.158** |
| days beating raw on RMSE (of 32) | 22 | 32 | 32 |
| test set size | 78 (hand-picked) | 78 (hand-picked) | **153 (full, real)** |

A clean, monotonic improvement with data volume across every metric --
exactly what you'd want to see before trusting the pipeline further, and
the strongest evidence yet (short of the still-open gaps in plan.md S6)
that the 3D U-Net transfer to India rainfall is picking up real signal
rather than an artifact of a small sample.

**Still not a citable result**: one calendar split (no blocked-CV
averaging, literature-review.md S7.1's ~30% relative-skill claim from
averaging models across folds is untested here), train-only
standardization rather than fold-internal week-specific climatology,
no orography/MJO features, and no comparison yet against the other
baselines literature-review.md S4 ranks above raw ECMWF (climatology,
per-cell regression, quantile mapping) -- beating raw ECMWF is necessary
but the literature review's actual bar (S4) is beating quantile mapping
at most leads, which isn't computed here.

## Artifacts in this directory

Same schema as the other two runs: `model_epoch_*.pt`, `best_epoch`,
`stats.json`, `train_history.pkl`, `test_rain_predictions.npz`
(153, 32, 48, 64), `test_rain_scores.npz`, `raw_ecmwf_scores.npz`.
