# Rain, medium-slice training run

Second real training run, on more data than `../rain_small/results.md`
and with the training-loop speedups (mixed precision, `cudnn.benchmark`,
pinned-memory DataLoaders — see `train.py`) added after that run. Still a
working-pipeline demo, not a citable result — see `../../plan.md` S6 for
the open gaps (train-only standardization, no fold-internal climatology,
no orography/MJO).

## Config

```
python train.py --target rain --model_path runs/rain_medium --unit 64 \
  --batch_size 11 --n_epochs 25 --limit_train 250 --limit_val 50
python test.py --target rain --model_path runs/rain_medium --unit 64
python scripts/compute_raw_baseline.py --target rain --out runs/rain_medium
```

- Model: same architecture/capacity as the small run (`unit=64`).
- Data: 250 train / 50 val (vs. 60/20 in the small run), same 78-sample
  real 2021-2023 test split.
- New this run: AMP (fp16 autocast on the conv stack, fp32 loss),
  `cudnn.benchmark=True`, `pin_memory=True` DataLoaders.
- Wall time: **~11.5s/epoch** (vs. 6-15 *min*/epoch before the speedups on
  this same size of run) — see `plan.md`'s conversation log for why the
  earlier runs were so much slower (disk I/O contention, not GPU-bound).

## Training curve

Early-stopped at epoch 17 (patience=10); best epoch 7 (val_loss=0.954).
Train loss kept falling well past epoch 7 while val loss climbed from
there — clear overfitting on 250 samples, correctly caught by early
stopping rather than masked by it.

| epoch | train loss | val loss |
|---|---|---|
| 1 | 1.336 | 1.297 |
| 3 | 1.017 | 0.987 |
| 7 (best) | 0.954 | **0.954** |
| 10 | 0.915 | 0.966 |
| 14 | 0.833 | 1.031 |
| 17 (stop) | 0.691 | 1.065 |

## Test-set skill (same 78 real 2021-2023 test samples as rain_small)

```
pattern_corr_mean, day 1-32:
[0.470 0.551 0.537 0.555 0.539 0.553 0.530 0.467 0.456 0.455 0.451 0.421
 0.439 0.418 0.418 0.383 0.380 0.367 0.372 0.381 0.376 0.356 0.370 0.363
 0.365 0.330 0.366 0.384 0.367 0.342 0.326 0.349]

rmse_mean (mm), day 1-32:
[6.226 5.645 5.538 5.882 6.185 6.542 6.659 6.926 6.843 5.985 6.200 6.668
 7.175 7.300 7.221 7.200 6.441 6.687 6.978 7.488 7.671 7.651 7.438 6.595
 6.682 7.208 7.597 7.750 7.706 7.449 6.985 7.215]
```

## Skill relative to raw ECMWF

| lead (day) | model corr | raw corr | gain | model RMSE | raw RMSE | MSSS |
|---|---|---|---|---|---|---|
| 1 | 0.470 | 0.408 | +0.062 | 6.226 | 6.898 | +0.185 |
| 4 | 0.555 | 0.447 | +0.108 | 5.882 | 6.478 | +0.176 |
| 7 | 0.530 | 0.421 | +0.109 | 6.659 | 6.848 | +0.055 |
| 14 | 0.418 | 0.346 | +0.072 | 7.300 | 7.462 | +0.043 |
| 21 | 0.376 | 0.317 | +0.059 | 7.671 | 7.804 | +0.034 |
| 28 | 0.384 | 0.308 | +0.077 | 7.750 | 7.898 | +0.037 |
| 32 | 0.349 | 0.296 | +0.053 | 7.215 | 7.276 | +0.017 |

- **Correlation: 32/32 lead days beat raw ECMWF**, mean gain +0.074 (up
  from +0.061 on 60 samples).
- **RMSE: 32/32 lead days beat raw ECMWF too**, mean MSSS +0.088 (up from
  a near-wash +0.029 on 60 samples). This is the real change from more
  data: the small run corrected *pattern* well before it had learned to
  correct *magnitude*; here it's doing both, consistently.
- Full arrays: `raw_ecmwf_scores.npz`.

## Small vs. medium, side by side

| | rain_small (60 train) | rain_medium (250 train) |
|---|---|---|
| day-1 pattern corr | 0.416 | 0.470 |
| day-32 pattern corr | 0.356 | 0.349 |
| mean corr gain over raw ECMWF | +0.061 | +0.074 |
| mean MSSS over raw ECMWF | +0.029 (near-wash) | **+0.088** |
| days beating raw on RMSE | 22/32 | **32/32** |

More data mainly bought RMSE calibration, not correlation (correlation
gain barely moved). Consistent with the shrinkage story in
`literature-review.md` S3: pattern is easier to learn than magnitude, and
magnitude needs the extra samples.

**Still not a result to report**: 250 of ~724 cached train samples (and
~724 of the full ~1039×20yr record), one calendar split, no CV averaging,
same open gaps as `rain_small`. The next honest step is the full dataset
build + full `TRAIN_YEARS` split, now that the speedups make that
tractable in hours rather than potentially all day.

## Artifacts in this directory

Same schema as `../rain_small/`: `model_epoch_*.pt`, `best_epoch`,
`stats.json`, `train_history.pkl`, `test_rain_predictions.npz`,
`test_rain_scores.npz`, `raw_ecmwf_scores.npz`.
