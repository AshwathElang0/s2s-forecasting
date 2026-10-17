# Rain, small-slice training run

First real (full-capacity) training run of the 3D U-Net on cached ECMWF/IMD
rain samples — a working-pipeline demo, not a result to cite. See
`../../plan.md` for what's implemented vs. still scoped, and `../../
plan.md` S6 for the honest gaps (train-only standardization, no
fold-internal climatology yet, no orography/MJO).

## Config

```
python train.py --target rain --model_path runs/rain_small --unit 64 \
  --batch_size 11 --n_epochs 15 --limit_train 60 --limit_val 20
python test.py --target rain --model_path runs/rain_small --unit 64
```

- Model: full paper capacity (`unit=64`), same architecture as the
  reference 3D U-Net (`model/unet3d.py`).
- Data: 60 train / 20 val samples (capped via `--limit_train`/`--limit_val`
  to keep load time short — `load_raw` decompresses every cached `.npz` up
  front, which dominated wall time at larger sample counts), 78 test
  samples (the full, real 2021-2023 test split — not capped).
- Loss: masked MSE + masked latitude-weighted pattern correlation.
- Wall time: ~18 min total (15 epochs, GPU).

## Training curve

Train and val loss both decrease steadily and plateau around epoch 10-13;
best val loss at epoch 13 (0.9421). No divergence, no NaNs, no
overfitting blowup even at only 60 train samples -- see
`train_history.pkl` for the full per-epoch (loss, mse, corr) tuples.

| epoch | train loss | val loss |
|---|---|---|
| 1 | 1.676 | 1.473 |
| 5 | 1.090 | 1.107 |
| 10 | 0.995 | 0.949 |
| 13 (best) | 0.953 | **0.942** |
| 15 | 0.947 | 0.954 |

## Test-set skill (78 real 2021-2023 test samples)

Pattern correlation by lead day, full 32-day array in `test_rain_scores
.npz["pattern_corr_mean"]`:

```
day  1- 7: 0.416 0.468 0.482 0.509 0.500 0.500 0.496
day  8-14: 0.431 0.445 0.444 0.443 0.419 0.418 0.415
day 15-21: 0.410 0.390 0.378 0.376 0.369 0.363 0.385
day 22-28: 0.365 0.373 0.362 0.372 0.335 0.350 0.397
day 29-32: 0.375 0.350 0.336 0.356
```

RMSE by lead day (mm, on the raw rain scale) is in the same file under
`rmse_mean`, roughly flat at 6-8mm across all leads.

**What's encouraging**: the correlation curve has the physically sensible
shape the literature review predicts (S3) -- high at short lead (~0.5 at
day 1-7), decaying smoothly toward ~0.35-0.37 by day 28-32, rather than
noise. That's a qualitatively real pattern, not an artifact of a tiny
sample.

## Skill relative to raw ECMWF (the comparison that actually matters)

`scripts/compute_raw_baseline.py` scores the raw ECMWF ensemble-mean `tp`
forecast (channel 0 of every cached sample, unstandardized mm/day) against
the same 78 test samples with the same masked metrics -- the fair
comparison, not absolute RMSE (literature-review.md S3/S6.1).

| lead (day) | model corr | raw ECMWF corr | gain | model RMSE | raw RMSE | MSSS |
|---|---|---|---|---|---|---|
| 1 | 0.416 | 0.408 | +0.008 | 6.97 | 6.90 | -0.022 |
| 4 | 0.509 | 0.447 | +0.062 | 6.27 | 6.48 | +0.063 |
| 7 | 0.496 | 0.421 | +0.075 | 7.00 | 6.85 | -0.044 |
| 14 | 0.415 | 0.346 | +0.069 | 7.52 | 7.46 | -0.017 |
| 21 | 0.385 | 0.317 | +0.068 | 7.81 | 7.80 | -0.001 |
| 28 | 0.397 | 0.308 | +0.090 | 7.82 | 7.90 | +0.021 |
| 32 | 0.356 | 0.296 | +0.060 | 7.30 | 7.28 | -0.006 |

- **Pattern correlation: model beats raw ECMWF at all 32/32 lead days**,
  mean gain +0.061. A real, consistent effect even at 60 training samples.
- **RMSE (MSSS = 1 - MSE_model/MSE_raw): a near-wash**, model wins 22/32
  days, mean MSSS only +0.029. This is the shrinkage signature
  literature-review.md S3 predicts: the model is correcting *pattern*
  well before it's learned to correct *magnitude*, which is exactly what
  you'd expect from 60 samples and 15 epochs, not evidence the approach
  doesn't work.
- Full per-lead arrays: `raw_ecmwf_scores.npz` (same schema as
  `test_rain_scores.npz`).

**What this is not**: a result to report. 60 train samples is far short of
the ~724 available in the current cache (let alone the full ~1039-date x
20-year record), no CV averaging, no hyperparameter tuning, and the
train-only standardization gap in plan.md S6 is still open. Treat these
numbers as "the wiring produces sensible output," not "this is the
model's skill."

## Artifacts in this directory

- `model_epoch_1.pt` ... `model_epoch_15.pt` -- per-epoch checkpoints
- `best_epoch` -- 12 (0-indexed; epoch 13 in the table above)
- `stats.json` -- train-only feature/target standardization stats used
- `train_history.pkl` -- full per-epoch (loss, mse, corr) for train/val
- `test_rain_predictions.npz` -- `preds`, `obs`, `mask` arrays, (78, 32, 48, 64)
- `test_rain_scores.npz` -- `pattern_corr_mean/std`, `rmse_mean/std`,
  `epre_mean/std`, each (32,)
