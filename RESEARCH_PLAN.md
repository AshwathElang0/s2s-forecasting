# ECMWF-Only Daily Subseasonal Rainfall Post-Processing

## Research objective

Develop a new post-processing method that augments ECMWF ensemble precipitation forecasts for days 1-28. The method must use ECMWF forecast information only at prediction time. IMD rainfall is permitted as a historical supervision target and as independent evaluation truth, but no observed rainfall history may enter the forecast-time feature vector.

The primary outputs are 28 daily rainfall predictions. Weekly totals for weeks 1-4 are derived from those daily predictions and remain important secondary evaluation targets.

## Proposed contribution

The first direction is a **shared temporal residual post-processor**:

1. Convert the ECMWF accumulated precipitation fields into daily increments for all 28 valid days.
2. Derive permutation-invariant ensemble features for each day: mean, spread, lower quantile, median, upper quantile, and wet-member fraction.
3. Use a compact temporal convolutional network to predict a correction to the log-transformed ECMWF ensemble mean across all 28 days jointly.
4. Penalize abrupt changes in the learned correction between adjacent forecast days.
5. Preserve nonnegative rainfall through an inverse-log output transform.
6. Aggregate the daily predictions into weekly totals for comparison with Standard ABC and existing baselines.

The correction is initialized at zero, so the model starts from the ECMWF ensemble mean and must earn each change through training.

## Why this is a new research direction

The current ABC implementation predicts four weekly totals and uses observed rainfall history through Persistence++. The proposed method instead addresses daily temporal placement directly and treats the forecast trajectory as the object being corrected. It does not simply replace one set of weekly mixture weights with another.

The research hypothesis is:

> A shared temporal correction model can improve the daily trajectory and the resulting weekly accumulations by learning lead-dependent, state-dependent ECMWF bias while preserving temporal coherence.

## Baselines

The experiment must include:

- Raw ECMWF control forecast.
- Raw ECMWF ensemble mean.
- Seasonal climatology estimated from historical training observations.
- Existing weekly Standard ABC, clearly labeled as an observation-history baseline if Persistence++ remains enabled.
- ECMWF-only daily bias correction without temporal convolution.
- The proposed ECMWF-only temporal residual model.

The observation-history Standard ABC baseline must not be presented as deployment-equivalent to the proposed method. Its Persistence++ input is unavailable under the new deployment constraint.

## Experimental protocol

### Data split

Use chronological initialization-date splits:

- Training dates: earlier years.
- Validation dates: a later block inside the training period.
- Test dates: strictly later years, untouched until final evaluation.

All normalization constants, climatologies, bias corrections, thresholds, and early-stopping decisions must be fit using training data only.

### No-observation-input audit

Every forecast-time model input must be traceable to:

- ECMWF control or perturbed forecasts.
- Calendar metadata.
- Static location metadata.
- Historical training climatology or learned parameters.

The following must be excluded from prediction inputs:

- Prior seven-day observed rainfall.
- Any observation from the current forecast case.
- IMD-derived recent rainfall anomalies.

### Daily metrics

Evaluate:

- Daily RMSE and MAE.
- Daily anomaly correlation.
- Wet-day occurrence metrics.
- Continuous ranked probability score if an ensemble/distributional extension is added.
- Calibration and Brier Score for daily exceedance events.

### Weekly metrics

Aggregate the 28 daily predictions into four seven-day totals and evaluate:

- Weekly RMSE and MAE.
- ACC.
- Brier Score and Brier Skill Score for local 99th-percentile events.
- Timing/placement error, comparing the predicted and observed daily rainfall profiles within each week.

### Uncertainty

Use initialization-date block bootstrap or rolling-origin variation. IID resampling of grid cells is not sufficient because spatial cells from one forecast initialization are correlated.

## Upgrade sequence

### Stage 1: establish the daily baseline

Implement the daily data adapter and verify that accumulated ECMWF fields are converted into 28 daily increments. Compare raw ensemble mean, control, and climatology.

### Stage 2: validate the residual model

Run the current `ECMWFDailyPostprocessor` with only ensemble-derived features. Compare it with a no-learning identity model and a day-independent bias correction.

Success requires improvement on held-out dates in either daily RMSE or weekly RMSE without materially worsening calibration.

### Stage 3: add spatial context

Extend the model to see neighboring ECMWF grid cells or a compact spatial convolution. The current first version treats each grid-cell trajectory independently; spatial coherence is a planned extension.

### Stage 4: add probabilistic output

Replace the point correction with a zero-inflated Gamma, hurdle, quantile, or ensemble-generating head. Evaluate both rainfall amounts and event probabilities.

### Stage 5: compare against stronger post-processing methods

Test permutation-invariant ensemble networks, analog-CNN post-processing, and spatially coherent precipitation post-processing ideas. The comparison should focus on whether the proposed method provides a useful accuracy/complexity tradeoff for the India domain.

### Stage 6: rolling-origin paper evaluation

Run several chronological origins and report mean skill differences with block uncertainty intervals. A single favorable test period is not sufficient evidence.

## Falsification criteria

The proposed method should not be claimed as an improvement if:

- It only improves training loss but not held-out daily or weekly metrics.
- Its gains disappear under rolling-origin validation.
- It improves weekly totals only by destroying daily timing.
- It requires observed rainfall at inference time.
- A simple bias correction matches it with substantially lower complexity.

## Literature anchors

Relevant directions include:

- Permutation-invariant neural post-processing of ensemble weather forecasts, DOI: 10.1175/AIES-D-23-0070.1.
- Hybrid analog-ensemble and CNN precipitation post-processing, DOI: 10.1175/MWR-D-21-0154.1.
- Spatially correlated CNN post-processing for ensemble precipitation, DOI: 10.1175/MWR-D-25-0219.1.
- Deep learning enhancement of dynamical-statistical subseasonal precipitation forecasts, DOI: 10.1088/1748-9326/ad5370.
- Deep learning with spatio-temporal conditioning for global subseasonal precipitation, DOI: 10.1038/s41612-026-01430-8.

These works motivate the direction but do not replace the need for a clean ECMWF-only comparison on the India domain.
