# 3D U-Net for India rainfall — results so far

A 3D U-Net that corrects ECMWF's 32-day rainfall forecast over India. The architecture, loss and training are ported from a published model built for the western US.

## References

| | link |
|---|---|
| Reference paper | Ryu, Kim, Wang & Yoon (2026), [*Increasing resolution and accuracy in sub-seasonal forecasting through 3D U-Net: the western US*](https://gmd.copernicus.org/articles/19/27/2026/), Geoscientific Model Development 19, 27–39 |
| Reference code (what we ported from) | [github.com/W-US-BCSR/IRnAF](https://github.com/W-US-BCSR/IRnAF) · archived release on [Zenodo](https://zenodo.org/records/14776781) |
| Forecast input data | [ECMWF S2S reforecasts](https://apps.ecmwf.int/datasets/data/s2s/levtype=sfc/type=pf/) |
| Observed rainfall (truth) | [India Meteorological Department, Pune](https://imdpune.gov.in), gridded daily rainfall, downloaded with [imdlib](https://imdlib.readthedocs.io/en/latest/) |
| Only published U-Net for Indian monsoon S2S rainfall | *U-Net calibration of subseasonal Indian summer monsoon rainfall forecasts*, Climate Dynamics (2026), [doi:10.1007/s00382-026-08286-x](https://doi.org/10.1007/s00382-026-08286-x) |

---

## 1. Inputs and outputs

Each training sample is one ECMWF forecast, identified by its **issue date** (the day the forecast was made). Example: issue date **2015-01-11**.

### Inputs — 17 channels, each a 32-day × 48 × 64 block

The grid is 1.5° over India and the surrounding ocean. Every channel has one map per future day (day 1 = 2015-01-11, day 32 = 2015-02-11).

| # | channel | what it is | type |
|---|---|---|---|
| 1 | rain mean | ECMWF rainfall forecast, averaged over its 11 ensemble members | forecast |
| 2 | rain spread | how much the 11 members disagree | forecast |
| 3–6 | humidity | at 200, 500, 700, 850 hPa | forecast |
| 7–10 | winds | east–west and north–south, at 200 and 850 hPa | forecast |
| 11 | pressure | sea-level pressure | forecast |
| 12–13 | heights | geopotential height at 200 and 1000 hPa | forecast |
| 14 | SST | sea-surface temperature | forecast |
| 15 | yesterday's rain | IMD **observed** rain on 2015-01-10, copied into all 32 days | observation |
| 16–17 | calendar | sin/cos of day-of-year, changes along the 32 days | date |

Channels 1–14 are ECMWF's predictions **for each future day**, not past observations. Channel 15 is the only real observation, and it covers one day. **It is dropped in the headline model** (see [Section 8](#8-review-response-professors-comments)): observed IMD rain would not be available in real time. The input domain is 21–109.5°E, 19.5°S–39°N; Section 8.2 explains how much of it feeds each prediction.

### Output — 1 channel, 32 days × 48 × 64

Corrected daily rainfall (mm/day) for 2015-01-11 → 2015-02-11: one map per day.

### Truth

IMD's observed daily rainfall on those same 32 days, only over Indian land cells (ocean and neighbouring countries are masked out).

### Data split

| | years | samples |
|---|---|---|
| train | 2004–2017 | 724 |
| validation (for early stopping) | 2018–2020 | 155 |
| test (never seen in training) | 2021–2023 | 153 |

Results are reported at **forecast days 7, 14, 21 and 28** (weeks 1–4).

### Scaling the data

Before training, every input channel and the rainfall target are rescaled to average 0 and spread 1. Rainfall is first converted to `log(1 + rain)`, because rain is heavily skewed. The same average and spread are applied to every sample. This is just unit conversion: it tells the model nothing about any specific day.

The question is which years those averages come from:

| | years used to compute the scaling | includes test years? |
|---|---|---|
| Reference paper ([code](https://github.com/W-US-BCSR/IRnAF), `preprocess/ECMWF_data.py`) | train + validation + test combined | **yes** |
| This project | train only (2004–2017) | no |

The paper computes its scaling over all three periods at once, including the 2022 test year it reports results on, and scales the observed rainfall the same way. We fixed this when porting: our scaling uses training years only. The practical effect is small in both cases, because it's a single number per variable, but it means test-period data touched the paper's pipeline and not ours.

---

## 2. Architecture

The "3D" is **lead-time × latitude × longitude**. A single network sees the whole 32-day forecast at once, so errors at day 21 can be corrected using what it saw at days 1–20.

### The reference paper's diagram

![3D U-Net architecture, Figure 1 of Ryu et al. 2026](https://gmd.copernicus.org/articles/19/27/2026/gmd-19-27-2026-f01-web.png)

*Figure 1 from [Ryu et al. 2026](https://gmd.copernicus.org/articles/19/27/2026/), reproduced under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The left side (green) shrinks the data, the centre (teal) is the bottleneck, and the right side (blue) expands it back. The dashed grey arrows are skip connections that pass fine detail straight across.*

Our network has the same structure. The differences are the sizes: their grid is 72 × 72 at 0.25° over the western US, ours is 48 × 64 at 1.5° over India, and our input has 17 channels.

### Our version, with our sizes

Each row shows: stage, number of channels, and data size (lead days × lat × lon).

```
input               17 ch   32 × 48 × 64    (lead × lat × lon)
│
├─ Encoder 1        64 ch   32 × 48 × 64  ───────────────┐ skip
│  avg-pool ÷2                                           │
├─ Encoder 2       128 ch   16 × 24 × 32  ─────────┐ skip│
│  avg-pool ÷2                                     │     │
├─ Encoder 3       256 ch    8 × 12 × 16  ───┐ skip│     │
│  avg-pool ÷2                               │     │     │
├─ Bottleneck      512 ch    4 ×  6 ×  8     │     │     │
│  up-conv ×2                                │     │     │
├─ Decoder 1       256 ch    8 × 12 × 16  ◄──┘     │     │
│  up-conv ×2                                      │     │
├─ Decoder 2       128 ch   16 × 24 × 32  ◄────────┘     │
│  up-conv ×2                                            │
├─ Decoder 3        64 ch   32 × 48 × 64  ◄──────────────┘
│
output               1 ch   32 × 48 × 64    corrected rainfall
```

- Each encoder/decoder block: `[3×3×3 conv → BatchNorm → GELU] × 2`
- Down: average pooling. Up: learned transposed convolution.
- Skip connections carry fine detail from the encoder straight to the decoder.
- **22.4 M parameters.** Training samples: 724.

**Loss** = MSE + (1 − pattern correlation), computed only over IMD land cells.

**Training:** Adam, lr 1e-4, batch 11, stops after 10 epochs with no improvement on validation.

This is why the input needs all 32 days: the lead axis is halved three times (32 → 16 → 8 → 4). With only 4 days it would fail at the third halving.

---

## 3. How we measure

| metric | meaning | perfect |
|---|---|---|
| pattern correlation | does the predicted rainfall map match the real one | 1 |
| RMSE | average error, mm/day | 0 |
| E_pre | paper's combined score (correlation + variability) | 0 |
| SD ratio | variability of prediction ÷ variability of reality | 1 (below 1 = too flat) |

---

## 4. Experiments

### Experiment 1 — Does more training data help?

Same model trained on 60, 250 and 724 samples. The first two used a smaller, hand-picked test set (78 samples).

| training samples | day-1 corr | average corr gain over raw ECMWF | RMSE, mm/day (32-day avg) | raw ECMWF RMSE, same test set |
|---|---|---|---|---|
| 60 | 0.416 | +0.061 | 7.08 | 7.19 |
| 250 | 0.470 | +0.074 | 6.87 | 7.19 |
| **724 — the Baseline model** | **0.553** | **+0.111** | **6.37** | 6.94 |

RMSE is only comparable within a row: the first two rows use the smaller 78-sample test set.

More data helps steadily. The 724-sample model beats raw ECMWF on all 32 days. It is called **the Baseline model** from here on. It overfits early: best at epoch 7.

---

### Experiment 2 — Is the score honest?

Two reference forecasts were added:

- **Climatology**: "predict the long-term average rain for this date and place." It contains no forecast skill at all.
- **Debiased ECMWF**: ECMWF with its average bias for each date and lead removed.

Each forecast was then scored twice: normally, and with climatology subtracted from both prediction and truth, so only the departure from normal is judged.

Every climatology in this report uses only years **before** the one being forecast. A 2010 forecast uses the average of 1951–2009, and a 2022 forecast uses 1951–2021. No later year ever feeds into an earlier year's "normal".

> **Experiments 2–7 are scored on aligned days** (forecast day d = ECMWF lead d−1 against IMD date issue+d; 31 days). They were first scored one day off, see Experiment 9. The rescoring is [`experiments/rescore_aligned.py`](experiments/rescore_aligned.py), with full output in [`experiments/aligned_rescore.md`](experiments/aligned_rescore.md). Every conclusion survived except Experiment 6's.

| | corr, normal scoring | corr, climatology removed | RMSE, mm/day (31-day avg) |
|---|---|---|---|
| raw ECMWF | 0.346 | 0.205 | 6.84 |
| climatology | **0.361** | 0.000 | 6.85 |
| debiased ECMWF | 0.371 | 0.246 | 6.52 |
| Baseline model | 0.436 | **0.325** | **6.40** |

(One day off, as first reported: raw 0.329 / 0.183 / 6.94, debiased 0.353 / 0.222 / 6.62, Baseline 0.440 / 0.332 / 6.37. Alignment raises the ECMWF references but barely moves the U-Net: it was trained against the same IMD dates it is scored on, so only the first forecast day drops out.)

**Under normal scoring, climatology beats raw ECMWF.** Most of India's rainfall map is predictable from the calendar alone (the west coast is wet every July), so normal scoring partly rewards knowing the month.

With climatology removed, the Baseline model scores 0.325. That is its real forecasting skill on daily maps, and it is still clearly better than every reference. (On weekly totals it is not: see Experiment 9.)

---

### Experiment 3 — Predicting the departure from normal

**Idea:** instead of predicting the whole rainfall map, predict how far rain will depart from normal for that date and place, then add the normal back.

**Formula.** Per grid cell and forecast day, on the `log(1 + rain)` scale:

```
obs         = log(1 + IMD observed rain)
ecmwf       = log(1 + ECMWF rain, 11-member mean)
normal      = average obs for this day of year, years before this one only
ecmwf_usual = average ecmwf for this issue week and forecast day, earlier years only

obs_dev     = (obs − normal) / σ_obs          target the network learns
ecmwf_dev   = (ecmwf − ecmwf_usual) / σ_ecmwf  replaces the ECMWF rain input channel

predicted rain = exp(normal + σ_obs × network output) − 1
```

Same U-Net and same 17 inputs as the Baseline; yesterday's rain is also given as its departure from normal. Subtracting ECMWF's usual forecast from the input but adding the *observed* normal back also removes ECMWF's average bias. To avoid leakage, both averages for year Y use only years before Y; 2004 has no earlier forecast year, so it was dropped, leaving 672 training samples (Baseline: 724).

**Results, 3 seeds each (0, 1, 2).** Baseline is the 17-input `all_inputs_s{0,1,2}` ([Exp 5 folder](experiments/exp05_input_variables/)); departure runs are `departure_s{0,1,2}` ([Exp 3 folder](experiments/exp03_departure_from_normal/)).

| run | corr, clim. removed | day 7 | day 14 | day 21 | day 28 | corr, standard | RMSE |
|---|---|---|---|---|---|---|---|
| Baseline (17 inputs) | 0.315 (0.305–0.323) | 0.486 | 0.308 | 0.163 | 0.137 | 0.440 | 6.42 |
| **Departure from normal** | **0.340** (0.335–0.343) | 0.506 | 0.312 | 0.222 | 0.174 | 0.438 | 6.43 |

Mean of 3 seeds; brackets are lowest–highest seed. Day columns are climatology-removed correlation.

| comparison (clim. removed, 3-seed averages) | difference | 95% range | seeds overlap? | real? |
|---|---|---|---|---|
| departure from normal vs Baseline | **+0.026** | +0.015 to +0.037 | no | **yes** |

**Predicting the departure from normal is a real gain on daily maps (+0.026)**: every one of its runs beats every Baseline run, the gain grows with lead (+0.04 at day 28), and it uses fewer training samples. Standard correlation and RMSE are identical, so the gain is in placing wetter/drier-than-normal areas.

**But not on weekly totals.** Scored as in Experiment 9, it ties the Baseline at weeks 1–2 and is *worse* at weeks 3–4 per cell (−0.032 [−0.058, −0.003] and −0.038 [−0.064, −0.013]). The daily log-space gain does not carry over to weekly amounts.

**Note — trust gate, dropped.** We also tried a two-output head, `d = sigmoid(g) × ecmwf_dev + correction`, letting the network set how much of ECMWF's departure to pass through (`gate_s{0,1,2}`). Over 3 seeds it scored 0.342, the same as the plain departure model (+0.001, −0.003 to +0.005). An earlier single run credited it with +0.014; that was a lucky seed. A variant feeding ECMWF rain only through the gate scored at Baseline level.

---

### Experiment 4 — The paper's E_pre metric, reported at weeks 1–4

**Bug found:** our port of E_pre gave 1.386 for a perfect forecast instead of 0 (a constant of 16 where it should be 4). Now fixed. The error is a constant offset, so no ranking changes.

**Baseline model:**

| | day 7 | day 14 | day 21 | day 28 |
|---|---|---|---|---|
| pattern corr | 0.508 | 0.423 | 0.374 | 0.374 |
| RMSE (mm) | 6.11 | 6.50 | 6.65 | 6.72 |
| E_pre | 2.17 | 2.99 | 3.11 | 3.27 |
| SD ratio (median over forecasts) | **0.39** | **0.29** | **0.28** | **0.27** |

The SD ratio was first reported as a mean over forecasts (0.65 / 0.44 / 0.86 / 0.46). That mean is dominated by a few near-dry maps whose observed spread is close to zero; their ratios exceed 200. The median is the reliable number.

**The predictions are too flat.** They carry roughly a third of the real day-to-day spatial variability. The paper reported the same problem for rain in mountain terrain. It is caused by the MSE loss, which pushes the model to hedge toward the average.

---

### Experiment 5 — Do the extra input variables help?

This is the paper's test. Channel groups were added one at a time, with the same seed and all 724 training samples.

Pattern correlation (normal scoring):

| run | inputs | channels | day 7 | day 14 | day 21 | day 28 | avg (31 days) |
|---|---|---|---|---|---|---|---|
| **Rain-only** | rain mean, spread, yesterday, calendar | 5 | 0.497 | 0.415 | 0.375 | 0.384 | 0.439 |
| + Humidity | + 4 humidity levels | 9 | 0.501 | 0.423 | 0.375 | 0.381 | 0.439 |
| + Winds | + 4 wind fields | 13 | 0.501 | 0.427 | 0.371 | 0.385 | 0.438 |
| **All inputs** | + pressure, heights, SST | 17 | 0.509 | 0.430 | 0.381 | 0.391 | 0.441 |

Climatology removed, 31-day average: 0.321 / 0.331 / 0.320 / 0.323.

RMSE (mm/day):

| run | channels | day 7 | day 14 | day 21 | day 28 | avg (31 days) |
|---|---|---|---|---|---|---|
| Rain-only | 5 | 6.19 | 6.60 | 6.71 | 6.76 | 6.49 |
| + Humidity | 9 | 6.14 | 6.53 | 6.69 | 6.74 | 6.45 |
| + Winds | 13 | 5.98 | 6.44 | 6.62 | 6.67 | 6.36 |
| All inputs | 17 | 6.21 | 6.57 | 6.72 | 6.74 | 6.50 |

**All four are the same within noise.** Average correlation ranges 0.438–0.441 (seed spread of one setup: up to 0.009), climatology-removed 0.320–0.331 (seed spread up to 0.018), RMSE 6.36–6.50 mm/day (seed spread up to 0.15). Adding humidity, then winds, then pressure/heights/SST never produces a gain that stands out. The 12 atmospheric channels add nothing measurable, which matches the paper's finding.

One thing to check: on weekly totals (Experiment 9 scores), the + Winds run is the best U-Net from Experiments 1–7 (per cell 0.643 / 0.449 / 0.265 / 0.174 vs 0.603 / 0.371 / 0.212 / 0.145 for all inputs). It is a single seed, scored from an early checkpoint (see below), so this is not established.

(The + Winds run hit a transient GPU fault at epoch 10, before its test step. It was scored from its saved best checkpoint, epoch 5. Validation loss had worsened every epoch after that, and every other run in this experiment also peaked at epochs 4–5.)

---

### Experiment 6 — Does yesterday's observed rain help?

Yesterday's IMD rain (the day before the forecast is issued) is the only real observation among the inputs. No paper we have recommends it, and the reference paper doesn't use it. Every model in Experiments 1–5 had it, so it had never been tested.

We trained the 17-input setup with and without it (16 inputs), same 724 training forecasts, **3 seeds each** (`all_inputs_s{0,1,2}`, `no_obs_s{0,1,2}`). The first version of this section had one seed each and was scored one day off; this is the aligned 3-seed version.

| score | span | with (17) | without (16) | with − without | 95% range |
|---|---|---|---|---|---|
| corr, standard | all 31 days | 0.440 | 0.440 | +0.000 | −0.001 to +0.002 |
| corr, climatology removed | days 1–3 | 0.639 | 0.640 | −0.001 | −0.004 to +0.002 |
| | days 8–14 | 0.380 | 0.382 | −0.002 | −0.008 to +0.003 |
| | days 15–31 | 0.176 | 0.189 | −0.012 | −0.020 to −0.005 |
| | all 31 days | 0.315 | 0.322 | −0.007 | −0.012 to −0.004 |
| RMSE, mm/day | all 31 days | 6.42 | 6.41 | +0.01 | +0.01 to +0.02 |

Paired bootstrap over the 153 test forecasts, on seed-averaged scores. Seed range, climatology removed: with 0.305–0.323, without 0.311–0.329.

- **No difference on standard scoring, at short lead, or on RMSE.**
- **The earlier "slightly better with it at long lead" (+0.018, days 15–32) is gone.** With 3 seeds and aligned days, it goes the other way (−0.012), and that difference is smaller than the seed spread (0.018), so read it as no effect.
- On weekly totals the two setups trade small wins (all-India week 1: with it +0.072 [+0.049, +0.101]; per cell week 2: without it +0.015). Nothing consistent.

**Verdict:** yesterday's observed rain does nothing measurable. Dropping it (the 16-input headline model) is right.

---

### Experiment 7 — Does the day-of-year input help? (3 seeds each)

The reference paper gives the network only ECMWF forecasts. We also give it the day of the year (sin/cos, 2 channels). We trained the paper-style setup (the 14 ECMWF inputs only, `ecmwf_only`) against the 16-input headline model (`no_obs`, the same 14 plus calendar). The historical mean was never an input. First experiment with **3 seeds per setup** (0, 1, 2); same 724 training and 153 test forecasts.

| score, 31-day avg | with calendar (16) | without (14) | difference |
|---|---|---|---|
| corr, standard | 0.440 (0.437–0.442) | 0.439 (0.434–0.443) | +0.000 |
| corr, climatology removed | 0.322 (0.311–0.329) | 0.330 (0.320–0.336) | −0.007 |
| RMSE, mm/day | 6.41 (6.37–6.44) | 6.41 (6.39–6.45) | −0.01 |

Mean of 3 seeds; brackets are the lowest and highest seed.

| corr, standard | day 7 | day 14 | day 21 | day 28 |
|---|---|---|---|---|
| with calendar | 0.507 | 0.423 | 0.376 | 0.386 |
| without | 0.504 | 0.424 | 0.372 | 0.381 |

**The calendar adds nothing measurable.** Identical on standard correlation and RMSE. With climatology removed, *without* is 0.007 higher; the paired bootstrap on seed-averaged scores calls it real (−0.012 to −0.002), but seeds of the same setup differ by up to 0.017, so it is read as no difference. On weekly totals, *with* is slightly better at weeks 2–3 per cell (+0.018, +0.020) and worse on all-India week 1 (−0.044): mixed again. ECMWF's fields already carry the season.

**Retraining noise, measured.** Spread across 3 seeds of one setup (Experiments 3, 5–7):

| score | seed spread |
|---|---|
| corr, standard | 0.001–0.009 |
| corr, climatology removed | **0.009–0.018** |
| RMSE, mm/day | 0.05–0.15 |

Earlier sections assumed ~0.006, from one accidental repeat on the standard score. On the climatology-removed score it is about 2.5× that. The bootstrap only covers which test forecasts were drawn, not training randomness. Large results stand (beating raw/debiased ECMWF, flat predictions, more data helping); the single-run climatology-removed gains in Experiments 3 and 6 (+0.011, +0.014, +0.011) were within this noise. Rerunning Experiment 3 with seeds 1 and 2 confirmed the departure-from-normal gain (+0.026 aligned) and removed the gate's (+0.001). Experiment 6's single-run +0.011 did not survive 3 seeds.

---

*Experiment 8 is the review response, [Section 8](#8-review-response-professors-comments).*

### Experiment 9 — How far does plain bias correction get?

**Dyn++** (from the ABC paper, Mouatadid et al. 2023) multiplies ECMWF's rain by one correction factor per grid cell, lead week and calendar month: (observed weekly rain + 1) / (ECMWF weekly rain + 1), fitted on 2004–2017 over issue dates within ±1 month, capped at ×0.25–×4. No neural network: 12 × 4 × 187 numbers. Scored on the same 153 test forecasts, **on correctly aligned days** (ECMWF lead *t* against IMD date issue+1+*t*; Experiments 1–8 were scored one day off, see [`preprocess/alignment.py`](preprocess/alignment.py)), days 1–28. Folder: [`experiments/exp09_bias_correction/`](experiments/exp09_bias_correction/).

Weekly totals with the 1951–2020 normal removed. "Per cell" pools every land cell and forecast; "all-India" correlates the India-average anomaly across forecasts. U-Nets: mean of 3 seeds.

| | per cell, wk 1 | wk 2 | wk 3 | wk 4 | all-India, wk 1 | wk 2 | wk 3 | wk 4 |
|---|---|---|---|---|---|---|---|---|
| raw ECMWF | 0.556 | 0.382 | 0.210 | 0.130 | 0.890 | 0.790 | 0.567 | 0.396 |
| debiased ECMWF (Exp 2) | 0.612 | 0.406 | 0.212 | 0.141 | 0.753 | 0.522 | 0.269 | 0.145 |
| **Dyn++** | **0.668** | **0.512** | **0.330** | **0.239** | **0.910** | **0.837** | **0.590** | 0.389 |
| U-Net, 16 inputs (headline) | 0.591 | 0.386 | 0.224 | 0.142 | 0.535 | 0.398 | 0.203 | 0.095 |
| U-Net, departure from normal | 0.603 | 0.374 | 0.181 | 0.106 | 0.607 | 0.432 | 0.183 | 0.079 |

| daily, days 1–28 | corr, clim. removed | corr, standard | RMSE, mm/day | bias, mm/day | SD ratio |
|---|---|---|---|---|---|
| raw ECMWF | 0.223 | 0.356 | 6.82 | −0.03 | 0.60 |
| debiased ECMWF (Exp 2) | 0.266 | 0.382 | 6.50 | −1.20 | 0.48 |
| Dyn++ | 0.238 | 0.406 | 6.57 | −0.06 | 0.63 |
| U-Net, 16 inputs | 0.342 | 0.448 | 6.39 | −1.55 | 0.39 |
| U-Net, departure from normal | 0.360 | 0.445 | 6.40 | −1.53 | 0.40 |

- **Dyn++ beats raw ECMWF by about +0.11 per cell at every week** (95% range +0.09 to +0.14 at week 1, +0.08 to +0.17 at week 3), keeps ECMWF's all-India signal, and has almost no bias.
- **Both U-Nets are worse than Dyn++ on weekly totals**: −0.07 to −0.15 per cell, −0.3 to −0.4 all-India (bootstrap ranges exclude 0 at every week).
- **The U-Nets still win on daily climatology-removed correlation** (0.34–0.36 vs 0.24), the measure used in Experiments 2–7. That score is computed per day in log(1 + rain), which rewards getting the wet/dry pattern right, including light rain. Weekly totals in mm reward getting the amounts right where it rains, and are the usual S2S unit. The U-Net's gain does not survive the change of measure.
- **Corrections made in log space come out dry** (Exp 2 debiasing −1.2 mm/day, U-Nets −1.5); Dyn++ works in mm and does not.

Dyn++ is now the reference to beat.

### Experiment 10 — A U-Net on top of Dyn++

Same 16 inputs as the headline model, trained on aligned days, scored as in Experiment 9. Three variants. Folder: [`experiments/exp10_unet_on_bias_correction/`](experiments/exp10_unet_on_bias_correction/).

- **`plain_aligned`**: the old U-Net (log space) retrained on aligned days. This is the control.
- **`residual`**: Dyn++ rain replaces raw ECMWF rain as input, and the output is Dyn++ plus a log-space correction that starts at zero. Same loss as before.
- **`residual_mm`**: output = Dyn++ × exp(correction), trained in mm/day with MSE + (1 − pattern correlation of the departure from normal). No all-India loss term.

`residual_mm`: mean of 3 seeds. The other two U-Nets: 1 seed each.

| | per cell, wk 1 | wk 2 | wk 3 | wk 4 | all-India, wk 1 | wk 2 | wk 3 | wk 4 |
|---|---|---|---|---|---|---|---|---|
| Dyn++ | 0.668 | 0.512 | 0.330 | 0.239 | **0.910** | 0.837 | 0.590 | 0.389 |
| `plain_aligned` | 0.551 | 0.358 | 0.207 | 0.136 | 0.459 | 0.331 | 0.170 | 0.073 |
| `residual` | 0.628 | 0.389 | 0.196 | 0.141 | 0.613 | 0.400 | 0.189 | 0.104 |
| **`residual_mm`** | **0.730** | **0.558** | **0.404** | **0.299** | 0.901 | **0.838** | **0.624** | **0.420** |

| weekly RMSE, mm/week | per cell, wk 1 | wk 2 | wk 3 | wk 4 | all-India, wk 1 | wk 2 | wk 3 | wk 4 |
|---|---|---|---|---|---|---|---|---|
| Dyn++ | 26.8 | 30.8 | 34.0 | 35.0 | **4.6** | **5.9** | **8.6** | **9.8** |
| **`residual_mm`** | **24.6** | **29.8** | **32.7** | **34.2** | 5.4 | 6.3 | 8.8 | 10.0 |

Daily, days 1–28, `residual_mm`: RMSE 6.23 mm/day (Dyn++ 6.57), bias −0.37 mm/day, SD ratio 0.60.

- **`residual_mm` beats Dyn++ per cell at every week**: +0.063 [+0.044, +0.082] at week 1, +0.046 / +0.074 / +0.060 at weeks 2–4. Every 95% range excludes 0, and the 3 seeds agree within 0.02.
- **It ties Dyn++ on all-India**: every range includes 0 (week 1: −0.009 [−0.024, +0.005]). No all-India loss term was needed.
- **Training in mm is what matters.** `residual` starts from the same Dyn++ but trains in log space, and it falls below Dyn++ (−0.04 to −0.13 per cell) and comes out ~50% too dry.
- **Aligning the dates alone does not help the old design**: `plain_aligned` ≈ the Experiment 6 model.
- **Caveats:** 3 test years only; `plain_aligned` and `residual` are single seeds; `residual_mm` was picked for extra seeds after seed 0's test score (seeds 1–2 reproduced it).

### Experiment 11 — Past-years-only Dyn++ factors for training

In Experiment 10, a training forecast from year Y got Dyn++ factors fitted on every training year except Y, so partly on future years. Here they use only 2004 … Y−1 (2004 gets factor 1). Validation and test factors are unchanged (2004–2017, all past). No IMD rain is a U-Net input in either case. Folder: [`experiments/exp11_past_only_factors/`](experiments/exp11_past_only_factors/). 3 seeds each. **This is the final `residual_mm`**; the Experiment 10 figures now use it.

| | per cell, wk 1 | wk 2 | wk 3 | wk 4 | all-India, wk 1 | wk 2 | wk 3 | wk 4 |
|---|---|---|---|---|---|---|---|---|
| Dyn++ | 0.668 | 0.512 | 0.330 | 0.239 | 0.910 | 0.837 | 0.590 | 0.389 |
| `residual_mm`, leave-one-year-out | 0.730 | 0.558 | 0.404 | 0.299 | 0.901 | 0.838 | 0.624 | 0.420 |
| **`residual_mm`, past only (final)** | **0.725** | **0.544** | **0.399** | **0.287** | **0.906** | **0.838** | **0.629** | **0.421** |

| weekly RMSE, mm/week | per cell, wk 1 | wk 2 | wk 3 | wk 4 |
|---|---|---|---|---|
| Dyn++ | 26.8 | 30.8 | 34.0 | 35.0 |
| `residual_mm`, leave-one-year-out | 24.6 | 29.8 | 32.7 | 34.2 |
| **`residual_mm`, past only (final)** | **24.8** | **30.1** | **32.8** | **34.4** |

- **No measurable cost**: past only − leave-one-year-out is −0.005 to −0.014 per cell, every 95% range includes 0, about the seed spread.
- **Still beats Dyn++** at every week per cell (+0.058, +0.033, +0.069, +0.048; all ranges above 0, week 4 only just) and ties it on all-India. Daily RMSE 6.23, bias −0.41 mm/day.

### Experiment 12 — Does ECMWF's bias follow the season? (no training)

Dyn++ pools issue months within ±1 month, assuming ECMWF's bias in July resembles June's and August's. Folder: [`experiments/exp12_seasonal_bias_check/`](experiments/exp12_seasonal_bias_check/).

- **The bias has a seasonal cycle**: ECMWF is too dry before the monsoon (Feb–Apr, land-median factor up to 1.3), about right on average in Jun–Sep (0.9–1.0) and too wet after it (0.5–0.7 in Oct–Dec).
- **It is a pattern, not noise**: factor maps fitted on even vs odd years (independent forecasts) correlate 0.68–0.75 for the same month, 0.53–0.59 one month apart, 0.34–0.43 two months apart and 0.12–0.17 six months apart.
- **±1 month is a sensible window**: on validation it ties same-month-only factors per cell; ±2, ±3 months and all-year are steadily worse.

---

## 5. Key findings

1. On daily maps, the model beats raw and debiased ECMWF at every lead, with real skill of 0.325 once climatology is removed (aligned days).
2. Normal scoring overstates skill: climatology alone beats raw ECMWF on it.
3. Predictions are too flat: the median SD ratio is 0.27–0.39 (Experiment 4, aligned).
4. 12 of the 17 input channels contribute nothing, and neither does the day-of-year input (Experiment 7, 3 seeds each).
5. Predicting the departure from normal is a confirmed gain on daily maps: +0.026 climatology-removed correlation over the Baseline, 3 seeds each. It does not carry over to weekly totals (worse at weeks 3–4). A "trust gate" on top added nothing and was dropped.
6. Yesterday's observed rain has no source in the literature and no measurable effect (3 seeds each, aligned). The earlier single-run +0.011 was noise.
7. Retraining the same setup moves the climatology-removed correlation by up to 0.018. Single-run differences smaller than that are not established.
8. On weekly totals, the usual S2S measure, plain bias correction (Dyn++) beats every U-Net from Experiments 1–7. Their gains on the daily log-space score do not carry over, and training in log space makes them about 50% too dry (Experiment 9).
9. A U-Net that multiplies Dyn++ by a learned correction and trains in mm (`residual_mm`) is the best model so far. It scores +0.05 to +0.07 weekly correlation over Dyn++ per cell at weeks 1–4 (3 seeds, all ranges exclude 0), is 1–2 mm/week more accurate, and ties Dyn++ on all-India (Experiment 10).
10. That result does not rely on future data: with Dyn++ training factors from past years only, `residual_mm` scores within noise of the original (3 seeds) and still beats Dyn++ in every week (Experiment 11). This past-only version is the final model. ECMWF's bias over India follows the season smoothly, which justifies Dyn++'s ±1-month window (Experiment 12).

## 6. Bugs found

| issue | effect | status |
|---|---|---|
| E_pre floor is 1.386, not 0 | scores read too high by a constant | fixed |
| Feature scaling upcast to float64 | 9 GB out-of-memory crash on the full training set | worked around |
| A module name clash in the gate code | training crashed on start | fixed |
| ECMWF and IMD dates one day apart | Experiments 1–8 scored one day off; ECMWF references looked worse | Experiments 2–7 rescored (`experiments/rescore_aligned.py`); Exps 9–10 aligned |
| SD ratio averaged over forecasts | a few near-dry maps (ratio > 200) inflated it | median reported |

## 7. Not done yet

| item | why it matters | cost |
|---|---|---|
| Seeds 1–2 for `plain_aligned` and `residual` (Exp 10) | their numbers are single-run | ~45 min |
| Departure-from-normal input or target on top of `residual_mm` | low priority: departure helped daily maps but not weekly totals | ~1 h, 3 seeds |
| Seeds 1–2 for the + Winds input set (Exp 5) | best weekly scores of Exps 1–7 on one seed | ~25 min |
| Fix flat predictions | still SD ratio 0.60 for `residual_mm` | loss change |
| Ensemble test (1 member vs 11 vs mean) | the paper's other test; we already use the mean, which they found best | cache rebuild, hours |
| Train on weeks 1–4 only | may help or hurt, since it gives less training signal | ~10 min |
| Quantile-mapping baseline | the literature's real bar | small |
| Tercile probability output | needed to compare with the [one Indian U-Net paper](https://doi.org/10.1007/s00382-026-08286-x) | medium |
| Blocked CV | year-to-year variation, only 3 test years | hours |

---

## 8. Review response (professor's comments)

Everything in this section uses the **16-input model without yesterday's observed IMD rain** (`experiments/exp06_yesterday_rain/runs/no_obs_s0`, the "without" run from Experiment 6). Figures are in [`experiments/exp08_review_diagnostics/figures/review/`](experiments/exp08_review_diagnostics/figures/review/); the numbers behind them are in `summary.json` and `sensitivity.json` there. Regenerate with `python experiments/exp08_review_diagnostics/make_review_figures.py` and `python experiments/exp08_review_diagnostics/input_sensitivity.py` (no retraining; they read the saved test predictions and checkpoint).

### 8.1 Comment 1: no IMD data as an input

Agreed. Yesterday's IMD rain (channel 15) would not exist at forecast time. Experiment 6 already showed removing it changes nothing on the standard score (corr 0.443 vs 0.443), so **the 16-input model is now the headline model**. The 17-input numbers in Experiments 1–5 should be read as "with an input we would not have in real time". IMD is now used only as the training target and for scoring. Test scores of the 16-input model at days 7 / 14 / 21 / 28: pattern corr 0.543 / 0.427 / 0.388 / 0.386, RMSE 6.03 / 6.60 / 6.79 / 6.84 mm/day.

Not an issue: the calendar channels (known in advance), and the "normal" in the departure-from-normal experiment (a climatology from past years, which is available in real time).

### 8.2 Which inputs are used to predict one value?

**The whole input block, but not equally.** The input domain is 21–109.5°E, 19.5°S–39°N (1.5° grid, padded to 48 × 64), for all 32 forecast days and all 16 channels. The network is convolutional, so there is no hand-picked window: every output cell is computed from the full domain.

How it works inside:

1. Each 3×3×3 convolution lets a cell see its neighbours one step away in day, latitude and longitude.
2. Three average-pool steps halve the grid each time (48×64 → 6×8, 32 days → 4). After that, one bottleneck cell summarises a block of 8 days × 8 × 8 cells (12° × 12°), and the 3×3×3 convolutions there look at the neighbouring blocks.
3. The decoder expands back to full size, and skip connections add the fine-scale detail from the encoder. So the output at one place and day combines local detail (skips) with broad patterns such as the monsoon flow (bottleneck).
4. The loss includes pattern correlation over the whole map, which pushes the network to get the map-wide pattern right, not only each cell.

We measured this instead of arguing from the architecture: for the day-14 output at one cell, the gradient of the output with respect to every input value, averaged over the 153 test forecasts ([`10_input_sensitivity.png`](experiments/exp08_review_diagnostics/figures/review/10_input_sensitivity.png)):

| | Nagpur (21°N, 79.5°E) | Mumbai (19.5°N, 72°E) |
|---|---|---|
| inputs that can influence it | entire domain, all 32 days | same |
| share of total influence within ±1 cell (1.5°) | 21% | 20% |
| within ±4 cells (6°) | 68% | 67% |
| within ±8 cells (12°) | 87% | 86% |
| from forecast days 11–17 (output is day 14) | 70% | 69% |
| from the output day itself | 13% | 13% |
| from ECMWF rain mean | 20% | 20% |
| from each of the other 15 channels | 4–9% each | 4–9% each |

So distant ocean and far-away cells can influence the prediction, but most of the influence sits within about 6° and within about three days of the target day. Caveat: gradient size is a sensitivity measure, not proof that a variable adds skill (Experiment 5 found the extra variables add none measurable).

### 8.3 Comment 2: learning curves

[`01_learning_curves.png`](experiments/exp08_review_diagnostics/figures/review/01_learning_curves.png): train and validation loss per epoch (total, MSE part, correlation part), with the test loss at the chosen checkpoint. Validation picks the epoch, so a test *curve* would be tuning on the test set; test is shown as one point at the chosen epoch. Only the best checkpoint of this run was kept, so a per-epoch test curve would need a rerun that saves every epoch.

| loss (standardized log-rain units) | train | validation | test |
|---|---|---|---|
| best epoch (7 of 17 run) | 0.946 | 0.961 | 0.979 |
| last epoch (17) | 0.597 | 1.105 | — |

It overfits: after about epoch 7 the training loss keeps falling (to 0.60) while validation rises (to 1.11). At the chosen epoch the three are close.

### 8.4 Comment 4: scatter plots

- [`02_scatter_gridcell.png`](experiments/exp08_review_diagnostics/figures/review/02_scatter_gridcell.png) (log-scale version: [`03_scatter_gridcell_logaxes.png`](experiments/exp08_review_diagnostics/figures/review/03_scatter_gridcell_logaxes.png)): observed vs predicted daily rain, every land cell and every test forecast, at days 7/14/21/28, U-Net above raw ECMWF.
- [`04_scatter_allindia_weekly.png`](experiments/exp08_review_diagnostics/figures/review/04_scatter_allindia_weekly.png): all-India mean rain, weekly averages, one dot per forecast.

Grid-cell daily rain (28,611 points per panel), U-Net / raw ECMWF:

| day | corr | RMSE (mm/day) | bias (mm/day) | regression slope |
|---|---|---|---|---|
| 7 | 0.65 / 0.57 | 7.87 / 8.20 | −1.25 / −0.25 | 0.30 / 0.35 |
| 14 | 0.54 / 0.47 | 8.65 / 8.81 | −1.57 / −0.13 | 0.20 / 0.26 |
| 21 | 0.51 / 0.43 | 8.93 / 9.03 | −1.65 / −0.11 | 0.17 / 0.23 |
| 28 | 0.48 / 0.41 | 8.99 / 9.09 | −1.71 / −0.14 | 0.15 / 0.21 |

These pooled numbers include the seasonal cycle and the wet-west / dry-northwest contrast, so they are higher than the per-map correlations reported earlier. Both models are scored the same way, so the comparison is fair.

What the scatter shows:

- **Both models squash the range, the U-Net more so at long range.** The mean prediction for heavy observed rain (40+ mm/day) is only about 10–18 mm/day at day 7 and 8–12 at day 28. The slope falls from 0.30 to 0.15. This is the "predictions too flat" finding of Experiment 4, now visible.
- **The U-Net is too dry on average by 1.25–1.7 mm/day; raw ECMWF's bias is 0.1–0.25.** This is new. A likely cause (not tested): the model predicts log(1 + rain), and converting an average in log space back to mm under-estimates average rain.
- **All-India weekly mean**: the U-Net's slope is 0.53–0.66 against ECMWF's 0.86–0.89, and its RMSE is higher (1.67 vs 0.71 mm/day in week 1, 2.33 vs 1.36 in week 4), because of the dry bias. Raw correlations of the weekly series look very high (0.92–0.98) for both, but that is the seasonal cycle: climatology alone scores 0.90.

### 8.5 Comment 3: test cases and time series

Cases were chosen by a rule, not by how good they look: the wettest 28-day period in the test set among Jun–Aug issues (2022-07-01), the driest (2023-08-05), the median (2021-08-26), and the wettest among Mar–May (2021-05-27), Oct–Dec (2022-10-07) and Dec–Feb (2021-12-27).

| figure | what |
|---|---|
| `08_case{1-6}_*_daily_maps.png` | per case: IMD, ECMWF, U-Net and U-Net minus observed, daily maps at days 7/14/21/28 |
| `09_case{1-6}_*_weekly_maps.png` | same, weekly means for weeks 1–4 (less noisy, the usual S2S unit) |
| [`07_case_timeseries_allindia.png`](experiments/exp08_review_diagnostics/figures/review/07_case_timeseries_allindia.png) | all-India mean rain over the 32 forecast days for the six cases |
| [`05_timeseries_allindia_by_week.png`](experiments/exp08_review_diagnostics/figures/review/05_timeseries_allindia_by_week.png) | all-India weekly mean, weeks 1–4, whole test period, with climatology |
| [`06_timeseries_cities.png`](experiments/exp08_review_diagnostics/figures/review/06_timeseries_cities.png) | one grid cell each at Delhi, Mumbai, Kolkata, Chennai, weeks 1 and 3 |

Reading them honestly:

- **Where it works**: the spatial pattern in the first one or two weeks. Example, issued 2022-07-01, week 1: map correlation 0.91 for the U-Net vs 0.73 for ECMWF.
- **Where it doesn't**: intensity. West-coast and central-India heavy rain is predicted at roughly a third to half of the observed size, and the time series show the U-Net undershooting every monsoon peak, more at longer lead.
- **All-India weekly mean, departures from climatology** (correlation after subtracting the 1951–2020 day-of-year normal):

| | week 1 | week 2 | week 3 | week 4 |
|---|---|---|---|---|
| U-Net | 0.63 | 0.44 | 0.24 | 0.14 |
| raw ECMWF | 0.88 | 0.81 | 0.61 | 0.42 |

On this measure **raw ECMWF is better than the U-Net**. The U-Net's gain over ECMWF (Experiments 2–5) is in the grid-cell spatial-pattern score, not in the area-average amount. At single cities the two are close (week 1 correlation: Kolkata 0.89 vs 0.83, Mumbai 0.93 vs 0.93; week 3: Chennai 0.53 vs 0.53, with climatology at 0.61).

### 8.6 What this changes

| finding | follow-up |
|---|---|
| Dry bias of 1.2–1.7 mm/day; monsoon peaks under-predicted | change the loss or the back-transform so the mean is preserved (e.g. train on rain directly). Makes "fix flat predictions" in Section 7 the first priority |
| All-India weekly departures worse than raw ECMWF | report area-mean skill next to pattern skill whenever comparing with ECMWF |
| Overfits after about epoch 7 | stronger regularisation, or more training years |
| One seed per setup | 2–3 seeds before claiming small differences |
