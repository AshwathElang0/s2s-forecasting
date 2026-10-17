# Climatology-anchored gated residual head

Why the current architecture wastes most of its capacity, what to change, and
how the change is grounded in
`../../../references/s2s_postprocessing_literature_review.md`.

This was written as `gated/README.md`, when the code lived in one `gated/`
folder; it now sits in the shared library and this experiment folder.
`exp01_training_size/runs/n724` and its scoring path are untouched, so the
A/B is against an unchanged baseline.

```
preprocess/climatology.py   observed + model climatologies, cached under data_cache/
preprocess/anomaly.py       the departures-from-climatology reformulation
model/gated_unet.py         GatedUNet3D (subclasses model/unet3d.py) + GatedDataset
experiments/exp03_departure_from_normal/
  DESIGN.md                 this file
  train_departure.py        training loop; same optimiser/loss/schedule as train.py
  test_departure.py         inference, scoring in mm, and the gate diagnostic
  score_seeds.py            3-seed comparison: Baseline vs departure vs gate
experiments/exp02_reference_baselines/compute_baselines.py
                            the baseline ladder + anomaly-space re-scoring
```

---

## 1. What the current architecture does

The network takes 17 stacked 3D blocks, each `32 lead days x 48 x 64` grid
cells (`preprocess/build_dataset.py`):

- ECMWF total-precipitation ensemble mean and spread (2 channels)
- 12 circulation channels: `msl`, `gh` at 200/1000, `q` at 200/500/700/850,
  `u`/`v` at 200/850, `wtmp`
- yesterday's observed IMD value for the target, broadcast across lead
- `doy_sin` / `doy_cos`

A 3-stage 3D U-Net (`model/unet3d.py`) pools three times, expands back, and
carries fine detail across via skip connections. The final `Conv3d(unit, 1)`
emits **one number per cell per lead day: the rainfall itself**.

The consequence worth stating plainly: **the network paints the entire
rainfall field from scratch on every forward pass.** ECMWF's own forecast is
not a starting point it adjusts — it is one of 17 input channels it may use or
ignore. So ~20M parameters spend most of their capacity re-deriving facts that
are free from 75 years of observations: India is wet in July and dry in
January, the Western Ghats are wet, the Thar is not.

It is like asking someone to repaint a picture from memory after glancing at
the original, instead of handing them the original and some correction fluid.

Three consequences, all visible in `../exp01_training_size/runs/n724/results.md`:

1. Capacity is spent on the seasonal cycle rather than on the forecast error.
2. It overfits hard — best at epoch 7 of 100, then val loss climbs
   (0.961 -> 1.103) while train loss keeps falling (0.941 -> 0.563).
3. It cannot say *"I don't know, predict normal."* It must commit to a full
   field even at day 30, where ECMWF has essentially no signal left.

Point 3 is the one the literature cares most about. Review S1.1: at S2S the
post-processor's job "shifts from *sharpening around a reliable signal* to
*deciding how far to deviate from climatology*... Methods that cannot learn
'when to be confident' tend to underperform."

---

## 2. The change, in plain terms

Two changes that only work together.

### 2.1 Change the question

Instead of *"how much rain will fall?"*, ask *"how much more or less than
normal for this date and place?"* Normal comes from 67 years of IMD
observations, already cached at 1.5 deg in
`data_cache/imd_targets_1p5deg.nc`.

Do the same on the input side: feed not ECMWF's raw predicted rain, but *"how
much wetter or drier than ECMWF usually predicts for this date and lead
time."*

Both sides now talk about departures. The seasonal cycle — the easy,
already-known part — leaves the problem entirely, and the network gets to
spend its capacity on the hard part.

Review S7.3: "Model departures from climatology, not absolute values. This
single reparameterisation is the most consistently reported design win in the
S2S post-processing literature." (Scheuerer et al. 2020, *MWR* 148, 3489-3506,
does the categorical version of this.)

### 2.2 Change the head to emit two numbers

Replace `Conv3d(unit, 1)` with `Conv3d(unit, 2)`, giving a **trust gate** in
`[0, 1]` and a **correction**, combined as:

```
answer = gate x (ECMWF's departure) + correction
```

That one line is the whole idea. What it can express:

| gate | correction | meaning |
|---|---|---|
| 1 | 0 | "ECMWF is right here — pass it through" |
| 0 | 0 | "ECMWF is useless here — predict normal" |
| 0.5 | 0 | "half-believe it" |
| any | != 0 | "...and it is systematically too wet/dry in this spot" |

The gate is emitted **per grid cell and per lead day**, so the network can
trust ECMWF at day 2 over the Ghats and ignore it at day 30 over the Deccan,
independently.

---

## 3. Why this should work better

**It starts from a good answer instead of from noise.** The head's weights are
zero-initialised and its bias set to `[0.0, 0.0]`, so at step 0 the model *is*
`0.5 x` anomaly-corrected ECMWF with zero learned correction — already a real
forecast. Training can only improve on that. Currently epoch 0 is random.
(Zero-init on a final projection is standard; weight gradients are still
non-zero, only the first step's upstream gradient vanishes.)

The gate init is set from measurement, not taste. On the train split the
correlation between the standardized model anomaly and the standardized
observed anomaly runs 0.534 at day 1, 0.470 at day 7, 0.242 at day 14, 0.071
at day 28 and 0.044 at day 32. Since the loss-optimal gate is exactly that
`r`, a confident init such as `sigmoid(2.0) = 0.88` would start the model
*worse than plain climatology*; `0.5` ties climatology on average and beats it
at short lead. Those same numbers are what the gate should be expected to
recover, and they are the reason a gate is worth having at all: the useful
trust in ECMWF genuinely falls by an order of magnitude across the lead axis.

**There is far less to learn, so far less to overfit.** The network only has
to model the residual, not the whole field. Review S5, "Sample size": ~1000
forecast cases is small relative to a deep network, which is "why global or
regional weight sharing, heavy regularisation, and small architectures
dominate the S2S literature."

**The sigmoid is the right parameterisation, not an arbitrary choice.** If the
model anomaly and the observed anomaly are each divided by their own
training-set standard deviation, the least-squares optimal coefficient on the
model anomaly is exactly the correlation `r`, which lies in `[0, 1]` for a
positively-skilful forecast. So a sigmoid gate spans precisely the useful
range, and as skill decays with lead, `r -> 0` drives `gate -> 0` and the
forecast falls back to climatology on its own. This is the classical EMOS/NGR
shrinkage coefficient (Gneiting et al. 2005, review S2.2) made
spatially-varying, lead-varying, and nonlinear.

**The gate is readable.** `test_gated.py` saves it. Map it by lead time and
season and you have a direct picture of where and when the post-processor
trusts the dynamical model — review S6.6's "windows of opportunity", and the
conditional reporting S5 asks for. That is a result in itself, not only a
means to one.

---

## 4. Provenance: what is published and what is not

Stated explicitly so this is not mistaken for a literature-backed method when
it is partly a synthesis.

**Published and standard:**

- Modelling departures from climatology — Scheuerer et al. 2020 (review S2.4,
  S7.3).
- A shrinkage coefficient on the ensemble mean that reverts to climatology as
  skill decays — the EMOS/NGR regression slope, Gneiting et al. 2005
  (review S2.2).
- Learned climatology plus learned model correction as the core of an S2S
  post-processor — ABC / Climatology++, Mouatadid et al. 2023, *Nat. Comms.*
  14, 3482 (review S2.3), which beat seven state-of-the-art DL methods.
- Residual prediction against a baseline field — general practice in weather ML.

**Not established, as far as this project has checked:** the specific
packaging — a per-cell, per-lead sigmoid gate inside a 3D U-Net head blending
an anomaly-corrected dynamical field against climatology, for S2S. No citation
is claimed for it. Three unread PDFs sit in `../../../references/papers/`
(notably the 2026 JGR-MLC Shi "Distribution-Guided Ensemble" paper) that may
bear on this; a proper search is required before any novelty claim.

---

## 5. Implementation

### 5.1 Climatologies (`climatology.py`) — strictly causal

For a sample issued in year Y, **every** number used to build its "normal"
comes from a date strictly before Y. No sample contributes to the climatology
it is scored against, and no later year leaks backwards into an earlier
training fold. This is the failure review S5 names directly: "model
climatology and observation climatology computed on the full record leak
future information into the training folds."

A fixed 1951-2017 climatology would already be leak-free *for the reported
test numbers* — test is 2021-2023, entirely after the window — but a 2010
training sample would have had its normal built partly from 2011-2017. That
does not inflate the headline score; it trains the model against targets it
could never reconstruct in deployment. The causal version removes both.

**Observed**, per cell per day-of-year, on the transformed scale (`log1p` for
rain, identity for temperature):

- Rain: expanding window `[1951, Y-1]` — more years give a steadier daily
  rainfall normal, and rain has no strong trend to average away.
- Temperature: rolling 30-year normal `[Y-30, Y-1]` — WMO practice, and it
  tracks the warming trend that a fixed window would suppress (review S2.1).
- Circular +/-10-day smoothing, so each day-of-year pools ~21 days x N years.
- -> `data_cache/climatology/obs_{target}_causal.npz`, one `(366, 48, 64)`
  array per year.

**Model**, per `(issue-DOY bin, lead day, cell)`, for the `tp_mean` anchor:

- Built from the existing `data_cache/samples/rain/*.npz` cache, channel 0.
  No GRIB or netCDF re-reads.
- Train-split issue dates from years `< Y` only. One chronological pass
  snapshots the running accumulation before each year is folded in. Val and
  test years get the full train accumulation, which is still strictly prior.
- 52 weekly bins, circular +/-3-bin (~+/-21-day) smoothing.
- `log1p` applied before averaging so the anchor shares the target's scale.
- -> `data_cache/climatology/model_{target}_causal.npz`.

**The cost, stated plainly.** The reforecast archive begins 2004-01-04, so a
2004 sample has *zero* prior forecast years and cannot be given a causal model
climatology at all. `usable_dates()` drops those 52 samples rather than let
them look ahead: **train falls from 724 to 672**, val and test are unchanged
at 155 and 153. The gated model therefore trains on ~7% less data than
`rain_full` did, which makes the A/B conservative — any win is despite the
handicap, not because of extra data.

Model climatology remains the binding constraint, and causality tightens it:
the 2005 fold has only 52 prior issue dates (~7 per bin after smoothing)
against 672 by 2017 (~90 per bin). Early folds have a noisier anchor.

### 5.2 Anomalies (`anomaly.py`)

With `sigma_obs`, `sigma_mdl` single scalars computed over the training split
(a global scalar, not a per-cell variance, to avoid dividing by near-zero
variance in dry cells and dates):

```
target_anom = (transform(y)   - obs_clim[doy_valid])       / sigma_obs
raw_anom    = (log1p(tp_mean) - mdl_clim[bin_issue, lead]) / sigma_mdl
```

`doy_valid` is the day-of-year of `issue_date + lead` — verified to be the
correct alignment: `ecmwf_features.py` maps `step = 24h` to lead index 0, and
`build_dataset.py` reads the IMD target starting at `issue_date + 0 days`, so
both sides agree.

`lag_obs` also becomes an anomaly, otherwise it smuggles the seasonal cycle
back in. `doy_sin`/`doy_cos` stay — they still identify the regime, they just
can no longer give away the answer. Other channels keep their existing
per-channel standardisation.

Inverse for scoring:
`y = inverse_transform(pred_anom * sigma_obs + obs_clim[doy_valid])`.

### 5.3 Head (`model.py`)

`GatedUNet3D` subclasses `UNet3D`. Since `UNet3D.forward` ends with
`return self.head(out)`, swapping `self.head` for a two-channel conv and
calling `super().forward()` reuses the entire trunk with no duplication:

```python
h = super().forward(x)              # (B, 2, lead, H, W)
gate = torch.sigmoid(h[:, 0:1])
pred = gate * anchor + h[:, 1:2]
```

Loss is unchanged (masked MSE + masked pattern correlation, imported from
`model/unet3d.py`) so the A/B isolates the architecture change.

### 5.4 Baselines (`compute_baselines.py`)

`experiments/exp01_training_size/runs/n724/results.md` compares only against **raw** ECMWF. Review S5:
"Many reported gains are gains over a weak reference." Having a climatology
makes two stronger references cheap, and both are reported:

1. **Climatology** — `pred_anom = 0`. Review S3.3 ranks this first on the
   baseline ladder: "At weeks 4-6, climatology is genuinely hard to beat."
2. **Anomaly-corrected ECMWF** — `expm1(obs_clim + (log1p(tp_mean) - mdl_clim))`.
   Lead-and-calendar-dependent bias removal, review S2.1. This is the honest
   bar for the gated model.

---

## 6. The comparison caveat

Measured as departures from normal, correlation numbers will read **lower**
than the 0.553 day-1 figure in `../exp01_training_size/runs/n724/results.md` — because that
number partly reflects the network knowing it is July, which `doy_sin`/
`doy_cos` hand it for free. Absolute-field pattern correlation over India is
inflated by the seasonal and orographic climatology common to both fields.

So the headline numbers dropping is the *expected* outcome, not a regression.
To keep the comparison honest, `compute_baselines.py --rescore` re-scores
`rain_full`'s saved `test_rain_predictions.npz` through the same climatology
into anomaly space. That re-scoring, not the raw numbers, is what says whether
the change worked.

---

## 7. Reproducing

```
python experiments/exp03_departure_from_normal/train_departure.py --target rain --model_path experiments/exp03_departure_from_normal/runs/gate_v1_first_try \
    --unit 64 --batch_size 11 --n_epochs 100 --patience 10
python experiments/exp03_departure_from_normal/test_departure.py  --target rain --model_path experiments/exp03_departure_from_normal/runs/gate_v1_first_try --unit 64
python experiments/exp02_reference_baselines/compute_baselines.py --target rain --out experiments/exp02_reference_baselines/results \
    --rescore experiments/exp01_training_size/runs/n724 experiments/exp03_departure_from_normal/runs/gate_v1_first_try
```

Same calendar split as `rain_full` (`config.TRAIN_YEARS`/`VAL_YEARS`/
`TEST_YEARS`), and the identical 153-sample test set. Train is 672 rather than
724 because the 2004 samples have no causal model climatology — see S5.1.

The climatology build is the slow step (~10 min, once); it caches under
`data_cache/climatology/` and later runs reuse it. Pass `--rebuild_clim` to
force a rebuild.
