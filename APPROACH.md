# S2S Prediction Enhancement — Approach

This document is meant to be fully self-contained: a reader comfortable with
machine learning and standard statistical post-processing (e.g. quantile
mapping) should be able to learn both the domain vocabulary and the actual
method from this file alone, without needing outside references. Section 1
is a glossary; everything after it uses those terms freely.

---

## 1. Glossary

**S2S (subseasonal-to-seasonal)**: the forecast lead-time window from
roughly 2 weeks to ~3 months out. It sits in a predictability gap: too far
out for the day-to-day atmospheric initial-condition information that makes
weather forecasting skillful to still matter much, but too soon for the
slow boundary conditions (ocean temperature patterns, etc.) that make
seasonal forecasting skillful to fully take over. Skill in this window is
real but weak, inconsistent across regions, and — per recent literature —
appears to come and go on decadal timescales as large-scale climate patterns
shift. This project targets **lead weeks 1, 2, 3, and 4** specifically
(days 1–7, 8–14, 15–21, 22–28 after a forecast is issued).

**Reforecast / hindcast**: a forecast run *after the fact*, using a fixed,
frozen version of a forecast model to simulate what it would have predicted
on some past date, using only the data available up to that date. This is
how you get a large, consistent sample of "forecast vs. what actually
happened" pairs for training/evaluating a correction method — you can't wait
around for 20 years of real forecasts, so you regenerate them retrospectively
with one unchanging model version instead.

**Ensemble / ensemble member**: instead of running a forecast model once,
you run it many times from slightly different (but all equally plausible)
starting conditions. Each run is a "member." The spread across members is a
built-in estimate of forecast uncertainty: if all members agree, the
atmosphere's future is predictable from where it's at now; if they diverge
wildly, small differences in the initial state are blowing up into very
different outcomes (chaotic/low-predictability regime). GEFSv12 (our
dynamical forecast source) has 5 members per date: one **control** run
("c00", from the best estimate of the true initial state, unperturbed) and
four **perturbed** runs ("p01"–"p04", each nudged slightly).

**GRIB / GRIB2**: a compact binary format meteorological centers use to
distribute forecast fields (the alternative to something like NetCDF).
Reading it requires the `ecCodes` C library (via the `cfgrib` Python
backend) rather than a plain NetCDF reader.

**Accumulated / per-interval precipitation**: forecast precipitation fields
are usually stored as an *accumulation over a time interval* (e.g. "the rain
that fell in the 3 hours ending at this timestamp"), not a running total
since the forecast started. We verified this empirically for GEFS (see
Section 4.4) since it's a common source of silent bugs if assumed wrong: a
weekly total is the **sum** of all the sub-daily accumulated values falling
in that week, not a difference of cumulative values.

**Teleconnection**: a statistical link between climate conditions in two
distant places — the classic example is El Niño (warm tropical Pacific
Ocean) reliably shifting storm tracks and temperatures across North America
months later. Teleconnections are the physical mechanism that makes any
S2S skill possible at all; without them, weeks 2–4 would be pure
climatology.

**ENSO / ONI**: El Niño–Southern Oscillation, the dominant tropical
Pacific Ocean temperature pattern that drives the best-known
teleconnections. The **Oceanic Niño Index (ONI)** is NOAA's standard scalar
summary of ENSO state (positive = El Niño-like/warm, negative = La
Niña-like/cool), reported as a running 3-month sea-surface-temperature
anomaly.

**MJO / RMM index**: the Madden-Julian Oscillation, an eastward-propagating
pulse of tropical rainfall/convection with a 30–60 day cycle — the dominant
*subseasonal* (as opposed to ENSO's seasonal-to-interannual) source of
predictability, especially at 1–4 week leads. The **RMM (Real-time
Multivariate MJO) index** summarizes it as two numbers (RMM1, RMM2) that
together define an **MJO phase** (1–8, which of 8 regions of the globe the
active convection is currently over) and an **amplitude** (how strong/active
the MJO signal currently is; a weak/small amplitude means "no coherent MJO
signal right now").

**Regime / window of opportunity**: informal terms in the S2S literature for
"a state of the atmosphere in which this particular forecast is more (or
less) predictable than average" — e.g., a strong, cleanly-phased MJO event
might make week-3 U.S. temperature unusually predictable, while a
disorganized tropical state might make the same lead time close to a coin
flip. Detecting this state, rather than assuming constant skill at every
lead time, is the central idea of this project (Section 5).

**Analog forecasting**: predict the future by finding a historical date
whose current conditions closely resemble today's, then using what actually
happened after that historical date as the forecast. A **learned similarity
mask** (used in one of our reference papers) is a set of learned weights
applied to the comparison so that "closeness" is measured only in the
regions that actually matter for the target forecast, rather than uniformly
across the whole globe.

**Brier Score (BS) / Brier Skill Score (BSS)**: the standard scoring rule
for a probabilistic binary forecast (e.g. "40% chance of an extreme week").
`BS = mean((p_forecast - y_observed)^2)` where `y_observed` is 0 or 1 — it's
literally mean-squared-error against a 0/1 label, lower is better. **BSS**
rescales it against a reference forecast (almost always climatology):
`BSS = 1 - BS_model / BS_climatology`. `BSS = 0` means "no better than
climatology," `BSS = 1` means "perfect," negative means "worse than just
guessing the climatological rate every time."

**CRPS (Continuous Ranked Probability Score)**: the Brier Score's
generalization to a continuous quantity (e.g. forecasting the *amount* of
precipitation, not just whether it exceeds a threshold) — integrates the
squared difference between the forecast's and the observation's cumulative
distribution functions. Reduces exactly to the Brier Score in the binary
case.

**Block bootstrap**: a resampling method for computing confidence
intervals/significance that resamples *contiguous blocks* of a time series
rather than individual points, so that autocorrelation (e.g. this week's
precipitation being correlated with last week's) doesn't cause the standard
i.i.d. bootstrap to understate uncertainty.

**Quantile mapping (assumed known) vs. what we do with quantiles here**: you
already know quantile mapping as a bias-correction technique — reshape a
forecast's marginal distribution to match the observed distribution by
mapping each forecast quantile to the corresponding observed quantile. We
are **not** doing that in the current pipeline. The only use of quantiles
right now is much simpler: compute the empirical 99th percentile of the
*observed* (CPC) weekly-precipitation distribution at each grid cell, and
use that fixed value as a **binary extreme-event threshold** — a label
definition, not a distribution transform. Actual quantile-mapping-style
multivariate bias correction (specifically Cannon's MBCn, Section 3) is a
reserved future upgrade for once the pipeline is generating ensembles that
need cross-variable consistency, not just a single precipitation field.

**Conservative vs. bilinear regridding**: two ways to move a gridded field
from one grid resolution/projection to another. **Bilinear** interpolates
each target point from its 4 nearest source points — cheap, but does not
guarantee that a spatial average/total is preserved. **Conservative**
remapping (e.g. via the ESMF library) computes exact area-weighted overlaps
between source and target grid cells so that domain integrals (e.g. total
water volume) are preserved exactly. We use bilinear for now (see Section
3), which is a known, documented simplification.

---

## 2. Problem statement

Enhance S2S forecasts of **extreme precipitation occurrence** for **lead
weeks 1–4 over CONUS**, as the empirical foundation for a research paper.
The central methodological bet (Section 5) is that *forecast confidence
itself* should be an explicit, learned, jointly-trained signal that gates
how much correction is applied — rather than correcting every forecast by
the same fixed amount regardless of how predictable that particular
situation actually is.

---

## 3. What the four reference papers established

| Paper | Core contribution | Role in this project |
|---|---|---|
| **Zhang et al. (2023)**, *Atmospheric Research* | Random Forest classifier post-processes a dynamical S2S model's (GEOS5) raw precipitation forecast to better predict extreme-occurrence. Fixed correction strength, one model per lead week. Inputs: raw precip forecast + auxiliary fields (surface temp, 500/850 hPa geopotential height). Found that adding those auxiliary predictors and tuning tree depth improved skill at all lead weeks, not just week 1. | Direct prior-art baseline (see Section 5.2, Baseline 3) |
| **Cannon (2017)**, *Climate Dynamics* | MBCn: iteratively (i) randomly rotate a multivariate dataset, (ii) apply plain univariate quantile mapping to each rotated dimension, (iii) rotate back; repeated until the *joint* multivariate distribution (not just each variable's marginal) matches observations, then a final pass restores marginals and the climate-change signal. Validated via a compound fire-weather index requiring several variables to be jointly realistic. | Reserved future upgrade for multivariate-consistent bias correction (Section 8) |
| **CAMulator / subCESMulator** (NCAR, arXiv:2504.06007) | Auto-regressive neural emulator of the CAM6 atmosphere model, trained on the CREDIT platform, taking prescribed sea-surface temperature + solar forcing as input. Explicitly enforces conservation of dry-air mass, moisture, and total energy to stay numerically stable over long rollouts (this constraint is *why* it doesn't blow up the way naive autoregressive ML weather models do). ~350x faster than running CAM6 directly, enabling large hindcast ensembles. Known bias: cold high-latitude winter bias from missing interactive land/sea-ice. | Reserved future component for cheap large-ensemble generation (Section 6) |
| **AI2ES interpretable S2S / model-analog work** (incl. arXiv:2506.14022) | Two related results: (a) a **learned spatial similarity mask** improves analog-based S2S forecasts by learning *where* initial-condition similarity actually predicts future outcome similarity, trained via: two candidate climate-state maps → each multiplied by a learnable mask (constrained to mean 1) → weighted MSE between them → single non-negative-weight linear layer → predicted future-error, trained end-to-end against realized skill. (b) A separate shallow-network + integrated-gradients analysis shows the network's own prediction confidence identifies "forecasts of opportunity" (top 20% most-confident predictions score ~65% accuracy vs. ~50% overall) — and that this confidence-skill relationship is **not stationary**, swinging between ~40% and ~80% accuracy across different decades as teleconnection strength changes. | Direct architectural template for the confidence/gating signal (Section 5.3), and the direct motivation for making that signal *learned and adaptive* rather than fixed |

---

## 4. Data

All sources below are free/open and confirmed working end-to-end
(`scripts/check_data_sources.py`, `s2s_data/`, `scripts/build_mvp_dataset.py`).

### 4.1 Shared analysis grid

Everything is regridded (bilinear interpolation, see glossary) onto the CPC
Unified precipitation grid, since that's the verification/label source and
therefore has to be the reference grid: **0.25° resolution, CONUS domain,
lat 20.125°–49.875°N, lon 230.125°–304.875°E** (i.e. ~130°W–55°W in the
more familiar ±180° convention), giving a **120 × 300** grid.

### 4.2 GEFSv12 reforecast — the dynamical forecast being corrected

NOAA's GEFSv12 reforecast, streamed from AWS Open Data (`s3://noaa-gefs-retrospective`)
as raw GRIB2 files, one per (init date, member, lead-time-range, variable).

**A hard, non-negotiable constraint found while building this**: the
archive has two lead-time products per init date:
- `Days:1-10` — every init date, 0.25° grid, 3-hourly accumulation.
- `Days:10-35` — **Wednesday-initialized dates only**, 0.5° grid, 6-hourly
  accumulation.

Every other day of the week stops at `Days:10-16`. Since a week-4 lead needs
day 22–28, **the entire week 1–4 study is necessarily restricted to
Wednesday init dates** — about 1/7 of calendar days. This is a real
sample-size constraint on the whole project, not a pipeline limitation.

Raw GRIB metadata for the precipitation variable (from an actual cached
file, `2019-01-16`, control member, `Days:1-10`):

```
Dimensions:  (step: 80, latitude: 721, longitude: 1440)
  step:      timedelta64, 0 days 03:00:00 ... 10 days 00:00:00  (3-hourly)
Data variable:
  tp         Total Precipitation, units "kg m**-2" (== mm)
Key attrs:
  GRIB_stepType: 'accum'   <- per-interval accumulation, confirmed empirically
                               (values don't grow monotonically with lead time)
  GRIB_dataType: 'cf'      <- control forecast; perturbed members are 'pf'
                               and require an explicit filter_by_keys to open,
                               since one file mixes both message types
```

Weekly totals are built by summing every 3-/6-hourly accumulated value whose
valid time falls in the target 7-day window (`s2s_data/sources/gefs.py`).

### 4.3 CPC Unified Gauge-Based Daily Precipitation — verification & labels

Real, gauge-observed daily precipitation over CONUS, 0.25° grid, 1948–present.
Two sub-archives with a hard cutover discovered while building the
downloader: **years ≤2006 live under one THREDDS path, years ≥2007 under a
different `RT/` path** — no single URL pattern covers an arbitrary year.

Example raw daily values (mm/day) at one grid cell (35.125°N, 99.875°W,
roughly north Texas), January 1–7, 2018 — a dry week, included precisely
*because* real data includes dry stretches, not just interesting events:

```
2018-01-01    0.0
2018-01-02    0.0
2018-01-03    0.0
2018-01-04    0.0
2018-01-05    0.0
2018-01-06    0.0
2018-01-07    0.0
```

**Weekly total** = plain 7-day sum of the daily field. **Extreme-event
label** = 1 if that weekly total exceeds the local, per-grid-cell 99th
percentile of a 2000–2019 (20-year) climatology of weekly totals, else 0 —
matching Zhang et al.'s definition exactly, for direct comparability. Real
threshold values from the built climatology:

```
Threshold (mm/week) at (35.125N, 99.875W): 88.6
CONUS-wide:  min 16.8, median 81.2, max 296.7 mm/week
```

(Thresholds vary ~18x across CONUS — a fixed, non-local threshold would be
meaningless; this is why the label is always computed per grid cell.)

### 4.4 A concrete worked example, straight from the built MVP dataset

This is the single most important illustration of *why* a confidence signal
should matter at all. Init date **2018-01-03**, lead week **2**, grid cell
(33.125°N, 117.375°W, near Los Angeles):

```
5-member GEFS ensemble forecast of weekly precip (mm):
  c00: 61.77   p01: 51.68   p02: 0.10   p03: 0.03   p04: 30.97

  -> ensemble mean:  28.9 mm   (well under the threshold)
  -> ensemble spread (std): 25.6 mm   (huge relative to the mean)

CPC-verified actual weekly precip:  67.1 mm
Local 99th-percentile threshold:    62.9 mm

  -> this WAS a real extreme event (67.1 > 62.9), labeled 1
```

The ensemble **mean** (28.9mm) badly misses this event — it would have been
scored as "no extreme" with high implied confidence if you only looked at
the mean. But the ensemble members wildly disagree (two near-zero, one at
~31mm, two around 50–62mm) — the **spread** is screaming "I don't actually
know what's going to happen here." This is exactly the situation the
confidence-gating idea (Section 5) is designed to exploit: a naive
correction model trained only on the ensemble mean has no way to know this
was a coin flip; a model that also sees the spread does.

### 4.5 CESM2 Large Ensemble (CESM2-LE) — confirmed accessible, not yet used

50-member climate model ensemble, 1850–2100, streamed as cloud-optimized
Zarr from AWS (`s3://ncar-cesm2-lens`). No single "total precipitation"
variable — it's split into `PRECC` (convective) + `PRECL` (large-scale),
both in m/s, summed and unit-converted (`× 1000 × 86400` → mm/day) in
`s2s_data/sources/cesm2le.py`. Reserved for the future regime-signal
pretraining step (Section 6); not part of the current MVP.

### 4.6 Physical attribution covariates: ONI and MJO RMM

Loaded per init date, used only for *interpreting* the confidence signal
after the fact (Section 7), never as a model input directly. Real sample
rows:

```
ONI (monthly, 3-month running):        MJO RMM (daily):
  date        oni                        date       rmm1     rmm2   phase  amplitude
  2026-04-01  0.46                       2024-02-22  0.236   0.249    6     0.343
  2026-05-01  0.95                       2024-02-23  0.202   0.325    6     0.383
  2026-06-01  1.39                       2024-02-24 -0.003   0.128    7     0.128
```

(Note: the BoM RMM file's most recent available row is from Feb 2024 — it's
a slow-updating static file, not a live real-time feed; fine for a
retrospective 2018–2019 study, worth knowing if this is ever extended to
near-real-time use.)

### 4.7 Data reliability issues found and fixed (worth knowing about)

- **IRI Data Library (ECMWF's S2S archive)** now requires a free account
  login — it used to be anonymous. This is why we use GEFS instead of
  ECMWF's S2S output.
- **CPC's OPeNDAP endpoint** would occasionally return a silently
  zero-filled field instead of erroring — confirmed by re-running the
  identical request multiple times with different outcomes. Fixed by
  switching to a plain HTTP file download with local caching
  (`data_cache/cpc/`) instead of OPeNDAP.
- That download itself intermittently **truncated a few hundred KB before
  the end** of a file (looks like a proxy/middlebox timing out a long-lived
  stream, not a NOAA-side outage). Fixed with HTTP Range-based
  resume-from-offset retries rather than restarting from scratch each time.
- GEFS perturbed-member files mix `cf`/`pf` GRIB message types in one file
  and need an explicit `filter_by_keys` to disambiguate when opening with
  `cfgrib`.

---

## 5. Method

### 5.1 Why not just fix the bias uniformly (the prior-art approach)

Zhang et al.'s approach — train a classifier to map raw forecast fields to
`P(extreme)`, same correction strength everywhere and every time — implicitly
assumes the forecast's *reliability* is constant. Section 4.4's worked
example shows why that's a bad assumption: the raw ensemble mean and a
correction model trained on it have no mechanism to distinguish "this
forecast is trustworthy" from "this forecast is a coin flip with a
misleadingly calm-looking mean." AI2ES's finding that confidence-skill
relationships shift by decade makes this worse: even if you got the average
correction strength right on your training period, it will be wrong again
once the teleconnection regime shifts.

### 5.2 Baselines (formal definitions)

Let `f_1, ..., f_N` be the `N=5` ensemble members' weekly precipitation
forecast at a given (init date, lead week, grid cell), `y` the CPC-verified
weekly total, `τ` the local climatological extreme threshold (Section 4.3),
and `p_clim` the *unconditional* climatological probability of exceeding
`τ` — which, by construction (`τ` is defined as the 99th percentile), is
**always ≈ 0.01 everywhere**. This is an important, slightly subtle point:
climatology alone is a nearly-uninformative baseline here precisely *because*
the threshold is locally calibrated to make it so — any real skill has to
come from actually discriminating extreme weeks from non-extreme ones, not
from getting the local base rate right.

- **Baseline 1 — raw ensemble**: `p_raw = (1/N) * Σᵢ 1[fᵢ > τ]`, the plain
  fraction of members exceeding the threshold.
- **Baseline 2 — climatology**: `p_clim ≈ 0.01` everywhere (see above).
- **Baseline 3 — fixed-strength correction (Zhang-style)**: a model (Random
  Forest, or any classifier) `p_RF = M(f_1,...,f_N, other features)`,
  trained once on the whole training set to predict `P(y > τ)`, applied
  identically regardless of how much the members agree.

### 5.3 The core idea: confidence-gated correction

Define a **confidence signal** `c ∈ [0, 1]` for that specific (init date,
lead week, grid cell) — high `c` means "trust the correction," low `c`
means "fall back toward climatology." The corrected forecast is:

```
p_gated = c * p_RF + (1 - c) * p_clim
```

This is a direct, minimal formalization of "operationalize predictability
detection instead of just diagnosing it after the fact" (Section 2): instead
of a separate step that *explains* when the model was right, `c` is wired
directly into what probability gets reported.

**Full version (future work, once the MVP validates the idea)**: `c` is not
hand-computed but **learned**, using the AI2ES analog-mask architecture as a
template, adapted to this problem:

```
inputs: two feature maps (e.g. current ensemble state; a candidate analog
        or simply the ensemble members themselves)
  -> elementwise multiply by a learnable spatial mask (mean-constrained to 1)
  -> reduce to a scalar (e.g. weighted variance across members)
  -> pass through a constrained (non-negative-weight) linear layer
  -> output: predicted skill / confidence for this case
```

Critically, `c`'s parameters and `p_RF`'s parameters are trained **jointly**
— gradients from the correction task flow into the confidence head — so `c`
calibrates against *realized* skill on real GEFS/CPC pairs, not a synthetic
proxy computed in isolation. This is what elevates the method past "bolt an
existing confidence trick onto an existing correction trick" (see Section 6
for why that distinction matters for publishability).

**Nonstationarity handling**: because `c` is learned rather than fixed, it
can in principle re-calibrate itself across the kind of decadal
teleconnection-strength shifts AI2ES documented — this is directly tested
in the validation plan (Section 7) via a train-on-one-decade,
test-on-another stress test.

**Multi-lead sharing**: one model, conditioned explicitly on lead week
(1–4) as an input, rather than four independent per-week models — so
representations learned at an easier, more-predictable lead can transfer to
a harder one, and the model can be directly probed for *how* its
representations shift across the lead-time axis (a genuinely new question
none of the four reference papers directly study).

### 5.4 MVP version of the confidence signal (what we're testing first)

Building and training the full architecture above is a substantial
investment. Before doing that, the MVP tests the *cheapest possible*
version of the same idea, using something we already have for free: **each
case's 5-member ensemble spread**.

```
spread = std(f_1, ..., f_N)                         (real example: 25.6 mm)
c      = a monotonically decreasing function of spread,
         normalized across the training set
         (e.g. c = 1 - percentile_rank(spread))
```

i.e. "if the members agree, trust the correction; if they don't, shrink
toward climatology" — precisely the intuition Section 4.4's worked example
was chosen to illustrate. This is deliberately *not* learned or jointly
trained; it's a fixed, hand-specified rule, used to answer one narrow
question first: **does gating on *any* confidence signal beat the
fixed-strength baseline at all**, before investing in learning that signal
properly.

---

## 6. Why this scope, and what would make it Q1-worthy

A bare version of Section 5.3 ("gate correction strength on a confidence
signal") risks reading as a recombination of two existing techniques
(Zhang's fixed correction + AI2ES's confidence masking). What would elevate
it to a stronger contribution:

1. **Joint training**, not a frozen two-stage pipeline (Section 5.3) — the
   confidence head and the correction model share gradients.
2. **Decadal nonstationarity as the actual scientific hook**, not a
   side-note — showing `c` adapts across a documented regime shift, not
   just that gating helps on average.
3. **A full week 1–4 sweep with strong baselines**, explaining physically
   (via MJO phase / ONI, Section 4.6) *where along the lead-time axis*
   gating earns its keep.
4. **Physical attribution of the learned gate** (Section 7) — showing it
   keys on known teleconnection patterns, contributing understanding of
   S2S predictability rather than only a metric improvement.

---

## 7. Validation plan

- **Temporal, rolling-origin splits** — never random splits, since this is
  a time series with autocorrelation and a forced climate trend.
- **Nonstationarity stress test**: train on one period, test on another
  with a documented different teleconnection regime (and vice versa) —
  directly probing whether `c` adapts as intended (Section 5.3).
- **Metrics**: Brier Skill Score and CRPS-based skill score relative to
  climatology, per lead week; reliability diagrams; spatial skill maps
  (given the ~18x spatial variation in thresholds, Section 4.3, aggregate
  skill numbers alone would hide a lot); block-bootstrap significance
  testing (Section 1) to respect weekly autocorrelation.
- **Physical attribution**: correlate the learned confidence signal `c`
  against MJO phase/amplitude and ONI (Section 4.6) to confirm it tracks
  real teleconnections rather than being a statistical artifact — mirroring
  AI2ES's own interpretability analysis.

---

## 8. Deliberate scope cuts for the MVP

- **Confidence proxy**: ensemble spread (Section 5.4) instead of a
  CESM2-LE-pretrained, learned regime signal.
- **No CESM2-LE in the loop yet** (Section 4.5) — confirmed accessible,
  reserved for once the MVP shows the core idea has legs.
- **No MBCn multivariate bias correction yet** (Section 3) — relevant once
  the pipeline needs jointly-consistent multiple variables, not just
  univariate precipitation occurrence.
- **No joint training yet** — the MVP compares baselines (raw ensemble,
  climatology, a plain correction, and a spread-gated correction), not the
  full end-to-end learned architecture.

---

## 9. MVP dataset (built)

- **Init dates**: all Wednesdays, 2018–2019 (104 dates) — the minimum
  needed for a first real test given the Wednesday-only constraint
  (Section 4.2).
- **Per init date**: 5-member GEFS ensemble weekly precipitation for lead
  weeks 1–4, matching CPC-verified weekly totals, extreme-event labels
  (20-year climatology, Section 4.3), and that date's ONI + MJO RMM values.
- **Result**: 104/104 dates built successfully, 0 failures, ~18GB total
  (mostly cached GRIB source files), fully resumable
  (`scripts/build_mvp_dataset.py`, one NetCDF file per init date under
  `data_cache/mvp_dataset/per_init_date/`).

---

## 10. Current status / next step

MVP dataset is built and validated (Section 9). Next step: implement and
compare the four baselines in Section 5.2/5.4 (raw ensemble, climatology,
fixed-strength correction, spread-gated correction) against this dataset —
a first read on whether the core idea shows any signal before investing in
CESM2-LE pretraining and the fully joint-trained architecture (Section 5.3).
