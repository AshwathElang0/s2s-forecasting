# Plan: 3D U-Net post-processing for ECMWF S2S over India

This document does two things: (1) explains exactly how the reference paper's
3D U-Net was built, from the actual released source (`github.com/W-US-BCSR/
IRnAF`, cloned locally during development but not part of this repo -- see
below), and (2) maps every piece of it onto our India/ECMWF/IMD problem as
defined in `../literature-review.md` §5.1 and §7 (one level up, repo root).

This folder (`india_3dunet/`) was developed as a standalone sibling
directory and moved into this repo on the `unet` branch once it had a
working result -- paths below assume its current location, one level under
the repo root, next to `data/`.

Paper: Ryu, Kim, Wang & Yoon, *"Increasing resolution and accuracy in
sub-seasonal forecasting through 3D U-Net: the western US"*, GMD 19, 27–39
(2026). Code: `github.com/W-US-BCSR/IRnAF` (Zenodo 10.5281/zenodo.14776781).

---

## 1. How the reference paper actually built it

Reading the code (not just the methods section) resolves several ambiguities
the paper text leaves open.

### 1.1 Architecture (`model/IRnAF.py`)

A plain 3D U-Net, 3 encoder/decoder stages, `unit=64` base filters:

```
input (batch, ens, var, lead=32, lat=72, lon=72)
  -> reshape to (batch, ens*var, 32, 72, 72)      # ensemble & variable folded into channels
  -> conv1  [C -> 64 -> 64]      (3x3x3, BN, GELU) x2
  -> pool1  (avg, 2x2x2)          -> 16 x 36 x 36
  -> conv2  [64 -> 128 -> 128]
  -> pool2                         -> 8 x 18 x 18
  -> conv3  [128 -> 256 -> 256]
  -> pool3                         -> 4 x 9 x 9
  -> bottleneck [256 -> 512 -> 512]
  -> upconv1 (transposed conv, 2x2x2) -> 8x18x18, concat with conv3 skip (256+256)
  -> ex_conv1 [512 -> 256 -> 256]
  -> upconv2 -> 16x36x36, concat with conv2 skip (128+128)
  -> ex_conv2 [256 -> 128 -> 128]
  -> upconv3 -> 32x72x72, concat with conv1 skip (64+64)
  -> ex_conv3 [128 -> 64 -> 64]
  -> final 3x3x3 conv -> 1 output channel
```

Every conv block is `Conv3d(k=3,p=1) -> BatchNorm3d -> GELU`, twice. Pooling
is **average**, not max (this matters — max-pool would sharpen extremes
further before the bottleneck compresses them away; the paper's own finding
that the model underestimates precipitation extremes is partly a
architecture-level choice, not just a loss-level one). Upsampling is a
learned `ConvTranspose3d`, not interpolation.

**The key trick, and the one our literature review calls out explicitly**:
the three spatial-looking dimensions fed to `Conv3d` are actually
**(lead time, lat, lon)**, not (depth, height, width) — `depth = lead time`.
This is *why* a 3D (volumetric) U-Net is the right tool for a 2D forecasting
problem: it lets one network see the entire lead-time trajectory at once and
learn how spatial error patterns evolve across lead, with skip connections
carrying resolved fine-lead structure into the decoding of the coarser,
harder long-lead output — instead of training 4+ independent per-lead-week
models, or conditioning a 2D U-Net on lead via a FiLM scalar.

Ensemble members and variables are **not** separate spatial or temporal
axes — they're folded into the channel dimension before the first conv
(`x.view(batch, ens*var, time, lat, lon)`). So "50 ensemble members × 8
variables" is just a 400-channel input image; the network never explicitly
knows which channel is which ensemble/variable, it just gets to use all of
them as input features. This is why the paper can freely try
`E01`/`E50`/`E50M` (1, 50, or mean-of-50 members) and `V1`/`V2`/`V4`/`V8`
(variable-count ablations) as pure channel-count swaps with no architecture
change.

### 1.2 Loss (`pearson_correlation_loss_with_weights` + `nn.MSELoss`)

```
loss = MSE(pred, target) + (1 - latitude_weighted_pearson_corr(pred, target))
```

The correlation term is computed **per (batch item, lead-day) 2D
lat/lon map**, latitude-cosine weighted, then averaged over batch and lead.
It is *not* a single scalar correlation over the whole 4D tensor — that
distinction matters, because a per-map correlation directly penalizes
"right on average, wrong pattern" for every individual forecast, whereas a
global correlation could be gamed by getting a few maps very right and
ignoring the rest. Equal weight (1.0 MSE + 1.0 corr-loss), chosen by
trial-and-error per the paper text, not tuned.

### 1.3 Data handling (`preprocess/ECMWF_data.py`, `PRISM_data.py`)

- ECMWF: 50-member forecast, 10 variables, conservative regrid for `pr`/`tcw`
  (mass-conserving), bilinear for everything else, cropped to a target box,
  then **cropped again to the CONUS west (`235.5–253.25E, 31.25–49N`, 72×72
  at 0.25°)** — i.e. two-stage regrid+crop, once broad then once tight.
- PRISM (target): read raw `.bil` rasters, assign lat/lon from the raster
  transform, regrid to the same 0.25° target grid (conservative for
  precip, bilinear for temperature/elevation).
- **Standardization is fold-scoped by ensemble-count context, not just
  train-only**: mean/std for `pr`/`t2m` are computed twice — once over the
  NWP-only sample (`_only_NWP`, used for pure-forecast diagnostics) and once
  over NWP **and** PRISM concatenated together (`_both_PRISM_NWP`, used as
  the actual training pair) — so predictions and labels sit on the same
  standardized scale and can be inverse-transformed together at test time.
  Precipitation and TCW get `log10(x + 1)` before standardizing (negative
  values from regridding noise are clipped to 0 first).
- Elevation is static: computed once, standardized once, then
  **broadcast** across every sample/ensemble/lead as if it were a time
  varying channel — free spatial context with zero extra data cost.
- Split is a **fixed calendar block**, not k-fold: train 2015–2020 (626
  samples), val Mar-2021–Feb-2022 (104), test 2022–2023 (104). No blocking
  logic beyond "pick a later, non-overlapping calendar range" — this is the
  one place our own plan goes further than theirs (§7.1 below).

### 1.4 Training (`code/model_train.py`)

Adam, `lr=1e-4`, batch size **11** (chosen only because of GPU memory, per
the paper text — not tuned), 100 epochs max, early stopping on val loss with
patience 10. No LR schedule, no warmup, no weight decay. Model checkpointed
every epoch; the best-val-loss epoch's path is looked up later by a
plain-text `best_epoch` log file — no automatic best-checkpoint symlink.

### 1.5 Evaluation (`model/evaluation.py`)

Three metrics, all latitude-cosine weighted and NaN-masked:
- **`scorr`**: spatial (pattern) correlation between the two full lat/lon
  fields, per (sample, lead).
- **`rmses`**: latitude-weighted RMSE, per (sample, lead).
- **`Epre`** (Eq. 1 in the paper — a Taylor-diagram-style combined
  variance-ratio + correlation score): `E = log[ ((σ1/σ2 + σ2/σ1)² · 16) /
  (1+r)⁴ ]`, 0 for a perfect forecast, growing with either variance mismatch
  or decorrelation. This is exactly the `SD(pred)/SD(obs)` shrinkage
  diagnostic our own literature review flags in §3/§6.1, folded into one
  number with correlation.

### 1.6 What the paper found, and why it matters for us

- Ensemble mean (**E50M**) ≈ full 50-member ensemble (**E50**) on every
  metric; the single control member (**E01**) alone is clearly worse. →
  Our literature review already drew this conclusion for us: collapse our
  11 members to **mean + spread**, don't feed all 11 raw.
- More auxiliary variables (**V1→V8**) gave **no significant additional
  skill** once elevation/target-variable and one or two circulation fields
  were in. → We should not assume our full feature list (gh×2, msl, q×4,
  u/v×2×2, wtmp) all earns its keep; ablate variable count the same way.
- Precipitation is systematically **underestimated** in coastal/mountain
  terrain at every lead — a direct instance of the shrinkage-toward-mean
  behavior our literature review's §3 predicts from pure MSE, not fixed by
  adding the correlation term alone. Confirms weighted-MAE or a
  distribution-aware loss (§6.2 of the lit review) is worth ablating for
  rain specifically, exactly as planned.

---

## 2. Mapping onto our problem

| | Reference paper (western US) | This project (India) |
|---|---|---|
| Input | ECMWF S2S, 1.5°→0.25° downscaled | ECMWF S2S, **1.5°, no downscaling** |
| Target | PRISM, 0.042° (4 km) | IMD, regridded **up to** 1.5° |
| Grid | 72×72 @ 0.25° | **40×60 @ 1.5°**, padded to **48×64** |
| Lead axis | 32 days | **32 days** (day 1–32 of the available 46; covers weeks 1–4 and a few spare days for clean 2× pooling three times) |
| Members | 50 (1 control + 49 perturbed) | **11** (1 control + 10 perturbed) |
| Variables | t2m/pr + 7 auxiliary | tp/tmax/tmin + gh(2)/msl/q(4)/u(2)/v(2)/wtmp + orography |
| Ensemble handling | E01/E50/E50M ablation | same ablation, but literature review already predicts **mean+spread** wins — spread is a *new* channel the paper didn't have (their ablation was mean-vs-raw-members, never spread-as-a-feature) |
| Split | fixed calendar blocks | fixed calendar blocks **plus** blocked CV averaging (lit. review §7.1 — 1039 samples is much tighter than their ~830, so we can't afford a single train/val/test split to be the only evidence) |
| Loss | MSE + pattern corr | same, **masked** (India domain has ocean/foreign-territory cells the US domain didn't need to mask this aggressively) |
| Output | separate downscaled grid | **same grid as input** — this is a pure post-processing problem, no super-resolution, which simplifies padding/cropping a lot |

Three deliberate departures, each justified above:
1. **Masking everywhere.** Their US box is (almost) all land; ours spans
   ocean/Tibet/Myanmar where IMD has no gauges. Every loss term, every
   metric, and the correlation computation must respect the IMD-derived
   valid mask, or the network will happily fit noise over the Arabian Sea.
2. **Spread as an explicit channel**, not just more ensemble members. Their
   ablation showed raw members add nothing beyond the mean; our literature
   review's baseline-2 (per-cell regression) and the whole confidence-gating
   thread in `APPROACH.md` says spread is exactly the signal that should
   drive how much correction to trust. We add it as a first-class engineered
   feature rather than re-deriving it from 11 raw member channels.
3. **Fold-internal climatology**, computed inside each training fold, never
   over the full 20 years. The paper doesn't have this problem in the same
   way (PRISM target isn't expressed as an anomaly), but our target *is* an
   anomaly from a week-specific climatology, so this is a hard requirement,
   not a nicety — the literature review flags it as "the most common bug in
   this literature."

---

## 3. What's implemented in this folder

```
india-3dunet/
  plan.md                       (this file)
  config.py                     grid/lead/padding/split constants
  preprocess/
    regrid.py                   dependency-free conservative + bilinear regridder
                                 for regular lat/lon grids (no xesmf/ESMF on this machine)
    ecmwf_features.py           ingest .nc, fix step/valid-time bug, difference tp,
                                 concat cf+pf -> 11 members, per-variable arrays
    imd_targets.py               regrid IMD rain/tmax/tmin to 1.5°, build valid mask
    build_dataset.py             assemble one (features, target, mask) sample per issue
                                 date: circulation channels + lagged obs + day-of-year,
                                 padded to 48x64
    dataset_loader.py            fixed calendar-block split + train-only
                                 standardization (log1p for rain) applied to val/test
  model/
    unet3d.py                    the 3D U-Net + masked MSE/pattern-corr loss
    evaluation.py                masked pattern corr / RMSE / E_pre, ported from IRnAF
  scripts/
    repair_corrupt_nc.py         reconvert 0-byte ECMWF .nc files from their .grib
                                 siblings (S4 below)
    cache_imd_targets.py         one-time IMD regrid, run first
    build_all_samples.py         cache one sample per issue date x target, resumable
  train.py                       training loop (mirrors code/model_train.py)
  test.py                        inference + skill scoring (mirrors code/model_test.py)
```

`xesmf` (the paper's regridding library) needs the ESMF C library, which
isn't installable in this environment — `preprocess/regrid.py` reimplements
conservative and bilinear regridding directly for regular lat/lon grids
(exact area-weighted overlap along each axis, separable since both grids are
axis-aligned), which is all conservative regridding actually needs here.

**Standardization today is train-only mean/std (log1p first for rain), not
yet the week-specific, fold-internal climatology anomaly S7.1 of the
literature review calls for.** That's a real gap, not a rounding error —
see S6 below.

---

## 4. A real data bug found while building this

`data.md` already flags that gh/q/tp are each missing one file relative to
the full 1040/1040 set. Building the sample cache surfaced a related but
different problem: **88 of the `q` (specific humidity) `.nc` files are
0 bytes** -- an interrupted or failed `grib_to_netcdf` conversion, not a
missing-data gap. Their `.grib` originals are intact. `scripts/
repair_corrupt_nc.py` reconverts them with `cfgrib` (added to this
environment; not yet in `pyproject.toml`) rather than skipping those 88
forecast dates outright. 84 of 88 were recovered this way; the remaining 4
have a 0-byte `.grib` too (genuinely missing source data, matching
data.md's note) and are skipped by `scripts/build_all_samples.py`.
Worth porting this check into `scripts/check_data_sources.py` on `main`.

Two more things found only by actually running this against real data,
worth recording since they'd silently corrupt results otherwise:

- **`data.md`'s variable-level table has a typo.** u and v are documented
  as levels "200, 1000 mb", but the actual `.nc`/`.grib` files carry
  **200/850**, not 200/1000 (gh really is 200/1000 — only u/v are wrong).
  `config.py` uses the verified levels and flags this inline.
- **`wtmp` (sea surface temperature) is legitimately NaN over land** — not
  a data error, just what SST means outside the ocean — and needs an
  explicit fill (`build_dataset.py` zeroes it) before it can be a model
  input. Silently missing this produced NaN losses through every stage.

## 6. Honest scope of this first pass

- The model, losses, evaluation metrics, regridding, and the ECMWF/IMD
  ingestion (including both bugs above and the two S1.1 data bugs) are not
  just written but **verified against the real files in
  `../data/`**: the IMD-to-1.5° regrid cache was built for
  the full 1951–2025 record, and a real (if tiny — `unit=8`, 5 epochs, 66
  samples spanning 2004/2005 train, 2019 val, 2022 test) end-to-end run of
  `train.py` → `test.py` produced a finite, decreasing training loss and
  sane pattern-correlation/RMSE-by-lead numbers. That run is not a result
  to report — it's a wiring proof, and its checkpoints under `runs/` are
  disposable.
- The full feature list (§7.2 of the literature review: MJO/MISO indices,
  orography, fold-internal *climatology* standardization — today's
  standardization is train-only mean/std, log1p'd for rain, not a
  week-specific anomaly — across multiple CV blocks, per-fold model
  averaging) is a substantial second phase. Orography and MJO/MISO
  specifically have **no local source file** under
  `../data/` at all — not implemented because there's
  nothing to implement against yet, not an oversight.
- This pass wires up the core post-processing loop (ensemble mean+spread +
  circulation variables + lagged IMD obs + day-of-year → corrected value,
  one fixed calendar split) so it can train and be evaluated today, with
  the rest of §7.2 added as ablations on top of a working baseline rather
  than attempted all at once.
- **Next step to get a real result**: run `scripts/cache_imd_targets.py`
  (already done, cached under `data_cache/`), then
  `scripts/build_all_samples.py` for the full 1039 dates x 3 targets (this
  is the long step — budget a few hours; ~4% of `q` files needed
  `scripts/repair_corrupt_nc.py` first, also already run), then
  `train.py --target rain --unit 64` (the paper's real capacity) for each
  of rain/tmax/tmin.
