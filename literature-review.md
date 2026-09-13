# Post-Processing ECMWF S2S Forecasts to Corrected Values over India

**Task.** Given the ECMWF 11-member S2S reforecast for an init date and lead week, output
**one corrected value per grid cell** for weekly-aggregated rainfall and for tmax/tmin over
India, minimising error against IMD observations. Lead weeks 1–4.

**One grid.** All observations are remapped **up to ECMWF's native 1.5°**, and everything —
training, evaluation, the final product — lives there. This is a pure bias-correction
problem: 40×60 in, 40×60 out, no downscaling anywhere in the project.

**Success criterion.** Beat raw ECMWF at every lead, and beat quantile mapping at most
leads, on the metric set in §5.

Survey date: 2026-09-13. Supersedes `LITERATURE_REVIEW_ECMWF_INDIA.md` and
`LITERATURE_REVIEW_DETERMINISTIC_S2S.md`.

---

## 1. Data

### 1.1 ECMWF S2S reforecasts (input)

Seven variables, paired `.grib` + `.nc`: total precipitation, water/surface temperature,
mean sea-level pressure, geopotential height (2 levels), specific humidity (4 levels),
zonal and meridional wind (2 levels each).

- **1.5°**, domain 21–109.5°E / 19.5°S–39°N → **40 × 60**. The domain is much larger than
  India by design: it carries the monsoon trough and MISO/MJO propagation that supply the
  subseasonal signal.
- **46 daily steps** → leads day 1–46; weeks 1–6 reachable, weeks 1–4 targeted.
- **11 members** (1 control `_cf_` + 10 perturbed `_pf_`). Ensemble mean *and* spread are
  both predictors.
- **1039 init dates, 2004–2023**, roughly weekly. This is the sample size, and it is the
  binding constraint on everything (§7.1).

**Two data bugs to fix before modelling:**

1. **The `step` coordinate is wrong.** `grib_to_netcdf` wrote absolute dates in *2024* for
   a file whose init date is 2004. Reconstruct valid times from the **filename** init date
   plus step index. Trusting the coordinate misaligns every forecast–observation pair.
2. **`tp` is a running accumulation.** Weekly totals are `tp[day_end] − tp[day_start]`,
   **not** a sum. This is the opposite convention from GEFSv12 in `APPROACH.md` §4.4.

### 1.2 IMD observations (truth)

| File | Variable | Native grid | Domain | Period |
|---|---|---|---|---|
| `merged_imd_rain_data_1951_2025.nc` | `rain` | 0.25°, 129×135 | 6.5–38.5°N, 66.5–100°E | 1951–2025 daily |
| `merged_imd_tmax_data_1951_2025.nc` | `tmax` | 1.0°, 31×31 | 7.5–37.5°N, 67.5–97.5°E | same |
| `merged_imd_tmin_data_1951_2025.nc` | `tmin` | 1.0°, 31×31 | same | same |

### 1.3 Regridding to 1.5° — the only grid decision

- **Rain (0.25° → 1.5°):** conservative, area-weighted remap. Not a 6×6 block mean — IMD
  cell centres (6.5°N, …) are not aligned to the 1.5° grid, so cells must be genuinely
  area-weighted.
- **tmax/tmin (1.0° → 1.5°):** conservative is also correct here; bilinear is acceptable.
- **Mask:** derive a valid-data mask from IMD's own NaN pattern. The rain grid extends over
  ocean and foreign territory where IMD has no gauges. Every score must be mask-weighted,
  and the loss must be masked.

**Say this in the paper:** coarsening the truth to 1.5° averages away observational noise
and *raises* apparent skill. Scores here are not comparable to work verifying at 0.25°.
This is the honest cost of the cleaner problem, and stating it pre-empts the objection.

---

## 2. Framing: why a value, not terciles

The field calls this **statistical post-processing** (or calibration, or MOS). Two output
conventions dominate:

- **Tercile probabilities**, scored by RPSS. The convention in most published S2S work.
- **A corrected value**, scored by correlation and error metrics. **Our objective.**

Terciles dominate the literature for four reasons, only one of which binds us:

1. **The tercile map is the operational product.** WMO's SVS-LRF standard, NOAA CPC's
   8–14 day outlooks and IMD's extended-range outlooks are all below/near/above-normal
   probability maps. A tercile paper is a drop-in replacement for a real product.
2. **The benchmark specified it.** Horat & Lerch built on the WMO/ECMWF S2S AI Challenge,
   which pre-specified biweekly windows, tercile targets and RPSS. Much of the apparent
   consensus is one competition's spec being reused.
3. **Discretising rescues signal-to-noise at long lead**, and gives a bounded loss with an
   unambiguous no-skill reference (⅓,⅓,⅓). *This is the real reason, and the one worth
   being uncomfortable about.*
4. **Tercile edges normalise a heterogeneous domain for free** — the Western Ghats at
   500 mm/week and Rajasthan at 0 mm/week contribute equally. We solve this instead with
   per-cell standardisation and a transformed loss (§6).

**Our counter-argument** (make it explicitly; PBC's authors make the opposite case):
a point estimate still improves substantially against the raw forecast even where it
cannot beat climatology by much — see §3. And IMD's own operational extended-range product
is verified with correlation, RMSE, MBE and MAE (§4), so the deterministic framing is the
one Indian hydrological stakeholders actually use.

---

## 3. What is achievable, and the shrinkage trap

Let observed and raw-forecast weekly anomalies both have SD 1.0, with correlation *r*:

| Prediction | RMSE at r = 0.3 (≈ week 3) | RMSE at r = 0.9 (≈ week 1) |
|---|---|---|
| Raw ECMWF: `f` | 1.18 | 0.45 |
| Climatology: `0` | 1.00 | 1.00 |
| Optimal post-processed: `r·f` | **0.95** | **0.44** |

Against the **raw forecast** — the thing we are post-processing — the week-3 gain is
**19%**. Against **climatology** the same model is only 5% better. Both readings are true;
report both.

**The trap.** Under squared-error loss the optimal prediction is `E[y | forecast]`, which
**shrinks toward climatology** as predictability falls. Consequences, none of them bugs:

- Week 3–4 fields will look smooth and damped. Few heavy events, few heatwaves.
- A model producing sharp, realistic-looking fields at long lead will have **worse** RMSE.
- The model under-predicts extremes by construction.
- `SD(pred)/SD(obs)` should fall from ~1 at week 1 toward ~*r* at week 4. If it stays near
  1, the model is overconfident.

**This is why §5 does not lead with RMSE.** DL-DSPM refuses to report RMSE for
deterministic subseasonal precipitation at all, on the grounds that it "is highly
correlated with the standard deviation of subseasonal precipitation forecasts" and can be
lowered by simply scaling predictions down. An RMSE gain at week 4 is consistent with
having learned nothing except how far to shrink.

---

## 4. Baselines to beat, in order

Each is scored on the full §5 metric set.

| Rung | Method | Why it's here |
|---|---|---|
| 0 | Raw ECMWF ensemble mean | The thing being corrected |
| 1 | Climatology (fold-internal, week-specific) | The floor |
| 2 | **Per-cell linear regression of obs on ensemble mean** | Captures the optimal shrinkage of §3 exactly. Surprisingly strong; the honest must-beat |
| 3 | Quantile mapping, per cell / lead / season | Operational default |
| 4 | EMOS / NGR mean | Canonical distributional baseline, mean taken as the point estimate |
| 5 | **ABC** — Climatology++ / Persistence++ (§5.1) | Released code; Persistence++ subsumes our lagged-obs feature plan into a citable baseline |
| 6 | LightGBM with past-obs features | The Das et al. configuration for India |

Two India-specific reference points to cite rather than implement:

- **IMD operational extended-range forecast** (*Pure Appl. Geophys.*, `10.1007/s00024-022-02998-x`)† —
  CFSv2-based, week 1–4 cumulative rainfall, verified with correlation, RMSE, MBE, MAE over
  river basins; reasonable skill to weeks 2–3.
- **Shao et al. (2022)**, *Int. J. Climatol.* — ECMWF subseasonal reforecasts fail to
  reproduce observed warming trends. Handle the 30-year trend **outside** the network; a
  20-year sample cannot learn it.

---

## 5. The literature

### 5.1 The deterministic branch — our objective

| Paper | Output | Vars | Leads | Domain | Metric |
|---|---|---|---|---|---|
| **ABC** — Mouatadid et al. 2023, *Nat. Commun.* 14:3482 | value | T, P | wk 3–4, 5–6 | CONUS **1.5°** | uncentered ACC |
| **3D U-Net** — *GMD* 19, 27 (2026) | value | T, P | **d 1–32** | W. US | pattern corr., RMSE, `E_pre` |
| **DL-DSPM** — *ERL* 2024, `10.1088/1748-9326/ad5370` | value (anomaly) | P | pentads 1–6 | SW China | TCC, PCC, TS |
| **BiConvLSTM** — Liu et al. 2025, *QJRMS* `10.1002/qj.4989`† | value | T | S2S | China | MAE, MSSS, ACC |
| **Season-Net** — Nikraftar et al. 2025, `10.1029/2025JH000957` | value + ensemble | T (daily) | 3 mo | N. Am., Africa | MAE, Kendall τ |
| **GMD 16, 535 (2023)** — customized loss | value | P | — | Gulf coast | KGE (3 components) |
| **Rodeo** — Hwang et al. 2019, arXiv:1809.07394 | value | T, P | wk 3–4, 5–6 | W. US | correlation skill |
| **Das et al. 2024** — `10.1029/2024JH000187` | value | wind | S2S | **India** | RMSE + CRPSS |

**ABC is our benchmark.** Explicitly "a more accurate point estimate"; CONUS at 1.5° with
376 grid points, i.e. our resolution and a comparable grid count. Three components
averaged: **Dynamical++** (adaptively picks the observation window and member range from
recent performance, then bias-corrects), **Climatology++** (a learned climatology using no
dynamical forecast at all — the explicit off-ramp, as a separate inspectable model),
**Persistence++** (regression on lagged obs + climatology + forecast). Temperature skill
+60–90% over baselines of 0.18–0.25; precipitation +40–69% over 0.11–0.15. Code:
`github.com/microsoft/subseasonal_toolkit`. Those absolute numbers are the realistic
ceiling to have in mind; Indian monsoon rainfall will be lower.

**3D U-Net is the closest architecture.** Same source model, same 1.5° input, both our
variables, our exact lead range, deterministic. Two transfers: (i) **lead time as the third
convolutional axis** rather than a FiLM scalar, so the network sees the whole lead
trajectory and learns how error structure evolves — with 1039 init dates × 46 steps this
beats four per-week models; (ii) **loss = MSE + spatial pattern correlation**, which
penalises the degenerate "output climatology everywhere" solution pure MSE rewards. Also a
useful negative result: their ensemble mean performed comparably to all 50 members, which
suggests our 11 can be collapsed to mean + spread. *(Their study also downscales to 0.25°;
that part is not ours.)*

**DL-DSPM corrects the predictors, not the target.** A ResNet corrects OLR and humidity,
a U-Net corrects u/v; the corrected circulation fields then feed a statistical
precipitation model. Beats raw ECMWF beyond pentad 4 and its own baseline beyond pentad 5,
with circulation correction contributing almost nothing before pentad 4 and most of the
gain after. We have all those fields. Worth an ablation.

**BiConvLSTM** adds **terrain as an input channel** and finds gains concentrated **beyond
16 days' lead**. Our feature list had no orography; over India it should matter more than
over China. Also the source of **MSSS** as the right skill-score framing of RMSE.

**Season-Net** and **GMD 2023** matter mainly for their losses (§6).

### 5.2 The probabilistic branch — architectures transfer, numbers do not

| Paper | Output | Note |
|---|---|---|
| **Horat & Lerch 2024**, *MWR* 152(3), arXiv:2306.15956 | terciles | **The architecture template.** Code: `github.com/HoratN/pp-s2s` |
| **Clim. Dyn. 2026**, `10.1007/s00382-026-08286-x` | terciles | **Our nearest neighbour.** Weekly ISM rainfall, leads 1–4, India |
| **PBC** — arXiv:2604.16238 | quintile-bin CDFs | Current SOTA. **Probabilistic**, not our task — cite for positioning |
| **RUFCO** — *AIES* 3(4), `AIES-D-24-0020` | binned | FiLM conditioning + learned climatology off-ramp |
| **Distributional Regression U-Nets** — arXiv:2407.02125 | distribution params | CRPS loss; mean is a deterministic product |

**Horat & Lerch, in detail** — this is the encoder/decoder to copy, head aside. Global
1.5°. Inputs for precip: gh500, gh850, MSLP, total column water, plus the ensemble-mean
target encoded as **distance from the tercile edges** (a per-cell standardisation trick
that works fine with a continuous output — worth trying instead of plain anomalies for
rain). U-Net: 3 encoder blocks (2 conv layers each, `4·2^i` filters, batchnorm, average
pooling), 64-filter bottleneck, mirrored decoder with transposed convs and skips. 10-fold
CV on 2000–2019 in two-year blocks; ~1000 samples; 50 epochs, batch 16. **Averaging models
across CV folds gave ~30% relative skill gain** — mandatory at this sample size.
Precipitation RPSS 0.024 (14 d) / 0.010 (28 d) vs temperature 0.064 / 0.026 — expect rain
to be much harder.

**On the 40×60 grid:** three 2× poolings give 40→5 and 60→7.5. Pad to 48×64 so the
bottleneck is 6×8.

**PBC vs ABC — do not confuse them.** Same group, near-identical names, opposite outputs.
PBC corrects CDFs over five climatological quintile bins and is scored by RPSS/BSS; ABC
outputs a value. PBC's authors argue that beyond two weeks "a single deterministic forecast
cannot capture this uncertainty" — engage with that in the introduction (§2) rather than
omitting it.

**Out of scope:** generative downscaling (DiffScale — note its published application is
surface *wind speed* only), and the AI forecast models that replace the dynamical model
outright (FuXi-S2S, AIFS-SUBS). One paragraph of positioning each, no more.

### 5.3 The India gap

There is **no published deterministic, gridded, deep-learning post-processor for Indian
subseasonal rainfall or temperature at weeks 1–4.** The Clim. Dyn. 2026 U-Net occupies the
tercile ground; the entire deterministic branch above is CONUS, China and the western US.

---

## 6. Metrics and loss

### 6.1 What to report — per lead week, season, and IMD homogeneous region

All mask-weighted, all with **block-bootstrap** CIs (~1000 correlated samples; differences
of 0.005 are not real).

1. **Anomaly correlation** — temporal (per cell → map) and pattern (per date → series).
   **The headline.** Scale-invariant, so shrinkage cannot game it; it is also what the
   deterministic literature compares on.
2. **MSSS against both raw ECMWF and climatology** — `1 − MSE_model/MSE_ref`. Same content
   as RMSE on a 0-centred scale, and naming both references pre-empts "beating climatology
   is a low bar."
3. **KGE with all three components shown separately** — correlation, bias ratio, and
   **variability ratio**. The variability ratio *is* the `SD(pred)/SD(obs)` shrinkage
   diagnostic of §3, under an established name.

Keep raw RMSE in mm/°C in an appendix — hydrological users want it, and the IMD ERF
literature reports it — but do not lead with it. Add scatter/QQ plots so the damping is
visible rather than buried in a number.

### 6.2 Loss — run these as an ablation, don't pick one *a priori*

| Loss | Source | Why |
|---|---|---|
| MSE on anomalies (T) / on `log1p(rain)` | — | Baseline. Simplest, and the most damping-prone |
| **MSE + pattern correlation** | 3D U-Net | Penalises the "output climatology" degenerate solution. *Try first for temperature* |
| **Weighted MAE** | GMD 16, 535 | Weight each cell's absolute error by its precipitation value, clipped to `[log(0.1+1), log(100+1)]`. *Try first for rain* |
| Sliding-window quantile-mapping loss | Season-Net | Penalises distributional mismatch over 21-day windows; cannot be minimised by shrinking variance. Most principled, most work |

**Note on MAE.** *Plain* MAE is wrong for rainfall — it targets the conditional median,
which is ~0 outside the monsoon, so the model collapses to zero. **Weighted** MAE fixes
this: dry cells get a floor weight, wet cells dominate in proportion to magnitude. The GMD
paper reports >70% improvement on climatology and >50% on the 99th percentile versus
unweighted MAE. This is the cheapest high-value change available.

Always **latitude-cosine weight** the loss and mask cells with no IMD data.

---

## 7. Plan

### 7.1 The binding constraint: 1039 init dates

- Prefer **one network conditioned on lead time and season** over specialised networks.
- **Blocked multi-year CV**, never random splits — nearby init dates are correlated.
- **Compute climatologies and any distribution statistics inside each training fold only.**
  Computing them once over 2004–2023 and evaluating on a held-out year is a leak. This is
  the most common bug in this literature.
- **Average models across CV folds** at inference (~30% relative gain, Horat & Lerch).

### 7.2 Pipeline

1. **Ingest and fix.** Rebuild valid times from filename init dates. Difference `tp` to
   per-day rain, aggregate to weekly totals. Concatenate `_cf_` + `_pf_` to 11 members.
   Use the `.nc` files (no `cfgrib` in the env); `wtmp` needs `cftime` or
   `decode_times=False`.
2. **Regrid** all IMD variables to 1.5° (§1.3). Build the valid-data mask.
3. **Features**, all as anomalies from a week-specific, fold-internal climatology,
   standardised by grid-cell σ:
   - **Ensemble mean and spread** of the target variable — spread is the model's own
     uncertainty estimate and is what lets the network learn *how much* to shrink per
     forecast rather than applying a fixed per-lead shrinkage.
   - gh (2 levels), msl, q (4 levels), u/v (2 levels each), wtmp.
   - **Orography** — static, free, and absent from the earlier plan.
   - **Lagged observed IMD state** — the single biggest cheap win in Das et al.
   - Day-of-year sin/cos; MJO RMM and/or a MISO index.
4. **Target.** Weekly-aggregated anomaly from the fold-internal climatology, so "no skill"
   is output 0. Rain trained under §6.2's transform, scored back in mm.
5. **Model.** Horat & Lerch's U-Net (§5.2), padded to 48×64, with a **single continuous
   output channel** in place of the softmax head. Three additions to ablate:
   - **Lead-time handling:** FiLM conditioning vs. lead as a 3D convolutional axis.
   - **Climatology off-ramp:** learned per-cell gate `α ∈ [0,1]`, output
     `α · correction + (1 − α) · climatology` — the shrinkage of §3 made explicit and
     learnable per forecast rather than fixed per lead.
   - **Season/regime conditioning** via FiLM on monsoon phase (pre-monsoon / JJAS /
     post-monsoon / winter) rather than raw month.
6. **Evaluate** per §6.1.
7. **Second head (ablation, not assumption).** A CRPS head on the shared encoder costs
   little and recovers the extremes information MSE damps out — and lets us report
   literature-comparable RPSS by integrating the predicted CDF across tercile edges. But
   GMD 16, 535 found their auxiliary head *hurt* extremes indices, attributed to the extra
   parameters. At 1039 samples, test it; don't assume it.
8. **Optional ablation:** correct the circulation fields first, DL-DSPM style (§5.1).

### 7.3 Where the novelty sits

Three components, none of which exist together in the published record:

- **A deterministic, gridded, deep-learning post-processor for Indian subseasonal rainfall
  and temperature at weeks 1–4** (§5.3).
- **Learned per-forecast confidence gating** (`APPROACH.md` §5.3) — generalising ABC's
  fixed three-component blend and RUFCO's climatology off-ramp into one learned α.
- **A systematic loss ablation for deterministic S2S rainfall** (§6.2) — four published
  losses, never compared on one architecture and dataset.

---

## 8. Reading order

1. **GMD 19, 27 (2026)** — 3D U-Net, western US. *Closest architecture.*
2. **Mouatadid et al. (2023)** — ABC, arXiv:2209.10666. *The deterministic benchmark.*
3. **Horat & Lerch (2024)** — arXiv:2306.15956. *The U-Net to copy.*
4. **Clim. Dyn. (2026)** — `10.1007/s00382-026-08286-x`. *Our nearest neighbour.*
5. **Environ. Res. Lett. (2024)** — DL-DSPM. *Read §3 and §6.1 here first.*
6. **GMD 16, 535 (2023)** — customized precipitation loss. *For §6.2 alone.*
7. **Das et al. (2024)** — `10.1029/2024JH000187`. *India, deterministic, RMSE-reported.*
8. **PBC (2026)** — arXiv:2604.16238. *For positioning, not as our objective.*
9. **Shao et al. (2022)** — trend-aware post-processing.
10. **IMD ERF (2022)** — `10.1007/s00024-022-02998-x`. *The operational baseline.*

† Read from publisher abstract and search summaries only — verify against the PDF before
citing numbers.

---

## 9. Sources

- [ABC — *Nat. Commun.* 14:3482 (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10272189/) · [arXiv:2209.10666](https://arxiv.org/abs/2209.10666) · [code](https://github.com/microsoft/subseasonal_toolkit)
- [3D U-Net sub-seasonal forecasting, western US — *GMD* 19, 27 (2026)](https://gmd.copernicus.org/articles/19/27/2026/)
- [Horat & Lerch — *MWR* 152(3) (2024)](https://journals.ametsoc.org/view/journals/mwre/152/3/MWR-D-23-0150.1.xml) · [arXiv:2306.15956](https://arxiv.org/pdf/2306.15956)
- [U-Net calibration of subseasonal ISM rainfall — *Clim. Dyn.* (2026)](https://link.springer.com/article/10.1007/s00382-026-08286-x)
- [DL-DSPM — *Environ. Res. Lett.* (2024)](https://iopscience.iop.org/article/10.1088/1748-9326/ad5370)
- [BiConvLSTM S2S temperature over China — *QJRMS* (2025)](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.4989)
- [Season-Net — *JGR:MLC* (2025)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000957)
- [Customized deep learning for precipitation bias correction — *GMD* 16, 535 (2023)](https://gmd.copernicus.org/articles/16/535/2023/)
- [Forecast Rodeo — KDD 2019](https://arxiv.org/abs/1809.07394) · [SubseasonalClimateUSA — arXiv:2109.10399](https://arxiv.org/pdf/2109.10399)
- [PBC — arXiv:2604.16238](https://arxiv.org/html/2604.16238)
- [RUFCO — *AIES* 3(4) (2024)](https://dx.doi.org/10.1175/AIES-D-24-0020.1)
- [Distributional Regression U-Nets — arXiv:2407.02125](https://arxiv.org/abs/2407.02125)
- [Bias-corrected extended-range forecast over India — *Pure Appl. Geophys.* (2022)](https://link.springer.com/article/10.1007/s00024-022-02998-x)
- [ReST — *npj Clim. Atmos. Sci.* (2026)](https://www.nature.com/articles/s41612-026-01430-8)
