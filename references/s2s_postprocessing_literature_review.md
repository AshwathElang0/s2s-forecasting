# Post-Processing of Numerical Model Outputs for Precipitation and Temperature at Sub-Seasonal to Seasonal Lead Times

### A Literature Review

*Scope: statistical, machine learning, and deep learning methods for reducing error in dynamical model forecasts of precipitation and 2 m temperature at lead times from roughly 10 days to several months.*

*Compiled August 2026*

---

## 1. Framing the Problem

### 1.1 Why this lead time is different

The 10-day to 2-month window is the "predictability desert" between numerical weather prediction (NWP) and seasonal climate prediction. Two things change relative to the medium range, and both matter for how post-processing should be designed:

1. **Signal-to-noise collapses.** By week 3, the deterministic signal from initial conditions has largely decayed. What remains comes from slowly varying boundary conditions and low-frequency modes: ENSO, the Madden-Julian Oscillation (MJO), the stratospheric polar vortex, soil moisture, sea ice, and snow cover. Raw ensemble spread often exceeds the useful signal, so the post-processor's job shifts from *sharpening around a reliable signal* to *deciding how far to deviate from climatology*.

2. **The relevant target changes.** Daily fields become largely unpredictable; the operationally meaningful predictands are weekly, biweekly, or monthly aggregates, and often categorical (tercile or quintile probabilities) rather than continuous amounts.

A consequence widely noted in the literature: a well-calibrated post-processed forecast at week 4 may be one that mostly *reverts to climatology* except in specific windows of opportunity (active MJO phases, strong ENSO states, stratospheric events). Methods that cannot learn "when to be confident" tend to underperform.

### 1.2 Sources of error the post-processor is asked to fix

- **Systematic (unconditional) bias**, in the mean and in higher moments, varying by grid cell, lead time, and season.
- **Conditional bias**, where forecast errors depend on the forecast value itself, on the ensemble spread, or on the large-scale regime.
- **Dispersion errors**, typically under-dispersion (overconfidence) at short leads and over-dispersion or degenerate spread at long leads. ECMWF work has documented a genuine signal-to-noise paradox in subseasonal ensembles, where reliability diagnostics behave counterintuitively.
- **Representativeness and resolution mismatch**, since GCM grid cells (often 0.5° to 1.5°) do not correspond to point observations or basin-scale quantities.
- **Lost multivariate structure**, when univariate calibration destroys the spatial, temporal, and inter-variable coherence that downstream applications (hydrology, crop models, energy) require.

A useful mental model: post-processing at S2S is roughly 20% bias removal and 80% *skill-aware recalibration toward climatology*.

---

## 2. Method Families

### 2.1 Simple bias correction and climatological calibration

The operational baseline almost everywhere. These methods are the benchmark any proposed method must beat, and they are surprisingly hard to beat at long leads.

| Method | Idea | Notes |
|---|---|---|
| Mean bias removal / anomaly correction | Subtract lead-dependent, calendar-dependent model climatology; add observed climatology | Requires a reforecast (hindcast) climatology. Standard operational practice at ECMWF and most centres |
| Variance scaling | Additionally match the variance of anomalies | Cheap way to fix over-confident spread |
| Quantile mapping (QM) / empirical quantile mapping (EQM) | Map the forecast CDF onto the observed CDF | Widely used for seasonal GCM output and for downscaling. Fixes marginal distributions but is applied member-wise and does not use ensemble spread as information |
| Trend-preserving / trend-aware QM | Correct while preserving the long-term climate signal | Important when hindcast periods span decades of warming; naive QM can suppress trends |
| Multiplicative or ratio-based scaling for precipitation | Simple factor correction | Used in several Indian basin studies as a first-order fix |

**Documented limits.** For NMME seasonal precipitation over CONUS, quantile mapping has been found to *degrade* the reliability of multi-model mean forecasts in most regions, while copula-based approaches performed better. For Australian catchments, Bayesian post-processing of ACCESS-S was more skilful and more reliable than quantile-mapped forecasts, especially for monthly and seasonal accumulations. The general lesson: QM corrects marginals but does not create skill, and it can harm calibration when the raw ensemble is unreliable.

### 2.2 Parametric distributional regression

These fit a full predictive distribution whose parameters are regressed on ensemble summary statistics (typically the ensemble mean and spread). This is the classical statistical post-processing paradigm.

- **EMOS / Non-homogeneous Gaussian Regression (NGR)** (Gneiting et al. 2005). Gaussian for temperature; for precipitation, censored and shifted gamma (Scheuerer and Hamill 2015), censored logistic, or zero-inflated variants. Fitted by minimising CRPS.
- **Boosted EMOS (EMOS-GB)** (Messner et al. 2017). Gradient boosting selects among many candidate predictors while retaining the parametric distributional form. Useful when you want to add teleconnection indices without overfitting.
- **Extended Logistic Regression (ELR)**. Produces coherent probabilities across multiple thresholds from one fit. Still the standard baseline for tercile probability forecasts at S2S, including in recent Indian monsoon calibration work.
- **Bayesian Model Averaging (BMA)** (Raftery et al. 2005). Mixture of member-conditional kernels; heavier to fit than EMOS and generally not better at S2S.
- **Bayesian Joint Probability (BJP) and RPP-S** (Wang, Robertson, Schepen and co-authors). A Gaussian joint model after data transformation (log-sinh, Yeo-Johnson, normal quantile transform), designed explicitly for seasonal and sub-seasonal rainfall and streamflow. RPP-S combines BJP for daily calibration with the Schaake shuffle to reconnect lead times. Evaluated on 12 Australian catchments across 12 initialisation dates, it reduced bias substantially and improved both skill and reliability relative to raw and to quantile-mapped ACCESS-S.
- **Seasonally Coherent Calibration (SCC)** (Wang et al. 2019 and follow-ups). Addresses the requirement that calibrated monthly forecasts aggregate coherently to the seasonal total.

**Where they excel.** Short training samples, interpretability, guaranteed calibration, easy integration of a small number of physically motivated predictors. For seasonal precipitation over Java, a logistic-distribution EMOS using SEAS5 output plus climate indices (Niño-3.4, DMI, MJO indices, regional SST) gave more skilful and more spatially homogeneous forecasts of upper quantiles than EQM bias correction alone. This "bridging" idea, feeding teleconnection indices in as predictors alongside the model output, is one of the highest-value and lowest-cost design choices at seasonal range.

### 2.3 Classical machine learning

- **Quantile Regression Forests (QRF)** (Meinshausen 2006; Taillardat et al. 2016). Non-parametric, predicts the full conditional distribution from the empirical distribution in terminal nodes. Proven for temperature, wind, and precipitation, and used operationally at Météo-France. Tail extensions (Taillardat et al. 2019) combine QRF with extreme value theory to fix its inability to extrapolate, giving significant gains for heavy rainfall. Known limitations: it is a *local* model (one fit per grid point, lead time, variable), so it can be spatially inconsistent, memory-hungry on gridded data, and cannot extrapolate beyond the training range.
- **Random forest classifiers** for categorical S2S targets. A CONUS study used RF classifiers with auxiliary predictors (surface temperature, 500 and 850 hPa geopotential height) to improve S2S prediction of weekly extreme precipitation events above the 99th percentile, explicitly exploiting the *ensemble spread* rather than just the mean.
- **Gradient boosting (XGBoost, LightGBM)**. Strong tabular baselines, widely used in operational-style ML seasonal forecast products (for example CW3E's ML seasonal precipitation outlooks, which blend RF, XGBoost, LSTM, and neural nets with NMME members).
- **Analog methods and the Analog Ensemble (AnEn)**. Match the current forecast to similar past forecasts and use the corresponding observations to build a predictive distribution. Naturally regime-aware and needs no distributional assumption; requires a long reforecast archive.
- **Self-organising maps (SOM)**. A hybrid clustering-analog method has been applied to Indian extended-range rainfall, correcting bias and downscaling simultaneously, at a resolution set by the observations rather than the model.
- **Adaptive Bias Correction (ABC)** (Mouatadid et al. 2023, *Nature Communications*). Deliberately low-cost. Applied to the leading ECMWF subseasonal model over CONUS, ABC improved temperature forecasting skill by 60 to 90% (over baseline skills of 0.18 to 0.25) and precipitation skill by 40 to 69% (over baselines of 0.11 to 0.15). Its components (Climatology++, model++ variants) are learned climatologies and learned model corrections rather than deep networks, which is a strong argument that much of the achievable gain at S2S comes from *smart, adaptive climatology* rather than model complexity.

### 2.4 Deep learning

**Point-wise / station-wise distributional networks**

- **Distributional Regression Networks (DRN)** (Rasp and Lerch 2018, *MWR* 146, 3885-3900). A feed-forward network outputs the parameters of a predictive distribution, trained by minimising CRPS. It learns nonlinear links automatically and, via station embeddings, achieves local adaptivity within a single globally trained model. This paper is the reference point for essentially all subsequent neural post-processing.
- **Quantile function regression with Bernstein polynomials** (Bremnes 2020). Distribution-free alternative that outputs a monotone quantile function.
- **Bin/categorical classification networks**. Output tercile or multi-category probabilities with cross-entropy loss. This is the dominant formulation at S2S because the operational product *is* categorical.

**Spatial architectures (CNN, U-Net)**

- **Scheuerer et al. (2020), *MWR* 148, 3489-3506.** The first substantial S2S neural post-processing study. ANN and CNN frameworks for week 2, 3, and 4 accumulated precipitation over California from ECMWF IFS reforecasts. Two design ideas worth borrowing: (a) the network models the *increase or decrease of category probabilities relative to climatological frequencies*, which is exactly the right inductive bias when signal-to-noise is low; and (b) location-specific normalisation plus geographic predictors let a single network serve the whole domain, reducing overfitting.
- **Horat and Lerch (2024), *MWR* 152, 667-687.** Four CNN-based post-processing models producing *global* probabilistic tercile forecasts of biweekly temperature and precipitation for weeks 3-4 and 5-6, built on the WMO S2S AI Challenge data. They outperformed ECMWF's bias-corrected forecasts and beat climatology in various regions. Includes a basis-function CNN (extending Scheuerer's approach to global patches) and both patch-wise and global U-Nets. Code at `github.com/HoratN/pp-s2s`.
- **Worsnop et al. (2024), RUFCO, *AIES* 3(4).** ResU-Net-based framework for subseasonal precipitation accumulation post-processing.
- **U-Net for multi-model ensemble precipitation over East Asia.** A modified U-Net with TimeDistributed convolutional layers post-processing six S2S models and their MME, targeting both precipitation amount and occurrence.
- **U-Net calibration of Indian Summer Monsoon Rainfall** (*Climate Dynamics*, 2026). Directly relevant to South Asian work: U-Net bias correction applied to three S2S GCMs (GEFSv12, ECMWF, IITM ERPv2), calibrating tercile probabilities of weekly accumulated rainfall at leads of 1 to 4 weeks for JJAS over India, consistently outperforming an Extended Logistic Regression baseline.
- **Distributional regression U-Nets** (2024). Combine the spatial coherence of U-Nets with a full predictive distribution per grid cell, benchmarked against QRF and tail-extended QRF for precipitation.
- **Spatial heterogeneity-aware U-Net** (2025/2026, *J. Hydrology*). Argues that standard U-Nets implicitly assume spatial homogeneity through weight sharing and therefore struggle with location-dependent error structure, and proposes architecture changes to fix it. Worth reading if you are working over India, where error characteristics differ sharply between the Western Ghats, the core monsoon zone, and the northeast.

**Set-structured and attention-based architectures**

- **Permutation-invariant / set-based networks** (Höhlein et al. 2024, *AIES* 3(1); Mlakar et al. 2023). Treat the ensemble as an unordered set rather than collapsing it to mean and spread. The Mlakar et al. attention-based model performed best in the EUPPBench benchmark study.
- **Hierarchical ensemble transformers** (Ben-Bouallègue et al. 2023) post-process each member individually while modelling member interactions.
- **PoET.** A transformer embedded in a CNN that takes a dynamical ensemble as input and outputs a new ensemble with all members simultaneously corrected, i.e. a hybrid AI-plus-dynamical system.
- **Graph neural networks** for post-processing with irregular station geometry and learned spatial information.
- **LSTM and recurrent models** for lagged-ensemble and temporal-context exploitation.

**Ensemble-structure exploitation matters.** Work on subseasonal forecasting has specifically found that exploiting the ensemble structure, rather than only the ensemble mean, enhances predictive performance. A CONUS study using *lagged* ensembles (members with different initialisation dates) plus observational predictors (relative humidity, MSLP, geopotential height) compared linear models, random forests, CNNs, and stacked multi-model combinations for monthly mean precipitation and 2 m temperature two weeks ahead, across regression, quantile regression, and tercile classification tasks.

### 2.5 Generative models

The newest family, and increasingly credible.

- **SEEDS** (Li et al. 2024, *Science Advances*). Diffusion-based generative emulation of forecast ensembles. The post-processing variant (SEEDS-GPP) learns to emulate a blended GEFS-plus-ERA5 distribution, correcting biases in the process. The generated ensembles were more reliable and better at forecasting extreme-event probabilities than the operational GEFS ensemble, at under one tenth the compute.
- **Vision transformer plus generative post-processing for extreme precipitation** (*AIES* 4(2), 2025). A 3D ViT for bias correction combined with a generative step to expand the ensemble, improving probabilistic warnings for extreme precipitation over CONUS.
- **Conditional GANs** for generating synthetic ensembles from deterministic model output, preserving inter-variable correlations.
- **Generative ML for multivariate post-processing** (Chen et al. 2022/2024). Directly targets the multivariate dependence problem rather than treating it as a post-hoc reordering step.

Caveat: most generative work so far targets the medium range and nowcasting. Application at S2S is still thin, which is itself an opportunity.

### 2.6 Restoring multivariate dependence

Univariate calibration applied grid-cell-by-grid-cell and lead-time-by-lead-time destroys spatial, temporal, and inter-variable coherence. If your downstream use is hydrological, agricultural, or energy-related, this step is not optional.

Two-step approach: calibrate marginals, then reorder samples according to a dependence template.

- **Ensemble Copula Coupling (ECC)** (Schefzik et al. 2013). Template is the rank structure of the raw ensemble. Cheap and strong.
- **Schaake shuffle** (Clark et al. 2004). Template is the rank structure of historical observations. In a systematic simulation-based comparison, the Schaake shuffle proved a compelling benchmark that was difficult to outperform, while parametric copula approaches and ECC variants were sensitive to the specific misspecification present.
- **SimSchaake** (Schefzik 2016) and date-selection variants: choose historical dates similar to the current forecast situation rather than at random.
- **Member-by-member post-processing** (Van Schaeybroeck and Vannitsem 2015). Corrects each member so that dependence is preserved by construction.
- **Gaussian copula approaches** (Möller et al. 2013) for low-dimensional settings.
- **Joint calibration for multivariate seasonal forecasts** (Schepen, Everingham and Wang 2020, *MWR* 148, 437-456), which explicitly addresses the fact that most seasonal post-processing methods ignore inter-variable dependence.

For your use case (precipitation *and* temperature jointly, across weeks), ECC or Schaake are the pragmatic defaults, with the Schaake shuffle preferred if the raw ensemble's dependence structure is itself suspect at long leads.

### 2.7 Multi-model combination

Multi-model ensembles (NMME, SubX, the WMO S2S database, C3S) often beat any single model. Post-processing and MME combination interact: calibrating each model before pooling generally beats pooling then calibrating, and learned weights (regression, stacking, Bayesian model combination) generally beat equal weighting when enough hindcast years are available. Note the counterexample in the literature: for NMME over parts of CONUS, the multi-model mean did *not* significantly improve precipitation skill where member models shared similar error patterns, so check for error-correlation before assuming MME gains.

---

## 3. What the Evidence Says Works Best

### 3.1 The current state of the art at subseasonal range

The strongest single result available is **Probabilistic Bias Correction (PBC)** (Guan, Mouatadid, Orenstein, Cohen, Weyn, Mackey et al., arXiv:2604.16238, 2026). PBC is an ML framework that learns to correct *historical probabilistic forecasts* rather than deterministic fields. Reported results:

- Applied to ECMWF's operationally debiased dynamical model, it improved skill for 91% of mean sea level pressure targets, 92% of temperature targets, and 98% of precipitation targets.
- Applied to ECMWF's AI Forecasting System (AIFS-SUBS), it roughly doubled the model's modest subseasonal skill, tripling positive skill for week-4 temperature and week-3 precipitation.
- Applied to the hybrid AI-plus-dynamical PoET system, it added a further 70 to 96% for precipitation, 15 to 24% for MSLP, and 5 to 7% for temperature.
- In ECMWF's real-time AI Weather Quest, the entry combining PBC-ECMWF and PBC-PoET (team MicroDuet / MicroEnsemble) placed first for all variables and lead times in the SON 2025 period and again in DJF 2025/26, outperforming six national forecasting centres, their dynamical multi-model ensemble, ECMWF's AI models, and 34 other teams. Code: `github.com/mouatadid/pbc`.

The headline lesson from PBC and from its predecessor ABC is that **post-processing a dynamical or AI model is currently more productive at S2S than replacing it**. Pure ML forecast models have started to be competitive (FuXi-S2S extended skilful MJO prediction from 30 to 36 days and improved global precipitation; AIFS-SUBS scores comparably to the IFS), but the largest verified skill gains in real-time competition have come from correction layers applied on top.

### 3.2 Method-vs-method comparisons

Synthesising across studies:

| Comparison | Finding |
|---|---|
| Neural networks vs EMOS (medium range, temperature) | DRN significantly outperforms EMOS and is computationally cheaper |
| Neural networks vs QRF | Roughly comparable for long training periods; QRF can win at specific locations because of its local adaptivity |
| CNN/U-Net vs point-wise NN at S2S | Spatial architectures win, because they retain spatial relationships, produce spatially homogeneous fields, and share statistical strength across grid cells (critical when signal-to-noise per cell is tiny) |
| U-Net vs Extended Logistic Regression (Indian monsoon, weeks 1-4) | U-Net consistently better for tercile probabilities |
| Deep learning vs simple adaptive corrections (S2S, CONUS) | ABC's simple learned corrections beat seven state-of-the-art DL/ML methods on the SubseasonalClimateUSA benchmark; complexity did not pay |
| Quantile mapping vs Bayesian / EMOS calibration (seasonal) | Bayesian and EMOS approaches better for skill *and* reliability, especially for aggregated monthly/seasonal totals and for upper quantiles |
| ECC vs Schaake shuffle | Schaake shuffle is a hard benchmark to beat; advanced ECC/Schaake extensions rarely add significantly over the basic versions |
| Weather-scale DL emulators transferred to S2S | They collapse to unskilled climatology; S2S requires ensembles, error-propagation control, and physics-informed constraints (ChaosBench) |

### 3.3 A defensible ranking for your task

For precipitation and temperature, 10 days to 2 months, with a decent reforecast archive:

1. **Establish honest baselines first**: raw ensemble, lead-and-calendar-dependent mean bias correction, climatology, and adaptive/learned climatology. At weeks 4-6, climatology is genuinely hard to beat and many published "improvements" disappear against a well-constructed climatological benchmark.
2. **Then a calibrated statistical baseline**: EMOS (censored shifted gamma or gamma-hurdle for precipitation, Gaussian for temperature) or ELR for tercile probabilities, with ensemble mean and spread as predictors.
3. **Then classical ML**: QRF with tail extension, or gradient-boosted EMOS, adding teleconnection indices (Niño-3.4, IOD/DMI, MJO amplitude and phase via RMM, soil moisture, SST) as predictors. This "bridging" step is where seasonal-range skill actually comes from.
4. **Then spatial deep learning**: a U-Net or basis-function CNN producing categorical (tercile/quintile) probabilities, trained globally or regionally, with the Scheuerer-style parameterisation of *departures from climatological frequencies*.
5. **Then dependence restoration**: ECC or Schaake shuffle if you need coherent multivariate fields.
6. **Optionally**: generative/diffusion ensemble expansion if extremes and tail probabilities matter to you.

---

## 4. Datasets, Benchmarks, and Competitions

| Resource | Content | Use |
|---|---|---|
| **WMO S2S Prediction Project database** (Vitart et al. 2017) | Reforecasts and forecasts from ~11 centres | Primary source for S2S post-processing training data |
| **S2S AI Challenge** (WMO, 2021) | ECMWF global forecasts/reforecasts of precipitation and temperature; 11-member weekly reforecasts 2000-2019 for training, 51-member 2020 forecasts for testing | The standard testbed for global S2S post-processing; used by Horat and Lerch |
| **SubX** (Pegion et al. 2019) | North American multi-model subseasonal experiment | Multi-model S2S |
| **NMME** (Kirtman et al. 2014) | North American Multi-Model Ensemble, monthly to seasonal | Seasonal-range post-processing |
| **SubseasonalClimateUSA** (Mouatadid et al., NeurIPS 2023) | Curated CONUS dataset plus four standard tasks (temperature and precipitation, weeks 3-4 and 5-6), with dynamical, ML, and DL baselines | Best-designed benchmark for ML-based S2S post-processing; `subseasonal_toolkit` on GitHub |
| **ChaosBench** (Nathaniel et al., NeurIPS 2024) | Multi-channel physics-based S2S benchmark including ocean, ice, land reanalysis over 45 years; operational baselines; physics-based metrics | Benchmarking DL emulators at S2S |
| **EUPPBench** (Demaeyer et al. 2023, *ESSD* 15, 2635-2653) | Standardised European post-processing benchmark, station and gridded | The reference benchmark for post-processing methodology (medium range, but the protocols transfer) |
| **AI Weather Quest** (ECMWF, 2025-2026) | Real-time competition: global quintile probabilistic forecasts of weekly-mean 2 m temperature (land), MSLP, and total precipitation (land) at weeks 3 and 4; 42 teams from 15 countries; scored by RPSS | The best current picture of who is actually winning at S2S, in operational conditions. Submitted forecasts are released openly after each 13-week period |
| **ERA5 / IMD gridded / CHIRPS / MSWEP / CPC** | Ground truth | Choice of reference dataset materially affects measured skill |

---

## 5. Verification: Getting the Evaluation Right

This is where S2S post-processing papers most often go wrong, so it deserves explicit design attention.

**Metrics**
- **CRPS / CRPSS** for continuous probabilistic forecasts, and as the natural training loss for distributional methods.
- **RPS / RPSS** for tercile or quintile categorical forecasts. This is the AI Weather Quest scoring metric.
- **Brier score / BSS** for threshold-exceedance and extremes.
- **Reliability diagrams and rank histograms / PIT histograms** for calibration.
- **Anomaly correlation coefficient (ACC)**, spatial correlation, and RMSE for deterministic summaries.
- **Skill decomposition** (reliability, resolution, uncertainty) to distinguish "I fixed the bias" from "I added information".

**Reference forecast choice is decisive.** Skill scores must be computed against a *properly constructed climatological benchmark*, ideally a probabilistic climatology built to match the ensemble configuration and validity time. Many reported gains are gains over a weak reference.

**Fair vs unfair scores.** A 2026 ECMWF result (Roberts, arXiv:2602.15830) shows that deep-learning post-processing methods that minimise an *unfair* score exhibit ensemble-size-dependent behaviour, which can produce misleading conclusions. If you train on CRPS with a finite ensemble, use the fair (bias-corrected) estimator.

**Cross-validation.** Use leave-one-year-out or blocked/rolling-origin splits, never random shuffling, since adjacent forecasts are strongly autocorrelated. Guard against the subtler leak: model climatology and observation climatology computed on the full record leak future information into the training folds.

**Sample size.** Weekly reforecasts over 20 years give roughly 1000 forecast cases per lead time, which is small relative to the number of parameters in a deep network. This is why global or regional weight sharing, heavy regularisation, and small architectures dominate the S2S literature, and why Rasp and Lerch found no meaningful benefit from more than one hidden layer.

**Report conditionally.** Skill at S2S is strongly regime-dependent. Reporting improvement *conditioned on* weather regime, MJO phase, ENSO state, and season is more informative and more publishable than a single global average. One European study found that ML models beat NWP for temperature at 14, 21, and 28 days, and for cold-wave days at 21 and 28 days, but that the quality of forecasts depended on the weather regime present at forecast initialisation.

---

## 6. Gaps and Opportunities

Areas where the literature is thin, in rough order of tractability:

1. **S2S post-processing outside North America, Europe, and East Asia.** Horat and Lerch noted a "remarkable lack" of post-processing approaches for S2S relative to the medium range, and the geographic concentration is stark. South Asia in particular is under-served: the U-Net ISMR calibration study (2026) is one of very few, and the operational Indian extended-range system (IITM ERPv2, CFSv2-based) has mostly been treated with linear or analog methods.
2. **Monsoon-specific structure.** Active-break cycles, intraseasonal oscillation propagation (BSISO), and the strongly non-Gaussian, spatially heterogeneous rainfall distribution over India are poorly served by methods designed for mid-latitude winter precipitation. Regime-conditioned or regime-embedding post-processors are an open direction.
3. **Extremes at S2S.** Most post-processors are trained on average-case losses and therefore regress extremes toward the mean. Tail-aware approaches (QRF with EVT tail extension, weighted scoring rules, generative ensemble expansion) exist but have barely been applied at S2S.
4. **Post-processing AI forecast models.** A brand-new question: AI weather models have different error structures from dynamical models (blurring, drift toward climatology, spectral deficits). PBC's result that correction still helps AIFS substantially suggests this is a productive and largely unexplored direction.
5. **Multivariate and cross-timescale coherence.** Producing calibrated fields that are simultaneously coherent in space, time, and across precipitation-temperature is still hard, and it is exactly what hydrological and agricultural users need.
6. **Explainability and windows of opportunity.** Using post-processor diagnostics to *identify when* the model has exploitable skill, rather than only to correct it, is emerging (weather regime conditioning, teleconnection attribution, XAI on trained CNNs).
7. **Short training data.** Methods that work with 10 to 20 years of reforecasts, transfer learning across regions, and pre-training on reanalysis before fine-tuning on reforecasts.

---

## 7. Practical Recommendations for Your Study Design

1. **Fix the target early.** Weekly or biweekly aggregates, categorical (tercile/quintile) probabilities for weeks 3+, continuous distributions for days 10-20. Do not try to post-process daily precipitation at week 5.
2. **Use the ensemble, not just the mean.** Spread, quantiles, member-level fields, and lagged members all carry information. Set-based or permutation-invariant architectures are designed for this.
3. **Model departures from climatology, not absolute values.** This single reparameterisation is the most consistently reported design win in the S2S post-processing literature.
4. **Add slow predictors.** ENSO indices, IOD/DMI, MJO RMM amplitude and phase, SST anomalies, soil moisture, snow cover, stratospheric polar vortex strength. At seasonal range these are often more informative than the model's own precipitation field.
5. **Share weights spatially, adapt locally.** Global or regional training with location embeddings and location-specific normalisation, following Scheuerer et al. and Rasp and Lerch.
6. **Build the baseline ladder** (raw, debiased, climatology, adaptive climatology, EMOS/ELR, QRF) before you build the deep model, and report all of it.
7. **Restore dependence at the end** with ECC or the Schaake shuffle if any downstream model consumes your output.
8. **Verify with fair scores, blocked CV, and regime-conditioned reporting.**

---

## 8. Key References

**Reviews and foundational**

- Vannitsem, S., et al. (2021). Statistical postprocessing for weather forecasts: review, challenges, and avenues in a big data world. *Bulletin of the American Meteorological Society*, 102(3), E681-E699.
- Vannitsem, S., Wilks, D. S., and Messner, J. W., eds. (2018). *Statistical Postprocessing of Ensemble Forecasts*. Elsevier. See especially Ch. 4 (Schefzik and Möller, dependence structures) and Ch. 10 (Van Schaeybroeck and Vannitsem, long-range forecasts from months to decades).
- Li, W., et al. (2017). A review on statistical postprocessing methods for hydrometeorological ensemble forecasting. *WIREs Water*, 4, e1246.
- Robertson, A. W., Vitart, F., and Camargo, S. J. (2020). Subseasonal to seasonal prediction of weather to climate. *JGR Atmospheres*, 125(6).
- Vitart, F., et al. (2017). The Subseasonal to Seasonal (S2S) Prediction Project database. *BAMS*, 98(1), 163-173.

**Parametric statistical post-processing**

- Gneiting, T., Raftery, A. E., Westveld, A. H., and Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *MWR*, 133, 1098-1118.
- Raftery, A. E., Gneiting, T., Balabdaoui, F., and Polakowski, M. (2005). Using Bayesian model averaging to calibrate forecast ensembles. *MWR*, 133, 1155-1174.
- Scheuerer, M., and Hamill, T. M. (2015). Statistical postprocessing of ensemble precipitation forecasts by fitting censored, shifted gamma distributions. *MWR*, 143, 4578-4596.
- Messner, J. W., Mayr, G. J., and Zeileis, A. (2017). Nonhomogeneous boosting for predictor selection in ensemble postprocessing. *MWR*, 145, 137-147.
- Bremnes, J. B. (2020). Ensemble postprocessing using quantile function regression based on neural networks and Bernstein polynomials. *MWR*, 148, 403-414.

**Machine learning**

- Taillardat, M., Mestre, O., Zamo, M., and Naveau, P. (2016). Calibrated ensemble forecasts using quantile regression forests and ensemble model output statistics. *MWR*, 144, 2375-2393.
- Taillardat, M., Fougères, A.-L., Naveau, P., and Mestre, O. (2019). Forest-based and semiparametric methods for the postprocessing of rainfall ensemble forecasting. *Weather and Forecasting*, 34(3), 617-634.
- Taillardat, M., and Mestre, O. (2020). From research to applications: examples of operational ensemble post-processing in France using machine learning. *NPG*, 27, 329-347.
- Schulz, B., and Lerch, S. (2022). Machine learning methods for postprocessing ensemble forecasts of wind gusts: a systematic comparison. *MWR*, 150, 235-257.

**Deep learning**

- Rasp, S., and Lerch, S. (2018). Neural networks for postprocessing ensemble weather forecasts. *MWR*, 146, 3885-3900. Code: `github.com/slerch/ppnn`.
- Scheuerer, M., Switanek, M. B., Worsnop, R. P., and Hamill, T. M. (2020). Using artificial neural networks for generating probabilistic subseasonal precipitation forecasts over California. *MWR*, 148, 3489-3506.
- Horat, N., and Lerch, S. (2024). Deep learning for postprocessing global probabilistic forecasts on subseasonal time scales. *MWR*, 152(3), 667-687. Code: `github.com/HoratN/pp-s2s`.
- Höhlein, K., Schulz, B., Westermann, R., and Lerch, S. (2024). Postprocessing of ensemble weather forecasts using permutation-invariant neural networks. *AIES*, 3(1).
- Worsnop, R. P., Scheuerer, M., Hamill, T. M., Smith, T. A., and Schlör, J. (2024). RUFCO: a deep learning framework to postprocess subseasonal precipitation accumulation forecasts. *AIES*, 3(4).
- Li, W., Pan, B., Xia, J., and Duan, Q. (2022). Convolutional neural network-based statistical post-processing of ensemble precipitation forecasts. *Journal of Hydrology*, 605, 127301.

**S2S-specific ML and current state of the art**

- Mouatadid, S., Orenstein, P., Flaspohler, G., Cohen, J., Oprescu, M., Fraenkel, E., and Mackey, L. (2023). Adaptive bias correction for improved subseasonal forecasting. *Nature Communications*, 14, 3482.
- Mouatadid, S., et al. (2023). SubseasonalClimateUSA: a dataset for subseasonal forecasting and benchmarking. *NeurIPS* 36.
- Guan, H., Mouatadid, S., Orenstein, P., Cohen, J., Weyn, J. A., Mackey, L., et al. (2026). Enhancing AI and dynamical subseasonal forecasts with probabilistic bias correction. arXiv:2604.16238. Code: `github.com/mouatadid/pbc`.
- Nathaniel, J., Qu, Y., Nguyen, T., Yu, S., Busecke, J., Grover, A., and Gentine, P. (2024). ChaosBench: a multi-channel, physics-based benchmark for subseasonal-to-seasonal climate prediction. *NeurIPS* 37, 43715-43729.
- Chen, L., et al. (2024). A machine learning model that outperforms conventional global subseasonal forecast models (FuXi-S2S). *Nature Communications*, 15.
- Loegel, O., Talib, J., Vitart, F., Hoffmann, J., and Chantry, M. (2025). The AI Weather Quest: an international competition for sub-seasonal forecasting with AI. *Environmental Research: Climate / IOP*.
- Roberts, C. D., and Vitart, F. (2026). Ensemble reliability and the signal-to-noise paradox in ECMWF subseasonal forecasts. *QJRMS*, e70224.
- Roberts, C. D. (2026). Ensemble-size-dependence of deep-learning post-processing methods that minimize an (un)fair score. arXiv:2602.15830.

**Seasonal-range calibration**

- Schepen, A., Zhao, T., Wang, Q. J., and Robertson, D. E. (2018). A Bayesian modelling method for post-processing daily sub-seasonal to seasonal rainfall forecasts from global climate models. *HESS*, 22, 1615-1628.
- Schepen, A., Everingham, Y., and Wang, Q. J. (2020). On the joint calibration of multivariate seasonal climate forecasts from GCMs. *MWR*, 148, 437-456.
- Tian, D., et al. (2021). Calibration of ECMWF seasonal ensemble precipitation reforecasts in Java (Indonesia) using bias-corrected precipitation and climate indices. *Weather and Forecasting*, 36(4).
- Khajehei, S., Ahmadalipour, A., and Moradkhani, H. (2018). Copula-based bias correction of NMME precipitation forecasts.

**Multivariate dependence**

- Clark, M., Gangopadhyay, S., Hay, L., Rajagopalan, B., and Wilby, R. (2004). The Schaake shuffle. *Journal of Hydrometeorology*, 5, 243-262.
- Schefzik, R., Thorarinsdottir, T. L., and Gneiting, T. (2013). Uncertainty quantification in complex simulation models using ensemble copula coupling. *Statistical Science*, 28, 616-640.
- Schefzik, R. (2016). A similarity-based implementation of the Schaake shuffle. *MWR*, 144, 1909-1921.
- Lerch, S., Baran, S., Möller, A., Groß, J., Schefzik, R., Hemri, S., and Graeter, M. (2020). Simulation-based comparison of multivariate ensemble post-processing methods. *NPG*, 27, 349-371.
- Van Schaeybroeck, B., and Vannitsem, S. (2015). Ensemble post-processing using member-by-member approaches. *QJRMS*, 141, 807-818.

**Generative**

- Li, L., Carver, R., Lopez-Gomez, I., Sha, F., and Anderson, J. (2024). Generative emulation of weather forecast ensembles with diffusion models (SEEDS). *Science Advances*, 10.
- Chen, J., Janke, T., Steinke, F., and Lerch, S. (2024). Generative machine learning methods for multivariate ensemble post-processing. *Annals of Applied Statistics*.
- Improving ensemble extreme precipitation forecasts using generative artificial intelligence (2025). *AIES*, 4(2).

**India and South Asia**

- U-Net calibration of subseasonal Indian summer monsoon rainfall forecasts (2026). *Climate Dynamics*. doi:10.1007/s00382-026-08286-x.
- Abhilash, S., et al. (2016). A bias-correction and downscaling technique for operational extended range forecasts based on self-organizing map. *Climate Dynamics*.
- Bias-corrected extended-range forecast over India for hydrological applications during monsoon 2020 (2022). *Pure and Applied Geophysics*. doi:10.1007/s00024-022-02998-x.
- Evaluation of performance of seasonal precipitation prediction at regional scale over India (2018). *Theoretical and Applied Climatology*. doi:10.1007/s00704-018-2421-9.
