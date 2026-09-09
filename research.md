# Research Papers

- https://link.springer.com/article/10.1007/s00382-017-3580-6
- https://ncarprojects.ucar.edu/ai/science/s2s
- https://www.ai2es.org/s2s-prediction/
- Shi et al. (2026), *JGR: Machine Learning and Computation*, https://doi.org/10.1029/2025JH000993 (local PDF)
- Zhang, Yang, Gao, Hong, Zhang, Wen & Cheng (2023), *Atmospheric Research*, https://doi.org/10.1016/j.atmosres.2022.106502 (local PDF)

## Three Resources Explained: Climate Bias Correction and AI for S2S Prediction

This document walks through three resources on climate/weather modeling. The first is a peer-reviewed methods paper on statistically correcting climate model output. The second and third are lab/program pages describing ongoing machine learning research programs focused on subseasonal-to-seasonal (S2S) prediction. Each is covered separately below, followed by a short synthesis at the end.

### Resource 1: Multivariate Quantile Mapping Bias Correction (MBCn)

**Source:** Cannon, A. J. (2017). "Multivariate quantile mapping bias correction: an N-dimensional probability density function transform for climate model simulations of multiple variables." *Climate Dynamics*, 50, 31-49. Link: https://link.springer.com/article/10.1007/s00382-017-3580-6

**What is this about**

Climate models (global or regional) have systematic errors, called biases, in the variables they simulate (temperature, precipitation, humidity, etc.). Before model output can be used for real applications, like flood or wildfire risk assessments, these biases usually need to be corrected. The most common fix is "quantile mapping," a statistical technique that adjusts a model's simulated values so that their statistical distribution matches observed historical data.

The problem is that quantile mapping is normally applied to one variable at a time. This ignores the relationships between variables. For example, if wind speed and humidity are correlated in the real world but a model gets that relationship wrong, correcting each variable separately won't fix the joint relationship. This matters a lot for downstream applications that combine several variables, like fire-risk indices or drought indices.

**What was done**

The author adapted a technique from image processing called the N-dimensional probability density function transform (N-pdft), originally developed to transfer color characteristics from one photo to another. He turned it into a climate bias-correction method called MBCn (Multivariate Bias Correction, N-dimensional). The core idea: repeatedly rotate the multivariate dataset in random directions, apply standard one-variable quantile mapping along each rotated axis, then rotate back. Doing this over many iterations gradually reshapes the model's full joint (multivariate) distribution to match the observed one, not just each variable individually.

He also extended the method (quantile delta mapping, QDM) so that future climate-change trends in the model are preserved even as the historical bias is corrected, since a bias-correction method that accidentally erases a real warming or drying trend would be dangerous for projections.

Three test cases were used to demonstrate and validate MBCn:
- An image recoloring example, chosen because its color-channel data behaves mathematically like a small climate dataset. This let the author check that the method converges properly.
- A real climate application: correcting seven weather variables from a regional climate model (CanRCM4) over North America, then using the corrected values to compute Canada's Fire Weather Index (FWI) system, which estimates wildfire risk from temperature, humidity, wind, and precipitation.
- A spatial precipitation example, testing whether MBCn can jointly bias-correct and effectively downscale precipitation across a grid of 25 points, addressing the well-known problem that naive correction of interpolated precipitation creates unrealistically smooth, unrealistic rain fields.

**Main findings**

- MBCn consistently outperformed three alternative methods: plain univariate quantile mapping (QDM), and two other multivariate methods that only correct Pearson or Spearman correlation (MBCp and MBCr).
- For the fire weather index, MBCn had the best spatial pattern correlation with observations (above 0.98) and, on average, the best skill scores across error and correlation metrics, clearly ahead of the other three methods.
- MBCn was especially strong for extreme values (like the annual maximum daily FWI), where the simpler methods actually got worse, while MBCn stayed the best performer.
- For spatial precipitation, MBCn produced the most realistic spatial autocorrelation (a measure of how "patchy" versus "smooth" rainfall fields are), correctly avoiding both the unrealistic smoothness produced by naive quantile mapping and the unrealistic patchiness produced by one of the correlation-only methods.
- The method has practical costs: convergence can require many iterations, and when the number of variables being corrected is large relative to the amount of data, there is a real risk of overfitting to noise in the historical calibration period. The author suggests techniques like early stopping (borrowed from machine learning) as a safeguard.

**How to apply / learn from it**

- When correcting more than one climate variable at once, especially when those variables feed into a combined index (fire risk, drought, flood, agricultural yield), it's worth checking whether the correction method preserves inter-variable relationships, not just each variable's own distribution. Univariate correction can look fine variable-by-variable while still producing an unrealistic combined index.
- The general recipe (rotate, correct each axis, rotate back, repeat) is a reusable pattern anytime a real-world dataset needs to be reshaped to match a target multivariate distribution while preserving trends. It generalizes beyond climate: any domain with a "make simulated multivariate data statistically match observed multivariate data" problem could reuse this approach.
- Guard against overfitting when the number of variables/locations you're correcting jointly is large relative to your training data; techniques like early stopping, or reducing dimensionality first, help keep the correction from fitting noise instead of real bias.
- Cross-validation matters. The paper's fire-index and precipitation tests both used out-of-sample verification (calibrate on one period, test on another), which is a good template for validating any bias-correction or downscaling workflow before trusting it for future projections.

### Resource 2: ML for Subseasonal-to-Seasonal (S2S) Prediction (NSF NCAR)

**Source:** NSF NCAR Strategic Projects, "Our Science Using AI" program page. Link: https://ncarprojects.ucar.edu/ai/science/s2s

**What is this about**

This is a project page from NSF's National Center for Atmospheric Research (NCAR) describing their ongoing effort to use machine learning to improve subseasonal-to-seasonal forecasting, the notoriously hard prediction window of roughly two weeks to a few months out, which sits between short-term weather forecasting and long-term seasonal/climate prediction.

**What was done**

NCAR is building ML-based "emulators," models trained to imitate the behavior of full physics-based climate models but at a fraction of the computational cost.

- **CAMulator:** an autoregressive ML emulator of NCAR's Community Atmosphere Model (CAM6). It predicts the next atmospheric state (for example, specific humidity) from inputs like sea surface temperature, solar radiation, moisture, and atmospheric energy. It stays stable over long simulated runs and executes about 350 times faster than the original physics-based model. NCAR entered a modified real-time version of CAMulator into the AI Weather Quest, an international S2S forecasting competition, generating 341 possible future realizations of Earth's atmosphere per forecast. A single real-time subseasonal forecast now takes under 3 minutes, and a 20-year hindcast suite (used to evaluate historical forecast skill) takes about 12 hours, work that would be far more expensive with a traditional model.
- **subCESMulator:** a newer, in-development emulator meant specifically for S2S timescales, which require modeling the coupled Earth system (atmosphere, land, and ocean together), rather than the atmosphere alone. It's being trained on output from the CESM2 Large Ensemble, a large set of climate model simulations, along with an AI-ready dataset with additional high-frequency output compatible with NCAR's CREDIT framework.

**Main findings**

The page is a program summary rather than a paper reporting statistical results, but the key reported facts are:

- CAMulator remains numerically stable over long autoregressive rollouts (many models drift or blow up after too many predicted steps), and runs roughly 350 times faster than CAM6.
- The speed gain lets NCAR generate large ensembles (341 realizations) and full 20-year hindcast suites in about 12 hours, something that would be computationally prohibitive with a conventional dynamical model.
- Subseasonal prediction is described as needing coupled representation of atmosphere, land, and ocean together, which is why a separate, more complex emulator (subCESMulator) is being built rather than just extending an atmosphere-only model.

**How to apply / learn from it**

- ML emulation is a practical way to trade a small amount of accuracy for a large amount of speed, enabling ensemble sizes and hindcast lengths that would otherwise be computationally out of reach. This is broadly useful anywhere large ensembles or rapid iteration matter more than perfect physical fidelity.
- Autoregressive stability is a real engineering challenge worth planning for early: an emulator that predicts "next state from current state" repeatedly can drift or diverge over long rollouts, so stability testing over long horizons should be part of any similar emulator-building effort.
- S2S problems need multi-component (coupled) models. If you're trying to predict something that spans days to months out, atmosphere-only inputs are probably insufficient; land and ocean states carry the slower-evolving memory that gives skill at those longer lead times.
- Public hindcast datasets lower the barrier for downstream researchers. NCAR's plan to release the 20-year hindcast database is a good practice to emulate: sharing a large, standardized evaluation dataset makes it much easier for the broader community to benchmark their own S2S methods.

### Resource 3: S2S Prediction of Extreme Weather (NSF AI2ES)

**Source:** NSF AI2ES (Artificial Intelligence for Earth System Science), "S2S Prediction" research page. Link: https://www.ai2es.org/s2s-prediction/

**What is this about**

AI2ES is an NSF-funded institute focused on applying AI to Earth system science. This page describes their "use-inspired" research into subseasonal-to-seasonal prediction, again targeting the hard-to-predict window between short-term weather and long-term seasonal outlooks, with an emphasis on interpretability (understanding why a model makes a given prediction, not just getting the prediction) and on extreme weather specifically. It covers three related sub-projects.

**What was done**

- **Interpretable AI for S2S (temperature prediction).** The team built an interpretable neural network approach, based on "prototype" pattern-matching (their "This Looks Like That There" architecture), to forecast temperature along the U.S. West Coast at S2S timescales. Because long, reliable observational records are scarce at this timescale, they trained on climate model output (CESM2) first and then fine-tuned using reanalysis/observational data (ERA5), a transfer-learning strategy meant to compensate for limited real-world training data. They also built a separate interpretable "analog forecasting" method, which predicts the future by finding historical patterns that closely resemble the current state.
- **AI for S2S scientific discovery (severe convective weather).** The team used a convolutional neural network to estimate the likelihood of tornadoes, hail, and damaging wind one to three weeks out, across different U.S. regions. They then applied explainable AI (XAI) techniques to the trained model to identify which input features and spatial locations most influenced its predictions, using the model as a discovery tool for the underlying atmospheric dynamics, not just a forecasting tool.
- **Water level prediction (Gulf of Mexico).** In collaboration with the University of Valencia, the team built a supervised learning model to predict average regional water levels along the Northwest Gulf of Mexico at S2S-to-multi-year timescales, relevant for coastal inundation risk.

**Main findings**

- The prototype-based interpretable model showed real predictive skill for West Coast temperature at S2S timescales, an especially hard target since this window is well known to have limited underlying predictability.
- Using transfer learning (pretrain on model output, then fine-tune on real observations) let the team get skillful results despite lacking large amounts of real-world S2S training data.
- The interpretability techniques successfully surfaced physically meaningful patterns, specifically, they identified representative atmospheric patterns linked to how the tropical Madden-Julian Oscillation (a slow-moving band of tropical rainfall and wind) influences midlatitude weather. This is an example of AI doing genuine scientific discovery, not just prediction.
- Their controlled-abstention approach lets the model flag "forecasts of opportunity", situations where the atmosphere is in an unusually predictable state, so forecasters can know when to trust the model more.
- For water levels, the initial expectation was that deep Gulf of Mexico heat content would be the useful predictor, but the results so far show that past water level values themselves are the more important driver at S2S scale, an example of a hypothesis being revised by the data.

**How to apply / learn from it**

- Interpretability is not just a nice-to-have here; it's used as a discovery tool. Building explainability into a model from the start (rather than bolting it on afterward) let this team find real physical relationships (like the MJO-midlatitude teleconnection), a pattern worth copying in other data-scarce, physically complex domains.
- Transfer learning helps when observational data is limited. Pretraining on a large but imperfect simulated dataset (climate model output), then fine-tuning on smaller amounts of real observations, is a reusable strategy anywhere ground-truth data is scarce relative to the complexity of the problem.
- "Abstention" or confidence-flagging is valuable operationally. A model that can say "I'm confident here" versus "this is a low-skill situation" is more usable for real decision-making than one that always outputs a prediction with no sense of its own reliability.
- Let the data challenge your assumptions. The Gulf of Mexico example, where deep-ocean heat content turned out to be less predictive than expected, is a reminder to let empirical results, not just physical intuition, guide which predictors actually matter.

### Resource 4: GAN-W2C — Distribution-Guided Ensemble Postprocessing for S2S Precipitation

**Source:** Shi, W., Pan, B., Huang, J., Dou, T., Feng, J., Yuan, H., Wang, A., Xia, X., Huang, L., & Luo, Y. (2026). "Distribution-guided ensemble postprocessing for S2S precipitation forecasts: A seamless pathway using deep generative models." *Journal of Geophysical Research: Machine Learning and Computation*, 3, e2025JH000993. https://doi.org/10.1029/2025JH000993

**What is this about**

This paper tackles the same core difficulty our project's `APPROACH.md` calls out (Section 5.1): a post-processing correction that works at short lead times can fail at long lead times, because the dynamical forecast and the verifying observation are still "paired" (traceable to the same evolving trajectory) at short leads but become effectively unrelated at long leads once chaos scrambles the forecast's initial-condition information. A regression-based corrector trained across all lead times at once ends up either discarding most of its training data (the long-lead majority) or fitting to what is mostly noise. The authors propose a single unified deep-generative framework, GAN-W2C ("weather-to-climate"), that is valid across the whole lead-time range at once, tested on ECMWF S2S reforecasts over Southern China.

**What was done**

GAN-W2C combines two GAN variants into one architecture, trained jointly:

- A **weather-scale component** (paired, short-lead regime): a direct prediction loss plus an adversarial loss (an image-to-image-translation-style setup) sharpens the correction and avoids blurring; a VAE-style KL-divergence loss injects controlled perturbations so each raw ensemble member is expanded into a small "mini-ensemble" of plausible corrected outcomes.
- A **climate-scale component** (unpaired, long-lead regime): a CycleGAN, which doesn't need paired forecast/observation samples at all — it just needs the forecast's overall distribution and the observation's overall distribution to look alike, enforced via a cycle-consistency loss (map forecast→observation→forecast and require you land back where you started).
- These two are unified by **tapering the weight of the paired prediction loss down to zero as lead time grows** (a quadratic decay), so short-lead samples get the full stack of constraints while long-lead samples are governed only by the distribution-matching (unpaired) constraints. This is a single continuous training objective across all lead times, not two separate models glued together after the fact.
- The framework transforms the *whole* atmospheric state (circulation + precipitation jointly), not precipitation alone, with an "invariance loss" anchoring the large-scale circulation close to the raw dynamical forecast — precipitation is generated conditional on that anchored circulation, which is what keeps the output physically self-consistent (e.g., a precipitation anomaly and the moisture-transport pattern that would produce it stay coherent) instead of "looking realistic but meteorologically nonsensical."
- Baselines: raw dynamical output, a GAN-free MOS regression baseline (trained only on the first 3 lead days, MSE loss), EMOS (a parametric per-grid-point ensemble statistical model), and probabilistic climatology.

**Main findings**

- GAN-W2C beat every baseline on CRPS (92.9% of the raw dynamical model's score, i.e., ~7% better) and on Brier score for 90th/95th-percentile extremes (89.4%/91.0% of the raw model's score), extending the skillful *probabilistic* forecast horizon out to roughly week 3 — versus pentad 2–3 for the raw dynamical model and pentad 3 for MOS.
- The regression-based MOS baseline exhibited the classic "blurring" failure mode: it predicts moderate precipitation almost everywhere, underestimating extremes and spatial heterogeneity, and past ~2 weeks its ensemble mean becomes implausibly weak. GAN-W2C preserved realistic spatial detail (power spectra much closer to observations) and extreme magnitudes.
- GAN-W2C calibrated ensemble spread substantially better than MOS at short leads (+22.5% in a spread-to-error ratio metric) and stayed reasonable at long leads, whereas MOS's under-dispersion problem *worsened* with lead time.
- A joint EOF (empirical orthogonal function) analysis of precipitation vs. 850 hPa moisture transport showed GAN-W2C's generated fields stayed physically self-consistent even where they missed the exact observed pattern, whereas MOS's precipitation output frequently contradicted the moisture-transport conditions implied by its own (unmodified) circulation fields — a concrete demonstration that univariate regression correction can silently break multivariate physical relationships.
- The framework decomposes total forecast uncertainty into two components with distinct behavior: `std_dyn'` (spread across corrected versions of the different dynamical ensemble members — this grows with lead time and saturates, since it inherits the dynamical model's own chaotic-divergence estimate) and `std_post` (the internal spread of the mini-ensemble generated from a *single* dynamical member — this stays roughly flat across lead times, since it reflects postprocessing/parameterization uncertainty rather than chaos). Both correlate meaningfully with actual local forecast error (r ≈ 0.71–0.74), i.e., the decomposition is not just a diagnostic curiosity — it's a usable advance indicator of where the forecast is likely to be wrong.
- An ablation study confirmed *both* halves are necessary: a weather-scale-only model underestimates intensity and ensemble spread everywhere (even at short leads it never learns to see); a climate-scale-only model matches marginal distributions well but has spread-error correlation no better than the raw dynamical model, meaning it fails to produce genuinely useful uncertainty information.

**How to apply / learn from it**

- This paper is direct, up-to-date evidence for the exact critique in `APPROACH.md` Section 5.1: a regression-based, uniform-strength corrector (functionally the same category as our Baseline 3 / Zhang-style Random Forest, see Resource 5 below) blurs extremes and gets worse — not just stagnant — at longer lead times if trained naively across all leads. GAN-W2C's finding that MOS-style regression correction "further disrupts physical consistency" strengthens the case for why a fixed-strength correction is the wrong default, and is worth citing alongside the Zhang et al. result when motivating the confidence-gating idea.
- The **lead-time-dependent loss-tapering** mechanism is a genuinely different way of handling "correction reliability degrades with lead time" than our project's planned approach (Section 5.3: gate the *output* between the correction and climatology using a confidence signal `c`). GAN-W2C instead changes *what the model is trained to do* as lead time grows — shifting the training objective itself from paired regression toward unpaired distribution-matching. These are complementary, not mutually exclusive: one could imagine gating at inference time (our approach) using a model whose training itself already tapers its reliance on paired-sample assumptions (this paper's approach). Worth keeping in mind as a variant for Section 6 ("what would make it Q1-worthy").
- The **uncertainty decomposition** into a dynamically-propagated component and a postprocessing-inherent component is a stronger, more principled analogue of our project's MVP confidence proxy (raw 5-member ensemble spread, Section 5.4). If/when we build the "full version" learned confidence signal (Section 5.3), this paper's `std_dyn'` / `std_post` split is a good template for what the learned signal should actually be trying to separate — chaotic/dynamical uncertainty (which should, in principle, be *learnable and lead-time-dependent*) versus a roughly constant postprocessing-model uncertainty floor.
- The "ensemble-agnostic" design — training and inferring member-by-member rather than needing the whole ensemble at once, yet still achieving implicit ensemble calibration — is a practical engineering pattern worth reusing regardless of whether we adopt GANs: it reduces memory/compute requirements and would let a future model be trained on GEFS's 5 members without hard-coding that number anywhere in the architecture.
- A caution for scope: the authors are explicit that this method requires *complete retraining* for a new region (it learns region-specific bias and climatology, not a transferable physical model) and that a GAN's notoriously unstable training adds real engineering cost. If a generative postprocessing approach is ever added to this project (per `APPROACH.md` Section 8's "reserved future upgrades"), budget for that instability and region-specific retraining rather than assuming a CONUS model would need only light tuning.

### Resource 5: Random Forest Post-Processing of S2S Extreme Precipitation Occurrence over CONUS

**Source:** Zhang, L., Yang, T., Gao, S., Hong, Y., Zhang, Q., Wen, X., & Cheng, C. (2023). "Improving Subseasonal-to-Seasonal forecasts in predicting the occurrence of extreme precipitation events over the contiguous U.S. using machine learning models." *Atmospheric Research*, 281, 106502. https://doi.org/10.1016/j.atmosres.2022.106502

**Note:** this is the exact paper already cited as **"Baseline 3"** in this project's `APPROACH.md` (Section 3's reference table, and Section 5.2's formal baseline definitions). Our project's extreme-event label definition — local, per-grid-cell 99th percentile of a weekly-total climatology — is deliberately copied from this paper's methodology "for direct comparability," per `APPROACH.md` Section 4.3. The summary below explains what that baseline actually does and found.

**What is this about**

Most S2S evaluation work looks at the accuracy of the forecast *ensemble mean*. This paper instead asks a narrower, more decision-relevant question: how well does an S2S model's full ensemble *spread* capture the occurrence of extreme (99th-percentile) weekly precipitation events over the contiguous U.S. (CONUS), and can a machine-learning correction improve that specifically? The forecast source is NASA's GEOS5 (a contributing model to the North America Multi-Model Ensemble Phase II), a 10-member ensemble reforecast dataset, 1982–2011, verified against the PRISM gauge/radar precipitation product.

**What was done**

- **Model**: independent Random Forest (RF) classifiers trained separately at *each* 0.25° grid pixel over CONUS, and separately for each lead week (week 1 = days 1–7, ... week 4 = days 22–28), to predict whether that week's precipitation will exceed the local 99th-percentile threshold.
- **Cross-validation**: "leave-10-years-out" — the 30-year record is split into three 10-year blocks; when one block is the evaluation target, the RF is trained on the other 20 years.
- **Ten experiment scenarios** varying two things independently: (a) which input variables feed the RF — precipitation alone, or combined with surface air temperature (T), 500 hPa geopotential height (G500), and/or 850 hPa geopotential height (G850); and (b) the RF's maximum tree depth (3, 5, 7, 9, 12, 15), since deeper trees can capture more complex, spatially-varying relationships but risk overfitting.
- **New ensemble-aware evaluation metrics**, since standard categorical metrics (POD/FAR/CSI) don't account for a multi-member ensemble: **EPOD** (a "hit" if *any* member correctly predicts an extreme), **EFAR** (a "false alarm" only if *all* members incorrectly predict one), and **ECSI** (a combined hit-vs-false-alarm score) — designed specifically to ask "does the ensemble's spread of possibilities envelop the eventual outcome?" They also computed an event-reweighted Brier Skill Score (to avoid the well-known problem that rare-event skill scores can look artificially good by mostly forecasting "no event"), and regional AUROC (Area Under the ROC Curve) across the nine NCEI climate regions.

**Main findings**

- Using precipitation alone as the RF input significantly improved skill at week 1, but that improvement faded fast by weeks 3–4 — consistent with the general S2S predictability-decay problem.
- Adding the auxiliary atmospheric variables (T, G500, G850) as additional RF inputs produced modest but real gains specifically at the *longer* lead times (weeks 2–4), where precipitation alone had stopped helping — the physical reasoning being that precipitation itself is a noisy, parameterized model output, while circulation variables like geopotential height are more directly resolved by the dynamical model's physics and therefore carry more genuine signal at long leads.
- Tuning the RF's maximum tree depth mattered a lot: skill increased steadily up to a tree depth of about 9–12, after which further depth gave essentially no additional benefit (and RF's overfitting behavior became suspected at very deep trees). With the fully-tuned configuration, ECSI improved over the raw GEOS5 baseline by roughly 120% at week 1, 155% at week 2, 187% at week 3, and 250% at week 4 relative to the untouched dynamical forecast.
- The correction's benefit was strongly *regional*: skill gains were much larger and more consistent along the U.S. West Coast and East Coast, and weaker in the interior — specifically the Southwest, South, and West North Central NCEI regions. The authors' explanation is mechanistic: coastal extreme events tend to be driven by large-scale, multi-day synoptic systems (atmospheric rivers, extratropical cyclones) that dynamical models — even at coarse resolution — represent reasonably well, whereas interior extremes are more often driven by small-scale, sub-daily mesoscale convection that GCMs at ~100 km resolution simply cannot resolve, so there's less real signal for any post-processing method (ML or otherwise) to extract.

**How to apply / learn from it**

- This paper is the direct precedent for our project's "Baseline 3" (fixed-strength correction) — its own reported result that RF correction helps substantially at week 1 but needs extra engineering (auxiliary variables, deeper trees) just to stay useful at weeks 2–4, without ever closing the gap back to week-1-level skill, is itself evidence supporting our project's hypothesis: a single fixed correction strength is not well matched to how forecast reliability actually degrades across lead times, which is the whole motivation for the confidence-gating approach in `APPROACH.md` Section 5.3.
- The **regional skill split by dominant precipitation mechanism** (synoptic/coastal vs. convective/interior) is a useful additional lens for our project's own week 1–4 sweep and physical-attribution plan (`APPROACH.md` Section 7, currently framed only around MJO phase/ONI). It suggests that our confidence signal `c` might end up correlating with *region* (or a region/mechanism proxy) as much as with time-varying teleconnection state — worth checking for in the validation plan, since a spatially-varying — not just temporally-varying — component of confidence would be a genuinely new finding beyond what AI2ES's own interpretability work (Resource 3) surfaced.
- The **ensemble-aware metrics** (EPOD/EFAR/ECSI) and the event-reweighted Brier Skill Score are directly reusable for our own evaluation, and are a good sanity check against a known failure mode: a naive Brier Score on a 99th-percentile rare event can look deceptively good just by mostly predicting "no event," which is exactly why this paper reweights it and why our own validation plan (Section 7) should consider doing the same rather than using an unweighted Brier Score.
- The hyperparameter-tuning result (depth ~9–12 is the sweet spot; beyond that is wasted complexity and mild overfitting risk) is a concrete, directly-transferable data point if a Random-Forest-based fixed-strength baseline is implemented literally as specified in our own `APPROACH.md` Section 5.2 — it saves us from re-deriving that tuning curve from scratch.
- One limitation worth carrying forward: this paper trains one independent RF *per grid pixel per lead week* (no spatial or cross-lead sharing at all). Our project's planned single model conditioned on lead week as an input (`APPROACH.md` Section 5.3, "multi-lead sharing") is explicitly designed to let information transfer across lead times in a way this baseline architecture cannot — this paper's per-pixel/per-week independence is a reasonable methodological point of contrast to highlight when explaining why our design choice matters.

### Putting the three together

These resources sit at different points of the same broader challenge: producing trustworthy predictions and projections of the Earth system, and making the most of imperfect models and limited data.

- The Cannon (2017) paper is a foundational statistical technique: a way to fix the errors that physics-based climate models make, especially when several variables need to be jointly consistent (as in a fire-risk or drought index). It is a tool used after a physical model has already produced output.
- The NCAR S2S page shows a complementary, more recent approach: instead of correcting a slow, expensive physical model's output, replace parts of the modeling process itself with a fast ML emulator, trading some fidelity for dramatically more speed and larger ensembles.
- The AI2ES page shows a third angle: using ML not just for speed or bias correction, but to extract new scientific understanding (interpretable patterns, teleconnections, confidence estimates) from data that is otherwise very hard to predict.

Together they illustrate a natural pipeline that a research or applications team could borrow from: run or emulate a physical model (NCAR-style emulators), statistically correct its remaining biases so it matches reality (MBCn-style multivariate bias correction), and then build interpretable, uncertainty-aware ML on top to extract forecasts and scientific insight (AI2ES-style approach), especially for the notoriously hard S2S prediction window.

### Where Resources 4 and 5 fit in

Resources 4 (GAN-W2C) and 5 (Zhang et al.'s Random Forest correction) are both concrete, published instances of the "statistically correct a dynamical model's remaining biases" step in the pipeline above — but they sit at opposite ends of a spectrum this project's own `APPROACH.md` is explicitly trying to navigate:

- **Zhang et al. (Resource 5)** is a fixed-strength, regression/classification-style corrector — the same category of method as this project's Baseline 3. It works, especially at short lead times, but its own results show the benefit shrinking (and needing more engineering to sustain) as lead time grows, and its skill is strongly gated by region/precipitation mechanism rather than being uniform across CONUS.
- **GAN-W2C (Resource 4)** is a more sophisticated response to exactly the same underlying problem — the corrector's assumptions about "the forecast and observation are still related" quietly stop being true at long lead times — but solves it by changing the *training objective* itself (tapering from paired regression to unpaired distribution-matching as lead time grows) and by explicitly decomposing forecast uncertainty into dynamical and postprocessing components, rather than gating a single output between "corrected" and "climatology" at inference time.

This project's own approach (confidence-gated correction, `APPROACH.md` Section 5) is a third, deliberately simpler path toward the same goal: rather than changing the training objective (GAN-W2C) or accepting a fixed correction strength (Zhang et al.), it keeps a fixed-strength corrector but learns *when to trust it*, gating its output against climatology using a confidence signal. Reading these two papers side by side is a useful check on where this project's contribution actually sits: it is neither reinventing Zhang et al.'s baseline, nor duplicating GAN-W2C's generative machinery, but testing whether a much cheaper confidence-gating mechanism can capture some of the same benefit GAN-W2C gets from a full generative rearchitecture — and, per `APPROACH.md` Section 8, GAN-style generative postprocessing remains a plausible future upgrade path if the simpler gating idea proves the underlying hypothesis correct.

## What is quantile mapping?

Quantile mapping is the statistical technique underlying both the plain "QDM" baseline and Cannon's MBCn method above. The problem it solves: a climate model's simulated values for some variable (e.g., temperature) don't statistically match real observed values, maybe the model runs a few degrees too warm on average, or its distribution is too narrow, too wide, or skewed differently than reality. Quantile mapping corrects the simulated values so their statistical distribution lines up with the observed one.

**How it works:**

1. Take the model's simulated values and the observed values over some shared historical period.
2. Build a cumulative distribution (essentially, a ranking from lowest to highest) for each dataset.
3. For any simulated value, find its **percentile/quantile** within the simulated distribution (e.g., "this value sits at the 90th percentile of simulated temperatures").
4. Replace it with whatever value sits at that *same* percentile in the **observed** distribution.

Because the correction is applied percentile-by-percentile rather than as one fixed offset (like "subtract 2 degrees from everything"), it can fix the mean, the spread, and the shape of the distribution (including skew and extreme values) all at once, not just a simple average bias.

**Two refinements that show up in the resources above:**

- **Quantile Delta Mapping (QDM):** Plain quantile mapping, applied naively to *future* projections, can accidentally erase a real climate-change trend, because it forces future values back onto the shape of the historical observed distribution. QDM instead corrects the *bias* while preserving the *change* relative to the model's own historical simulation, so a projected warming or drying trend survives the correction.
- **MBCn (multivariate extension):** Ordinary quantile mapping only works one variable at a time, which can leave the *relationships between* variables wrong even if each variable looks fine individually. MBCn repeatedly rotates the data into random directions, applies one-variable quantile mapping along each rotated axis, then rotates back, iterating until the entire joint (multivariate) distribution matches observations, not just each variable in isolation.
