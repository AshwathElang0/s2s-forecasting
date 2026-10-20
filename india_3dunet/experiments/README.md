# Experiments

One folder per experiment: the scripts that ran it, its runs (checkpoints,
predictions, training logs) and its results. Full write-ups are in
[`../results.md`](../results.md); each folder's README says what it holds and
how to rerun it. Shared code lives outside this folder: `model/`,
`preprocess/`, `train.py`, `test.py`, `train_ablation.py`.

Experiments 1–8 were first scored one day off (ECMWF lead t against IMD
issue+t). [`rescore_aligned.py`](rescore_aligned.py) rescores every saved
Exp 2–7 run on aligned days; the output is
[`aligned_rescore.md`](aligned_rescore.md), and results.md uses those numbers.

| # | folder | question | short answer |
|---|---|---|---|
| 0 | [exp00_pipeline_checks](exp00_pipeline_checks/) | does the pipeline run end to end? | yes (smoke runs only) |
| 1 | [exp01_training_size](exp01_training_size/) | does more training data help? | yes: 60 → 250 → 724 samples keeps improving |
| 2 | [exp02_reference_baselines](exp02_reference_baselines/) | is the score honest? | U-Net beats raw, climatology and log-debiased ECMWF on daily maps |
| 3 | [exp03_departure_from_normal](exp03_departure_from_normal/) | predict the departure from normal instead? | +0.026 climatology-removed corr on daily maps, 3 seeds, but not on weekly totals; the trust gate adds nothing |
| 4 | [exp04_epre_weekly_scores](exp04_epre_weekly_scores/) | the paper's E_pre metric at weeks 1–4 | bug fixed (floor was 1.386); predictions carry ~1/3 of the real variability |
| 5 | [exp05_input_variables](exp05_input_variables/) | do the 12 atmospheric inputs help? | no, all within noise |
| 6 | [exp06_yesterday_rain](exp06_yesterday_rain/) | does yesterday's observed rain help? | no effect (3 seeds); dropped, giving the 16-input headline model |
| 7 | [exp07_calendar_input](exp07_calendar_input/) | does the day-of-year input help? | no |
| 8 | [exp08_review_diagnostics](exp08_review_diagnostics/) | professor's review: figures, scatter, case studies | U-Net is too dry and loses the all-India signal |
| 9 | [exp09_bias_correction](exp09_bias_correction/) | how far does plain bias correction (Dyn++) get? | beats every U-Net from Exps 1–7 on weekly totals |
| 10 | [exp10_unet_on_bias_correction](exp10_unet_on_bias_correction/) | does a U-Net on top of Dyn++ beat Dyn++? | yes, if trained in mm (`residual_mm`): +0.05–0.07 weekly corr per cell, 3 seeds; ties on all-India |
| 11 | [exp11_past_only_factors](exp11_past_only_factors/) | does `residual_mm` still win with Dyn++ training factors from past years only? | yes: within noise of Exp 10 (3 seeds), still beats Dyn++ per cell at weeks 1–4; this is the final model |
| 12 | [exp12_seasonal_bias_check](exp12_seasonal_bias_check/) | does ECMWF's bias follow the season, justifying the ±1-month window? | yes: neighbouring months' bias maps are alike, distant months' are not; ±1 month ties same-month, wider is worse |

## Run names

Runs were renamed when this folder was set up. Old names appear in older logs
and in the `ablation_set` field of their `stats.json`:

| experiment | new | old |
|---|---|---|
| 0 | `smoke`, `first_full_attempt` | `rain_smoke`, `rain_real` |
| 1 | `n60`, `n250`, `n724` | `rain_small`, `rain_medium`, `rain_full` |
| 3 | `departure_s{0,1,2}` | `rain_anom_plain{,_s1,_s2}` |
| 3 | `gate_s{0,1,2}` | `rain_gated_keep{,_s1,_s2}` |
| 3 | `gate_v1_first_try`, `gate_v2_rain_only_via_gate` | `rain_gated`, `rain_gated_v2` |
| 5 | `rain_only_s0`, `humidity_s0`, `winds_s0`, `all_inputs_s{0,1,2}` | `abl_tp_s0`, `abl_moisture_s0`, `abl_wind_s0`, `abl_all_s{0,1,2}` |
| 6 | `no_obs_s{0,1,2}` (headline model) | `abl_no_lag_s{0,1,2}` |
| 7 | `ecmwf_only_s{0,1,2}` | `abl_no_lag_no_doy_s{0,1,2}` |

`config.find_run(name)` returns a run's folder from its name alone.

## Day alignment

The sample cache's target is IMD date issue+t, one day early relative to ECMWF
lead t (`preprocess/alignment.py`). Experiments 1–8 were trained that way and
are scored with the `obs_side`/`ecmwf_side` slices. Experiments 9 and 10 read
IMD issue+1+t directly and are aligned throughout.
