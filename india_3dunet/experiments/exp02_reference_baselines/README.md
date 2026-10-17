# Experiment 2 — is the score honest?

Adds two reference forecasts (climatology and log-space debiased ECMWF) and
scores everything twice: normally, and with climatology removed
([results.md, Experiment 2](../../results.md#experiment-2--is-the-score-honest)).

- `compute_baselines.py`: scores raw / climatology / debiased ECMWF and rescores
  any run's saved predictions in both spaces.
- `results/aligned/`: the rescoring on correctly aligned days (`baselines.log`
  and one score file per method). The original, one-day-off scores are in
  `../exp03_departure_from_normal/runs/gate_v1_first_try/baseline_*.npz`.

```
python experiments/exp02_reference_baselines/compute_baselines.py --target rain \
    --out experiments/exp02_reference_baselines/results/aligned \
    --rescore experiments/exp05_input_variables/runs/all_inputs_s0 experiments/exp06_yesterday_rain/runs/no_obs_s0
```

**Answer:** under normal scoring, climatology beats raw ECMWF. With
climatology removed, the U-Net (0.32) still beats debiased ECMWF (0.25) on
daily maps. Experiment 9 adds a stronger, weekly bias correction (Dyn++).
