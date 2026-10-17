# Experiment 7 — does the day-of-year input help? (3 seeds)

The 16-input headline model (`../exp06_yesterday_rain/runs/no_obs_s*`) against
the paper-style 14 ECMWF-only inputs
([results.md, Experiment 7](../../results.md#experiment-7--does-the-day-of-year-input-help-3-seeds-each)).

- `runs/ecmwf_only_s{0,1,2}`: 14 inputs, no calendar.
- `score_seeds.py`: 3-seed comparison + bootstrap; `results/aligned_rescore.log`
  is its output on aligned days.
- `logs/seeds_batch.log`: timing log of the seed batch.

```
python train_ablation.py --set ecmwf_only --seed 0
python experiments/exp07_calendar_input/score_seeds.py
```

**Answer:** the calendar adds nothing measurable. This experiment also
measured retraining noise: seeds of one setup differ by up to 0.016 on the
climatology-removed correlation.
