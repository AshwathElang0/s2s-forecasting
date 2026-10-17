# Experiment 6 — does yesterday's observed rain help?

The 17-input model with and without yesterday's IMD rain
([results.md, Experiment 6](../../results.md#experiment-6--does-yesterdays-observed-rain-help)).
The "with" runs are `../exp05_input_variables/runs/all_inputs_s*`.

- `runs/no_obs_s{0,1,2}`: 16 inputs, no observed rain. **This is the headline
  model** used in Experiments 7–10 and the review figures. Seed 0 was trained
  here; seeds 1–2 were added for Experiment 7.

```
python train_ablation.py --set no_obs --seed 0
```

**Answer:** no difference on the standard score; a small, unconfirmed gain
with climatology removed. Dropped anyway, because observed IMD rain would not
exist at forecast time.
