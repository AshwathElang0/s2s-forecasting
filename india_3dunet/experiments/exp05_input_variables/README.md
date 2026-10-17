# Experiment 5 — do the extra input variables help?

The reference paper's test: add channel groups one at a time
([results.md, Experiment 5](../../results.md#experiment-5--do-the-extra-input-variables-help)).

| run | inputs | channels |
|---|---|---|
| `runs/rain_only_s0` | rain mean, spread, yesterday's rain, calendar | 5 |
| `runs/humidity_s0` | + 4 humidity levels | 9 |
| `runs/winds_s0` | + 4 wind fields | 13 |
| `runs/all_inputs_s{0,1,2}` | + pressure, heights, SST (seeds 1–2 added for Exp 3) | 17 |

```
python train_ablation.py --set rain_only        # or humidity / winds / all_inputs, --seed N
```

**Answer:** all four are the same within noise. The 12 atmospheric channels
add nothing measurable.
