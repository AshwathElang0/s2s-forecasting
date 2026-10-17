# Experiment 3 — predicting the departure from normal

Predict how far rain departs from the normal for that date and place, then add
the normal back. `DESIGN.md` is the full design write-up, including the trust
gate ([results.md, Experiment 3](../../results.md#experiment-3--predicting-the-departure-from-normal)).

| runs | what |
|---|---|
| `runs/departure_s{0,1,2}` | departure from normal, plain head — the confirmed gain |
| `runs/gate_s{0,1,2}` | departure + trust gate, ECMWF rain also kept as an input |
| `runs/gate_v1_first_try` | the first gate run (single seed) |
| `runs/gate_v2_rain_only_via_gate` | gate with ECMWF rain only through the gate; scored at Baseline level |

The comparison Baseline is `../exp05_input_variables/runs/all_inputs_s{0,1,2}`.

```
python experiments/exp03_departure_from_normal/train_departure.py --target rain --plain_head --keep_anchor \
    --seed 1 --model_path experiments/exp03_departure_from_normal/runs/departure_s1
python experiments/exp03_departure_from_normal/test_departure.py --target rain \
    --model_path experiments/exp03_departure_from_normal/runs/departure_s1
python experiments/exp03_departure_from_normal/score_seeds.py      # 3-seed comparison + bootstrap
```

`results/aligned_rescore.log` is `score_seeds.py` on aligned days.

**Answer:** +0.026 climatology-removed correlation over the Baseline (95%
+0.015 to +0.037), every seed beating every Baseline seed. The gate adds
nothing (+0.001).
