# Experiment 1 — does more training data help?

The same 17-input U-Net trained on 60, 250 and 724 forecasts
([results.md, Experiment 1](../../results.md#experiment-1--does-more-training-data-help)).

| run | training forecasts | trained with |
|---|---|---|
| `runs/n60` | 60 | `train.py --limit_train 60` |
| `runs/n250` | 250 | `train.py --limit_train 250` |
| `runs/n724` | 724 — "the Baseline model" | `train.py` |

Each run folder has its own `results.md`. `compute_raw_baseline.py` scores raw
ECMWF on the same test set:

```
python train.py --target rain --model_path experiments/exp01_training_size/runs/n724
python test.py  --target rain --model_path experiments/exp01_training_size/runs/n724
python experiments/exp01_training_size/compute_raw_baseline.py --target rain --out experiments/exp01_training_size/runs/n724
```

**Answer:** yes. Skill improves steadily with more data, and the 724 model
beats raw ECMWF on all 32 days. It overfits early (best epoch 7).
