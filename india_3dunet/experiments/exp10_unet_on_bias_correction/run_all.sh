#!/bin/bash
# Train every Experiment 10 run (3 variants x 3 seeds, ~12-15 min each), then score.
# Seed 0 of every variant first, so early results cover all three variants.
# A run with test predictions already saved is skipped, so this can be restarted.
# Usage, from india_3dunet/:  bash experiments/exp10_unet_on_bias_correction/run_all.sh
set -u
HERE=experiments/exp10_unet_on_bias_correction
mkdir -p $HERE/results
for seed in 0 1 2; do
  for variant in plain_aligned residual residual_mm; do
    run=$HERE/runs/${variant}_s${seed}
    if [ -f "$run/test_rain_predictions.npz" ]; then echo "skip $variant s$seed"; continue; fi
    mkdir -p "$run"
    for attempt in 1 2; do
      python $HERE/train_residual.py --variant $variant --seed $seed > "$run/train.log" 2>&1 && break
      echo "$variant s$seed attempt $attempt failed"
    done
    echo "$variant s$seed done $(date +%H:%M)"
  done
done
python $HERE/score_runs.py > $HERE/results/score.log 2>&1
echo "scored $(date +%H:%M)"
