#!/bin/bash
# Experiment 11: residual_mm with past-years-only Dyn++ factors on the training
# dates, seeds 0-2 (~20 min each; only seed 0 has been run), then score against Exp 10's residual_mm.
# A run with test predictions already saved is skipped, so this can be restarted.
# Usage, from india_3dunet/:  bash experiments/exp11_past_only_factors/run_all.sh
set -u
HERE=experiments/exp11_past_only_factors
python $HERE/prepare_factors.py || exit 1
for seed in 0 1 2; do
  run=$HERE/runs/residual_mm_past_s${seed}
  if [ -f "$run/test_rain_predictions.npz" ]; then echo "skip s$seed"; continue; fi
  mkdir -p "$run"
  for attempt in 1 2; do
    python -u experiments/exp10_unet_on_bias_correction/train_residual.py --variant residual_mm --seed $seed \
      --dynpp_fit past --out_dir $HERE/runs > "$run/train.log" 2>&1 && break
    echo "s$seed attempt $attempt failed"
  done
  echo "residual_mm_past s$seed done $(date +%H:%M)"
done
python $HERE/score_runs.py > $HERE/results/score.log 2>&1
echo "scored $(date +%H:%M)"
