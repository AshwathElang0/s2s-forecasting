# India 3D U-Net

Post-processing ECMWF S2S rainfall reforecasts over India with a 3D U-Net,
verified against IMD gridded rainfall. Write-up: [`results.md`](results.md).
Design and port notes: [`plan.md`](plan.md).

## Layout

```
india_3dunet/
  config.py            grid, lead days, splits, paths; config.find_run(name)
  train.py, test.py    the original training / test entry points (Experiment 1)
  train_ablation.py    training with a chosen input set (Experiments 5-7)
  model/               the networks and the metrics
    unet3d.py            3D U-Net + masked MSE / pattern-correlation loss
    gated_unet.py        trust-gate head (Exp 3)
    residual_unet.py     U-Net on top of Dyn++ (Exp 10)
    evaluation.py        daily per-map metrics; evaluation_fixed.py: E_pre fix
    weekly_scores.py     weekly / all-India scores on aligned days (Exp 9, 10)
  preprocess/          data -> model-ready arrays
    ecmwf_features.py, imd_targets.py, regrid.py, build_dataset.py, dataset_loader.py
    alignment.py         ECMWF lead t <-> IMD issue+1+t
    climatology.py       causal observed and model climatologies
    anomaly.py           departure-from-normal reformulation (Exp 3)
    bias_correction.py   Dyn++ (Exp 9, 10)
  scripts/             one-off data jobs: build the cache, regrid IMD, repair files
  experiments/         one folder per experiment: scripts, runs, results
  data_cache/          sample cache + climatologies (not in git, rebuilt by scripts/)
  plots/               domain map
```

Raw data lives in `../data/` (see `../data/data.md`); papers and the reference
code in `../references/`.

## Running things

Run everything from this folder, e.g.

```
python train_ablation.py --set no_obs --seed 0
python experiments/exp09_bias_correction/run_bias_correction.py
bash experiments/exp10_unet_on_bias_correction/run_all.sh
```

Each experiment's README lists its exact commands.
