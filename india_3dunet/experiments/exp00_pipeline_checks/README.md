# Experiment 0 — pipeline checks

Not an experiment: the first end-to-end runs that proved the cache build,
training and test steps work.

- `runs/smoke/`: 5-epoch run on a handful of samples.
- `runs/first_full_attempt/`: the first full-size attempt (only `stats.json`
  and its log survived).
- `logs/build_full.log`: the full sample-cache build (`scripts/build_all_samples.py`).
