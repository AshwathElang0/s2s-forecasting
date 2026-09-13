# CLAUDE.md

This is the `s2s-forecasting` repository — remote:
`github.com/AshwathElang0/s2s-forecasting`.

## Branches

- **`main`** — the codebase (`s2s_data/`, `scripts/`, `pyproject.toml`), owned by
  Ashwath. Data-source loaders (GEFS, CPC, CESM2-LE, teleconnection indices) and the
  regridding utility live here.
- **`research-notes`** — this branch. Planning and literature-review documents: the
  problem statement, the method (`APPROACH.md`), and the literature survey
  (`literature-review.md`). No code changes belong here; open PRs against `main` for
  that.

## What this project is

Post-process ECMWF S2S reforecasts (11-member ensemble, 1.5° resolution, India domain)
into a bias-corrected value per grid cell for weekly precipitation and tmax/tmin,
verified against IMD gridded observations. Lead weeks 1–4. See `literature-review.md`
for the full framing, baselines, and plan, and `APPROACH.md` for the confidence-gating
method this project is building toward.

Read `literature-review.md` before proposing an architecture or metric — it documents
why RMSE alone is an insufficient headline metric here (the shrinkage/damping property
of squared-error post-processing at long lead) and what the field reports instead.
