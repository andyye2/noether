# ABUPT Adaptive K1024 Handoff

This handoff records the current runnable state for the ShapeNet car AB-UPT adaptive volume-anchor sampling work.

## Canonical GitHub source

- Repository: https://github.com/andyye2/noether
- Branch for adaptive sampling: `adaptive-volume-anchor-sampling`
- NCSA project checkout used by the Slurm scripts: `/scratch/$USER/ABUPT/noether-adaptive`
- Final Slurm scripts are tracked in this branch under `slurm/`.

The default `main` branch does not contain all adaptive volume-sampling code. For this project, clone or checkout `adaptive-volume-anchor-sampling`.

## What is in Git

- Adaptive sampler implementation:
  - `recipes/aero_cfd/src/aero_cfd/pipeline/sample_processors/score_aware_anchor_point_sampling.py`
  - `recipes/aero_cfd/src/aero_cfd/pipeline/sample_processors/sampling_score.py`
  - `recipes/aero_cfd/src/aero_cfd/callbacks/volume_residual_score_refresh.py`
  - `recipes/aero_cfd/src/aero_cfd/trainers/sampling_weighted_loss.py`
- Unit tests: `tests/unit/aero_cfd/test_adaptive_sampling.py`
- Final Slurm entrypoints: `slurm/`
- This Codex handoff tree: `.codex/abupt-adaptive-k1024/`

Large data and model checkpoints are not stored in GitHub. The ShapeNet data must exist at `/scratch/$USER/data/shapenet_car`, and trained checkpoints must either be regenerated from the scripts or already exist on NCSA scratch.

## NCSA recovery from GitHub

```bash
mkdir -p /scratch/$USER/ABUPT
cd /scratch/$USER/ABUPT

git clone -b adaptive-volume-anchor-sampling https://github.com/andyye2/noether.git noether-adaptive
cd noether-adaptive

module purge || true
module load anaconda3/2024.10 || module load anaconda3

# If the shared venv is missing, recreate it for the Slurm scripts.
python -m pip install --user -U uv
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/ABUPT/.venv
uv sync

mkdir -p /scratch/$USER/ABUPT/slurm
cp -v slurm/*.sbatch slurm/*.sh /scratch/$USER/ABUPT/slurm/
```

Confirm data and environment:

```bash
ls -ld /scratch/$USER/data/shapenet_car
/scratch/$USER/ABUPT/.venv/bin/python --version
```

## Current final run sequence

Run the pure uniform K1024 baseline that saves a volume-MSE-best checkpoint:

```bash
sbatch /scratch/$USER/ABUPT/slurm/official_k1024_uniform_volume_best_seed0.sbatch
```

Run the final adaptive sweep grid. The tracked default grid is only:

```text
u=0.5, g=1.0, r=10
u=0.7, g=1.0, r=10
u=0.9, g=1.0, r=10
```

Submit it:

```bash
bash /scratch/$USER/ABUPT/slurm/submit_official_k1024_adaptive_sweep_grid.sh
```

After all four training jobs finish, run compact posthoc eval:

```bash
export POSTHOC_COMPACT_EVAL=1
export BASELINE_RUN_NAME=official_k1024_uniform_volume_best_seed0
export ADAPTIVE_RUN_NAMES='official_k1024_ipw_u0p5_g1_r10_seed0:official_k1024_ipw_u0p7_g1_r10_seed0:official_k1024_ipw_u0p9_g1_r10_seed0'
export EVAL_SUFFIX="volbest_baseline_allu_$(date +%Y%m%d_%H%M)"

sbatch --export=ALL /scratch/$USER/ABUPT/slurm/evaluate_official_k1024_sweep_fixed.sbatch
```

Use `POSTHOC_COMPACT_EVAL=1` so missing checkpoints fail loudly instead of silently skipping wake/wall/top10 metrics.

## Important caveat

An uploaded local volume-best baseline checkpoint produced an anomalous posthoc compact baseline Global MSE around `5.00785`, while the same run's training-log volume MSE was around `0.037611`. Do not use the 99% compact improvement printed against that anomalous baseline as a final scientific claim. Regenerate the baseline on NCSA with `official_k1024_uniform_volume_best_seed0.sbatch`, then rerun compact eval.
