# Recovery Checklist

Use this checklist after cloning on NCSA or after deleting local files.

## GitHub

- Clone repository `andyye2/noether`.
- Checkout branch `adaptive-volume-anchor-sampling`.
- Confirm these files exist:
  - `slurm/official_k1024_uniform_volume_best_seed0.sbatch`
  - `slurm/official_k1024_adaptive_sweep.sbatch`
  - `slurm/submit_official_k1024_adaptive_sweep_grid.sh`
  - `slurm/evaluate_official_k1024_sweep_fixed.sbatch`
  - `.codex/abupt-adaptive-k1024/README.md`

## NCSA scratch

- Confirm data exists: `/scratch/$USER/data/shapenet_car`.
- Confirm or recreate venv: `/scratch/$USER/ABUPT/.venv`.
- Copy tracked Slurm files to `/scratch/$USER/ABUPT/slurm` if you want to use the historical command paths.

## Runs to regenerate if scratch outputs are missing

1. `sbatch /scratch/$USER/ABUPT/slurm/official_k1024_uniform_volume_best_seed0.sbatch`
2. `bash /scratch/$USER/ABUPT/slurm/submit_official_k1024_adaptive_sweep_grid.sh`
3. `sbatch --export=ALL /scratch/$USER/ABUPT/slurm/evaluate_official_k1024_sweep_fixed.sbatch` after exporting the variables shown in `.codex/abupt-adaptive-k1024/README.md`.

## Evidence of success

- Baseline run log ends with `official_k1024_uniform_volume_best_seed0_ok`.
- Adaptive run logs end with `official_k1024_ipw_u..._ok` and `Encountered 0 errors`.
- Eval log prints `run_posthoc_compact_eval True` and `posthoc compact MSE on volume-MSE-best checkpoints:`.
- Eval log prints best adaptive params; the expected current winner is `u=0.9,g=1.0,r=10`.
