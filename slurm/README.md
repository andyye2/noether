# ABUPT K1024 Slurm Scripts

These are the current scripts for the adaptive volume-anchor sampling K1024 run. They are tracked here so the NCSA `/scratch/$USER/ABUPT/slurm` directory can be recreated from GitHub.

## Scripts

- `official_k1024_uniform_volume_best_seed0.sbatch`: pure uniform K1024 baseline, saving `loss/test/volume_velocity_mse` best checkpoint.
- `official_k1024_adaptive_sweep.sbatch`: one adaptive K1024 run, configured by `U`, `G`, and `R`.
- `submit_official_k1024_adaptive_sweep_grid.sh`: submits the final grid `u=0.5,0.7,0.9`, `g=1.0`, `r=10`.
- `evaluate_official_k1024_sweep_fixed.sbatch`: compares baseline and adaptive volume-best checkpoints, including Global/Wake/Wall/Top10 compact MSE.
- `official_k1024_uniform_seed0.sbatch`: older uniform K1024 baseline script that does not guarantee a volume-MSE-best checkpoint. Keep for reference, but use the `volume_best` script for compact eval.

## Copy to historical NCSA path

From the repo root on NCSA:

```bash
mkdir -p /scratch/$USER/ABUPT/slurm
cp -v slurm/*.sbatch slurm/*.sh /scratch/$USER/ABUPT/slurm/
```

See `.codex/abupt-adaptive-k1024/README.md` for the full recovery and run sequence.
