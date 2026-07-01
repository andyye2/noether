# Codex Conversation Summary

This is a compact summary of the relevant decisions and results from the Codex work on adaptive K1024 sampling.

## Main debugging path

1. Early adaptive-vs-official comparisons were misleading because the adaptive branch baseline was not the same path as the original official run.
2. A clean original official K1024 local run was completed for 500 epochs and confirmed to have `Encountered 0 errors`.
3. The adaptive K1024 sweep was corrected to save `best_model.loss.test.volume_velocity_mse_model.th` and to evaluate with 1024 surface and 1024 volume anchors.
4. Three final adaptive runs completed for `u=0.5`, `u=0.7`, and `u=0.9`, with `g=1.0`, `r=10`, seed `0`.
5. `u=0.9` gave the best volume metrics among the adaptive runs.
6. The fixed eval script was updated to use lower-is-better for compact MSE best selection and to split `ADAPTIVE_RUN_NAMES` on `:`, avoiding Slurm comma parsing problems.
7. For posthoc wake/wall/top10 metrics, every run needs a real `best_model.loss.test.volume_velocity_mse_model.th` checkpoint. The old `official_k1024_uniform_seed0` baseline did not have one, so compact eval skipped until a volume-best baseline checkpoint was supplied.
8. The uploaded local baseline checkpoint allowed compact eval to run, but its baseline compact MSE was anomalous. The next clean step is to rerun the uniform volume-best baseline on NCSA using the tracked script.

## Important commands discovered

Use `:` instead of commas inside `ADAPTIVE_RUN_NAMES` when passing through Slurm export variables:

```bash
export ADAPTIVE_RUN_NAMES='official_k1024_ipw_u0p5_g1_r10_seed0:official_k1024_ipw_u0p7_g1_r10_seed0:official_k1024_ipw_u0p9_g1_r10_seed0'
```

Set `POSTHOC_COMPACT_EVAL=1` when you want wake/wall/top10; otherwise `auto` can skip if any checkpoint is missing.

## Current interpretation

If the target is volume only, adaptive sampling is promising: the training-log volume MSE improves by about 27% for `u=0.9` relative to the cluster K1024 uniform baseline. Surface metrics degrade in some comparisons; that was accepted as outside the target for this volume-focused experiment.
