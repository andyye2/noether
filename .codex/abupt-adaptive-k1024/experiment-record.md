# Experiment Record

## Final adaptive K1024 sweep

All adaptive runs below used:

- `num_surface_anchor_points=1024`
- `num_volume_anchor_points=1024`
- `g=1.0`
- `r=10`
- seed `0`
- checkpoint selected by `loss/test/volume_velocity_mse`

Training logs found `Encountered 0 errors` for the three final adaptive runs.

| run | job | volume MSE min | volume L2 min | compact Global | compact Wake | compact Non-wake | compact Wall005 | compact Wall01 | compact Wall02 | compact Top10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `official_k1024_ipw_u0p5_g1_r10_seed0` | 9244316 | 0.032163 | 0.014308 | 0.04445265 | 0.19674043 | 0.02575862 | 0.30632559 | 0.19265960 | 0.13930300 | 0.23416604 |
| `official_k1024_ipw_u0p7_g1_r10_seed0` | 9244317 | 0.029419 | 0.013646 | 0.04597291 | 0.23224704 | 0.02310690 | 0.40014452 | 0.19145898 | 0.12053208 | 0.24453529 |
| `official_k1024_ipw_u0p9_g1_r10_seed0` | 9244320 | 0.027501 | 0.013570 | 0.03855058 | 0.18818382 | 0.02018241 | 0.26317658 | 0.15049879 | 0.10299121 | 0.18297334 |

Among the three adaptive runs, `u=0.9,g=1,r=10` is best on every compact MSE region listed above.

## Training-log comparison against baselines

Against the cluster uniform baseline `official_k1024_uniform_seed0/9208936`, training-log volume MSE improved as follows:

| adaptive run | adaptive volume MSE | baseline volume MSE | relative change |
| --- | ---: | ---: | ---: |
| u0.5 | 0.032163 | 0.037818 | -14.95% |
| u0.7 | 0.029419 | 0.037818 | -22.21% |
| u0.9 | 0.027501 | 0.037818 | -27.28% |

Against the later local volume-best baseline training log, the baseline volume MSE was about `0.037611`; u0.9 was still better by about `26.88%`.

## Compact baseline warning

The compact posthoc run using the uploaded local volume-best baseline printed:

- baseline compact Global MSE: `5.007847634871421`
- baseline training-log volume MSE: `0.037611`

This mismatch is too large to treat as valid. Use it only as evidence that the eval pipeline ran and that the adaptive compact rows were produced. For a final baseline-vs-adaptive compact comparison, rerun the baseline directly on NCSA with `slurm/official_k1024_uniform_volume_best_seed0.sbatch` and rerun `slurm/evaluate_official_k1024_sweep_fixed.sbatch`.

## What not to keep using

Do not use the early exploratory grids as the default final sweep. The tracked `submit_official_k1024_adaptive_sweep_grid.sh` has been narrowed to the final three `u` values: `0.5`, `0.7`, and `0.9`.
