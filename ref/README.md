# Adaptive Sampling Reference Pack

This folder is meant to support the first project path: replace hard-coded wake sampling with score/residual-driven adaptive anchor sampling in the original noether AB-UPT ShapeNet car pipeline.

Baseline source: exported from local `ABUPT/noether` commit `5c225a360565ef97442cf3cde2b40d0bf22ca93c`. The new project baseline commit is `93f216f` (`chore: clean original noether baseline`).

## Local inventory

### Papers

| File | Main use for this project | Source |
| --- | --- | --- |
| `papers/wu_2023_residual_adaptive_sampling_RAD_RAR-D.pdf` | Core backing for residual-adaptive distribution sampling. Most directly maps to sampling anchors with probability proportional to error/residual score. | https://arxiv.org/abs/2207.10289 |
| `papers/gao_2023_failure_informed_adaptive_sampling_FI-PINNs.pdf` | Stronger theoretical framing: posterior error/failure indicator, enrich points in high-failure regions. Useful for project motivation. | https://arxiv.org/abs/2210.00279 |
| `papers/mao_2023_residual_gradient_adaptive_sampling.pdf` | Adds residual-gradient/variation information. Useful if pure residual over-focuses on noisy points. | https://arxiv.org/abs/2302.08035 |
| `papers/lu_2021_deepxde_RAR_library.pdf` | Library-level RAR precedent. Useful because DeepXDE has compact code that repeatedly evaluates residual and adds anchors. | https://arxiv.org/abs/1907.04502 |
| `papers/qi_2017_pointnetplusplus_FPS_set_abstraction.pdf` | Coverage-aware sampling and local neighborhoods via FPS/set abstraction. Useful for optional score-plus-coverage selection. | https://arxiv.org/abs/1706.02413 |
| `papers/kashefi_2021_pointnet_cfd_irregular_geometries.pdf` | Point-cloud CFD surrogate precedent on irregular geometries. Supports staying in point space instead of rasterizing fields. | https://arxiv.org/abs/2010.09469 |
| `papers/nvidia_2025_domino_external_aerodynamics.pdf` | External aerodynamics, point-cloud neural operator, SDF/local geometry, surface and volume fields. Useful context close to ShapeNet/DrivAer style data. | https://arxiv.org/abs/2501.13350 |
| `papers/lang_2020_samplenet_differentiable_point_sampling.pdf` | Learnable task-specific point sampler. Not first milestone, but a second-stage idea if fixed residual-score sampling works. | https://arxiv.org/abs/1912.03663 |
| `papers/yan_2020_pointasnl_adaptive_point_sampling.pdf` | Adaptive sampling after FPS with neighbor reweighting/adjustment. Useful for learnable supernode/point selection later. | https://arxiv.org/abs/2003.00492 |

### Code and snippets

| Path | Why it matters |
| --- | --- |
| `code/pinn-sampling-main/` | Official RAD/RAR-D implementation. Look at `src/burgers/RAD.py` and `src/burgers/RAR_D.py`: residual score -> normalized probability -> resample/add anchors. |
| `snippets/pinn_sampling_burgers_RAD.py` | Small standalone entry point for probability-resampling by residual magnitude. |
| `snippets/pinn_sampling_burgers_RAR_D.py` | Small standalone entry point for adding a quota of high-score anchors. |
| `snippets/deepxde_burgers_RAR.py` | Simple RAR loop: evaluate residual on candidates, select high-error point, add anchor, retrain. |
| `code/pointnet2-master/` and `snippets/pointnet2_tf_sampling.py` | Official FPS/probability sampling entry points. Useful for coverage correction after score sampling. |
| `code/PointNetCFD-main/` | Point-cloud CFD supervised surrogate code. Useful for data shape and point-field prediction examples. |
| `code/SampleNet-master/` | Learnable differentiable sampler. Keep as future option, not first implementation. |
| `code/PointASNL-master/` | Adaptive sampling around FPS seeds. Useful for future point/supernode distribution work. |

Zip archives are kept beside the extracted folders for reproducibility.

## Why this is enough to start

The noether ShapeNet AB-UPT path already trains on randomly sampled anchors. The key local facts are:

| noether component | Relevant file | Current behavior |
| --- | --- | --- |
| Anchor sampling | `recipes/aero_cfd/src/aero_cfd/pipeline/sample_processors/anchor_point_sampling.py` | `torch.randperm` selects uniform surface/volume anchors. |
| Pipeline wiring | `recipes/aero_cfd/src/aero_cfd/pipeline/multistage_pipelines/aero_multistage.py` | Builds geometry points, supernodes, then anchor processors. |
| AB-UPT preset | `recipes/aero_cfd/src/aero_cfd/presets/shapenet_car.py` | AB-UPT override uses 256 surface anchors and 256 volume anchors; query supervision is off. |
| Loss | `recipes/aero_cfd/src/aero_cfd/trainers/aerodynamics_cfd.py` | Mean MSE over selected anchors, so under-sampled hard regions have weak gradient contribution. |

This makes anchor distribution the lowest-risk first lever. Query distribution is not useful until query outputs are supervised. Supernodes affect geometry encoding indirectly and should be second-stage.

## Recommended first implementation

Start with an offline score cache plus a score-aware anchor sampler.

1. Train or load a baseline checkpoint.
2. Run dense/candidate inference on each training sample and compute a per-volume-point score:
   `score = mean((pred_velocity - target_velocity) ** 2, dim=-1)`.
3. Store scores beside sample identity, preferably with the same point order as `volume_position`.
4. Replace only volume anchor sampling first:
   `prob = uniform_fraction / N + (1 - uniform_fraction) * normalize((score + eps) ** gamma)`.
5. Sample `num_volume_anchor_points` with `torch.multinomial(prob, replacement=False)`.
6. Keep `uniform_fraction` around `0.2-0.4` at first to preserve global coverage.
7. Evaluate global MSE plus region MSE. If global MSE worsens, add inverse-probability or region-balanced loss weighting.

Good first defaults:

```text
adaptive_score_key: volume_sampling_score
num_volume_anchor_points: 256
uniform_fraction: 0.30
gamma: 1.0
score_eps: 1e-8
replacement: false
fallback: uniform randperm when score is missing
```

## Borrowable logic

RAD/RAR-D gives the direct probability transform:

```text
candidate_score = abs(residual)
prob = normalize(candidate_score ** k / mean(candidate_score ** k) + c)
selected_ids = random_choice(candidate_ids, p=prob)
```

In noether this becomes:

```text
candidate_score = cached per-point velocity error, local gradient score, or uncertainty score
prob = uniform_mix + adaptive_mix * normalize(candidate_score ** gamma)
selected_ids = torch.multinomial(prob, num_volume_anchor_points)
```

DeepXDE RAR gives the training loop concept:

```text
train baseline
evaluate residual/error on many candidate points
add or replace anchors in high-error regions
continue training
```

PointNet++ gives the optional coverage correction:

```text
select a larger score-biased candidate pool
run FPS inside that pool to keep spatial spread
use the FPS result as anchors
```

## First milestone proposal

Implement only these files first:

1. Add `ScoreAwareAnchorPointSamplingSampleProcessor` next to the existing anchor sampler.
2. Add config fields to `AeroCFDPipelineConfig`: `volume_score_sampling`, `volume_score_key`, `volume_score_uniform_fraction`, `volume_score_gamma`.
3. Add a script that writes `volume_sampling_score` into cached samples or a sidecar map.
4. Add an evaluation script/table for global, wake, near-wall SDF-band, and top-score-bin errors.

Avoid a learnable sampler in milestone 1. It is interesting, but it changes training dynamics and adds more moving parts before we prove that score-biased anchors help this model.
