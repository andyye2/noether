# 01 Score-Aware Volume Anchor Sampler

## Intent

新增一个并列 sample processor，用当前 sample 中的 `volume_sampling_score` 改变 volume anchor 的采样分布。原随机 sampler 保持不动，fallback 行为必须与原 `torch.randperm` contract 一致。

## Implementation Target

新增：

- `recipes/aero_cfd/src/aero_cfd/pipeline/sample_processors/score_aware_anchor_point_sampling.py`
- class `ScoreAwareAnchorPointSamplingSampleProcessor`

保持与 `AnchorPointSamplingSampleProcessor` 相同的主要构造参数：

- `items`
- `num_points`
- `to_prefix_and_postfix`
- `to_prefix_midfix_postfix`
- `keep_queries`
- `seed`

新增参数：

- `score_key: str = "volume_sampling_score"`
- `uniform_fraction: float = 0.3`
- `gamma: float = 1.0`
- `eps: float = 1e-8`
- `prob_key: str = "volume_anchor_sampling_prob"`
- `weight_key: str = "volume_anchor_sampling_weight"`

## Sampling Rule

对长度为 `N` 的 score：

```text
adaptive = clamp(score, min=0) ** gamma + eps
adaptive = adaptive / adaptive.sum()
uniform = full_like(adaptive, 1 / N)
prob = uniform_fraction * uniform + (1 - uniform_fraction) * adaptive
perm = torch.multinomial(prob, num_points, replacement=False, generator=generator)
weight = 1 / (N * prob[perm])
```

`uniform_fraction=0.3` 时，单点采样概率有下界，`weight` 上界约为 `1 / 0.3`。v1 使用 sampler probability 的 inverse-probability style correction；严格 without-replacement inclusion probability 后续再做，不进入第一版复杂度。

## Output Contract

对所有 `items` 使用同一组 `perm`，保证 `volume_position`、`volume_velocity`、features 完全对齐。

除原有 `volume_anchor_position`、`volume_anchor_velocity` 等 anchor outputs 外，新增：

- `volume_anchor_sampling_prob`: shape `(num_points,)`
- `volume_anchor_sampling_weight`: shape `(num_points,)`

当 `keep_queries=True` 时，query 仍为未被选中的剩余点，保持原 sampler contract。

## Fallback

以下情况退回原 `torch.randperm` 采样，并输出 prob/weight 为全 1：

- `score_key` 缺失。
- score 非 tensor。
- score shape 与 `volume_position` 第一维不匹配。
- score 中有 NaN/Inf。
- score clamp 后全零或 sum 不可用。
- `num_points >= N`。
- 上游 loader 因 fingerprint mismatch 未注入 score。

fallback 必须使用与原 sampler 一致的 deterministic seed：`sample["index"] + seed`。

## Code Backing

- 原 sampler：`recipes/aero_cfd/src/aero_cfd/pipeline/sample_processors/anchor_point_sampling.py`
- pipeline volume branch：`recipes/aero_cfd/src/aero_cfd/pipeline/multistage_pipelines/aero_multistage.py`
- RAD 概率思想：`ref/code/pinn-sampling-main/src/burgers/RAD.py`
- RAR-D 概率思想：`ref/code/pinn-sampling-main/src/burgers/RAR_D.py`

## Acceptance Notes

这一阶段只负责按 score 改 anchor 分布和产出 prob/weight，不负责生成 score，也不负责 loss weighting。
