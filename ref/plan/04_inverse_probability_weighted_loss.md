# 04 Inverse-Probability Weighted Loss

## Intent

对监督回归而言，按 residual 过采样 hard points 会改变训练目标。v1 必须把 inverse-probability style weighting 放进主路径，不能只作为后续补丁。

## Recommended Route

采用 Route C+：

- 新增 recipe-local trainer：`aero_cfd.trainers.SamplingWeightedLossTrainer`。
- 新增 recipe-local config：`SamplingWeightedLossTrainerConfig`，继承 noether 的 `WeightedLossTrainerConfig`。
- 不修改 `src/noether/core/schemas/trainers.py`。
- 不修改 `src/noether/training/trainers/weighted_loss.py`。
- 不重写 `train_step`。

选择原因：

- `BaseTrainer` 文档明确 `loss_compute` 是自定义 loss 的原生扩展点。
- recipe 已有 `aero_cfd/trainers/`，把实验 trainer 放在 recipe 内符合现有结构。
- 改 core trainer/schema 会把 ShapeNet adaptive 实验字段扩散到共享组件，blast radius 更大。
- 全仓没有实际业务 trainer 重写 `train_step` 的先例；复制 forward + `TrainerResult` 逻辑不划算。

## Config

`SamplingWeightedLossTrainerConfig` 增加：

```python
sample_weight_keys: dict[str, str] = {}
```

ShapeNet adaptive 主实验配置：

```python
trainer_kind = "aero_cfd.trainers.SamplingWeightedLossTrainer"
sample_weight_keys = {
    "volume_velocity": "volume_anchor_sampling_weight",
}
```

默认空时行为等价 `WeightedLossTrainer`。

## Batch Delivery

sampling weight 必须先进入 batch：

- score-aware sampler 输出 `volume_anchor_sampling_weight`。
- train pipeline 把 `volume_anchor_sampling_weight` 加入 `default_collator_items`。

不要把 weight 伪装成普通 target property。`SamplingWeightedLossTrainer` 轻量 override `_split_batch`：

- 正常 split `forward_properties` 和 `target_properties`。
- 额外把 `sample_weight_keys.values()` 对应 batch keys 放入 `targets`。
- 避免 base `_split_batch` 因 batch 多出 weight key 而 warning。

这样 weight 只在 recipe trainer 内被解释，不污染 model forward，也不污染 core trainer。

## Weighted Loss Rule

无 sample weight 时保持当前逻辑：

```python
loss = F.mse_loss(target, pred) * field_weight
```

有 sample weight 时：

```python
elementwise = (pred - target) ** 2
point_loss = elementwise.mean(dim=-1)
weighted = point_loss * sample_weight
loss = weighted.mean() * field_weight
```

`sample_weight` shape 应为 `(B, N)` 或可 squeeze 成 `(B, N)`。权重全 1 时结果应与原 mean MSE 等价。

## Why This Is Required

PINN collocation residual 的目标处处为 0，过采样 residual 点主要改变优化重点；但 noether ShapeNet 是监督回归，训练 loss 是 anchor 上的 empirical mean MSE。改变 anchor 分布但不加权，会把目标从原始 volume distribution 改成 score-biased distribution。

因此主结果应是：

```text
adaptive-weighted-v1
```

而不是 unweighted adaptive。

## Ablation

保留 unweighted adaptive，但只用于诊断：

- hard region 是否更快下降。
- global MSE 是否因目标偏置受损。
- inverse-probability weighting 是否缓解 global regression drift。

## Code Backing

- current trainer：`src/noether/training/trainers/weighted_loss.py`
- trainer extension point：`src/noether/training/trainers/base.py`
- recipe trainer precedent：`recipes/aero_cfd/src/aero_cfd/trainers/aerodynamics_cfd.py`
- PointNet++ per-point supervised weights：`ref/code/pointnet2-master/models/pointnet2_sem_seg.py`

## Acceptance Notes

v1 不修改 noether core trainer，不修改老 `AerodynamicsCFDTrainer`。如果后续 YAML 老路径也要支持，再单独扩展。
