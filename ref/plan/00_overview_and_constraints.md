# 00 Overview and Constraints

## Goal

在 noether ShapeNet AB-UPT 路径上实现第一版最小侵入 adaptive sampling：训练期间周期性用当前模型计算 volume per-point residual score，再用该 score 改变 train volume anchor 采样分布，并用 inverse-probability style weighting 修正监督回归 loss 的分布偏置。

默认行为必须完全等价 clean baseline。所有新功能默认关闭。

## Scope

- 只作用于 ShapeNet AB-UPT 的 train volume anchor sampling。
- 不改模型结构、decoder、query 训练语义、surface anchor sampling、geometry point/supernode sampling。
- 不改原 `AnchorPointSamplingSampleProcessor`；新增并列实现。
- 不把 score 写回原始 dataset；score 通过 sidecar 文件注入。
- 不把 recipe 实验字段扩散进 noether core trainer/schema。
- eval/test/test_repeat 保持原始均匀或 dense query inference，不使用 score-biased sampler。

## Non-goals

- v1 不引入 PDE residual；score 使用监督残差。
- v1 不把一次性 baseline checkpoint score 称为 adaptive sampling。
- v1 不修改通用 `noether.training.trainers.WeightedLossTrainer`。
- v1 不修改老的 `aero_cfd.trainers.AerodynamicsCFDTrainer`；优先支持当前 Python preset 使用的 weighted-loss 路径。
- `top-score-bin MSE` 不作为主判据，只作为诊断采样是否覆盖 hard points。

## Verified Code Facts

- ShapeNet AB-UPT 默认 `num_surface_anchor_points=256`、`num_volume_anchor_points=256`、query 数为 0。
- volume anchor sampler 直接从 `volume_position` / `volume_velocity` 采样。
- `AeroABUPT.forward()` 会收集 `surface_anchor_position`、`volume_anchor_position` 和 `query_volume_position`。
- `QueryInferenceCallback` 固定 anchors、分块 queries，并注释说明 anchor outputs 会因 decoder self-attention over `[anchors, queries]` 略有变化。
- `WeightedLossTrainer` 当前对每个 field 直接调用 `F.mse_loss(..., mean)`，adaptive sampling 会改变监督目标。
- `BaseTrainer` 文档明确把 override `loss_compute` 作为自定义 loss 的原生扩展点。
- recipe 已有自己的 trainer 目录和 aero trainer；新增 recipe-local trainer 符合既有结构。
- `PeriodicDataIteratorCallback` 可按 epoch/update/sample 周期性遍历 dataset，适合做 score refresh。
- `extra_datasets` 已被 `test_repeat` 使用，作为 `score_refresh_train` 的注册方式是原生路径。

## Method Positioning

RAD/RAR-D 的核心是周期性：

1. 用当前模型在候选点上计算 residual/error。
2. 按 residual 构造概率分布。
3. 替换或添加训练点。
4. 继续训练，再重复。

因此主方案必须是 dynamic score refresh。static score cache 只能作为 warm-start 或 ablation。

## Success Criteria

- baseline 配置、旧 checkpoint、旧训练命令不受影响。
- adaptive 打开后只影响 train volume anchors。
- score refresh 用 query-mode residual，避免 full volume anchor chunk 的上下文碎片化和离训练分布问题。
- score-refresh anchors 使用固定 seed，使 score 变化主要来自模型变化，而不是评估上下文抖动。
- weighted adaptive 作为主实验；unweighted adaptive 只作为偏置诊断。
- 主指标是 global / wake / non-wake / near-wall，top-score-bin 仅辅助解释。
