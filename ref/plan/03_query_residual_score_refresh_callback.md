# 03 Query Residual Score Refresh Callback

## Intent

用 noether 原生 callback 机制实现真正 dynamic adaptive sampling：训练过程中周期性用当前模型重算 volume per-point residual score，并写入 sidecar，供 train sampler 下一轮读取。

## Implementation Target

新增：

- `recipes/aero_cfd/src/aero_cfd/callbacks/volume_residual_score_refresh.py`
- `VolumeResidualScoreRefreshCallbackConfig`
- `VolumeResidualScoreRefreshCallback`

callback 继承 `PeriodicDataIteratorCallback`，使用 `every_n_epochs` / `every_n_updates` / `every_n_samples` 控制刷新频率。

## Dataset and Pipeline

使用独立 dataset key：

```text
score_refresh_train
```

不要复用已经 adaptive sampling 的 `train` pipeline。score refresh pipeline 需要提供：

- `geometry_position`
- `geometry_supernode_idx`
- `geometry_batch_idx`
- fixed-seed training-sized `surface_anchor_position`
- fixed-seed training-sized `volume_anchor_position`
- full candidate `volume_score_position`
- full candidate `volume_score_velocity`
- `index`

其中 training-sized anchors 使用原随机 anchor sampler 生成，但 seed 固定，保证同一个样本跨 refresh 的 anchor 上下文稳定。full candidate keys 通过 refresh 专用 emit-candidates 机制在 anchor sampler 前复制出来，不被 anchor sampler 改写或重排。

## Refresh Candidate Keys

普通 AB-UPT anchor pipeline 的 collator 不会带 full `volume_position/volume_velocity`，并且 anchor target rename 会覆盖原 full target。因此 refresh pipeline 需要一个专用配置，例如：

```python
emit_volume_score_candidates: bool = False
```

当仅在 `score_refresh_train` pipeline 中打开时：

- 在 volume anchor sampler 前复制 `volume_position -> volume_score_position`。
- 在 volume anchor sampler 前复制 `volume_velocity -> volume_score_velocity`。
- 将 `volume_score_position` 和 `volume_score_velocity` 加入 refresh collator items。

该机制只服务 score refresh，不进入默认 train/eval pipeline。

## Query-Mode Residual

打分时不能把完整点云切成 `volume_anchor_position` chunks。正确路径是：

1. 固定 `surface_anchor_position` 和 `volume_anchor_position`。
2. 将 `volume_score_position` 按 `query_chunk_size` 分块，作为 `query_volume_position` 输入。
3. 每个 chunk forward 后取 `query_volume_velocity`。
4. 与同顺序 `volume_score_velocity` 计算 residual。
5. 拼回原始 full volume 顺序。

score：

```text
volume_sampling_score = mean((query_volume_velocity - volume_score_velocity) ** 2, dim=-1)
```

使用 query-mode 的原因：

- `QueryInferenceCallback` 已证明 AB-UPT decoder 对 `[anchors, queries]` 做 self-attention。
- full point cloud chunk-as-anchor 会让每个点只 condition 在同 chunk 邻居上，并且 anchor 数量离训练分布。
- 固定 256 anchors + query chunks 更接近 noether 的 dense inference 语义，残差场更可比。

## Refresh Frequency

配置化，不硬编码。

默认建议：

- ShapeNet 500 epochs：`every_n_epochs=5`，约 100 次 refresh。
- 小规模 smoke test：允许 `every_n_epochs=1`。
- 大训练也可改为 `every_n_updates`，保持约 50-100 次 refresh。

这比每 100 epochs 刷新一次更接近 RAD/RAR-D 的迭代重采样精神。

## Sidecar Write

每次 refresh 覆盖旧 sidecar：

- 写 score。
- 写当前 `volume_score_position` fingerprint。
- 写 `index` 和可选 `run_name`。

若某个 sample 打分失败，不写半成品。v1 建议保留旧 sidecar 并记录 warning，避免训练突然退化为全随机。

## Code Backing

- query inference 语义：`recipes/aero_cfd/src/aero_cfd/callbacks/query_inference.py`
- periodic iterator：`src/noether/core/callbacks/periodic.py`
- model query key：`src/noether/modeling/models/aerodynamics.py`
- extra dataset 先例：`recipes/aero_cfd/src/aero_cfd/presets/shapenet_car.py`
- RAD dynamic resampling：`ref/code/pinn-sampling-main/src/burgers/RAD.py`
- RAR-D dynamic add anchors：`ref/code/pinn-sampling-main/src/burgers/RAR_D.py`

## Acceptance Notes

该 callback 只负责刷新 score，不直接改 train dataset，不直接采样 anchor，不计算 loss。
