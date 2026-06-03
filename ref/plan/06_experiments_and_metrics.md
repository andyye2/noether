# 06 Experiments and Metrics

## Experiment Matrix

第一轮只做能回答机制问题和偏置问题的最小矩阵：

1. `baseline`: 原随机 volume anchors。
2. `adaptive-weighted-v1`: dynamic score refresh + score-aware sampler + recipe-local weighted loss。
3. `adaptive-unweighted`: dynamic score refresh + score-aware sampler，但不加 IPW，仅诊断目标偏置。
4. `static-score`: 一次性 baseline score，只作为 warm-start/ablation，不命名为 adaptive。

## Default Hyperparameters

主实验默认：

```text
num_volume_anchor_points = 256
uniform_fraction = 0.3
gamma = 1.0
eps = 1e-8
refresh = every 5 epochs
query_chunk_size = 10000
score_refresh_anchor_seed = fixed
```

ShapeNet 默认 500 epochs 时，every 5 epochs 约 100 次 refresh，贴近 RAD/RAR-D 的周期性重采样。

## Primary Metrics

主判据必须来自不依赖 score 自身定义的区域或全局指标：

- global volume velocity MSE。
- wake region MSE。
- non-wake MSE。
- near-wall SDF bands：`0.005L`、`0.01L`、`0.02L`。

如果 wake region 需要第一版区域定义，先使用已有 `ref/README.md` 记录的物理分区思路；不要把 baseline residual top bins 当主判据。

## Diagnostic Metrics

诊断指标用于解释机制，不用于单独宣称成功：

- sampled score mean / quantile。
- selected anchor score vs full candidate score。
- top-score-bin MSE。
- sidecar missing count。
- fingerprint mismatch count。
- sampler fallback count。
- effective sample weight mean/max。
- refresh score mean/max/quantiles。

## Evaluation Order

结论顺序：

1. 确认 score refresh 正常产生非退化 score。
2. 确认 sampler 实际提高 high-score 点被选中概率。
3. 看 independent region metrics 是否改善。
4. 同时监控 global MSE，不能等局部有效后才看 global。
5. 用 weighted vs unweighted 判断目标偏置是否影响全局表现。

## Code and Paper Backing

- RAD/RAR-D dynamic sampling：`ref/code/pinn-sampling-main/src/burgers/RAD.py`、`RAR_D.py`
- DeepXDE RAR 概念：`ref/snippets/deepxde_burgers_RAR.py`
- Point cloud CFD supervised context：`ref/code/PointNetCFD-main/`
- aerodynamic volume/surface/SDF evaluation context：`ref/papers/nvidia_2025_domino_external_aerodynamics.pdf`

## Acceptance Notes

如果 adaptive-unweighted 的 wake/near-wall 更好但 global 变差，不能算主方案成功；需要以 adaptive-weighted-v1 的 independent metrics 和 global tradeoff 为主要判断。
