# 05 Train-Only Wiring and Configs

## Intent

把 score-aware sampler、score loader、score refresh callback 和 recipe-local weighted loss 接到 ShapeNet AB-UPT 训练路径，但只影响 train split。baseline 和 eval/test 必须保持不变。

## Pipeline Config

在 `AeroCFDPipelineConfig` 增加默认关闭字段：

```python
use_volume_score_sampling: bool = False
volume_score_key: str = "volume_sampling_score"
volume_score_dir: str | None = None
volume_score_uniform_fraction: float = 0.3
volume_score_gamma: float = 1.0
volume_score_eps: float = 1e-8
emit_volume_score_candidates: bool = False
```

`AeroMultistagePipeline.__init__()` 保存这些字段。

## Train Pipeline Wiring

只改 `_get_anchor_point_sampling_sample_processor()` 中的 volume branch：

- `use_volume_score_sampling=False`：继续使用原 `AnchorPointSamplingSampleProcessor`。
- `use_volume_score_sampling=True`：先插入 `LoadSamplingScoreSampleProcessor`，再使用 `ScoreAwareAnchorPointSamplingSampleProcessor`。
- surface branch 继续使用原随机 sampler。
- geometry point/supernode sampling 不变。

当 `use_volume_score_sampling=True` 时，train collator 必须额外收集：

- `volume_anchor_sampling_prob`
- `volume_anchor_sampling_weight`

如果 `volume_score_dir is None`，loader 不注入 score，score-aware sampler fallback random；这样打开开关但未生成 score 时仍可运行。

## Score Refresh Pipeline Wiring

`emit_volume_score_candidates=True` 只用于 `score_refresh_train` pipeline：

- 在 anchor sampler 前复制 full `volume_position` / `volume_velocity` 到 refresh candidate keys。
- refresh collator 收集 `volume_score_position` / `volume_score_velocity`。
- refresh anchor sampler 使用固定 seed，保持评估上下文稳定。
- 不打开 `use_volume_score_sampling`，避免 refresh dataset 本身被 adaptive sampler 影响。

## Train-Only Rule

必须只修改：

```python
config.datasets["train"].pipeline
```

不能修改：

- `config.datasets["test"].pipeline`
- `config.datasets["test_repeat"].pipeline`
- val/eval callback 的 dataset pipeline

score refresh 使用额外 dataset：

```python
extra_datasets["score_refresh_train"]
```

该 dataset 的 pipeline 是 refresh 专用 pipeline，不是 adaptive train pipeline。

## Script Target

新增实验脚本：

- `recipes/aero_cfd/scripts/train_shapenet_score_sampling_ablation.py`

脚本职责：

- 构建 baseline config。
- 为 adaptive run 只 patch train pipeline。
- 添加 `score_refresh_train` dataset。
- 添加 `VolumeResidualScoreRefreshCallback`。
- 使用 `aero_cfd.trainers.SamplingWeightedLossTrainer`。
- 设置 `sample_weight_keys`。

不要修改 `ShapeNetCarPreset` 默认值，避免 clean baseline drift。

## Code Backing

- pipeline config/class：`recipes/aero_cfd/src/aero_cfd/pipeline/multistage_pipelines/aero_multistage.py`
- preset split 构造：`src/noether/core/presets/base.py`
- ShapeNet defaults：`recipes/aero_cfd/src/aero_cfd/presets/shapenet_car.py`
- Python training path：`recipes/aero_cfd/scripts/train_shapenet_car.py`
- extra dataset precedent：`recipes/aero_cfd/src/aero_cfd/presets/shapenet_car.py`

## Acceptance Notes

打开 adaptive 后，eval metric 仍应衡量原始分布或 dense query inference 结果，不能被 score-biased anchor sampler 污染。
