# 02 Score Sidecar Loading and Alignment

## Intent

训练 dataloader 有 workers，callback 内存状态不适合作为 train sampler 的直接数据源。v1 使用 sidecar 文件承载 per-sample score，train pipeline 每次加载当前 sidecar，并用轻量 fingerprint 校验点序对齐。

## Implementation Target

新增：

- `LoadSamplingScoreSampleProcessor`
- `make_volume_position_fingerprint()` 小工具，可放在同一模块或 `aero_cfd/pipeline/sample_processors/sampling_score.py`

processor 插入位置：

- 仅在 train pipeline 且 `use_volume_score_sampling=True` 时插入。
- 插在 volume score-aware anchor sampler 前。

## Sidecar Location and Schema

每个 sample 一份 `.pt` 文件，文件名采用 index 定位：

```text
score_{index}.pt
```

内容：

```python
{
    "index": int,
    "run_name": str | None,
    "volume_sampling_score": Tensor[N],
    "volume_position_fingerprint": dict[str, object],
}
```

`run_name` 可从 `ShapeNetCarDataset.sample_info()` 取得；它用于诊断和额外校验，但 v1 不把 lookup 改成 run-name manifest，避免扩大实现复杂度。

## Alignment Rule

loader 对当前 sample 中已有的 normalized `volume_position` 重新计算 fingerprint。只有以下条件都满足时，才写入 `sample["volume_sampling_score"]`：

- sidecar `index` 与 sample `index` 一致。
- score 是 tensor。
- score 长度与当前 `volume_position` 第一维一致。
- `volume_position_fingerprint` 与当前 sample 匹配。

不匹配时保持 sample 原样，不注入 score；后续 sampler 会 fallback random。这样能捕捉“shape 仍匹配但点序错位”的危险情况。

## Fingerprint Definition

v1 使用当前 pipeline 已经加载的 normalized `volume_position`，不额外从磁盘加载 raw `volume_points.pt`，避免训练热路径额外 IO。

fingerprint 内容建议包括：

- dtype string。
- shape。
- point count。
- 32 个等间隔采样点的 float64 值。
- 全 tensor 的 float64 `sum` 和 `sum of squares`。

不对每个 `__getitem__` 做 full tensor sha256。仓库中已有 sha256 idiom 可作为写法参考，但这里是训练热路径，轻量 fingerprint 更合适。

## Missing Score Behavior

- sidecar 文件不存在：不报错，不注入 score。
- sidecar 缺 key：不报错，不注入 score。
- score shape 不匹配：不注入 score。
- fingerprint 不匹配：不注入 score，并计入 diagnostic。

## Code Backing

- dataset `index` 来源：`src/noether/data/base/dataset.py`
- ShapeNet run metadata：`src/noether/data/datasets/cfd/shapenet_car/dataset.py`
- dataloader workers：`src/noether/data/container.py`
- existing sha256 idiom：`src/noether/data/datasets/cfd/drivaernet/preprocessing.py`
- sample processor style：`src/noether/data/pipeline/sample_processors/duplicate_keys.py`

## Acceptance Notes

sidecar loader 不改变原始 dataset 文件，不缓存跨样本状态，不依赖 callback 和 dataloader worker 共享 Python 对象。
