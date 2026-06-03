# 07 Tests and Acceptance

## Unit Tests

新增小而明确的单元测试，优先覆盖行为 contract，不做大规模训练。

### Sampler

覆盖：

- high-score 点在固定 seed 下更容易被选中。
- score 缺失时 shape/key 行为与原 random sampler 一致。
- score shape 不匹配、NaN/Inf、全零时 fallback random。
- 多个 `items` 使用同一组 indices，保证 `volume_position` 和 `volume_velocity` 对齐。
- `keep_queries=True` 时 anchor/query 无重叠，合并后覆盖原点集合。
- `seed + index` deterministic 行为与原 sampler 一致。
- `volume_anchor_sampling_prob` 和 `volume_anchor_sampling_weight` shape 正确。
- fallback 时 weight 全 1。

### Sidecar Alignment

覆盖：

- fingerprint 和 shape 匹配时注入 `volume_sampling_score`。
- shape 匹配但 fingerprint 不匹配时不注入 score。
- sidecar 缺失时不报错。
- score key 缺失时不报错。
- `run_name` 存在但不匹配时记录 warning，并不注入 score。

### Sampling Weighted Trainer

覆盖：

- `sample_weight_keys={}` 时完全等价当前 `WeightedLossTrainer`。
- 权重全 1 时 weighted loss 等价原 mean MSE。
- 非均匀权重只影响配置了 sample weight key 的 field。
- `_split_batch` 能把 collated `volume_anchor_sampling_weight` 放进 targets，同时不把它传给 model forward。
- 开启 weighted loss 但 batch 缺权重 key 时给出清晰配置错误。

## Integration Tests

最小集成测试：

- `use_volume_score_sampling=False` 时 pipeline 仍实例化原 volume sampler。
- `use_volume_score_sampling=True` 时只替换 volume sampler，surface sampler 不变。
- train pipeline collates `volume_anchor_sampling_weight`。
- `ShapeNetCarPreset` AB-UPT 默认配置不变。
- train config patch 不影响 `test` / `test_repeat` pipeline。
- mock callback 写 sidecar 后，train pipeline 能读取并采样。

## Callback Behavior Tests

用 mock model / tiny batch 验证：

- callback 按 query chunks 输出 full-length score。
- fixed refresh anchor seed 让同一 sample 的 anchor 上下文稳定。
- sidecar score 顺序与原 full `volume_score_position` 一致。
- 新 refresh 覆盖旧 sidecar。
- query chunk size 不改变 score 顺序。

## Acceptance Criteria

实现完成后必须满足：

- 默认训练命令和默认 preset 与 clean baseline 行为一致。
- adaptive 打开但无 score sidecar 时可运行，并 fallback random。
- fingerprint mismatch 不会静默使用错位 score。
- eval/test metrics 不使用 score-biased sampler。
- adaptive-weighted-v1 可通过 smoke test 完成至少一次 score refresh、一次 score-aware sampling、一次 weighted loss update。

## Suggested Test Locations

- sampler/sidecar：`tests/unit/aero_cfd/pipeline/sample_processors/`
- recipe trainer loss：`tests/unit/aero_cfd/trainers/`
- callback/wiring：`tests/unit/aero_cfd/callbacks/` 或轻量 integration test。
