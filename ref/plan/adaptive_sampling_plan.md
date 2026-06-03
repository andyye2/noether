# noether Adaptive Sampling Plan Index

这个目录把 adaptive sampling 计划拆成按步骤编号的文档。旧的单文件计划不再作为实现依据，避免把动态刷新、score 注入、loss 修正和实验验证挤在一起。

当前权威方案采用已核实的最原生路径：query-mode dynamic score refresh、train-only score-aware volume anchors、recipe-local `SamplingWeightedLossTrainer`、轻量 fingerprint sidecar 对齐，以及 fixed-seed refresh anchors。所有新增功能默认关闭，不把实验复杂度扩散进 noether core。

阅读顺序：

1. `00_overview_and_constraints.md`
2. `01_score_aware_volume_anchor_sampler.md`
3. `02_score_sidecar_loading_and_alignment.md`
4. `03_query_residual_score_refresh_callback.md`
5. `04_inverse_probability_weighted_loss.md`
6. `05_train_only_wiring_and_configs.md`
7. `06_experiments_and_metrics.md`
8. `07_tests_and_acceptance.md`

`latter.md` 保留为暂缓项记录，不并入主实施计划。
