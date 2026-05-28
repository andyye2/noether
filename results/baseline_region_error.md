# Baseline Region Error

- checkpoint: `/home/feng/Projects/ABUPT/outputs/2026-04-25_7d0mv/train/checkpoints/ab_upt_cp=best_model.loss.test.total_model.th`
- split: `test`

|region|point_count|point_fraction|velocity_mse|velocity_mae|relative_l2|mse_over_global|
|---|---|---|---|---|---|---|
|global|409600|1|5.77237|1.34866|0.224039|1|
|wake|44623|0.108943|21.5101|2.5415|0.730777|3.72639|
|non_wake|364977|0.891057|3.84824|1.20282|0.176067|0.666665|
|near_wall_sdf_0.005L|754|0.00184082|45.3335|3.97188|3.30999|7.85353|
|near_wall_sdf_0.01L|5650|0.0137939|32.1965|3.30233|1.28715|5.5777|
|near_wall_sdf_0.02L|20095|0.0490601|23.2297|2.82469|0.877229|4.02429|
