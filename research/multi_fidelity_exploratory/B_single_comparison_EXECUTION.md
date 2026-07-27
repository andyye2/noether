# 唯一一对一对照执行手册（A = 现有 P-FT vs B = 源匹配几何渲染 P-FT）

命名空间：exploratory。**不修改、不覆盖**
`/scratch/andyye2/ABUPT/outputs/multi_fidelity/cf974078…`、
`/scratch/andyye2/ABUPT/multi_fidelity_artifacts/cf974078…`、
`/scratch/andyye2/ABUPT/multi_fidelity_stats/cf974078…` 下的任何内容。
全程只提交**一个** B 训练作业。

本手册的所有远端路径都来自 2026-07-27 对 NCSA 的只读核验，不是假设。核验结论见 §2。

## 0. 预注册冻结（在看到任何 B 结果之前，本文件即冻结声明）

- 数据单元：task=`common`，N=`100`，replicate=`0`（subset seed 1103，model seed 7103），
  frame=`shapenet`，budget=`compute_matched`（40,000 updates），DataLoader seed=model seed，
  训练 pipeline seed=None，val 评估 seed=4242。
- **A**：现有 P-FT checkpoint（训练实现 `cf974078…`，复用，不重训）。
  run_id `mf-common-finetune-r0-n100-m9d5e7c8f12-s7103-shapenet-compute_matched`。
- **B**：同一单元 + `--position-scale 10000 --supernode-radius 1.0`，其余与 A **逐字节一致**
  （同一 manifest 文件、同一 statistics 文件、同一 protocol、同一 source checkpoint、同一 seeds）。
  两个常数由审计报告 §2 的实测导出（图平均度 5.3 ≈ 源 5.8；cap32 0.001 ≈ 源 0.000；
  域内最大归一化坐标 9,971 < 10,000 无混叠），不是调参结果。
- **主 endpoint**：`best_model.loss.val.total`（A、B 同规则）；`latest`（final-raw）同时报告。
- 评估 split：只用 `val`。禁止 test、禁止 grid search、禁止追加 replicate、禁止第二个候选配置。
- 成功判据与归因边界见审计报告 §4.4 / §4.5。

## 1. 本地离线检查（无联网）

```bash
cd /home/feng/Projects/ABUPT/multi_fidelity
export UV_PROJECT_ENVIRONMENT=/home/feng/Projects/ABUPT/noether/.venv
export PYTHONPATH="$PWD:$PWD/src:$PWD/recipes/aero_cfd/src"
export PYTHONDONTWRITEBYTECODE=1
UV=/home/feng/.local/bin/uv

$UV run --project "$PWD" --no-sync ruff check --no-cache <改动的四个文件>
$UV run --project "$PWD" --no-sync ruff format --check --no-cache <同上>
$UV run --project "$PWD" --no-sync pytest -q -p no:cacheprovider \
  --basetemp=/tmp/mf-exploratory-pytest \
  tests/unit/recipes/aero_cfd/ research/multi_fidelity/tools/
```

`build_eval_config` 设计上要求 worktree 干净，因此 `test_frozen_eval_config.py`
只有在 clean commit 之后才能通过。**先提交 clean commit，再在干净 worktree 上复跑全套。**
以下 `$NEWC` 指该 commit。

## 2. NCSA 实际结构（2026-07-27 只读核验，覆盖此前的错误假设）

| 用途 | 实际路径 | 备注 |
|---|---|---|
| 代码仓库 | `/scratch/andyye2/ABUPT/multi_fidelity` | 分支 `multi-fidelity`，核验时 HEAD=`88a9d09e`、clean。**不是** `…/noether` |
| Python 环境 | `/scratch/andyye2/ABUPT/shapenet-divu0/5-14-restored/.venv` | `uv run --no-sync` |
| 数据 | `/scratch/andyye2/data/drivaerml_subsampled_10x` | |
| 源 checkpoint | `/scratch/andyye2/ABUPT/outputs/2026-04-25_7d0mv/train/checkpoints/ab_upt_cp=latest_model.th` | sha256 `261a46b7…`，size 28147342 |
| `--source-output-path` | `/scratch/andyye2/ABUPT/outputs` | |
| 作业脚本 + 日志 | `/scratch/andyye2/ABUPT/slurm/multi_fidelity/` | 已有 `drivaerml_transfer_array.sbatch`、`drivaerml_statistics_array.sbatch`。**作业脚本不在 repo 内** |
| artifacts（按 commit 分目录） | `/scratch/andyye2/ABUPT/multi_fidelity_artifacts/<commit>/{commands,dry_runs,reports,staging,metrics}` | 代码上传的既有约定是 `staging/<name>_<shortsha>.bundle` |
| statistics（按 commit 分目录） | `/scratch/andyye2/ABUPT/multi_fidelity_stats/<commit>/seed<subset_seed>/n<N>_<task>_<frame>.json` | |
| 训练输出 | `/scratch/andyye2/ABUPT/outputs/multi_fidelity/<commit>/<run_id>/` | A 在 `cf974078…` 下 |

日志命名由 `#SBATCH --output=/scratch/andyye2/ABUPT/slurm/multi_fidelity/%x-%A_%a.out`
配合提交时的 `-J <name>` 决定，历史上是 `mf-p0-*`、`mf-p1-*`。

**节点排除**：`/scratch/andyye2/ABUPT/slurm/*.sbatch` 中共 28 处使用
`#SBATCH --exclude=ccc0390`（不是 ccc0389）。`sinfo` 显示 `ccc0389` 与 `ccc0390`
都存在于 `IllinoisComputes-GPU`。本手册排除 **`ccc0389,ccc0390`** 两者：
沿用既有约定，同时覆盖用户点名的节点；多排除一个节点只减少候选，不改变作业语义。

## 3. 代码 staging（`git bundle` 直传，不 push GitHub）

```bash
# 本地
git -C /home/feng/Projects/ABUPT/multi_fidelity bundle create \
  /tmp/geometry_rendering_${NEWC:0:7}.bundle multi-fidelity
# 上传到既有 staging 约定位置
#   /scratch/andyye2/ABUPT/multi_fidelity_artifacts/$NEWC/staging/geometry_rendering_<shortsha>.bundle
```

远端（不执行 `git pull` / `uv sync` / 联网）：

```bash
export REPO=/scratch/andyye2/ABUPT/multi_fidelity
git -C "$REPO" bundle verify <bundle>
git -C "$REPO" fetch <bundle> multi-fidelity
git -C "$REPO" switch multi-fidelity && git -C "$REPO" merge --ff-only <fetched>
git -C "$REPO" rev-parse HEAD                                  # == $NEWC
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
```

`REPO` 推进前必须为 clean 且处于 `multi-fidelity` 分支；只允许 fast-forward。

## 4. NCSA 变量（exploratory 专用）

```bash
set -euo pipefail
export REPO=/scratch/andyye2/ABUPT/multi_fidelity
export ENV_ROOT=/scratch/andyye2/ABUPT/shapenet-divu0/5-14-restored/.venv
export DATA_ROOT=/scratch/andyye2/data/drivaerml_subsampled_10x
export SOURCE_ROOT=/scratch/andyye2/ABUPT/outputs
export PROTOCOL="$REPO/research/multi_fidelity/experiment_protocol.yaml"
export PYTHONPATH="$REPO:$REPO/src:$REPO/recipes/aero_cfd/src"

export NEWC="$(git -C "$REPO" rev-parse HEAD)"
export OLDC=cf974078ea633f1fb6ff2178de7fa56ac60c89d5     # A，只读引用

export EXP_ART=/scratch/andyye2/ABUPT/multi_fidelity_artifacts/$NEWC
export EXP_CMDS=$EXP_ART/commands
export EXP_DRYRUN=$EXP_ART/dry_runs
export EXP_REPORTS=$EXP_ART/reports
export EXP_METRICS=$EXP_ART/metrics
export EXP_STAGING=$EXP_ART/staging
export TRAIN_B_ROOT=/scratch/andyye2/ABUPT/outputs/multi_fidelity_exploratory/$NEWC/n100-r0-single-comparison
export SLURM_ROOT=/scratch/andyye2/ABUPT/slurm/multi_fidelity

mkdir -p "$EXP_CMDS" "$EXP_DRYRUN" "$EXP_REPORTS" "$EXP_METRICS" "$EXP_STAGING" "$TRAIN_B_ROOT"
```

## 5. 输入复用（**不重算 manifest / stats**）

B 直接复用 A 的冻结 artifact，这比重新物化更强：输入是同一个文件，不可能漂移。

```bash
export MANIFEST=/scratch/andyye2/ABUPT/multi_fidelity_artifacts/$OLDC/manifests/drivaerml_nested_seed1103.json
export STATS=/scratch/andyye2/ABUPT/multi_fidelity_stats/$OLDC/seed1103/n100_common_shapenet.json

# 与 A 的 sidecar 逐项核对（数值取自 A 的 training_provenance.json）
sha256sum "$MANIFEST"    # raw   == 99da18fbdf1b7daf0a802fd9223f81124138740fbda560e2b6a6447b5bb5d3f3
sha256sum "$STATS"       #       == 38a3ef71deaf5659d39b6decb3665b72b6e160e43b739f7f7c2e90f8211b2c01
sha256sum "$PROTOCOL"    #       == c9721eac22f5519a4713bab13f3d17dd0622a82d0487c36589be0bc2a9e14ac4
```

三个 sha256 中任何一个不符 → STOP。
`position_scale` 是 preset 常数，不进入 stats artifact，因此 stats 复用不产生语义冲突。
manifest 内记录的 `implementation_git_commit` 仍是 `$OLDC`，这是**有意**的：
A 和 B 共用同一份数据划分，划分本身没有被本次改动影响。

## 6. B 的 dry-run 与 graph preflight

```bash
B_CMD="env PYTHONPATH=$PYTHONPATH uv run --project $REPO --no-sync python \
  $REPO/recipes/aero_cfd/scripts/run_drivaerml_transfer_strict.py \
  --protocol $PROTOCOL \
  --dataset-root $DATA_ROOT \
  --manifest $MANIFEST \
  --target-statistics $STATS \
  --output-path $TRAIN_B_ROOT \
  --source-output-path $SOURCE_ROOT \
  --task common --strategy finetune --sample-size 100 --budget compute_matched \
  --coordinate-frame shapenet --position-scale 10000 --supernode-radius 1.0 \
  --replicate 0 --model-seed 7103 --eval-point-seed 4242"

DRY="$EXP_DRYRUN/B_n100_r0_ps10000_sr1.yaml"
bash -lc "$B_CMD --dry-run" > "$DRY"
grep -q 'test_dataset_in_training_config: false' "$DRY"
grep -q 'scale: 10000' "$DRY"       # position normalizer
grep -q 'radius: 1.0' "$DRY"        # supernode_pooling_config
```

除 `--position-scale` / `--supernode-radius` / `--output-path` 外，`$B_CMD` 与
`multi_fidelity_artifacts/$OLDC/commands/P1_commands.txt` 第 6 行（A）逐 token 相同。

```bash
srun --partition=IllinoisComputes --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32GB --time=02:00:00 \
  bash -lc "env PYTHONPATH='$PYTHONPATH' uv run --project '$REPO' --no-sync python \
    '$REPO/research/multi_fidelity/tools/audit_radius_graph.py' '$DRY' \
    --output '$EXP_REPORTS/B_ps10000_sr1_radius_graph.json' \
    --point-seeds 42 43 44 --design-count 3 --device cpu --max-zero-fraction 0.001"
```

预期（已用真实 run_1 几何独立实测）：`zero_fraction = 0.0000` ≤ 0.001 通过；
`cap_fraction ≈ 0.001`（P0 基线为 **1.0000**）；平均度 ≈ 5.3（源侧 5.8）。
gate 失败或 `cap_fraction` 仍接近 1 → STOP，不提交训练。

## 7. 提交唯一一个 B 训练作业

作业脚本：`$SLURM_ROOT/drivaerml_transfer_exploratory_array.sbatch`（**新文件，不覆盖既有两个**），
格式照抄 `drivaerml_transfer_array.sbatch`，仅改动：

1. `COMMIT` 固定为 `$NEWC`；
2. `OUTPUT_ROOT` 固定为 `$TRAIN_B_ROOT`（守卫仍校验命令行含 `--output-path $OUTPUT_ROOT`）；
3. 新增 `#SBATCH --exclude=ccc0389,ccc0390`；
4. 新增守卫：命令行必须含 `--strategy finetune`、`--position-scale 10000`、
   `--supernode-radius 1.0`，且**必须不含** `outputs/multi_fidelity/cf974078`。

```bash
printf '%s\n' "$B_CMD" > "$EXP_CMDS/B_train.txt"
test "$(wc -l < "$EXP_CMDS/B_train.txt")" -eq 1

sbatch --test-only -J mf-p2b --array=1-1 \
  "$SLURM_ROOT/drivaerml_transfer_exploratory_array.sbatch" "$EXP_CMDS/B_train.txt"
sbatch -J mf-p2b --array=1-1 \
  "$SLURM_ROOT/drivaerml_transfer_exploratory_array.sbatch" "$EXP_CMDS/B_train.txt"
```

日志落在 `$SLURM_ROOT/mf-p2b-<jobid>_1.{out,err}`。
run_id 预期 `mf-common-finetune-r0-n100-m9d5e7c8f12-s7103-shapenet-compute_matched-ps10000-sr1`。

完成判据：`COMPLETED`、无 NaN/OOM、
`$TRAIN_B_ROOT/<run_id>/train/training_provenance.json` 存在且
`position_scale: 10000.0`、`supernode_radius: 1.0`、
`manifest_raw_sha256` / `target_statistics_sha256` 与 A 相同。

## 8. 冻结 val 评估（3 个 eval 作业，均 `--evaluation-split val`，无 test flag）

| # | 对象 | checkpoint tag | 几何参数 | 输出目录 |
|---|---|---|---|---|
| 1 | B | `latest` | `--position-scale 10000 --supernode-radius 1.0` | `$EXP_METRICS/val/common/P-FT-ps10000-sr1/compute_matched/latest/` |
| 2 | B | `best_model.loss.val.total` | 同上 | `.../best-model-loss-val-total/` |
| 3 | A | `best_model.loss.val.total` | 默认（1000 / 9） | `$EXP_METRICS/val/common/P-FT/compute_matched/best-model-loss-val-total/` |

A 的 `latest`（final-raw）val 指标已存在于
`multi_fidelity_artifacts/$OLDC/reports/P1_final_raw_val.csv`，不重跑。
每个 `.audit.json` 须核对：`evaluation_split=val`、`evaluation_design_count=34`、
`target_checkpoint_sha256 == sidecar_target_checkpoint_sha256`、
`test_release_confirmed=false`，以及 `position_scale` / `supernode_radius` 与训练一致。

## 9. 汇总对照报告（写入 `$EXP_REPORTS`）

对 A、B 各自报告，并给出 34 个 val design 的配对差分布：

- pressure / velocity relative L2；equal-field geometric aggregate；MAE
- best-val 所在 update；final-raw 与 best-val 的差
- train/val 学习曲线；前几千 update 的 grad-norm / update-norm 统计
- 参数量与可训练参数量（应同为 7,008,004）
- wall time、MaxRSS、GPU memory（sacct / 日志）
- 同单元 scratch（S, n100, r0）指标只读引用，并按审计报告 §4.5 标注这是**跨渲染对照**

## 10. STOP 条件

- `$MANIFEST` / `$STATS` / `$PROTOCOL` 的 sha256 与 A 的 sidecar 不符；
- `REPO` 推进前不 clean、不在 `multi-fidelity` 分支，或非 fast-forward；
- graph preflight 的 zero gate 失败，或 `cap_fraction` 仍接近 1；
- B 的 dry-run resolved config 中 position normalizer scale ≠ 10000，或 supernode radius ≠ 1.0；
- 任何步骤将要写入 `outputs/multi_fidelity/cf974078…`、
  `multi_fidelity_artifacts/cf974078…`、`multi_fidelity_stats/cf974078…`；
- 任何步骤将要覆盖 `$SLURM_ROOT` 下已存在的 `.sbatch`。
