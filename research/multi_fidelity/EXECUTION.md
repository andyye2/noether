# DrivAerML 迁移学习 staged 执行手册

本文把 `experiment_protocol.yaml` 中的预注册设计落实为可复制执行的 NCSA 流程。所有路径和命令均以 2026-07-16 的已审计状态为准；每一阶段只有在上一阶段的检查通过后才可释放。

## 0. 当前状态与下载边界

当前没有任何“下载失败但实验必需”的文件，也没有未解决的必需下载。此前 `uv` 访问 PyPI 的网络失败已由现有虚拟环境配合 `uv run --no-sync` 绕过；现有检查未发现缺失依赖，不需要 `uv sync` 或 `pip install`。

另外两项已知但不阻塞：

- WSL 环境没有 `pdftotext`，文献 PDF 已使用 bundled runtime 完成抽取和审计，不需要为此下载软件。
- GeoPT 只保留为可选的后备路线，不属于当前 S/P-FT 主实验。若未来确实要使用 GeoPT，必须先暂停，由用户自行下载并确认文件已就位后再继续。

截至本文写入时：

- 尚未向 NCSA staging 本 worktree 的代码；
- 尚未向 NCSA staging ShapeNet-Car 源 checkpoint；
- 尚未上传文件，也未提交任何统计、训练或评估作业。

硬性联网边界：

1. 以下流程不得运行 `uv sync`、`uv lock`、`pip install`、`git pull`、`git clone https://...`，也不得自动从 GitHub、PyPI 或其他站点下载。
2. 代码 staging 使用本地 `git bundle` 上传到 NCSA，不依赖 GitHub 下载。
3. 远端依赖只做 `uv run --project ... --no-sync` 检查。若检查失败，立即 **STOP**，记录完整错误并告知用户；不得尝试联网修复。

## 1. 冻结输入与目录

ShapeNet-Car confirmatory primary checkpoint：

```text
本地：
/home/feng/Projects/ABUPT/outputs/2026-04-25_7d0mv/train/checkpoints/ab_upt_cp=latest_model.th

NCSA 目标：
/scratch/andyye2/ABUPT/outputs/2026-04-25_7d0mv/train/checkpoints/ab_upt_cp=latest_model.th

size:
28147342 bytes

SHA256:
261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38
```

生产实验必须从一个 clean commit 运行。当前本地 `research/multi_fidelity/evidence/manifests/` 下的 dirty manifests 只用于本地证据审计，不能用于生产。生产 manifests 必须在 NCSA 完成代码 staging 后，由 clean commit 重新生成到 repo 外；这样 manifest 可记录同一个 commit 且 `implementation_git_dirty=false`，又不会因为生成文件把 repo 变脏。

登录 NCSA 后统一定义：

```bash
set -euo pipefail

export REPO=/scratch/andyye2/ABUPT/noether
export DATA_ROOT=/scratch/andyye2/data/drivaerml_subsampled_10x
export SOURCE_ROOT=/scratch/andyye2/ABUPT/outputs
export SOURCE_CKPT="$SOURCE_ROOT/2026-04-25_7d0mv/train/checkpoints/ab_upt_cp=latest_model.th"
export TRAIN_BASE=/scratch/andyye2/ABUPT/outputs_transfer
export EVAL_BASE=/scratch/andyye2/ABUPT/outputs_transfer_eval
export STATS_BASE=/scratch/andyye2/ABUPT/multi_fidelity_stats
export PROTOCOL="$REPO/research/multi_fidelity/experiment_protocol.yaml"
export PYTHONPATH="$REPO:$REPO/src:$REPO/recipes/aero_cfd/src"

export COMMIT="$(git -C "$REPO" rev-parse HEAD)"
export TRAIN_ROOT="$TRAIN_BASE/$COMMIT"
export EVAL_ROOT="$EVAL_BASE/$COMMIT"
export STATS_ROOT="$STATS_BASE/$COMMIT"
export ARTIFACT_ROOT="/scratch/andyye2/ABUPT/multi_fidelity_artifacts/$COMMIT"
export MANIFEST_ROOT="$ARTIFACT_ROOT/manifests"
export COMMAND_ROOT="$ARTIFACT_ROOT/commands"
export DRYRUN_ROOT="$ARTIFACT_ROOT/dry_runs"
export METRICS_ROOT="$ARTIFACT_ROOT/metrics"
export LOG_ROOT="$ARTIFACT_ROOT/logs"
export REPORT_ROOT="$ARTIFACT_ROOT/reports"

mkdir -p "$MANIFEST_ROOT" "$COMMAND_ROOT" "$DRYRUN_ROOT" \
  "$METRICS_ROOT" "$LOG_ROOT" "$REPORT_ROOT"
```

所有运行产物均写到 repo 外。任何阶段开始前都执行：

```bash
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" || {
  echo "STOP: production repository is dirty" >&2
  git -C "$REPO" status --short
  exit 1
}
test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"
```

## 2. 本地无联网检查

这些命令只使用已有 `/home/feng/Projects/ABUPT/noether/.venv`，不会同步或下载依赖：

```bash
set -euo pipefail
cd /home/feng/Projects/ABUPT/multi_fidelity

export UV_PROJECT_ENVIRONMENT=/home/feng/Projects/ABUPT/noether/.venv
export PYTHONPATH="$PWD:$PWD/src:$PWD/recipes/aero_cfd/src"
export PYTHONDONTWRITEBYTECODE=1

/home/feng/.local/bin/uv run --project "$PWD" --no-sync python -c \
  "from pathlib import Path; import yaml; p=Path('research/multi_fidelity/experiment_protocol.yaml'); assert isinstance(yaml.safe_load(p.read_text()), dict); print('protocol_ok')"

bash -n recipes/aero_cfd/jobs/drivaerml_statistics_array_ncsa.job
bash -n recipes/aero_cfd/jobs/drivaerml_transfer_array_ncsa.job

/home/feng/.local/bin/uv run --project "$PWD" --no-sync ruff check --no-cache \
  recipes/aero_cfd/scripts/run_drivaerml_transfer_strict.py \
  recipes/aero_cfd/scripts/eval_drivaerml_transfer_frozen.py \
  research/multi_fidelity/tools/materialize_study_manifests.py \
  research/multi_fidelity/tools/compute_subset_statistics.py \
  research/multi_fidelity/tools/generate_statistics_commands.py \
  research/multi_fidelity/tools/generate_training_commands.py \
  research/multi_fidelity/tools/generate_evaluation_commands.py \
  research/multi_fidelity/tools/audit_radius_graph.py \
  research/multi_fidelity/tools/merge_metric_csvs.py \
  research/multi_fidelity/tools/assess_validation_gate.py \
  research/multi_fidelity/tools/analyze_transfer_results.py

/home/feng/.local/bin/uv run --project "$PWD" --no-sync pytest -q -p no:cacheprovider \
  --basetemp=/tmp/multi-fidelity-execution-pytest \
  tests/unit/recipes/aero_cfd/test_transfer_subset_statistics.py \
  tests/unit/recipes/aero_cfd/test_strict_transfer_runner.py \
  tests/unit/recipes/aero_cfd/test_transfer_lr_modifiers.py \
  tests/unit/recipes/aero_cfd/test_frozen_eval_config.py \
  research/multi_fidelity/tools/test_generate_statistics_commands.py \
  research/multi_fidelity/tools/test_generate_evaluation_commands.py \
  research/multi_fidelity/tools/test_audit_radius_graph.py \
  research/multi_fidelity/tools/test_merge_metric_csvs.py \
  research/multi_fidelity/tools/test_assess_validation_gate.py \
  research/multi_fidelity/tools/test_analyze_transfer_results.py
```

任一命令失败都先 STOP。不要把失败解释为需要联网安装；先把错误交给用户判断。

## 3. 代码 staging：clean commit 的 bundle 上传

这一步是“本地上传到 NCSA”，不是从 GitHub 下载。本文只给命令，尚未执行。

用户已于 2026-07-18 明确批准 P0/P1 真实训练准备；当前 `experiment_protocol.yaml` 已冻结为：

```yaml
status: frozen_before_first_target_job
```

冻结状态只解除协议门，不跳过其余完整性门。必须重新执行第 2 节全部离线检查，审阅最终 diff，再把实现与冻结后的 YAML 一起提交为 clean commit。后续 production manifests 必须绑定这份冻结 YAML 的 SHA256。提交后在 WSL 中先验证冻结状态和 clean worktree，再创建 bundle：

```bash
set -euo pipefail
export LOCAL_REPO=/home/feng/Projects/ABUPT/multi_fidelity
export PROTOCOL_LOCAL="$LOCAL_REPO/research/multi_fidelity/experiment_protocol.yaml"
export UV_PROJECT_ENVIRONMENT=/home/feng/Projects/ABUPT/noether/.venv

PROTOCOL_STATUS="$(
  /home/feng/.local/bin/uv run --project "$LOCAL_REPO" --no-sync python -c \
  "from pathlib import Path; import yaml; print(yaml.safe_load(Path('$PROTOCOL_LOCAL').read_text(encoding='utf-8'))['status'])"
)"
test "$PROTOCOL_STATUS" = frozen_before_first_target_job || {
  echo "STOP: protocol status is '$PROTOCOL_STATUS'; explicit user-approved freeze is required" >&2
  exit 1
}

git -C "$LOCAL_REPO" status --short --branch
test -z "$(git -C "$LOCAL_REPO" status --porcelain --untracked-files=all)" || {
  echo "STOP: review and commit the local implementation first" >&2
  exit 1
}

export COMMIT="$(git -C "$LOCAL_REPO" rev-parse HEAD)"
export LOCAL_STAGING=/home/feng/Projects/ABUPT/staging
export BUNDLE="$LOCAL_STAGING/noether-$COMMIT.bundle"
mkdir -p "$LOCAL_STAGING"

git -C "$LOCAL_REPO" bundle create "$BUNDLE" HEAD
git -C "$LOCAL_REPO" bundle verify "$BUNDLE"
sha256sum "$BUNDLE"
printf 'COMMIT=%s\nBUNDLE=%s\n' "$COMMIT" "$BUNDLE"
```

在 Windows PowerShell 中上传 bundle：

```powershell
$Commit = (wsl.exe -d UbuntuD -- bash -lc "git -C /home/feng/Projects/ABUPT/multi_fidelity rev-parse HEAD").Trim()
$Bundle = "\\wsl.localhost\UbuntuD\home\feng\Projects\ABUPT\staging\noether-$Commit.bundle"
$HostName = "andyye2@cc-login.campuscluster.illinois.edu"
$RemoteBundle = "/scratch/andyye2/ABUPT/staging/noether-$Commit.bundle"

ssh.exe $HostName "mkdir -p /scratch/andyye2/ABUPT/staging"
scp.exe $Bundle "${HostName}:${RemoteBundle}"
ssh.exe $HostName "git -C /scratch/andyye2/ABUPT/noether bundle verify '$RemoteBundle'"
```

然后在 NCSA 上把 bundle fetch 到已有 repo。先确认远端 worktree clean；不执行 `git pull`：

```bash
set -euo pipefail
export REPO=/scratch/andyye2/ABUPT/noether
export COMMIT=<粘贴本地输出的40位commit>
export BUNDLE="/scratch/andyye2/ABUPT/staging/noether-$COMMIT.bundle"

test -f "$BUNDLE"
git -C "$REPO" bundle verify "$BUNDLE"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" || {
  echo "STOP: existing NCSA repository is dirty" >&2
  git -C "$REPO" status --short
  exit 1
}

git -C "$REPO" fetch "$BUNDLE" HEAD
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = "$COMMIT"
git -C "$REPO" switch --detach "$COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
```

## 4. 源 checkpoint staging 与校验

本文只给出用户可执行的上传命令，尚未上传。先在 Windows PowerShell 创建远端目录并上传：

```powershell
$HostName = "andyye2@cc-login.campuscluster.illinois.edu"
$LocalCheckpoint = "\\wsl.localhost\UbuntuD\home\feng\Projects\ABUPT\outputs\2026-04-25_7d0mv\train\checkpoints\ab_upt_cp=latest_model.th"
$RemoteDir = "/scratch/andyye2/ABUPT/outputs/2026-04-25_7d0mv/train/checkpoints"

ssh.exe $HostName "mkdir -p '$RemoteDir'"
scp.exe $LocalCheckpoint "${HostName}:${RemoteDir}/ab_upt_cp=latest_model.th"
```

在 NCSA 上严格核对 size 和 SHA256：

```bash
set -euo pipefail
export SOURCE_CKPT=/scratch/andyye2/ABUPT/outputs/2026-04-25_7d0mv/train/checkpoints/ab_upt_cp=latest_model.th
export EXPECTED_SIZE=28147342
export EXPECTED_SHA=261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38

test -f "$SOURCE_CKPT"
test "$(stat -c %s "$SOURCE_CKPT")" -eq "$EXPECTED_SIZE"
test "$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')" = "$EXPECTED_SHA"
printf 'checkpoint_ok size=%s sha256=%s\n' \
  "$(stat -c %s "$SOURCE_CKPT")" \
  "$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')"
```

任何不一致都 STOP，不得开始 finetune。

## 5. NCSA 依赖只检查

代码和 checkpoint 均 staging 完成后，在 NCSA 登录节点只验证已有环境。此处绝不安装：

```bash
set -euo pipefail
export REPO=/scratch/andyye2/ABUPT/noether
export PYTHONPATH="$REPO:$REPO/src:$REPO/recipes/aero_cfd/src"

command -v uv
uv --version

test -x "$REPO/.venv/bin/python" || {
  echo "STOP: existing remote venv is missing" >&2
  exit 1
}

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python -c \
  "import torch, yaml; from aero_cfd.presets.drivaerml_transfer import DrivAerMLTransferCommonPreset; from noether.training.runners import HydraRunner; print('dependency_check_ok', torch.__version__, torch.cuda.is_available(), DrivAerMLTransferCommonPreset.__name__, HydraRunner.__name__)"

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/recipes/aero_cfd/scripts/run_drivaerml_transfer_strict.py" --help >/dev/null

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/recipes/aero_cfd/scripts/eval_drivaerml_transfer_frozen.py" --help >/dev/null
```

若这一节失败，立即 STOP 并把错误告知用户。禁止运行 `uv sync`、`pip install` 或任何下载命令。

## 6. 在 repo 外重新生成生产 manifests

重新载入第 1 节变量，然后确认 repo clean。manifest 目录必须是 `$ARTIFACT_ROOT/manifests`，不能是 repo 内的本地证据目录：

```bash
set -euo pipefail
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/materialize_study_manifests.py" \
  --protocol "$PROTOCOL" \
  --output-dir "$MANIFEST_ROOT"

test "$(find "$MANIFEST_ROOT" -maxdepth 1 -type f -name 'drivaerml_nested_seed*.json' | wc -l)" -eq 8
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
```

校验八份 manifest 都绑定当前 clean commit：

```bash
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python - "$MANIFEST_ROOT" "$COMMIT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
paths = sorted(root.glob("drivaerml_nested_seed*.json"))
assert len(paths) == 8, len(paths)
for path in paths:
    payload = json.loads(path.read_text())
    assert payload["implementation_git_commit"] == commit, path
    assert payload["implementation_git_dirty"] is False, path
    assert len(payload["protocol_sha256"]) == 64, path
    assert len(payload["manifest_sha256"]) == 64, path
print("production_manifests_ok", len(paths), commit)
PY
```

## 7. P0/P1 train-only statistics：4 个单遍 CPU jobs，产出 16 个 artifacts

P0 与 P1 合并去重后恰好有 16 个统计 artifact；按 seed/task/frame 分成 4 个命令，每个命令单遍扫描最大 N 并在嵌套前缀处快照。只在生产 manifests 校验通过后生成命令：

```bash
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/generate_statistics_commands.py" \
  --protocol "$PROTOCOL" \
  --phases P0 P1 \
  --repo-root "$REPO" \
  --manifest-root "$MANIFEST_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --stats-root "$STATS_ROOT" \
  --output "$COMMAND_ROOT/statistics_P0_P1.txt"

test "$(wc -l < "$COMMAND_ROOT/statistics_P0_P1.txt")" -eq 4
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
```

Slurm 模板使用相对日志名 `slurm-%x_%A_%a.out`。因此日志目录必须预先存在，并且必须从该目录提交：

```bash
mkdir -p "$LOG_ROOT/statistics_P0_P1"
cd "$LOG_ROOT/statistics_P0_P1"
sbatch --array=1-4%4 \
  "$REPO/recipes/aero_cfd/jobs/drivaerml_statistics_array_ncsa.job" \
  "$COMMAND_ROOT/statistics_P0_P1.txt"
```

记录返回的 job ID。等全部 array tasks 为 `COMPLETED` 后，再检查：

```bash
test "$(find "$STATS_ROOT" -type f -name '*.json' | wc -l)" -eq 16
test "$(find "$STATS_ROOT" -type f -name '*.flat.yaml' | wc -l)" -eq 16
```

如果任何 task 失败，先检查对应 `$LOG_ROOT/statistics_P0_P1/slurm-*.out`，不得释放训练。

## 8. 生成 P0/P1 训练命令并做 P0 preflight

统计完整后生成训练命令；不要使用 `--allow-missing-stats`：

```bash
for PHASE in P0 P1; do
  env PYTHONPATH="$PYTHONPATH" \
    uv run --project "$REPO" --no-sync python \
    "$REPO/research/multi_fidelity/tools/generate_training_commands.py" \
    --protocol "$PROTOCOL" \
    --phase "$PHASE" \
    --repo-root "$REPO" \
    --manifest-root "$MANIFEST_ROOT" \
    --dataset-root "$DATA_ROOT" \
    --output-path "$TRAIN_ROOT" \
    --stats-root "$STATS_ROOT" \
    --source-output-path "$SOURCE_ROOT" \
    --output "$COMMAND_ROOT/${PHASE}_commands.txt"
done

test "$(wc -l < "$COMMAND_ROOT/P0_commands.txt")" -eq 6
test "$(wc -l < "$COMMAND_ROOT/P1_commands.txt")" -eq 30
```

从生成的 P0 命令文件中分别抽取 scratch 和 finetune，追加 `--dry-run`，把 resolved YAML 写到 repo 外：

```bash
SCRATCH_CMD="$(grep -m1 -- '--strategy scratch' "$COMMAND_ROOT/P0_commands.txt")"
FINETUNE_CMD="$(grep -m1 -- '--strategy finetune' "$COMMAND_ROOT/P0_commands.txt")"
test -n "$SCRATCH_CMD"
test -n "$FINETUNE_CMD"

bash -lc "$SCRATCH_CMD --dry-run" > "$DRYRUN_ROOT/P0_common_scratch.yaml"
bash -lc "$FINETUNE_CMD --dry-run" > "$DRYRUN_ROOT/P0_common_finetune.yaml"

grep -q 'test_dataset_in_training_config: false' "$DRYRUN_ROOT/P0_common_scratch.yaml"
grep -q 'test_dataset_in_training_config: false' "$DRYRUN_ROOT/P0_common_finetune.yaml"
```

随后用真实 resolved config 和真实数据 pipeline 审计 radius graph。以下命令使用 CPU allocation；两个审计均须以 exit code 0 结束：

```bash
mkdir -p "$LOG_ROOT/graph_audit"
cd "$LOG_ROOT/graph_audit"

srun --partition=IllinoisComputes --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32GB --time=02:00:00 \
  bash -lc "env PYTHONPATH='$PYTHONPATH' uv run --project '$REPO' --no-sync python '$REPO/research/multi_fidelity/tools/audit_radius_graph.py' '$DRYRUN_ROOT/P0_common_scratch.yaml' --output '$REPORT_ROOT/P0_common_scratch_radius_graph.json' --point-seeds 42 43 44 --design-count 3 --device cpu --max-zero-fraction 0.001"

srun --partition=IllinoisComputes --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32GB --time=02:00:00 \
  bash -lc "env PYTHONPATH='$PYTHONPATH' uv run --project '$REPO' --no-sync python '$REPO/research/multi_fidelity/tools/audit_radius_graph.py' '$DRYRUN_ROOT/P0_common_finetune.yaml' --output '$REPORT_ROOT/P0_common_finetune_radius_graph.json' --point-seeds 42 43 44 --design-count 3 --device cpu --max-zero-fraction 0.001"
```

审计报告必须保留 `zero_fraction`、`cap_fraction` 和 gate 结果。任一 gate 失败都 STOP。

## 9. P0 六个 jobs，然后 P1 三十个 jobs

先只释放 P0：

```bash
mkdir -p "$LOG_ROOT/P0"
cd "$LOG_ROOT/P0"
sbatch --array=1-6%2 \
  "$REPO/recipes/aero_cfd/jobs/drivaerml_transfer_array_ncsa.job" \
  "$COMMAND_ROOT/P0_commands.txt"
```

确认六个 tasks 全部 `COMPLETED`、无 NaN/OOM、每个成功训练目录都有 `train/training_provenance.json`。P0 有任何异常就 STOP。

P0 通过后才释放 P1：

```bash
mkdir -p "$LOG_ROOT/P1"
cd "$LOG_ROOT/P1"
sbatch --array=1-30%3 \
  "$REPO/recipes/aero_cfd/jobs/drivaerml_transfer_array_ncsa.job" \
  "$COMMAND_ROOT/P1_commands.txt"
```

P1 完成后必须有 30 个 `budget=compute_matched` 的 primary training sidecars。P0 的 6 个 smoke sidecars可以共存；后续评估会用 budget/frame 过滤排除它们。

## 10. P1 final-raw validation、合并与负迁移门控

这里只评估 P1 的 common、S/P-FT、replicate 0/1/2、N=50/100、`compute_matched`、`shapenet`，应生成 12 条 validation 命令。validation 不需要 test-release 确认：

```bash
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/generate_evaluation_commands.py" \
  --repo-root "$REPO" \
  --dataset-root "$DATA_ROOT" \
  --protocol "$PROTOCOL" \
  --training-output-root "$TRAIN_ROOT" \
  --eval-output-root "$EVAL_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --split val \
  --checkpoint-tag latest \
  --methods S P-FT \
  --reps 0 1 2 \
  --ns 50 100 \
  --tasks common \
  --budgets compute_matched \
  --frames shapenet \
  --output "$COMMAND_ROOT/P1_final_raw_val_commands.txt"

test "$(wc -l < "$COMMAND_ROOT/P1_final_raw_val_commands.txt")" -eq 12
! grep -q -- '--evaluation-split test' "$COMMAND_ROOT/P1_final_raw_val_commands.txt"
```

从预先创建的日志目录提交：

```bash
mkdir -p "$LOG_ROOT/P1_final_raw_val"
cd "$LOG_ROOT/P1_final_raw_val"
sbatch --array=1-12%3 \
  "$REPO/recipes/aero_cfd/jobs/drivaerml_transfer_array_ncsa.job" \
  "$COMMAND_ROOT/P1_final_raw_val_commands.txt"
```

十二个评估 jobs 全部完成后合并 CSV，并执行唯一的 validation gate：

```bash
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python - "$METRICS_ROOT/val/common" <<'PY'
import json, pathlib, sys
paths = sorted(pathlib.Path(sys.argv[1]).glob("*/compute_matched/latest/*.csv.audit.json"))
assert len(paths) == 12, len(paths)
for path in paths:
    audit = json.loads(path.read_text())
    assert audit["evaluation_split"] == "val"
    assert audit["datasets_in_eval_config"] == ["val"]
    assert audit["evaluation_design_count"] == 34
    assert len(audit["official_evaluation_design_ids"]) == 34
    assert audit["target_checkpoint_tag"] == "latest"
    assert audit["target_checkpoint_sha256"] == audit["sidecar_target_checkpoint_sha256"]
    assert len(audit["manifest_raw_sha256"]) == 64
    assert audit["test_release_confirmed"] is False
print("validation_audits_ok", len(paths))
PY

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/merge_metric_csvs.py" \
  "$METRICS_ROOT"/val/common/S/compute_matched/latest/*.csv \
  "$METRICS_ROOT"/val/common/P-FT/compute_matched/latest/*.csv \
  --output "$REPORT_ROOT/P1_final_raw_val.csv" \
  --provenance "$REPORT_ROOT/P1_final_raw_val_merge.json"

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/assess_validation_gate.py" \
  "$REPORT_ROOT/P1_final_raw_val.csv" \
  --confirm-final-raw-val \
  --output "$REPORT_ROOT/P1_validation_gate.json"

grep -E '"decision": "(continue|stop)"' "$REPORT_ROOT/P1_validation_gate.json"
```

- 若 decision 为 `stop`：不得运行 P2，不得生成或运行任何 test 命令；先审计坐标映射和 normalization。
- 只有 decision 为 `continue` 才进入下一节。

## 11. continue 后释放 P2

P2 新增 25 个 train-only statistics artifacts：

```bash
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/generate_statistics_commands.py" \
  --protocol "$PROTOCOL" \
  --phases P2 \
  --repo-root "$REPO" \
  --manifest-root "$MANIFEST_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --stats-root "$STATS_ROOT" \
  --output "$COMMAND_ROOT/statistics_P2.txt"

test "$(wc -l < "$COMMAND_ROOT/statistics_P2.txt")" -eq 5

mkdir -p "$LOG_ROOT/statistics_P2"
cd "$LOG_ROOT/statistics_P2"
sbatch --array=1-5%4 \
  "$REPO/recipes/aero_cfd/jobs/drivaerml_statistics_array_ncsa.job" \
  "$COMMAND_ROOT/statistics_P2.txt"
```

等 5 个单遍统计 jobs 全部完成并产出 25 个 artifacts 后，生成并释放 50 个 P2 training jobs：

```bash
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/generate_training_commands.py" \
  --protocol "$PROTOCOL" \
  --phase P2 \
  --repo-root "$REPO" \
  --manifest-root "$MANIFEST_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --output-path "$TRAIN_ROOT" \
  --stats-root "$STATS_ROOT" \
  --source-output-path "$SOURCE_ROOT" \
  --output "$COMMAND_ROOT/P2_commands.txt"

test "$(wc -l < "$COMMAND_ROOT/P2_commands.txt")" -eq 50

mkdir -p "$LOG_ROOT/P2"
cd "$LOG_ROOT/P2"
sbatch --array=1-50%3 \
  "$REPO/recipes/aero_cfd/jobs/drivaerml_transfer_array_ncsa.job" \
  "$COMMAND_ROOT/P2_commands.txt"
```

P2 的 50 个 tasks 必须全部成功，主网格才完整。P2a/P2b/P2c/P3 属于后续 gated controls/ablations，不得为了“顺便跑”而提前与 P2 或 test 一起释放。

## 12. 冻结所有决策后才释放最终 test

在生成 test 命令前，先书面冻结并归档：

- production commit、protocol SHA256、八个 manifest SHA256；
- primary task=`common`；
- methods=`S P-FT`；
- replicates=`0..7`；
- N=`25 50 100 200 400`；
- budget=`compute_matched`；
- frame=`shapenet`；
- target checkpoint tags=`latest` + `best_model.loss.val.total`；
- 分析阈值、bootstrap seed 和 common-field 定义；
- P1 validation gate 的原始 CSV、merge provenance 和 decision JSON。

确认不再根据 test 调参或改选择后，才允许在生成器层显式加入 `--confirm-test-release`。它会在每条 test 命令中再次加入 runner 层的 `--confirm-test-release`，形成双层 gate。主网格应恰好生成 80 条命令：

```bash
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/generate_evaluation_commands.py" \
  --repo-root "$REPO" \
  --dataset-root "$DATA_ROOT" \
  --protocol "$PROTOCOL" \
  --training-output-root "$TRAIN_ROOT" \
  --eval-output-root "$EVAL_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --split test \
  --checkpoint-tag latest \
  --methods S P-FT \
  --reps 0 1 2 3 4 5 6 7 \
  --ns 25 50 100 200 400 \
  --tasks common \
  --budgets compute_matched \
  --frames shapenet \
  --confirm-test-release \
  --output "$COMMAND_ROOT/primary_test_commands.txt"

test "$(wc -l < "$COMMAND_ROOT/primary_test_commands.txt")" -eq 80
test "$(grep -c -- '--evaluation-split test' "$COMMAND_ROOT/primary_test_commands.txt")" -eq 80
test "$(grep -c -- '--confirm-test-release' "$COMMAND_ROOT/primary_test_commands.txt")" -eq 80

# Freeze this sensitivity list in the same decision bundle before running either test array.
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/generate_evaluation_commands.py" \
  --repo-root "$REPO" \
  --dataset-root "$DATA_ROOT" \
  --protocol "$PROTOCOL" \
  --training-output-root "$TRAIN_ROOT" \
  --eval-output-root "$EVAL_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  --split test \
  --checkpoint-tag best_model.loss.val.total \
  --methods S P-FT \
  --reps 0 1 2 3 4 5 6 7 \
  --ns 25 50 100 200 400 \
  --tasks common \
  --budgets compute_matched \
  --frames shapenet \
  --confirm-test-release \
  --output "$COMMAND_ROOT/sensitivity_best_val_test_commands.txt"

test "$(wc -l < "$COMMAND_ROOT/sensitivity_best_val_test_commands.txt")" -eq 80
test "$(grep -c -- '--evaluation-split test' "$COMMAND_ROOT/sensitivity_best_val_test_commands.txt")" -eq 80
test "$(grep -c -- '--confirm-test-release' "$COMMAND_ROOT/sensitivity_best_val_test_commands.txt")" -eq 80
```

从预建日志目录提交最终 test：

```bash
mkdir -p "$LOG_ROOT/primary_test"
cd "$LOG_ROOT/primary_test"
sbatch --array=1-80%3 \
  "$REPO/recipes/aero_cfd/jobs/drivaerml_transfer_array_ncsa.job" \
  "$COMMAND_ROOT/primary_test_commands.txt"

mkdir -p "$LOG_ROOT/sensitivity_best_val_test"
cd "$LOG_ROOT/sensitivity_best_val_test"
sbatch --array=1-80%3 \
  "$REPO/recipes/aero_cfd/jobs/drivaerml_transfer_array_ncsa.job" \
  "$COMMAND_ROOT/sensitivity_best_val_test_commands.txt"
```

全部完成后合并并分析：

```bash
env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python - "$METRICS_ROOT/test/common" <<'PY'
import collections, json, pathlib, sys
paths = sorted(pathlib.Path(sys.argv[1]).glob("*/compute_matched/*/*.csv.audit.json"))
assert len(paths) == 160, len(paths)
counts = collections.Counter()
for path in paths:
    audit = json.loads(path.read_text())
    counts[audit["target_checkpoint_tag"]] += 1
    assert audit["evaluation_split"] == "test"
    assert audit["datasets_in_eval_config"] == ["test"]
    assert audit["evaluation_design_count"] == 50
    assert len(audit["official_evaluation_design_ids"]) == 50
    assert audit["official_test_design_ids"] == audit["official_evaluation_design_ids"]
    assert audit["target_checkpoint_sha256"] == audit["sidecar_target_checkpoint_sha256"]
    assert len(audit["manifest_raw_sha256"]) == 64
    assert audit["test_release_confirmed"] is True
assert counts == {"latest": 80, "best_model.loss.val.total": 80}, counts
print("test_audits_ok", dict(counts))
PY

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/merge_metric_csvs.py" \
  "$METRICS_ROOT"/test/common/S/compute_matched/latest/*.csv \
  "$METRICS_ROOT"/test/common/P-FT/compute_matched/latest/*.csv \
  --output "$REPORT_ROOT/primary_test.csv" \
  --provenance "$REPORT_ROOT/primary_test_merge.json"

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/analyze_transfer_results.py" \
  "$REPORT_ROOT/primary_test.csv" \
  --confirm-frozen-test \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260716 \
  --output "$REPORT_ROOT/primary_test_analysis.json"

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/merge_metric_csvs.py" \
  "$METRICS_ROOT"/test/common/S/compute_matched/best-model-loss-val-total/*.csv \
  "$METRICS_ROOT"/test/common/P-FT/compute_matched/best-model-loss-val-total/*.csv \
  --output "$REPORT_ROOT/sensitivity_best_val_test.csv" \
  --provenance "$REPORT_ROOT/sensitivity_best_val_test_merge.json"

env PYTHONPATH="$PYTHONPATH" \
  uv run --project "$REPO" --no-sync python \
  "$REPO/research/multi_fidelity/tools/analyze_transfer_results.py" \
  "$REPORT_ROOT/sensitivity_best_val_test.csv" \
  --confirm-frozen-test \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260716 \
  --output "$REPORT_ROOT/sensitivity_best_val_test_analysis.json"
```

`primary_test.csv`、merge provenance、analysis JSON、全部 command files、Slurm logs、training/evaluation provenance sidecars、protocol 和 manifest hashes 共同构成最终审计包。

## 13. 统一 STOP 条件

遇到以下任一情况立即停止当前及后续阶段：

- 出现任何必需下载、GitHub/PyPI 访问或安装需求；
- protocol 尚未经用户明确审阅同意，或 `status` 不是 `frozen_before_first_target_job`；
- repo 不是预期 commit 或 `git status --porcelain` 非空；
- checkpoint size/SHA256 不匹配；
- `uv run --project ... --no-sync` 依赖检查失败；
- manifest 不是八份、commit 不匹配或 `implementation_git_dirty` 不是 `false`；
- command count 与 16/6/30/12/25/50/80+80 任一预期不符；
- graph zero-degree gate 失败；
- 任一 Slurm task 失败、OOM、NaN 或缺少 provenance sidecar；
- P1 validation gate 返回 `stop`；
- test 前科学决策尚未冻结，或 test command 缺少任一层 `--confirm-test-release`。

STOP 后只收集日志、路径、size、SHA256、exit code 和最小错误上下文，并告知用户；不得用联网下载或扩大实验矩阵来自动绕过。
