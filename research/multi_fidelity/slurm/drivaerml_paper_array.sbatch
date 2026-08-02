#!/bin/bash -l
#SBATCH --job-name=mf-paper
#SBATCH --account=ssingh76-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=ccc0389,ccc0390
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --output=/scratch/andyye2/ABUPT/slurm/multi_fidelity/%x-%A_%a.out
#SBATCH --error=/scratch/andyye2/ABUPT/slurm/multi_fidelity/%x-%A_%a.err
#
# Run one line of a generated command file.
#
#   sbatch --array=1-4 drivaerml_paper_array.sbatch <commit> <command file>
#
# The generators own every scientific option, so this script only proves that
# the machine, the code, and the destinations are the intended ones. It never
# edits a command line.
#
# Always dry-run the submission first:
#   sbatch --test-only --array=1-4 drivaerml_paper_array.sbatch <commit> <file>

set -euo pipefail

PROJECT_ROOT=/scratch/andyye2/ABUPT/multi_fidelity
ENV_ROOT=/scratch/andyye2/ABUPT/shapenet-divu0/5-14-restored/.venv
DATA_ROOT=/scratch/andyye2/data/drivaerml_subsampled_10x
SOURCE_ROOT=/scratch/andyye2/ABUPT/outputs
SOURCE_CKPT="$SOURCE_ROOT/2026-04-25_7d0mv/train/checkpoints/ab_upt_cp=latest_model.th"
SLURM_ROOT=/scratch/andyye2/ABUPT/slurm/multi_fidelity
CACHE_ROOT=/scratch/andyye2/ABUPT/.cache
PROTOCOL="$PROJECT_ROOT/research/multi_fidelity/experiment_protocol.yaml"

EXPECTED_CHECKPOINT_SIZE=28147342
EXPECTED_CHECKPOINT_SHA=261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38

# Namespaces holding completed results. A command may not so much as name one.
# The artifacts root itself is not listed because this run writes its own
# manifests, statistics, and metrics under its own commit inside it; the two
# commits that already own results there are listed individually.
PROTECTED_ROOTS=(
    /scratch/andyye2/ABUPT/outputs/multi_fidelity/
    /scratch/andyye2/ABUPT/outputs/multi_fidelity_exploratory/
    /scratch/andyye2/ABUPT/multi_fidelity_artifacts/cf974078ea633f1fb6ff2178de7fa56ac60c89d5/
    /scratch/andyye2/ABUPT/multi_fidelity_artifacts/2ec68b91d5bd7573e464098eecf176e65f97f8e1/
    /scratch/andyye2/ABUPT/multi_fidelity_stats/
)

if [[ $# -ne 2 ]]; then
    echo "usage: sbatch --array=1-N $0 <expected commit> /absolute/path/to/commands.txt" >&2
    exit 2
fi

COMMIT=$1
COMMAND_FILE=$2

if [[ ! $COMMIT =~ ^[0-9a-f]{40}$ ]]; then
    echo "STOP: expected a full lowercase commit hash, got: $COMMIT" >&2
    exit 2
fi

if [[ -z ${SLURM_ARRAY_TASK_ID:-} ]]; then
    echo "STOP: this batch must be submitted as a Slurm array" >&2
    exit 2
fi

if [[ ! -f "$COMMAND_FILE" ]]; then
    echo "STOP: command file not found: $COMMAND_FILE" >&2
    exit 2
fi

command_count=$(wc -l < "$COMMAND_FILE")

if [[ $SLURM_ARRAY_TASK_ID -lt 1 || $SLURM_ARRAY_TASK_ID -gt $command_count ]]; then
    echo "STOP: array index $SLURM_ARRAY_TASK_ID is outside 1..$command_count" >&2
    exit 2
fi

command_line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMAND_FILE")

if [[ -z "$command_line" ]]; then
    echo "STOP: no command at array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

# The command must name this commit's output namespace, so a rerun of an older
# command file can never land on top of results from another implementation.
if [[ "$command_line" != *"/$COMMIT/"* ]]; then
    echo "STOP: command does not write into the namespace of $COMMIT" >&2
    echo "$command_line" >&2
    exit 2
fi

for protected in "${PROTECTED_ROOTS[@]}"; do
    if [[ "$command_line" == *"$protected"* ]]; then
        echo "STOP: command references a completed-result namespace: $protected" >&2
        echo "$command_line" >&2
        exit 2
    fi
done

if [[ "$command_line" == *"--evaluation-split test"* ]]; then
    echo "STOP: this batch never releases the held-out test split" >&2
    exit 2
fi

# Everything above validates the request itself and must work on any machine.
# Only now bring up the cluster environment.
if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi

if type module >/dev/null 2>&1; then
    module purge || true
    module load anaconda3/2024.10 || module load anaconda3
else
    echo "note: no module system found; using the environment as inherited" >&2
fi

for required_path in "$ENV_ROOT/bin/python" "$SOURCE_CKPT" "$PROTOCOL"; do
    if [[ ! -f "$required_path" ]]; then
        echo "STOP: required file missing: $required_path" >&2
        exit 2
    fi
done

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "STOP: dataset missing: $DATA_ROOT" >&2
    exit 2
fi

if [[ $(stat -c %s "$SOURCE_CKPT") -ne $EXPECTED_CHECKPOINT_SIZE ]]; then
    echo "STOP: source checkpoint size mismatch" >&2
    exit 2
fi

actual_checkpoint_sha=$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')

if [[ "$actual_checkpoint_sha" != "$EXPECTED_CHECKPOINT_SHA" ]]; then
    echo "STOP: source checkpoint SHA256 mismatch" >&2
    exit 2
fi

actual_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

if [[ "$actual_commit" != "$COMMIT" ]]; then
    echo "STOP: repository commit mismatch" >&2
    echo "expected=$COMMIT" >&2
    echo "actual=$actual_commit" >&2
    exit 2
fi

if [[ -n $(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all) ]]; then
    echo "STOP: production repository is dirty" >&2
    git -C "$PROJECT_ROOT" status --short >&2
    exit 2
fi

if ! grep -q '^status: frozen_before_first_target_job$' "$PROTOCOL"; then
    echo "STOP: experiment protocol is not frozen" >&2
    exit 2
fi

export PATH="/u/andyye2/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export XDG_CACHE_HOME="$CACHE_ROOT"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/recipes/aero_cfd/src"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$SLURM_ROOT" "$UV_CACHE_DIR" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID of $command_count"
echo "command_file=$COMMAND_FILE"
echo "command_file_sha256=$(sha256sum "$COMMAND_FILE" | awk '{print $1}')"
echo "commit=$actual_commit"
echo "project=$PROJECT_ROOT"
echo "environment=$ENV_ROOT"
echo "dataset=$DATA_ROOT"
echo "source_checkpoint=$SOURCE_CKPT"
echo "source_checkpoint_sha256=$actual_checkpoint_sha"
echo "protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')"
echo "python_version=$("$ENV_ROOT/bin/python" --version 2>&1)"
echo "uv_version=$(uv --version)"

nvidia-smi

"$ENV_ROOT/bin/python" -c \
'import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA is unavailable on the allocated GPU node"
assert torch.cuda.device_count() >= 1, "No CUDA device is visible"'

echo "command=$command_line"

srun --kill-on-bad-exit=1 bash -lc "$command_line"

echo "completed_at=$(date --iso-8601=seconds)"
echo "job_completed_ok"
