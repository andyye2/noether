#!/bin/bash -l
#SBATCH --job-name=mf-b2v-train
#SBATCH --account=ssingh76-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=ccc0389,ccc0390
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --output=/scratch/andyye2/ABUPT/slurm/multi_fidelity_B2_vtransfer/%x-%A_%a.out
#SBATCH --error=/scratch/andyye2/ABUPT/slurm/multi_fidelity_B2_vtransfer/%x-%A_%a.err
#
# sbatch --array=1-2 $0 <commit> <command-file> <command-file-sha256>

set -euo pipefail

PROJECT_ROOT=/scratch/andyye2/ABUPT/multi_fidelity_B2_vtransfer
ENV_ROOT=/scratch/andyye2/ABUPT/shapenet-divu0/5-14-restored/.venv
DATA_ROOT=/scratch/andyye2/data/drivaerml_subsampled_10x
SOURCE_ROOT=/scratch/andyye2/ABUPT/outputs
SOURCE_CKPT="$SOURCE_ROOT/2026-04-25_7d0mv/train/checkpoints/ab_upt_cp=latest_model.th"
SLURM_ROOT=/scratch/andyye2/ABUPT/slurm/multi_fidelity_B2_vtransfer
CACHE_ROOT=/scratch/andyye2/ABUPT/.cache
PROTOCOL="$PROJECT_ROOT/research/multi_fidelity/experiment_protocol.yaml"

EXPECTED_CHECKPOINT_SIZE=28147342
EXPECTED_CHECKPOINT_SHA=261a46b7464d50c26301db60758feaabc27ce3d4f30b511e826201d32154fe38

PROTECTED_ROOTS=(
    /scratch/andyye2/ABUPT/outputs/multi_fidelity/
    /scratch/andyye2/ABUPT/outputs/multi_fidelity_exploratory/
    /scratch/andyye2/ABUPT/outputs/multi_fidelity_paper/
    /scratch/andyye2/ABUPT/outputs/multi_fidelity_B2/
    /scratch/andyye2/ABUPT/multi_fidelity_artifacts/
    /scratch/andyye2/ABUPT/multi_fidelity_stats/
    /scratch/andyye2/ABUPT/multi_fidelity_B2_artifacts/
)

if [[ $# -ne 3 ]]; then
    echo "usage: sbatch --array=1-2 $0 <commit> <command-file> <command-file-sha256>" >&2
    exit 2
fi

COMMIT=$1
COMMAND_FILE=$2
EXPECTED_COMMAND_SHA=$3
ARTIFACT_ROOT="/scratch/andyye2/ABUPT/multi_fidelity_B2_vtransfer_artifacts/$COMMIT"
OUTPUT_ROOT="/scratch/andyye2/ABUPT/outputs/multi_fidelity_B2_vtransfer/$COMMIT"
MANIFEST="$ARTIFACT_ROOT/manifests/drivaerml_nested_seed1103.json"
STATISTICS="$ARTIFACT_ROOT/statistics/seed1103/n100_common_shapenet.json"

if [[ ! $COMMIT =~ ^[0-9a-f]{40}$ ]]; then
    echo "STOP: expected a full lowercase commit hash, got: $COMMIT" >&2
    exit 2
fi
if [[ ! $EXPECTED_COMMAND_SHA =~ ^[0-9a-f]{64}$ ]]; then
    echo "STOP: expected a lowercase command-file SHA256, got: $EXPECTED_COMMAND_SHA" >&2
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

actual_command_sha=$(sha256sum "$COMMAND_FILE" | awk '{print $1}')
if [[ $actual_command_sha != "$EXPECTED_COMMAND_SHA" ]]; then
    echo "STOP: command-file SHA256 mismatch" >&2
    echo "expected=$EXPECTED_COMMAND_SHA" >&2
    echo "actual=$actual_command_sha" >&2
    exit 2
fi

command_count=$(wc -l < "$COMMAND_FILE")
if [[ $command_count -ne 2 || $SLURM_ARRAY_TASK_ID -lt 1 || $SLURM_ARRAY_TASK_ID -gt 2 ]]; then
    echo "STOP: vtransfer training requires two commands and array indices 1..2; count=$command_count index=$SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
if [[ $(grep -c -- '--strategy scratch' "$COMMAND_FILE" || true) -ne 1 ]]; then
    echo "STOP: command file must contain exactly one scratch arm" >&2
    exit 2
fi
if [[ $(grep -c -- '--strategy finetune' "$COMMAND_FILE" || true) -ne 1 ]]; then
    echo "STOP: command file must contain exactly one finetune arm" >&2
    exit 2
fi
# This namespace exists for the wall-distance pair, and the two arms are only a
# pair if they read the same inputs. A plain arm submitted here would be filed
# next to feature arms it cannot be compared with.
feature_arms=$(grep -c -- '--wall-distance-feature' "$COMMAND_FILE" || true)
if [[ $feature_arms -ne 2 ]]; then
    echo "STOP: the vtransfer namespace requires --wall-distance-feature on both arms; matched=$feature_arms" >&2
    exit 2
fi

command_line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMAND_FILE")
EXPECTED_PREFIX="env PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/recipes/aero_cfd/src uv run --project $PROJECT_ROOT --no-sync python $PROJECT_ROOT/recipes/aero_cfd/scripts/run_drivaerml_transfer_strict.py"
if [[ $command_line != "$EXPECTED_PREFIX"* ]]; then
    echo "STOP: command is not the vtransfer strict-training entry point" >&2
    echo "$command_line" >&2
    exit 2
fi
for required in \
    "--protocol $PROTOCOL" \
    "--dataset-root $DATA_ROOT" \
    "--manifest $MANIFEST" \
    "--target-statistics $STATISTICS" \
    "--output-path $OUTPUT_ROOT" \
    "--source-output-path $SOURCE_ROOT" \
    "--task common" \
    "--sample-size 100" \
    "--budget compute_matched" \
    "--coordinate-frame shapenet" \
    "--position-scale 1000" \
    "--supernode-radius 0.1" \
    "--replicate 0" \
    "--model-seed 7103" \
    "--eval-point-seed 4242"; do
    if [[ $command_line != *"$required"* ]]; then
        echo "STOP: training command is missing the frozen request: $required" >&2
        echo "$command_line" >&2
        exit 2
    fi
done
if [[ $command_line != *'--strategy scratch'* && $command_line != *'--strategy finetune'* ]]; then
    echo "STOP: command is neither S-matched nor P-FT-matched" >&2
    exit 2
fi
if [[ $command_line == *'--run-id'* || $command_line == *'--dry-run'* ]]; then
    echo "STOP: production commands may not override the run ID or request a dry run" >&2
    exit 2
fi
if [[ $command_line == *';'* || $command_line == *'&&'* || $command_line == *'||'* || $command_line == *'`'* || $command_line == *'$('* ]]; then
    echo "STOP: command contains an unsupported shell control operator" >&2
    exit 2
fi
for protected in "${PROTECTED_ROOTS[@]}"; do
    if [[ $command_line == *"$protected"* ]]; then
        echo "STOP: command references a protected prior-result namespace: $protected" >&2
        echo "$command_line" >&2
        exit 2
    fi
done

if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi
module purge || true
module load anaconda3/2024.10 || module load anaconda3

for required_path in "$ENV_ROOT/bin/python" "$SOURCE_CKPT" "$PROTOCOL" "$MANIFEST" "$STATISTICS"; do
    if [[ ! -f $required_path ]]; then
        echo "STOP: required file missing: $required_path" >&2
        exit 2
    fi
done
if [[ ! -d $DATA_ROOT ]]; then
    echo "STOP: dataset missing: $DATA_ROOT" >&2
    exit 2
fi
if [[ $(stat -c %s "$SOURCE_CKPT") -ne $EXPECTED_CHECKPOINT_SIZE ]]; then
    echo "STOP: source checkpoint size mismatch" >&2
    exit 2
fi
actual_checkpoint_sha=$(sha256sum "$SOURCE_CKPT" | awk '{print $1}')
if [[ $actual_checkpoint_sha != "$EXPECTED_CHECKPOINT_SHA" ]]; then
    echo "STOP: source checkpoint SHA256 mismatch" >&2
    exit 2
fi

actual_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
if [[ $actual_commit != "$COMMIT" ]]; then
    echo "STOP: repository commit mismatch; expected=$COMMIT actual=$actual_commit" >&2
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

mkdir -p "$OUTPUT_ROOT" "$SLURM_ROOT" "$UV_CACHE_DIR" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID of $command_count"
echo "command_file=$COMMAND_FILE"
echo "command_file_sha256=$actual_command_sha"
echo "commit=$actual_commit"
echo "project=$PROJECT_ROOT"
echo "dataset=$DATA_ROOT"
echo "source_checkpoint=$SOURCE_CKPT"
echo "source_checkpoint_sha256=$actual_checkpoint_sha"
echo "output=$OUTPUT_ROOT"
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

echo "training_completed_at=$(date --iso-8601=seconds)"
echo "training_completed_ok"
