#!/bin/bash -l
#SBATCH --job-name=mf-b2v-stats
#SBATCH --account=ssingh76-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/andyye2/ABUPT/slurm/multi_fidelity_B2_vtransfer/%x-%A_%a.out
#SBATCH --error=/scratch/andyye2/ABUPT/slurm/multi_fidelity_B2_vtransfer/%x-%A_%a.err
#
# sbatch --array=1-1 $0 <commit> <command-file> <command-file-sha256>

set -euo pipefail

PROJECT_ROOT=/scratch/andyye2/ABUPT/multi_fidelity_B2_vtransfer
ENV_ROOT=/scratch/andyye2/ABUPT/shapenet-divu0/5-14-restored/.venv
DATA_ROOT=/scratch/andyye2/data/drivaerml_subsampled_10x
SLURM_ROOT=/scratch/andyye2/ABUPT/slurm/multi_fidelity_B2_vtransfer
CACHE_ROOT=/scratch/andyye2/ABUPT/.cache
PROTOCOL="$PROJECT_ROOT/research/multi_fidelity/experiment_protocol.yaml"

if [[ $# -ne 3 ]]; then
    echo "usage: sbatch --array=1-1 $0 <commit> <command-file> <command-file-sha256>" >&2
    exit 2
fi

COMMIT=$1
COMMAND_FILE=$2
EXPECTED_COMMAND_SHA=$3
ARTIFACT_ROOT="/scratch/andyye2/ABUPT/multi_fidelity_B2_vtransfer_artifacts/$COMMIT"
STATS_JSON="$ARTIFACT_ROOT/statistics/seed1103/n100_common_shapenet.json"
STATS_FLAT="$ARTIFACT_ROOT/statistics/seed1103/n100_common_shapenet.flat.yaml"

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
if [[ $command_count -ne 1 || $SLURM_ARRAY_TASK_ID -ne 1 ]]; then
    echo "STOP: statistics requires one command at array index 1; count=$command_count index=$SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
command_line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMAND_FILE")

EXPECTED_PREFIX="env PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/recipes/aero_cfd/src uv run --project $PROJECT_ROOT --no-sync python -m research.multi_fidelity.tools.compute_subset_statistics"
if [[ $command_line != "$EXPECTED_PREFIX"* ]]; then
    echo "STOP: command is not the vtransfer statistics entry point" >&2
    echo "$command_line" >&2
    exit 2
fi
for required in \
    "--root $DATA_ROOT" \
    "--protocol $PROTOCOL" \
    "--manifest $ARTIFACT_ROOT/manifests/drivaerml_nested_seed1103.json" \
    "--n 100" \
    "--fields surface_pressure volume_velocity volume_distance" \
    "--coordinate-frame shapenet" \
    "--output $STATS_JSON" \
    "--flat-stats-output $STATS_FLAT"; do
    if [[ $command_line != *"$required"* ]]; then
        echo "STOP: statistics command is missing the frozen request: $required" >&2
        echo "$command_line" >&2
        exit 2
    fi
done
if [[ $command_line == *';'* || $command_line == *'&&'* || $command_line == *'||'* || $command_line == *'`'* || $command_line == *'$('* ]]; then
    echo "STOP: command contains an unsupported shell control operator" >&2
    exit 2
fi
if [[ -e $STATS_JSON || -e $STATS_FLAT ]]; then
    echo "STOP: refusing to overwrite an existing vtransfer statistics artifact" >&2
    stat "$STATS_JSON" "$STATS_FLAT" 2>/dev/null || true
    exit 2
fi

if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi
module purge || true
module load anaconda3/2024.10 || module load anaconda3

for required_path in "$ENV_ROOT/bin/python" "$PROTOCOL"; do
    if [[ ! -f $required_path ]]; then
        echo "STOP: required file missing: $required_path" >&2
        exit 2
    fi
done
if [[ ! -d $DATA_ROOT ]]; then
    echo "STOP: dataset missing: $DATA_ROOT" >&2
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

mkdir -p "$ARTIFACT_ROOT/statistics" "$SLURM_ROOT" "$UV_CACHE_DIR" "$MPLCONFIGDIR"
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
echo "protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')"
echo "python_version=$("$ENV_ROOT/bin/python" --version 2>&1)"
echo "uv_version=$(uv --version)"
echo "command=$command_line"

srun --kill-on-bad-exit=1 bash -lc "$command_line"

test -s "$STATS_JSON"
test -s "$STATS_FLAT"
echo "statistics_sha256=$(sha256sum "$STATS_JSON" | awk '{print $1}')"
echo "statistics_completed_at=$(date --iso-8601=seconds)"
echo "statistics_completed_ok"
