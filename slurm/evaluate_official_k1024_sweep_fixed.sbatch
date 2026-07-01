#!/bin/bash
#SBATCH --job-name=eval-k1024-sweep
#SBATCH --account=ssingh76-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=ccc0390
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/ABUPT/slurm/%x-%j.out

set -eo pipefail

normalize_param() {
  awk -v x="$1" 'BEGIN { if (x == "" || x !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) print x; else printf "%.8g", x }'
}

param_tag() {
  printf '%s' "$(normalize_param "$1")" \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/-/m/g; s/\./p/g; s/\+//g; s/[^0-9A-Za-z_]+/_/g'
}

registry_value() {
  local env_file="$1"
  local key="$2"
  if [ -f "$env_file" ]; then
    grep "^${key}=" "$env_file" | tail -n 1 | cut -d= -f2-
  fi
}

best_volume_mse_checkpoint() {
  local run_dir="$1"
  find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'ab_upt_cp=best_model.loss.test.volume_velocity_mse*_model.th' -print 2>/dev/null \
    | sort \
    | head -n 1
}

latest_run_dir() {
  local root="$1"
  find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

split_run_names() {
  printf '%s\n' "$1" | tr ',;:' '\n' | tr ' ' '\n' | sed '/^$/d'
}

decode_param_tag() {
  printf '%s\n' "$1" | sed -E 's/^m/-/; s/p/./g'
}

run_name_param() {
  local name="$1"
  local key="$2"
  local tag
  tag="$(printf '%s\n' "$name" | sed -nE "s/.*_${key}([^_]+).*/\\1/p")"
  if [ -n "$tag" ]; then
    decode_param_tag "$tag"
  fi
}

discover_adaptive_siblings() {
  local name="$1"
  local suffix output_root
  suffix="$(printf '%s\n' "$name" | sed -nE 's/^official_k1024_ipw_u[^_]+(_g[^_]+_r[^_]+_seed[^_]+)$/\1/p')"
  if [ -z "$suffix" ]; then
    return
  fi
  output_root="$ADAPTIVE_OUTPUT_ROOT"
  {
    find "$SWEEP_REGISTRY" -maxdepth 1 -type f -name "official_k1024_ipw_u*${suffix}.env" -printf '%f\n' 2>/dev/null | sed 's/\.env$//'
    find "$output_root" -mindepth 1 -maxdepth 1 -type d -name "official_k1024_ipw_u*${suffix}" -printf '%f\n' 2>/dev/null
  } | sort -u
}

expand_single_adaptive_sweep_name() {
  local names="$1"
  local count only discovered discovered_count
  count="$(printf '%s\n' "$names" | sed '/^$/d' | wc -l | xargs)"
  if [ "$count" != "1" ] || [ "${EXPAND_SINGLE_ADAPTIVE_SWEEP:-1}" = "0" ]; then
    printf '%s\n' "$names"
    return
  fi
  only="$(printf '%s\n' "$names" | sed '/^$/d' | head -n 1)"
  discovered="$(discover_adaptive_siblings "$only")"
  discovered_count="$(printf '%s\n' "$discovered" | sed '/^$/d' | wc -l | xargs)"
  if [ "$discovered_count" -gt 1 ] && printf '%s\n' "$discovered" | grep -qx "$only"; then
    echo "ADAPTIVE_RUN_NAMES looked comma-truncated by Slurm; expanded '$only' to $discovered_count matching sweep runs. Quote ADAPTIVE_RUN_NAMES to pass an exact comma list." >&2
    printf '%s\n' "$discovered" | sed '/^$/d'
  else
    printf '%s\n' "$names"
  fi
}


source /etc/profile.d/modules.sh 2>/dev/null || true
module purge || true
module load anaconda3/2024.10 || module load anaconda3

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch/$USER/ABUPT/noether-adaptive}"
export ENV_ROOT="${ENV_ROOT:-/scratch/$USER/ABUPT/.venv}"
export DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data/shapenet_car}"
export EVAL_ROOT="${EVAL_ROOT:-/scratch/$USER/ABUPT/outputs/official_k1024_adaptive_sweep_eval}"
export BASELINE_REGISTRY="${BASELINE_REGISTRY:-/scratch/$USER/ABUPT/outputs/official_k1024_compare_registry}"
export SWEEP_REGISTRY="${SWEEP_REGISTRY:-/scratch/$USER/ABUPT/outputs/official_k1024_adaptive_sweep_registry}"
export ADAPTIVE_OUTPUT_ROOT="${ADAPTIVE_OUTPUT_ROOT:-/scratch/$USER/ABUPT/outputs/official_k1024_adaptive_sweep}"

export BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-official_k1024_uniform_seed0}"
export BASELINE_LABEL="${BASELINE_LABEL:-official_k1024_uniform_seed0}"
export ADAPTIVE_LABEL_PREFIX="${ADAPTIVE_LABEL_PREFIX:-}"

export U="${U:-${UNIFORM_FRACTION:-0.3}}"
export G="${G:-${GAMMA:-1.0}}"
export R="${R:-${REFRESH_EVERY:-10}}"
export SEED="${SEED:-0}"
export U_TAG="$(param_tag "$U")"
export G_TAG="$(param_tag "$G")"
export R_TAG="$(param_tag "$R")"
export ADAPTIVE_RUN_NAME="${ADAPTIVE_RUN_NAME:-official_k1024_ipw_u${U_TAG}_g${G_TAG}_r${R_TAG}_seed${SEED}}"

if [ ! -x "$ENV_ROOT/bin/python" ]; then
  export ENV_ROOT=/scratch/$USER/ABUPT/noether/.venv
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/recipes/aero_cfd/src"
export XDG_CACHE_HOME=/scratch/$USER/ABUPT/.cache
export MPLCONFIGDIR=/scratch/$USER/ABUPT/.cache/matplotlib

export MAX_SAMPLES="${MAX_SAMPLES:-100}"
export QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-1024}"
export TOP_SCORE_FRACTION="${TOP_SCORE_FRACTION:-0.10}"
export NUM_SURFACE_ANCHORS="${NUM_SURFACE_ANCHORS:-1024}"
export NUM_VOLUME_ANCHORS="${NUM_VOLUME_ANCHORS:-1024}"
export POSTHOC_COMPACT_EVAL="${POSTHOC_COMPACT_EVAL:-auto}"
export EXPAND_SINGLE_ADAPTIVE_SWEEP="${EXPAND_SINGLE_ADAPTIVE_SWEEP:-1}"
MISSING_VOLUME_BEST_CHECKPOINT="MISSING_VOLUME_MSE_BEST_CHECKPOINT"
MISSING_VOLUME_BEST_CHECKPOINTS=""

baseline_env="$BASELINE_REGISTRY/$BASELINE_RUN_NAME.env"
export BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$(registry_value "$baseline_env" RUN_DIR)}"
export BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$(latest_run_dir "/scratch/$USER/ABUPT/outputs/$BASELINE_RUN_NAME")}"
BASELINE_CHECKPOINT_OVERRIDE="${BASELINE_CHECKPOINT:-}"
BASELINE_VOLUME_BEST_CHECKPOINT="$(best_volume_mse_checkpoint "$BASELINE_RUN_DIR")"
if [ -n "$BASELINE_CHECKPOINT_OVERRIDE" ]; then
  export BASELINE_CHECKPOINT="$BASELINE_CHECKPOINT_OVERRIDE"
elif [ -n "$BASELINE_VOLUME_BEST_CHECKPOINT" ]; then
  export BASELINE_CHECKPOINT="$BASELINE_VOLUME_BEST_CHECKPOINT"
else
  export BASELINE_CHECKPOINT="$MISSING_VOLUME_BEST_CHECKPOINT"
  MISSING_VOLUME_BEST_CHECKPOINTS="${MISSING_VOLUME_BEST_CHECKPOINTS}${BASELINE_LABEL}
"
fi
export BASELINE_REFERENCE_CHECKPOINT="${BASELINE_REFERENCE_CHECKPOINT:-$BASELINE_CHECKPOINT}"

if [ "${ADAPTIVE_RUN_NAMES:-}" = "ALL" ]; then
  RAW_ADAPTIVE_NAMES="$(find "$SWEEP_REGISTRY" -maxdepth 1 -type f -name 'official_k1024_ipw_*.env' -printf '%f\n' 2>/dev/null | sed 's/\.env$//' | sort)"
elif [ -n "${ADAPTIVE_RUN_NAMES:-}" ]; then
  RAW_ADAPTIVE_NAMES="$(split_run_names "$ADAPTIVE_RUN_NAMES")"
else
  RAW_ADAPTIVE_NAMES="$ADAPTIVE_RUN_NAME"
fi
ADAPTIVE_NAMES="$(expand_single_adaptive_sweep_name "$RAW_ADAPTIVE_NAMES")"
ADAPTIVE_NAME_COUNT="$(printf '%s\n' "$ADAPTIVE_NAMES" | sed '/^$/d' | wc -l | xargs)"
export ADAPTIVE_NAME_COUNT

if [ -z "${EVAL_SUFFIX:-}" ]; then
  first_adaptive="$(printf '%s\n' "$ADAPTIVE_NAMES" | head -n 1)"
  export EVAL_SUFFIX="fixed_${BASELINE_RUN_DIR##*/}_${first_adaptive}_n${ADAPTIVE_NAME_COUNT}"
fi
export EVAL_RUN_ROOT="$EVAL_ROOT/$EVAL_SUFFIX"

mkdir -p "$EVAL_RUN_ROOT" "$MPLCONFIGDIR" /scratch/$USER/ABUPT/slurm
cd "$PROJECT_ROOT"

if [ -z "$BASELINE_RUN_DIR" ] || [ ! -d "$BASELINE_RUN_DIR" ]; then
  echo "Missing BASELINE_RUN_DIR. Set it manually or run $BASELINE_RUN_NAME first." >&2
  exit 1
fi
if [ "$BASELINE_CHECKPOINT" = "$MISSING_VOLUME_BEST_CHECKPOINT" ] || [ ! -f "$BASELINE_CHECKPOINT" ]; then
  echo "No baseline checkpoint file for loss/test/volume_velocity_mse:min was found; training-log comparison will still run." >&2
fi

RUN_CHECKPOINT_SPECS="${BASELINE_LABEL}|${BASELINE_RUN_DIR}|${BASELINE_CHECKPOINT}|baseline|||"
for adaptive_name in $ADAPTIVE_NAMES; do
  adaptive_env="$SWEEP_REGISTRY/$adaptive_name.env"
  if [ "$ADAPTIVE_NAME_COUNT" = "1" ] && [ -n "${ADAPTIVE_RUN_DIR:-}" ]; then
    adaptive_run_dir="$ADAPTIVE_RUN_DIR"
  else
    adaptive_run_dir="$(registry_value "$adaptive_env" RUN_DIR)"
  fi
  adaptive_run_dir="${adaptive_run_dir:-$(latest_run_dir "$ADAPTIVE_OUTPUT_ROOT/$adaptive_name")}"

  if [ "$ADAPTIVE_NAME_COUNT" = "1" ] && [ -n "${ADAPTIVE_CHECKPOINT:-}" ]; then
    adaptive_checkpoint="$ADAPTIVE_CHECKPOINT"
  else
    adaptive_checkpoint="$(best_volume_mse_checkpoint "$adaptive_run_dir")"
  fi
  if [ -z "$adaptive_checkpoint" ]; then
    adaptive_checkpoint="$MISSING_VOLUME_BEST_CHECKPOINT"
    MISSING_VOLUME_BEST_CHECKPOINTS="${MISSING_VOLUME_BEST_CHECKPOINTS}${adaptive_name}
"
  fi

  adaptive_u="$(registry_value "$adaptive_env" U)"
  adaptive_g="$(registry_value "$adaptive_env" G)"
  adaptive_r="$(registry_value "$adaptive_env" R)"
  adaptive_u="${adaptive_u:-$(run_name_param "$adaptive_name" u)}"
  adaptive_g="${adaptive_g:-$(run_name_param "$adaptive_name" g)}"
  adaptive_r="${adaptive_r:-$(run_name_param "$adaptive_name" r)}"
  adaptive_u="${adaptive_u:-$U}"
  adaptive_g="${adaptive_g:-$G}"
  adaptive_r="${adaptive_r:-$R}"
  adaptive_label="${ADAPTIVE_LABEL_PREFIX}${adaptive_name}"

  if [ -z "$adaptive_run_dir" ] || [ ! -d "$adaptive_run_dir" ]; then
    echo "Missing adaptive run dir for $adaptive_name. Expected registry $adaptive_env or output folder." >&2
    exit 1
  fi
  if [ "$adaptive_checkpoint" = "$MISSING_VOLUME_BEST_CHECKPOINT" ] || [ ! -f "$adaptive_checkpoint" ]; then
    echo "No adaptive checkpoint file for $adaptive_name loss/test/volume_velocity_mse:min was found; training-log comparison will still run." >&2
  fi

  RUN_CHECKPOINT_SPECS="${RUN_CHECKPOINT_SPECS}
${adaptive_label}|${adaptive_run_dir}|${adaptive_checkpoint}|adaptive|${adaptive_u}|${adaptive_g}|${adaptive_r}"
done
export RUN_CHECKPOINT_SPECS

if [ "$POSTHOC_COMPACT_EVAL" = "auto" ]; then
  if [ -z "$(printf '%s\n' "$MISSING_VOLUME_BEST_CHECKPOINTS" | sed '/^$/d')" ]; then
    export RUN_POSTHOC_COMPACT_EVAL=1
  else
    export RUN_POSTHOC_COMPACT_EVAL=0
  fi
elif [ "$POSTHOC_COMPACT_EVAL" = "1" ]; then
  if [ -n "$(printf '%s\n' "$MISSING_VOLUME_BEST_CHECKPOINTS" | sed '/^$/d')" ]; then
    echo "POSTHOC_COMPACT_EVAL=1 requires volume-MSE-best checkpoints for all runs. Missing:" >&2
    printf '%s\n' "$MISSING_VOLUME_BEST_CHECKPOINTS" | sed '/^$/d' >&2
    exit 1
  fi
  export RUN_POSTHOC_COMPACT_EVAL=1
else
  export RUN_POSTHOC_COMPACT_EVAL=0
fi

echo "host=$(hostname)"
echo "date=$(date)"
echo "project=$PROJECT_ROOT"
echo "env=$ENV_ROOT"
echo "data=$DATA_ROOT"
echo "eval_run_root=$EVAL_RUN_ROOT"
echo "suffix=$EVAL_SUFFIX"
echo "baseline_run_dir=$BASELINE_RUN_DIR"
echo "baseline_checkpoint=$BASELINE_CHECKPOINT"
echo "baseline_reference_checkpoint=$BASELINE_REFERENCE_CHECKPOINT"
echo "run_posthoc_compact_eval=$RUN_POSTHOC_COMPACT_EVAL"
echo "missing_volume_mse_best_checkpoints:"
printf '%s\n' "$MISSING_VOLUME_BEST_CHECKPOINTS" | sed '/^$/d' || true
echo "run_checkpoint_specs:"
echo "$RUN_CHECKPOINT_SPECS"
git branch --show-current
git log -1 --oneline
git status --short --branch
"$ENV_ROOT/bin/python" --version
nvidia-smi || true

cat > "$EVAL_RUN_ROOT/evaluate_official_k1024_sweep_fixed.py" <<'PYCODE'
from __future__ import annotations

import csv
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from aero_cfd.presets import ShapeNetCarPreset
from noether.core.factory import DatasetFactory, Factory
from noether.core.schemas.models import AnchorBranchedUPTConfig
from noether.modeling.models.aerodynamics import AeroABUPT


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: Path
    checkpoint: Path
    group: str
    u: str
    g: str
    r: str


class RegionStats:
    def __init__(self) -> None:
        self.point_count = 0
        self.sq_error_sum = 0.0
        self.abs_error_sum = 0.0
        self.delta_l2_sum = 0.0
        self.target_l2_sum = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        count = int(mask.sum().item())
        if count == 0:
            return
        delta = prediction[mask] - target[mask]
        self.point_count += count
        self.sq_error_sum += float(delta.square().mean(dim=-1).sum().item())
        self.abs_error_sum += float(delta.abs().mean(dim=-1).sum().item())
        self.delta_l2_sum += float(delta.square().sum().item())
        self.target_l2_sum += float(target[mask].square().sum().item())

    def row(self, total_points: int, global_mse: float) -> dict[str, float | int]:
        mse = self.sq_error_sum / self.point_count if self.point_count else math.nan
        mae = self.abs_error_sum / self.point_count if self.point_count else math.nan
        rel_l2 = math.sqrt(self.delta_l2_sum / self.target_l2_sum) if self.target_l2_sum > 0.0 else math.nan
        return {
            "point_count": self.point_count,
            "point_fraction": self.point_count / total_points if total_points else math.nan,
            "velocity_mse": mse,
            "velocity_mae": mae,
            "relative_l2": rel_l2,
            "mse_over_global": mse / global_mse if global_mse > 0.0 else math.nan,
        }


class RegionAccumulator:
    order = [
        "global",
        "wake",
        "non_wake",
        "near_wall_sdf_0p005L",
        "near_wall_sdf_0p01L",
        "near_wall_sdf_0p02L",
        "top_score_bin_10pct",
        "non_top_score_bin_90pct",
    ]

    def __init__(self) -> None:
        self.regions: dict[str, RegionStats] = {}

    def update(self, name: str, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        self.regions.setdefault(name, RegionStats()).update(prediction, target, mask)

    def rows(self) -> list[dict[str, Any]]:
        total = self.regions["global"].point_count
        global_mse = float(self.regions["global"].row(total, math.nan)["velocity_mse"])
        rows = []
        for name in self.order:
            row = self.regions.get(name, RegionStats()).row(total, global_mse)
            row["region"] = name
            rows.append(row)
        return rows


def denormalize_position(position: torch.Tensor, raw_pos_min: torch.Tensor, raw_pos_max: torch.Tensor) -> torch.Tensor:
    raw_min = raw_pos_min.to(dtype=position.dtype, device=position.device).flatten()
    raw_max = raw_pos_max.to(dtype=position.dtype, device=position.device).flatten()
    return position * (raw_max - raw_min) / 1000.0 + raw_min


def compute_wake_mask(
    surface_position: torch.Tensor,
    volume_position: torch.Tensor,
    raw_pos_min: torch.Tensor,
    raw_pos_max: torch.Tensor,
    box_lwh: tuple[float, float, float] = (0.47, 0.43, 0.31),
    axes: tuple[int, int, int] = (2, 0, 1),
) -> torch.Tensor:
    surface_raw = denormalize_position(surface_position, raw_pos_min, raw_pos_max)
    volume_raw = denormalize_position(volume_position, raw_pos_min, raw_pos_max)
    stream_axis, span_axis, vertical_axis = axes
    rear = surface_raw[:, stream_axis].amax()
    body_front = surface_raw[:, stream_axis].amin()
    length = rear - body_front
    if length <= 0.0:
        return torch.zeros(volume_position.shape[0], dtype=torch.bool)
    span_center = (surface_raw[:, span_axis].amin() + surface_raw[:, span_axis].amax()) / 2.0
    ground = surface_raw[:, vertical_axis].amin()
    wake_length, wake_width, wake_height = box_lwh
    stream = volume_raw[:, stream_axis]
    span = volume_raw[:, span_axis]
    vertical = volume_raw[:, vertical_axis]
    return (
        (stream >= rear)
        & (stream <= rear + wake_length * length)
        & (span >= span_center - 0.5 * wake_width * length)
        & (span <= span_center + 0.5 * wake_width * length)
        & (vertical >= ground)
        & (vertical <= ground + wake_height * length)
    )


def body_length(surface_position: torch.Tensor, raw_pos_min: torch.Tensor, raw_pos_max: torch.Tensor) -> torch.Tensor:
    surface_raw = denormalize_position(surface_position, raw_pos_min, raw_pos_max)
    return surface_raw[:, 2].amax() - surface_raw[:, 2].amin()


def migrate_legacy_readout_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
    model_state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    migrated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if ".domain_decoder_projections." in key and key.endswith(".project.weight"):
            key = key.replace(".project.weight", ".linear.project.weight")
        elif ".domain_decoder_projections." in key and key.endswith(".project.bias"):
            key = key.replace(".project.bias", ".linear.project.bias")
        migrated[key] = value
    for key, value in model_state_dict.items():
        if key not in migrated and ".domain_decoder_projections." in key and ".norm_final." in key:
            migrated[key] = value
    missing = set(model_state_dict) - set(migrated)
    unexpected = set(migrated) - set(model_state_dict)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint could not be migrated safely. "
            f"Missing keys: {sorted(missing)}. Unexpected keys: {sorted(unexpected)}."
        )
    return migrated


def load_model(checkpoint_path: Path, device: torch.device) -> AeroABUPT:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = AnchorBranchedUPTConfig(**checkpoint["model_config"])
    model = AeroABUPT(model_config).to(device)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError:
        state_dict = migrate_legacy_readout_state_dict(checkpoint["state_dict"], model.state_dict())
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def matched_indices(anchor_position: torch.Tensor, source_position: torch.Tensor) -> torch.Tensor:
    index_by_position = {tuple(row.tolist()): idx for idx, row in enumerate(source_position)}
    exact = [index_by_position.get(tuple(row.tolist())) for row in anchor_position]
    if all(idx is not None for idx in exact):
        result = torch.tensor(exact, dtype=torch.long)
    else:
        result = torch.cdist(anchor_position.float(), source_position.float()).argmin(dim=1)
    if result.unique().numel() != result.numel():
        raise RuntimeError("Matched anchor indices are not unique.")
    return result


def run_query_mode_volume_inference(
    model: AeroABUPT,
    batch: dict[str, torch.Tensor],
    full_volume_position: torch.Tensor,
    anchor_indices_cpu: torch.Tensor,
    query_chunk_size: int,
) -> torch.Tensor:
    n_points = full_volume_position.shape[1]
    device = full_volume_position.device
    anchor_indices = anchor_indices_cpu.to(device)
    query_mask = torch.ones(n_points, dtype=torch.bool, device=device)
    query_mask[anchor_indices] = False
    query_indices = torch.arange(n_points, device=device)[query_mask]
    base_kwargs = {
        "geometry_position": batch["geometry_position"],
        "geometry_supernode_idx": batch["geometry_supernode_idx"],
        "geometry_batch_idx": batch["geometry_batch_idx"],
        "surface_anchor_position": batch["surface_anchor_position"],
        "volume_anchor_position": batch["volume_anchor_position"],
    }
    if query_indices.numel() == 0:
        out = model(**base_kwargs)
        prediction = torch.empty((1, n_points, out["volume_velocity"].shape[-1]), dtype=out["volume_velocity"].dtype, device=device)
        prediction[:, anchor_indices] = out["volume_velocity"]
        return prediction
    anchor_prediction = None
    query_chunks = []
    for start in range(0, query_indices.numel(), query_chunk_size):
        idx = query_indices[start : start + query_chunk_size]
        out = model(**base_kwargs, query_volume_position=full_volume_position[:, idx])
        if anchor_prediction is None:
            anchor_prediction = out["volume_velocity"]
        query_chunks.append(out["query_volume_velocity"])
    assert anchor_prediction is not None
    prediction = torch.empty((1, n_points, anchor_prediction.shape[-1]), dtype=anchor_prediction.dtype, device=device)
    prediction[:, anchor_indices] = anchor_prediction
    prediction[:, query_indices] = torch.cat(query_chunks, dim=1)
    return prediction


def build_dataset_and_pipeline(data_root: Path, model_kind: str, num_surface_anchors: int, num_volume_anchors: int, seed: int):
    preset = ShapeNetCarPreset()
    dataset_config = preset.build_dataset(
        split="test",
        root=str(data_root),
        model_kind=model_kind,
        num_surface_anchor_points=num_surface_anchors,
        num_volume_anchor_points=num_volume_anchors,
        seed=seed,
    )
    dataset = DatasetFactory().instantiate(dataset_config)
    pipeline = Factory().create(dataset_config.pipeline)
    return dataset, pipeline


def sample_uri(dataset: Any, idx: int) -> Path:
    return Path(dataset.sample_info(idx)["sample_uri"])


def evaluate_sample(
    model: AeroABUPT,
    dataset: Any,
    pipeline: Any,
    sample_idx: int,
    device: torch.device,
    query_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample = dataset[sample_idx]
    batch = pipeline([sample])
    anchor_indices = matched_indices(batch["volume_anchor_position"][0].cpu(), sample["volume_position"].cpu())
    device_batch = to_device(batch, device)
    full_volume_position = sample["volume_position"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = run_query_mode_volume_inference(
            model=model,
            batch=device_batch,
            full_volume_position=full_volume_position,
            anchor_indices_cpu=anchor_indices,
            query_chunk_size=query_chunk_size,
        )
    target_norm = sample["volume_velocity"].unsqueeze(0)
    pred = dataset.denormalize("volume_velocity", pred_norm.cpu()).squeeze(0)
    target = dataset.denormalize("volume_velocity", target_norm).squeeze(0)
    raw_sdf = torch.load(sample_uri(dataset, sample_idx) / "volume_sdf.pt", weights_only=True).reshape(-1)
    return pred, target, sample["surface_position"], sample["volume_position"], raw_sdf


def update_regions(
    accumulator: RegionAccumulator,
    prediction: torch.Tensor,
    target: torch.Tensor,
    surface_position: torch.Tensor,
    volume_position: torch.Tensor,
    raw_sdf: torch.Tensor,
    raw_pos_min: torch.Tensor,
    raw_pos_max: torch.Tensor,
    top_score_mask: torch.Tensor | None,
) -> None:
    n_points = volume_position.shape[0]
    global_mask = torch.ones(n_points, dtype=torch.bool)
    wake_mask = compute_wake_mask(surface_position, volume_position, raw_pos_min, raw_pos_max)
    length = body_length(surface_position, raw_pos_min, raw_pos_max)
    accumulator.update("global", prediction, target, global_mask)
    accumulator.update("wake", prediction, target, wake_mask)
    accumulator.update("non_wake", prediction, target, ~wake_mask)
    for label, band in [
        ("near_wall_sdf_0p005L", 0.005),
        ("near_wall_sdf_0p01L", 0.01),
        ("near_wall_sdf_0p02L", 0.02),
    ]:
        accumulator.update(label, prediction, target, raw_sdf.abs() <= band * length)
    if top_score_mask is not None:
        accumulator.update("top_score_bin_10pct", prediction, target, top_score_mask)
        accumulator.update("non_top_score_bin_90pct", prediction, target, ~top_score_mask)


def build_top_score_masks(
    checkpoint_path: Path,
    dataset: Any,
    pipeline: Any,
    device: torch.device,
    num_samples: int,
    query_chunk_size: int,
    top_fraction: float,
) -> list[torch.Tensor]:
    print(f"building top-score masks from {checkpoint_path}", flush=True)
    model = load_model(checkpoint_path, device)
    masks = []
    for sample_idx in range(num_samples):
        prediction, target, *_ = evaluate_sample(model, dataset, pipeline, sample_idx, device, query_chunk_size)
        score = (prediction - target).square().mean(dim=-1)
        k = max(1, math.ceil(top_fraction * score.numel()))
        top_idx = torch.topk(score, k=k).indices
        mask = torch.zeros(score.numel(), dtype=torch.bool)
        mask[top_idx] = True
        masks.append(mask)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return masks


def checkpoint_tag(path: Path) -> str:
    name = path.name
    prefix = "ab_upt_cp="
    suffix = "_model.th"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def flatten_region_rows(region_rows: list[dict[str, Any]]) -> dict[str, Any]:
    flat = {}
    for row in region_rows:
        region = row["region"]
        for key, value in row.items():
            if key != "region":
                flat[f"{region}_{key}"] = value
    return flat


def compact_metric_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": summary["model"],
        "group": summary["group"],
        "u": summary["u"],
        "g": summary["g"],
        "r": summary["r"],
        "checkpoint": summary["checkpoint"],
        "checkpoint_path": summary["checkpoint_path"],
        "Global": summary.get("global_velocity_mse", math.nan),
        "Wake": summary.get("wake_velocity_mse", math.nan),
        "Non-wake": summary.get("non_wake_velocity_mse", math.nan),
        "Wall .005L": summary.get("near_wall_sdf_0p005L_velocity_mse", math.nan),
        "Wall .01L": summary.get("near_wall_sdf_0p01L_velocity_mse", math.nan),
        "Wall .02L": summary.get("near_wall_sdf_0p02L_velocity_mse", math.nan),
        "Top 10%": summary.get("top_score_bin_10pct_velocity_mse", math.nan),
        "elapsed_sec": summary.get("elapsed_sec", math.nan),
    }


COMPACT_MSE_METRICS = ['Global', 'Wake', 'Non-wake', 'Wall .005L', 'Wall .01L', 'Wall .02L', 'Top 10%']
TRAINING_SELECTION_METRIC = "loss/test/volume_velocity_mse"
SELECTED_LOG_METRIC_PREFIX = "selected@"
MISSING_CHECKPOINT_SENTINEL = "MISSING_VOLUME_MSE_BEST_CHECKPOINT"


def evaluate_checkpoint(
    run_spec: RunSpec,
    dataset: Any,
    pipeline: Any,
    device: torch.device,
    num_samples: int,
    query_chunk_size: int,
    top_score_masks: list[torch.Tensor],
    raw_pos_min: torch.Tensor,
    raw_pos_max: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start_time = time.time()
    model = load_model(run_spec.checkpoint, device)
    accumulator = RegionAccumulator()
    for sample_idx in range(num_samples):
        prediction, target, surface_position, volume_position, raw_sdf = evaluate_sample(
            model, dataset, pipeline, sample_idx, device, query_chunk_size
        )
        update_regions(
            accumulator=accumulator,
            prediction=prediction,
            target=target,
            surface_position=surface_position,
            volume_position=volume_position,
            raw_sdf=raw_sdf,
            raw_pos_min=raw_pos_min,
            raw_pos_max=raw_pos_max,
            top_score_mask=top_score_masks[sample_idx],
        )
    elapsed = time.time() - start_time
    region_rows = accumulator.rows()
    summary = {
        "model": run_spec.label,
        "group": run_spec.group,
        "u": run_spec.u,
        "g": run_spec.g,
        "r": run_spec.r,
        "run_dir": str(run_spec.run_dir),
        "checkpoint": checkpoint_tag(run_spec.checkpoint),
        "checkpoint_path": str(run_spec.checkpoint),
        "ok": True,
        "error": "",
        "elapsed_sec": round(elapsed, 3),
        **flatten_region_rows(region_rows),
    }
    long_rows = [
        {
            "model": run_spec.label,
            "group": run_spec.group,
            "u": run_spec.u,
            "g": run_spec.g,
            "r": run_spec.r,
            "checkpoint": checkpoint_tag(run_spec.checkpoint),
            "checkpoint_path": str(run_spec.checkpoint),
            "elapsed_sec": round(elapsed, 3),
            **row,
        }
        for row in region_rows
    ]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, long_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_run_specs(value: str) -> list[RunSpec]:
    specs = []
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        parts.extend([""] * (7 - len(parts)))
        label, run_dir, checkpoint, group, u, g, r = parts[:7]
        specs.append(
            RunSpec(
                label=label.strip(),
                run_dir=Path(run_dir.strip()),
                checkpoint=Path(checkpoint.strip()),
                group=group.strip(),
                u=u.strip(),
                g=g.strip(),
                r=r.strip(),
            )
        )
    return specs


def values(pattern: str, text: str) -> list[float]:
    return [float(match) for match in re.findall(pattern, text)]


TEST_LOG_METRIC_RE = re.compile(r"\b([A-Za-z0-9_.\-/]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
EPOCH_RE = re.compile(r"\bE=(\d+)/(?:\d+|\?)")
UPDATE_RE = re.compile(r"\bU=([^ ]+)")


def log_line_context(line: str) -> dict[str, Any]:
    context: dict[str, Any] = {}
    epoch_match = EPOCH_RE.search(line)
    update_match = UPDATE_RE.search(line)
    if epoch_match:
        context["selected_epoch"] = int(epoch_match.group(1))
    if update_match:
        context["selected_update"] = update_match.group(1)
    if len(line) >= 19 and line[4:5] == "-" and line[10:11] == "T":
        context["selected_time"] = line[:19]
    return context


def append_test_event(events: list[dict[str, Any]], metrics: dict[str, float], context: dict[str, Any]) -> None:
    if TRAINING_SELECTION_METRIC not in metrics:
        return
    event = dict(context)
    event.update(metrics)
    events.append(event)


def extract_test_metric_events(log_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_metrics: dict[str, float] = {}
    current_context: dict[str, Any] = {}
    for line in log_text.splitlines():
        metric_matches = [
            (match.group(1), float(match.group(2)))
            for match in TEST_LOG_METRIC_RE.finditer(line)
            if match.group(1).startswith("loss/test/")
        ]
        if not metric_matches:
            continue
        context = log_line_context(line)
        if not current_metrics:
            current_context = context
        for key, value in metric_matches:
            if key in current_metrics:
                append_test_event(events, current_metrics, current_context)
                current_metrics = {}
                current_context = context
            current_metrics[key] = value
            if context:
                current_context = {**current_context, **context}
    append_test_event(events, current_metrics, current_context)
    return events


def best_training_event_by_volume_mse(events: list[dict[str, Any]]) -> tuple[int | None, dict[str, Any]]:
    candidates = [(idx, event) for idx, event in enumerate(events) if is_finite_number(event.get(TRAINING_SELECTION_METRIC))]
    if not candidates:
        return None, {}
    return min(candidates, key=lambda item: float(item[1][TRAINING_SELECTION_METRIC]))


def read_training_diagnostics(run_spec: RunSpec) -> dict[str, Any]:
    hp_path = run_spec.run_dir / "hp_resolved.yaml"
    log_path = run_spec.run_dir / "log.txt"
    hp_text = hp_path.read_text(errors="ignore") if hp_path.exists() else ""
    log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    score_mean = values(r"score_refresh/score_refresh_train/mean_score: ([0-9.eE+-]+)", log_text)
    score_written = values(r"score_refresh/score_refresh_train/num_written: ([0-9.eE+-]+)", log_text)
    test_events = extract_test_metric_events(log_text)
    test_volume = [float(event[TRAINING_SELECTION_METRIC]) for event in test_events if is_finite_number(event.get(TRAINING_SELECTION_METRIC))]
    selected_event_index, selected_event = best_training_event_by_volume_mse(test_events)
    checkpoint_name = run_spec.checkpoint.name
    row: dict[str, Any] = {
        "model": run_spec.label,
        "group": run_spec.group,
        "u": run_spec.u,
        "g": run_spec.g,
        "r": run_spec.r,
        "run_dir": str(run_spec.run_dir),
        "checkpoint_path": str(run_spec.checkpoint),
        "checkpoint": checkpoint_tag(run_spec.checkpoint),
        "volume_mse_best_checkpoint_found": checkpoint_name != MISSING_CHECKPOINT_SENTINEL and run_spec.checkpoint.exists(),
        "checkpoint_is_volume_mse_best": "best_model.loss.test.volume_velocity_mse" in checkpoint_name,
        "hp_found": hp_path.exists(),
        "log_found": log_path.exists(),
        "surface_anchor_1024_detected": "num_surface_anchor_points: 1024" in hp_text,
        "volume_anchor_1024_detected": "num_volume_anchor_points: 1024" in hp_text,
        "sample_weight_keys_detected": "volume_anchor_sampling_weight" in hp_text,
        "training_selection_metric": TRAINING_SELECTION_METRIC,
        "training_selected_event_index": selected_event_index if selected_event_index is not None else math.nan,
        "training_selected_epoch": selected_event.get("selected_epoch", math.nan),
        "training_selected_update": selected_event.get("selected_update", ""),
        "training_selected_time": selected_event.get("selected_time", ""),
        "test_volume_mse_count": len(test_volume),
        "test_volume_mse_min": min(test_volume) if test_volume else math.nan,
        "test_volume_mse_last": test_volume[-1] if test_volume else math.nan,
        "score_refresh_mean_score_count": len(score_mean),
        "score_refresh_mean_score_last": score_mean[-1] if score_mean else math.nan,
        "score_refresh_num_written_last": score_written[-1] if score_written else math.nan,
    }
    for key in sorted(selected_event):
        if key.startswith("loss/test/"):
            row[f"{SELECTED_LOG_METRIC_PREFIX}{key}"] = selected_event[key]
    return row



def is_finite_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def best_params_by_metric(compact_rows: list[dict[str, Any]], group: str | None = None) -> list[dict[str, Any]]:
    candidate_rows = [row for row in compact_rows if group is None or row.get('group') == group]
    rows = []
    for metric in COMPACT_MSE_METRICS:
        finite = [row for row in candidate_rows if is_finite_number(row.get(metric))]
        if finite:
            best = min(finite, key=lambda row: float(row[metric]))
            rows.append(
                {
                    'metric': metric,
                    'group': best['group'],
                    'best_model': best['model'],
                    'u': best['u'],
                    'g': best['g'],
                    'r': best['r'],
                    'value': best[metric],
                    'checkpoint': best['checkpoint'],
                    'checkpoint_path': best['checkpoint_path'],
                }
            )
    return rows


def baseline_row(compact_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    baselines = [row for row in compact_rows if row.get('group') == 'baseline']
    return baselines[0] if baselines else None


def compact_mse_vs_baseline(compact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = baseline_row(compact_rows)
    if baseline is None:
        return []
    rows = []
    for row in compact_rows:
        if row.get('group') != 'adaptive':
            continue
        for metric in COMPACT_MSE_METRICS:
            baseline_value = baseline.get(metric)
            adaptive_value = row.get(metric)
            if not (is_finite_number(baseline_value) and is_finite_number(adaptive_value)):
                continue
            delta = float(adaptive_value) - float(baseline_value)
            pct_delta = 100.0 * delta / float(baseline_value) if float(baseline_value) != 0.0 else math.nan
            rows.append(
                {
                    'metric': metric,
                    'model': row['model'],
                    'u': row['u'],
                    'g': row['g'],
                    'r': row['r'],
                    'baseline_model': baseline['model'],
                    'baseline_value': baseline_value,
                    'adaptive_value': adaptive_value,
                    'delta': delta,
                    'pct_delta': pct_delta,
                    'improved': delta < 0.0,
                    'checkpoint': row['checkpoint'],
                    'checkpoint_path': row['checkpoint_path'],
                }
            )
    return rows


def selected_log_metric_keys(row: dict[str, Any]) -> set[str]:
    return {key[len(SELECTED_LOG_METRIC_PREFIX):] for key in row if key.startswith(SELECTED_LOG_METRIC_PREFIX)}


def training_metrics_vs_baseline_at_volume_min(diagnostic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = [row for row in diagnostic_rows if row.get('group') == 'baseline']
    if not baselines:
        return []
    baseline = baselines[0]
    rows = []
    for row in diagnostic_rows:
        if row.get('group') != 'adaptive':
            continue
        metric_keys = sorted(selected_log_metric_keys(baseline) & selected_log_metric_keys(row))
        for metric_key in metric_keys:
            baseline_value = baseline.get(f'{SELECTED_LOG_METRIC_PREFIX}{metric_key}')
            adaptive_value = row.get(f'{SELECTED_LOG_METRIC_PREFIX}{metric_key}')
            if not (is_finite_number(baseline_value) and is_finite_number(adaptive_value)):
                continue
            baseline_value = float(baseline_value)
            adaptive_value = float(adaptive_value)
            delta = adaptive_value - baseline_value
            pct_delta = 100.0 * delta / baseline_value if baseline_value != 0.0 else math.nan
            out = {
                'metric': metric_key,
                'selection_metric': TRAINING_SELECTION_METRIC,
                'model': row['model'],
                'u': row['u'],
                'g': row['g'],
                'r': row['r'],
                'baseline_model': baseline['model'],
                'baseline_value': baseline_value,
                'adaptive_value': adaptive_value,
                'delta': delta,
                'pct_delta': pct_delta,
                'improved': delta < 0.0,
                'baseline_selected_event_index': baseline.get('training_selected_event_index', math.nan),
                'adaptive_selected_event_index': row.get('training_selected_event_index', math.nan),
                'baseline_selected_epoch': baseline.get('training_selected_epoch', math.nan),
                'adaptive_selected_epoch': row.get('training_selected_epoch', math.nan),
                'baseline_checkpoint_path': baseline.get('checkpoint_path', ''),
                'adaptive_checkpoint_path': row.get('checkpoint_path', ''),
            }
            if metric_key == TRAINING_SELECTION_METRIC:
                out['baseline_test_volume_mse_min'] = baseline_value
                out['adaptive_test_volume_mse_min'] = adaptive_value
            rows.append(out)
    return rows


def training_volume_mse_vs_baseline(diagnostic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in training_metrics_vs_baseline_at_volume_min(diagnostic_rows)
        if row.get('metric') == TRAINING_SELECTION_METRIC
    ]


def main() -> None:
    data_root = Path(os.environ["DATA_ROOT"])
    eval_run_root = Path(os.environ["EVAL_RUN_ROOT"])
    suffix = os.environ["EVAL_SUFFIX"]
    run_specs = parse_run_specs(os.environ["RUN_CHECKPOINT_SPECS"])
    baseline_reference_checkpoint = Path(os.environ.get("BASELINE_REFERENCE_CHECKPOINT", MISSING_CHECKPOINT_SENTINEL))
    run_posthoc_compact_eval = os.environ.get("RUN_POSTHOC_COMPACT_EVAL", "0") == "1"
    max_samples = int(os.environ["MAX_SAMPLES"])
    query_chunk_size = int(os.environ["QUERY_CHUNK_SIZE"])
    top_score_fraction = float(os.environ["TOP_SCORE_FRACTION"])
    num_surface_anchors = int(os.environ["NUM_SURFACE_ANCHORS"])
    num_volume_anchors = int(os.environ["NUM_VOLUME_ANCHORS"])
    seed = int(os.environ["SEED"])
    model_kind = "noether.modeling.models.aerodynamics.AeroABUPT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_spec in run_specs:
        if not run_spec.run_dir.exists():
            raise FileNotFoundError(f"RUN_DIR does not exist: {run_spec.run_dir}")

    print("device", device, flush=True)
    print("run_posthoc_compact_eval", run_posthoc_compact_eval, flush=True)
    print("num_surface_anchors", num_surface_anchors, flush=True)
    print("num_volume_anchors", num_volume_anchors, flush=True)
    print("run_specs", run_specs, flush=True)

    summary_rows = []
    long_rows = []
    compact_rows = []
    if run_posthoc_compact_eval:
        if not data_root.exists():
            raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")
        if not baseline_reference_checkpoint.exists():
            raise FileNotFoundError(f"BASELINE_REFERENCE_CHECKPOINT does not exist: {baseline_reference_checkpoint}")
        for run_spec in run_specs:
            if not run_spec.checkpoint.exists():
                raise FileNotFoundError(f"CHECKPOINT does not exist: {run_spec.checkpoint}")

        dataset, pipeline = build_dataset_and_pipeline(data_root, model_kind, num_surface_anchors, num_volume_anchors, seed)
        num_samples = min(max_samples, len(dataset))
        raw_pos_min = torch.as_tensor(dataset.fetch_statistics()["raw_pos_min"])
        raw_pos_max = torch.as_tensor(dataset.fetch_statistics()["raw_pos_max"])
        print("num_samples", num_samples, flush=True)

        top_score_masks = build_top_score_masks(
            checkpoint_path=baseline_reference_checkpoint,
            dataset=dataset,
            pipeline=pipeline,
            device=device,
            num_samples=num_samples,
            query_chunk_size=query_chunk_size,
            top_fraction=top_score_fraction,
        )

        for run_spec in run_specs:
            print(f"evaluating {run_spec.label} {checkpoint_tag(run_spec.checkpoint)}", flush=True)
            try:
                summary, region_rows = evaluate_checkpoint(
                    run_spec=run_spec,
                    dataset=dataset,
                    pipeline=pipeline,
                    device=device,
                    num_samples=num_samples,
                    query_chunk_size=query_chunk_size,
                    top_score_masks=top_score_masks,
                    raw_pos_min=raw_pos_min,
                    raw_pos_max=raw_pos_max,
                )
            except Exception as exc:
                summary = {
                    "model": run_spec.label,
                    "group": run_spec.group,
                    "u": run_spec.u,
                    "g": run_spec.g,
                    "r": run_spec.r,
                    "run_dir": str(run_spec.run_dir),
                    "checkpoint": checkpoint_tag(run_spec.checkpoint),
                    "checkpoint_path": str(run_spec.checkpoint),
                    "ok": False,
                    "error": repr(exc),
                    "elapsed_sec": math.nan,
                }
                region_rows = []
                print(f"FAILED {run_spec.checkpoint}: {exc!r}", flush=True)
            summary_rows.append(summary)
            long_rows.extend(region_rows)
        compact_rows = [compact_metric_row(row) for row in summary_rows if row.get("ok") is True]
    else:
        print("posthoc compact eval skipped: missing volume-MSE-best checkpoint(s) or POSTHOC_COMPACT_EVAL=0", flush=True)
    diagnostic_rows = [read_training_diagnostics(run_spec) for run_spec in run_specs]

    compact_delta_rows = compact_mse_vs_baseline(compact_rows)
    training_metric_delta_rows = training_metrics_vs_baseline_at_volume_min(diagnostic_rows)
    training_delta_rows = [row for row in training_metric_delta_rows if row.get('metric') == TRAINING_SELECTION_METRIC]
    best_rows = best_params_by_metric(compact_rows)
    best_adaptive_rows = best_params_by_metric(compact_rows, group='adaptive')

    write_csv(eval_run_root / f'fixed_checkpoint_summary_{suffix}.csv', summary_rows)
    write_csv(eval_run_root / f'region_metrics_long_{suffix}.csv', long_rows)
    write_csv(eval_run_root / f'region_mse_compact_{suffix}.csv', compact_rows)
    write_csv(eval_run_root / f'training_diagnostics_{suffix}.csv', diagnostic_rows)
    write_csv(eval_run_root / f'compact_mse_vs_baseline_{suffix}.csv', compact_delta_rows)
    write_csv(eval_run_root / f'training_metric_vs_baseline_at_volume_mse_min_{suffix}.csv', training_metric_delta_rows)
    write_csv(eval_run_root / f'training_volume_mse_vs_baseline_{suffix}.csv', training_delta_rows)
    write_csv(eval_run_root / f'best_params_by_metric_{suffix}.csv', best_rows)
    write_csv(eval_run_root / f'best_adaptive_params_by_metric_{suffix}.csv', best_adaptive_rows)

    print("posthoc compact MSE on volume-MSE-best checkpoints:" if run_posthoc_compact_eval else "posthoc compact MSE skipped:")
    for row in compact_rows:
        print(
            row["model"],
            "u", row["u"],
            "g", row["g"],
            "r", row["r"],
            "Global", row["Global"],
            "Wake", row["Wake"],
            "Non-wake", row["Non-wake"],
            "Wall005", row["Wall .005L"],
            "Wall01", row["Wall .01L"],
            "Wall02", row["Wall .02L"],
            "Top10", row["Top 10%"],
            flush=True,
        )

    print('posthoc compact MSE vs baseline on volume-MSE-best checkpoints (lower is better):')
    for row in compact_delta_rows:
        status = 'better' if row['improved'] else 'worse'
        print(
            row['metric'],
            row['model'],
            'adaptive', row['adaptive_value'],
            'baseline', row['baseline_value'],
            'delta', row['delta'],
            'pct', row['pct_delta'],
            status,
            'u', row['u'],
            'g', row['g'],
            'r', row['r'],
            flush=True,
        )
    print(f'training-log volume MSE vs baseline at each run {TRAINING_SELECTION_METRIC}:min test (lower is better):')
    for row in training_delta_rows:
        status = 'better' if row['improved'] else 'worse'
        print(
            'Global(volume)',
            row['model'],
            'adaptive', row['adaptive_value'],
            'baseline', row['baseline_value'],
            'delta', row['delta'],
            'pct', row['pct_delta'],
            status,
            'baseline_epoch', row['baseline_selected_epoch'],
            'adaptive_epoch', row['adaptive_selected_epoch'],
            'u', row['u'],
            'g', row['g'],
            'r', row['r'],
            flush=True,
        )
    print(f'training-log selected test metrics vs baseline at each run {TRAINING_SELECTION_METRIC}:min test (lower is better for loss metrics):')
    for row in training_metric_delta_rows:
        status = 'better' if row['improved'] else 'worse'
        print(
            row['metric'],
            row['model'],
            'adaptive', row['adaptive_value'],
            'baseline', row['baseline_value'],
            'delta', row['delta'],
            'pct', row['pct_delta'],
            status,
            'u', row['u'],
            'g', row['g'],
            'r', row['r'],
            flush=True,
        )
    print('best overall by compact MSE metric:')
    for row in best_rows:
        print(row['metric'], row['best_model'], row['value'], 'group', row['group'], 'u', row['u'], 'g', row['g'], 'r', row['r'], flush=True)
    print('best adaptive params by compact MSE metric:')
    for row in best_adaptive_rows:
        print(row['metric'], row['best_model'], row['value'], 'u', row['u'], 'g', row['g'], 'r', row['r'], flush=True)
    print(f"official_k1024_sweep_fixed_eval_ok suffix={suffix}", flush=True)


if __name__ == "__main__":
    main()
PYCODE

"$ENV_ROOT/bin/python" -m py_compile "$EVAL_RUN_ROOT/evaluate_official_k1024_sweep_fixed.py"
"$ENV_ROOT/bin/python" "$EVAL_RUN_ROOT/evaluate_official_k1024_sweep_fixed.py"

