#!/bin/bash
#SBATCH --job-name=ad-k1024-sweep
#SBATCH --account=ssingh76-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=ccc0390
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/ABUPT/slurm/%x-%j.out

set -eo pipefail

normalize_param() {
  awk -v x="$1" 'BEGIN { if (x == "" || x !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) print x; else printf "%.8g", x }'
}

param_tag() {
  printf '%s' "$(normalize_param "$1")" \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/-/m/g; s/\./p/g; s/\+//g; s/[^0-9A-Za-z_]+/_/g'
}

export U="$(normalize_param "${U:-${UNIFORM_FRACTION:-0.3}}")"
export G="$(normalize_param "${G:-${GAMMA:-1.0}}")"
export R="$(normalize_param "${R:-${REFRESH_EVERY:-10}}")"
export SEED="${SEED:-0}"

export U_TAG="$(param_tag "$U")"
export G_TAG="$(param_tag "$G")"
export R_TAG="$(param_tag "$R")"
export RUN_NAME="${RUN_NAME:-official_k1024_ipw_u${U_TAG}_g${G_TAG}_r${R_TAG}_seed${SEED}}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge || true
module load anaconda3/2024.10 || module load anaconda3

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch/$USER/ABUPT/noether-adaptive}"
export ENV_ROOT="${ENV_ROOT:-/scratch/$USER/ABUPT/.venv}"
export DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data/shapenet_car}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/ABUPT/outputs/official_k1024_adaptive_sweep/$RUN_NAME}"
export SCORE_ROOT="${SCORE_ROOT:-/scratch/$USER/ABUPT/volume_scores/official_k1024_adaptive_sweep/$RUN_NAME/${SLURM_JOB_ID:-manual}}"
export RUN_REGISTRY="${RUN_REGISTRY:-/scratch/$USER/ABUPT/outputs/official_k1024_adaptive_sweep_registry}"
export PRECISION="${PRECISION:-float16}"
export VOLUME_ANCHORS="${VOLUME_ANCHORS:-1024}"
export SURFACE_ANCHORS="${SURFACE_ANCHORS:-1024}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
export TEST_EVERY="${TEST_EVERY:-1}"
export VOLUME_SCORE_UNIFORM_FRACTION="$U"
export VOLUME_SCORE_GAMMA="$G"
export REFRESH_EVERY="$R"
export VOLUME_SCORE_QUERY_CHUNK_SIZE="${VOLUME_SCORE_QUERY_CHUNK_SIZE:-1024}"

if [ ! -x "$ENV_ROOT/bin/python" ]; then
  export ENV_ROOT=/scratch/$USER/ABUPT/noether/.venv
fi

export UV_CACHE_DIR=/scratch/$USER/ABUPT/.cache/uv
export XDG_CACHE_HOME=/scratch/$USER/ABUPT/.cache
export MPLCONFIGDIR=/scratch/$USER/ABUPT/.cache/matplotlib
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/recipes/aero_cfd/src"

mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$OUTPUT_ROOT" "$SCORE_ROOT" "$RUN_REGISTRY" /scratch/$USER/ABUPT/slurm
cd "$PROJECT_ROOT"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="ad-k1024-u${U_TAG}-g${G_TAG}-r${R_TAG}" >/dev/null 2>&1 || true
fi

echo "host=$(hostname)"
echo "date=$(date)"
echo "project=$PROJECT_ROOT"
echo "env=$ENV_ROOT"
echo "data=$DATA_ROOT"
echo "output=$OUTPUT_ROOT"
echo "score=$SCORE_ROOT"
echo "registry=$RUN_REGISTRY"
echo "run_name=$RUN_NAME"
echo "u=$U"
echo "g=$G"
echo "r=$R"
git branch --show-current
git log -1 --oneline
git status --short --branch
"$ENV_ROOT/bin/python" --version
nvidia-smi

"$ENV_ROOT/bin/python" - <<'INNERPY'
from pathlib import Path
import os
import torch

from aero_cfd.presets import ShapeNetCarPreset
from noether.core.distributed.utils import accelerator_to_device
from noether.core.factory.base import Factory
from noether.core.factory.dataset import DatasetFactory
from noether.training.runners import HydraRunner

data_root = Path(os.environ["DATA_ROOT"])
output_root = Path(os.environ["OUTPUT_ROOT"])
score_root = Path(os.environ["SCORE_ROOT"])
registry_root = Path(os.environ["RUN_REGISTRY"])
precision = os.environ["PRECISION"]
volume_anchors = int(os.environ["VOLUME_ANCHORS"])
surface_anchors = int(os.environ["SURFACE_ANCHORS"])
seed = int(os.environ["SEED"])
refresh_every = int(os.environ["REFRESH_EVERY"])
uniform_fraction = float(os.environ["VOLUME_SCORE_UNIFORM_FRACTION"])
score_gamma = float(os.environ["VOLUME_SCORE_GAMMA"])
query_chunk_size = int(os.environ["VOLUME_SCORE_QUERY_CHUNK_SIZE"])
checkpoint_every = int(os.environ["CHECKPOINT_EVERY"])
test_every = int(os.environ["TEST_EVERY"])

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm job.")
if not data_root.exists():
    raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")
if not (0.0 <= uniform_fraction <= 1.0):
    raise ValueError(f"U/uniform_fraction must be in [0, 1], got {uniform_fraction}")
if score_gamma <= 0.0:
    raise ValueError(f"G/gamma must be positive, got {score_gamma}")
if refresh_every <= 0:
    raise ValueError(f"R/refresh_every must be positive, got {refresh_every}")

preset = ShapeNetCarPreset()
model_kind = "noether.modeling.models.aerodynamics.AeroABUPT"
physics_blocks = [
    "perceiver",
    "self",
    "cross",
    "self",
    "cross",
    "self",
    "cross",
    "self",
    "cross",
    "self",
]

callbacks_override = []
for callback in preset.standard_callbacks(
    save_every_n_epochs=checkpoint_every,
    batch_size=1,
    best_metric_key="loss/test/volume_velocity_mse",
):
    if callback.kind == "noether.training.callbacks.OfflineLossCallback":
        continue
    if callback.kind == "noether.core.callbacks.BestCheckpointCallback":
        callbacks_override.extend(
            preset.evaluation_callbacks(model_kind, batch_size=1, every_n_epochs=test_every)
        )
        callback.every_n_epochs = test_every
    callbacks_override.append(callback)

optimizer = preset.build_optimizer(clip_grad_norm=0.25)

config = preset.build_config(
    model_kind=model_kind,
    model_params=dict(
        hidden_dim=192,
        geometry_depth=1,
        physics_blocks=physics_blocks,
        num_domain_decoder_blocks={"surface": 2, "volume": 2},
    ),
    optimizer=optimizer,
    trainer_kind="noether.training.trainers.WeightedLossTrainer",
    trainer_params=dict(
        field_weights={"surface_pressure": 1.0, "volume_velocity": 1.0},
        precision=precision,
    ),
    dataset_root=str(data_root),
    output_path=str(output_root),
    datasets=["train", "test"],
    max_epochs=500,
    batch_size=1,
    accelerator="gpu",
    include_evaluation=False,
    callbacks_override=callbacks_override,
    use_volume_score_sampling=True,
    volume_score_dir=str(score_root),
    volume_score_uniform_fraction=uniform_fraction,
    volume_score_gamma=score_gamma,
    volume_score_refresh_every_n_epochs=refresh_every,
    volume_score_query_chunk_size=query_chunk_size,
    seed=seed,
    store_code_in_output=False,
)

for dataset_key, dataset_config in config.datasets.items():
    pipeline = dataset_config.pipeline
    if hasattr(pipeline, "num_volume_anchor_points"):
        pipeline.num_volume_anchor_points = volume_anchors
        pipeline.num_surface_anchor_points = surface_anchors

print("trainer_kind", config.trainer.kind)
print("max_epochs", config.trainer.max_epochs)
print("precision", config.trainer.precision)
print("sample_weight_keys", config.trainer.sample_weight_keys)
print("optimizer", config.model.optimizer_config)
print("model_geometry_depth", config.model.geometry_depth)
print("model_physics_blocks", config.model.physics_blocks)
print("model_domain_decoder_blocks", config.model.num_domain_decoder_blocks)
print("seed", config.seed)
print("dataset_keys", list(config.datasets.keys()))
print("callback_kinds", [callback.kind for callback in config.trainer.callbacks])
print("checkpoint_every", checkpoint_every)
print("test_every", test_every)
print("refresh_every", refresh_every)
print("volume_score_uniform_fraction", uniform_fraction)
print("volume_score_gamma", score_gamma)
for dataset_key, dataset_config in config.datasets.items():
    pipeline = dataset_config.pipeline
    print(
        f"anchors_{dataset_key}",
        pipeline.num_surface_anchor_points,
        pipeline.num_volume_anchor_points,
        "score_sampling",
        getattr(pipeline, "use_volume_score_sampling", None),
        "emit_score_candidates",
        getattr(pipeline, "emit_volume_score_candidates", None),
    )

assert config.model.geometry_depth == 1
assert config.model.num_domain_decoder_blocks == {"surface": 2, "volume": 2}
assert config.model.optimizer_config.clip_grad_norm == 0.25
assert config.trainer.kind == "aero_cfd.trainers.SamplingWeightedLossTrainer"
assert config.trainer.sample_weight_keys == {"volume_velocity": "volume_anchor_sampling_weight"}
assert config.trainer.precision == precision
assert config.datasets["train"].pipeline.num_surface_anchor_points == surface_anchors
assert config.datasets["train"].pipeline.num_volume_anchor_points == volume_anchors
assert config.datasets["train"].pipeline.volume_score_uniform_fraction == uniform_fraction
assert config.datasets["train"].pipeline.volume_score_gamma == score_gamma
assert any(callback.kind == "aero_cfd.callbacks.AeroMetricsCallback" for callback in config.trainer.callbacks)
assert any(callback.kind == "noether.core.callbacks.BestCheckpointCallback" for callback in config.trainer.callbacks)
assert all(callback.kind != "noether.training.callbacks.OfflineLossCallback" for callback in config.trainer.callbacks)

dataset = DatasetFactory().instantiate(config.datasets["train"])
raw_sample = dataset[0]
pipeline = Factory().create(config.datasets["train"].pipeline)
batch = pipeline([raw_sample])
print("precheck_surface_anchor_position", tuple(batch["surface_anchor_position"].shape))
print("precheck_volume_anchor_position", tuple(batch["volume_anchor_position"].shape))
print("precheck_sampling_prob", tuple(batch["volume_anchor_sampling_prob"].shape))
print("precheck_sampling_weight", tuple(batch["volume_anchor_sampling_weight"].shape))
print("precheck_surface_pressure_target", tuple(batch["surface_pressure_target"].shape))
print("precheck_volume_velocity_target", tuple(batch["volume_velocity_target"].shape))
assert batch["surface_anchor_position"].shape[1] == surface_anchors
assert batch["volume_anchor_position"].shape[1] == volume_anchors
assert batch["volume_anchor_sampling_weight"].shape[1] == volume_anchors

HydraRunner().main(device=accelerator_to_device("gpu"), config=config)

score_files = sorted(score_root.glob("score_*.pt"))
print("num_score_files", len(score_files))
if not score_files:
    raise RuntimeError(f"No score sidecars were written to {score_root}")

job_id = os.environ.get("SLURM_JOB_ID") or config.run_id
run_dir = output_root / str(job_id)
if not run_dir.exists():
    candidates = sorted(
        [path for path in output_root.iterdir() if path.is_dir() and (path / "hp_resolved.yaml").exists()],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"Could not locate run directory under {output_root}")
    run_dir = candidates[-1]

best_checkpoints = sorted((run_dir / "checkpoints").glob("ab_upt_cp=best_model.loss.test.volume_velocity_mse*_model.th"))
print("best_checkpoints", [str(path) for path in best_checkpoints])
if not best_checkpoints:
    raise RuntimeError(
        "No best checkpoint was written. Check that AeroMetricsCallback logs "
        "loss/test/volume_velocity_mse before BestCheckpointCallback."
    )
best_checkpoint = best_checkpoints[0]

registry_root.mkdir(parents=True, exist_ok=True)
(registry_root / f"{os.environ['RUN_NAME']}.env").write_text(
    "\n".join(
        [
            f"RUN_NAME={os.environ['RUN_NAME']}",
            f"RUN_DIR={run_dir}",
            f"OUTPUT_ROOT={output_root}",
            f"SCORE_ROOT={score_root}",
            f"BEST_CHECKPOINT={best_checkpoint}",
            f"U={uniform_fraction}",
            f"G={score_gamma}",
            f"R={refresh_every}",
            f"SEED={seed}",
            f"SLURM_JOB_ID={job_id}",
            "",
        ]
    )
)
print("run_dir", run_dir)
print("best_checkpoint", best_checkpoint)
print(f"{os.environ['RUN_NAME']}_ok")
INNERPY

