#!/bin/bash
#SBATCH --job-name=bl-k1024-vmse
#SBATCH --account=ssingh76-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=ccc0390
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/ABUPT/slurm/%x-%j.out

set -eo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge || true
module load anaconda3/2024.10 || module load anaconda3

export PROJECT_ROOT=/scratch/$USER/ABUPT/noether-adaptive
export ENV_ROOT=/scratch/$USER/ABUPT/.venv
export DATA_ROOT=/scratch/$USER/data/shapenet_car
export RUN_NAME="${RUN_NAME:-official_k1024_uniform_volume_best_seed0}"
export OUTPUT_ROOT=/scratch/$USER/ABUPT/outputs/$RUN_NAME
export RUN_REGISTRY=/scratch/$USER/ABUPT/outputs/official_k1024_compare_registry
export PRECISION=float16
export VOLUME_ANCHORS=1024
export SURFACE_ANCHORS=1024
export SEED=0
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
export TEST_EVERY="${TEST_EVERY:-1}"

if [ ! -x "$ENV_ROOT/bin/python" ]; then
  export ENV_ROOT=/scratch/$USER/ABUPT/noether/.venv
fi

export UV_CACHE_DIR=/scratch/$USER/ABUPT/.cache/uv
export XDG_CACHE_HOME=/scratch/$USER/ABUPT/.cache
export MPLCONFIGDIR=/scratch/$USER/ABUPT/.cache/matplotlib
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/recipes/aero_cfd/src"

mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$OUTPUT_ROOT" "$RUN_REGISTRY" /scratch/$USER/ABUPT/slurm
cd "$PROJECT_ROOT"

echo "host=$(hostname)"
echo "date=$(date)"
echo "project=$PROJECT_ROOT"
echo "env=$ENV_ROOT"
echo "data=$DATA_ROOT"
echo "output=$OUTPUT_ROOT"
echo "registry=$RUN_REGISTRY"
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
registry_root = Path(os.environ["RUN_REGISTRY"])
precision = os.environ["PRECISION"]
volume_anchors = int(os.environ["VOLUME_ANCHORS"])
surface_anchors = int(os.environ["SURFACE_ANCHORS"])
seed = int(os.environ["SEED"])
checkpoint_every = int(os.environ["CHECKPOINT_EVERY"])
test_every = int(os.environ["TEST_EVERY"])

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm job.")
if not data_root.exists():
    raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

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

optimizer = preset.build_optimizer(clip_grad_norm=0.25)

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
print("optimizer", config.model.optimizer_config)
print("model_geometry_depth", config.model.geometry_depth)
print("model_physics_blocks", config.model.physics_blocks)
print("model_domain_decoder_blocks", config.model.num_domain_decoder_blocks)
print("seed", config.seed)
print("dataset_keys", list(config.datasets.keys()))
print("callback_kinds", [callback.kind for callback in config.trainer.callbacks])
print("checkpoint_every", checkpoint_every)
print("test_every", test_every)
for dataset_key, dataset_config in config.datasets.items():
    pipeline = dataset_config.pipeline
    print(
        f"anchors_{dataset_key}",
        pipeline.num_surface_anchor_points,
        pipeline.num_volume_anchor_points,
    )

assert config.model.geometry_depth == 1
assert config.model.num_domain_decoder_blocks == {"surface": 2, "volume": 2}
assert config.model.optimizer_config.clip_grad_norm == 0.25
assert config.trainer.precision == precision
assert config.datasets["train"].pipeline.num_surface_anchor_points == surface_anchors
assert config.datasets["train"].pipeline.num_volume_anchor_points == volume_anchors

dataset = DatasetFactory().instantiate(config.datasets["train"])
raw_sample = dataset[0]
pipeline = Factory().create(config.datasets["train"].pipeline)
batch = pipeline([raw_sample])
print("precheck_surface_anchor_position", tuple(batch["surface_anchor_position"].shape))
print("precheck_volume_anchor_position", tuple(batch["volume_anchor_position"].shape))
print("precheck_surface_pressure_target", tuple(batch["surface_pressure_target"].shape))
print("precheck_volume_velocity_target", tuple(batch["volume_velocity_target"].shape))
assert batch["surface_anchor_position"].shape[1] == surface_anchors
assert batch["volume_anchor_position"].shape[1] == volume_anchors

HydraRunner().main(device=accelerator_to_device("gpu"), config=config)

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

registry_root.mkdir(parents=True, exist_ok=True)
(registry_root / f"{os.environ['RUN_NAME']}.env").write_text(
    "\n".join(
        [
            f"RUN_NAME={os.environ['RUN_NAME']}",
            f"RUN_DIR={run_dir}",
            f"OUTPUT_ROOT={output_root}",
            f"SLURM_JOB_ID={job_id}",
            "",
        ]
    )
)
best_checkpoints = sorted((run_dir / "checkpoints").glob("ab_upt_cp=best_model.loss.test.volume_velocity_mse*_model.th"))
print("run_dir", run_dir)
print("best_checkpoints", [str(path) for path in best_checkpoints])
if not best_checkpoints:
    raise RuntimeError(
        "Uniform baseline finished but did not save "
        "loss/test/volume_velocity_mse best checkpoint."
    )
print("best_checkpoint", best_checkpoints[0])
print("official_k1024_uniform_volume_best_seed0_ok")
INNERPY

