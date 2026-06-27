from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


FIELDS_TO_MATCH = [
    ("dataset_kind",),
    ("model", "kind"),
    ("model", "optimizer_config", "kind"),
    ("model", "optimizer_config", "lr"),
    ("model", "optimizer_config", "weight_decay"),
    ("model", "optimizer_config", "clip_grad_norm"),
    ("model", "optimizer_config", "schedule_config", "kind"),
    ("model", "optimizer_config", "schedule_config", "end_value"),
    ("model", "optimizer_config", "schedule_config", "max_value"),
    ("model", "optimizer_config", "schedule_config", "warmup_percent"),
    ("model", "forward_properties"),
    ("model", "supernode_pooling_config", "hidden_dim"),
    ("model", "supernode_pooling_config", "input_dim"),
    ("model", "supernode_pooling_config", "radius"),
    ("model", "transformer_block_config", "hidden_dim"),
    ("model", "transformer_block_config", "num_heads"),
    ("model", "transformer_block_config", "mlp_hidden_dim"),
    ("model", "transformer_block_config", "mlp_expansion_factor"),
    ("model", "transformer_block_config", "use_rope"),
    ("model", "geometry_depth"),
    ("model", "hidden_dim"),
    ("model", "physics_blocks"),
    ("model", "num_domain_decoder_blocks"),
    ("model", "data_specs"),
    ("trainer", "kind"),
    ("trainer", "max_epochs"),
    ("trainer", "effective_batch_size"),
    ("trainer", "precision"),
    ("trainer", "forward_properties"),
    ("trainer", "target_properties"),
    ("trainer", "surface_weight"),
    ("trainer", "volume_weight"),
    ("trainer", "surface_pressure_weight"),
    ("trainer", "volume_velocity_weight"),
    ("trainer", "use_physics_features"),
]

PIPELINE_FIELDS_TO_MATCH = [
    "kind",
    "num_surface_points",
    "num_volume_points",
    "num_surface_queries",
    "num_volume_queries",
    "use_physics_features",
    "num_geometry_supernodes",
    "num_geometry_points",
    "data_specs",
]


def get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        value = value[key]
    return value


def callback_summary(callbacks: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            callback["kind"],
            callback.get("every_n_epochs"),
            callback.get("batch_size"),
            callback.get("dataset_key"),
            callback.get("metric_key"),
        )
        for callback in callbacks
    ]


def resolve_config_with_noether_schema(repo_root: Path, raw_config: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "recipes" / "aero_cfd"))
    old_cwd = os.getcwd()
    os.chdir(repo_root / "recipes" / "aero_cfd")
    try:
        from noether.core.schemas.schema import ConfigSchema

        schema = ConfigSchema(**raw_config)
        resolved = schema.model_dump(exclude_unset=True, exclude_computed_fields=True)
        resolved["config_schema_kind"] = schema.config_schema_kind
        return resolved
    finally:
        os.chdir(old_cwd)


def compose_k1024_config(repo_root: Path, dataset_root: str, output_path: str) -> dict[str, Any]:
    config_dir = repo_root / "recipes" / "aero_cfd" / "configs"
    overrides = [
        "+experiment/shapenet=ab_upt",
        "tracker=disabled",
        f"dataset_root={dataset_root}",
        f"output_path={output_path}",
        "+accelerator=gpu",
        "+seed=0",
        "pipeline.num_surface_anchor_points=1024",
        "pipeline.num_volume_anchor_points=1024",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="train_shapenet", overrides=overrides)
    raw_config = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    return resolve_config_with_noether_schema(repo_root, raw_config)


def clean_archive(source_repo: Path, commit: str) -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix=f"noether_{commit}_")
    archive = subprocess.run(
        ["git", "-C", str(source_repo), "archive", commit],
        check=True,
        stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", temp_dir.name], check=True, input=archive.stdout)
    return temp_dir


def default_source_repo() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a clean original Noether commit composes the April official "
            "ShapeNet AB-UPT config with only the anchor counts changed to 1024/1024."
        )
    )
    parser.add_argument("--source-repo", default=str(default_source_repo()))
    parser.add_argument("--commit", default="0e1400b")
    parser.add_argument("--april-hp", required=True)
    parser.add_argument("--dataset-root", default="/tmp/shapenet_car_config_check")
    parser.add_argument("--output-path", default="/tmp/original_official_k1024_seed0")
    args = parser.parse_args()

    april = yaml.safe_load(Path(args.april_hp).read_text())
    with clean_archive(Path(args.source_repo), args.commit) as clean_repo:
        k1024 = compose_k1024_config(Path(clean_repo), args.dataset_root, args.output_path)

    mismatches: list[tuple[str, Any, Any]] = []
    for field in FIELDS_TO_MATCH:
        april_value = get_nested(april, field)
        k1024_value = get_nested(k1024, field)
        if april_value != k1024_value:
            mismatches.append((".".join(field), april_value, k1024_value))

    for dataset_key in ["train", "test", "test_repeat"]:
        for field in PIPELINE_FIELDS_TO_MATCH:
            april_value = april["datasets"][dataset_key]["pipeline"][field]
            k1024_value = k1024["datasets"][dataset_key]["pipeline"][field]
            if april_value != k1024_value:
                mismatches.append((f"datasets.{dataset_key}.pipeline.{field}", april_value, k1024_value))

    if callback_summary(april["trainer"]["callbacks"]) != callback_summary(k1024["trainer"]["callbacks"]):
        mismatches.append(
            (
                "trainer.callbacks.summary",
                callback_summary(april["trainer"]["callbacks"]),
                callback_summary(k1024["trainer"]["callbacks"]),
            )
        )

    print(f"clean_commit: {args.commit}")
    print("intentional_anchor_changes:")
    for dataset_key in ["train", "test", "test_repeat"]:
        april_pipeline = april["datasets"][dataset_key]["pipeline"]
        k1024_pipeline = k1024["datasets"][dataset_key]["pipeline"]
        print(
            f"  {dataset_key}: "
            f"surface {april_pipeline['num_surface_anchor_points']} -> {k1024_pipeline['num_surface_anchor_points']}, "
            f"volume {april_pipeline['num_volume_anchor_points']} -> {k1024_pipeline['num_volume_anchor_points']}"
        )
    print(f"field_mismatches_excluding_anchors: {len(mismatches)}")
    for name, april_value, k1024_value in mismatches:
        print(f"MISMATCH {name}: april={april_value!r} k1024={k1024_value!r}")
    print(f"trainer_kind: {k1024['trainer']['kind']}")
    print(f"model_kind: {k1024['model']['kind']}")
    print(f"precision: {k1024['trainer']['precision']}")
    print(f"optimizer: {k1024['model']['optimizer_config']['kind']}")
    print(f"clip_grad_norm: {k1024['model']['optimizer_config']['clip_grad_norm']}")
    print(f"geometry_depth: {k1024['model']['geometry_depth']}")
    print(f"physics_blocks: {k1024['model']['physics_blocks']}")
    print(f"decoder_blocks: {k1024['model']['num_domain_decoder_blocks']}")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
