#  Copyright (c) 2025 Emmi AI GmbH. All rights reserved.

"""
Create a wake-oversampled ShapeNet car CFD dataset variant.

This script follows the ShapeNet-Car preprocessing layout, but starts from an
existing preprocessed dataset. It preserves all original points and appends
additional copies of points inside a literature-defined wake box.
"""

import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from noether.data.datasets.cfd.shapenet_car.filemap import SHAPENET_CAR_FILEMAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_SIMULATION_COUNT = 889
NUM_PARAM_FOLDERS = 9
PREPROCESSED_FOLDER_NAME = "preprocessed"
DEFAULT_WAKE_OVERSAMPLING_FACTOR = 3
DEFAULT_WAKE_BOX_LWH = (0.47, 0.43, 0.31)
DEFAULT_WAKE_AXES = (2, 0, 1)  # ShapeNet-Car stores streamwise/spanwise/road-normal as z/x/y.


def resolve_preprocessed_root(root: Path) -> Path:
    """
    Resolve the source root to the preprocessed folder.

    Args:
        root: Dataset root or its preprocessed subfolder

    Returns:
        Path to the preprocessed data folder

    Raises:
        FileNotFoundError: If the resolved folder does not exist
    """
    preprocessed_root = root if root.name == PREPROCESSED_FOLDER_NAME else root / PREPROCESSED_FOLDER_NAME
    if not preprocessed_root.exists():
        raise FileNotFoundError(f"Preprocessed data folder does not exist: {preprocessed_root.as_posix()}")
    return preprocessed_root


def get_simulation_relative_paths(root: Path) -> list[Path]:
    """
    Get relative paths to all valid preprocessed simulation directories.

    Args:
        root: Path to the preprocessed dataset folder

    Returns:
        List of Path objects representing relative paths to valid simulations

    Raises:
        ValueError: If the expected number of simulations is not found
        FileNotFoundError: If a parameter folder does not exist
    """
    if not root.is_absolute():
        raise ValueError(f"Root path must be absolute: {root}")

    simulation_relative_paths = []
    for param_idx in range(NUM_PARAM_FOLDERS):
        param_folder = f"param{param_idx}"
        param_path = root / param_folder

        if not param_path.exists():
            raise FileNotFoundError(f"Parameter folder does not exist: {param_path}")

        for simulation_name in sorted(os.listdir(param_path)):
            simulation_path = param_path / simulation_name
            if simulation_path.is_dir():
                simulation_relative_paths.append(simulation_path.relative_to(root))

    if len(simulation_relative_paths) != EXPECTED_SIMULATION_COUNT:
        raise ValueError(f"Expected {EXPECTED_SIMULATION_COUNT} simulations but found {len(simulation_relative_paths)}")
    return simulation_relative_paths


def load_preprocessed_data(
    root: Path, simulation_relative_path: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load preprocessed tensors for a single simulation.

    Args:
        root: Root directory containing preprocessed data
        simulation_relative_path: Relative path to the simulation directory

    Returns:
        Tuple of (surface_pressure, surface_position, surface_normals,
                 volume_position, volume_velocity, volume_sdf, volume_normals)
    """
    sample_path = root / simulation_relative_path
    try:
        surface_pressure = torch.load(sample_path / SHAPENET_CAR_FILEMAP.surface_pressure, weights_only=True)
        surface_position = torch.load(sample_path / SHAPENET_CAR_FILEMAP.surface_position, weights_only=True)
        surface_normals = torch.load(sample_path / SHAPENET_CAR_FILEMAP.surface_normals, weights_only=True)
        volume_position = torch.load(sample_path / SHAPENET_CAR_FILEMAP.volume_position, weights_only=True)
        volume_velocity = torch.load(sample_path / SHAPENET_CAR_FILEMAP.volume_velocity, weights_only=True)
        volume_sdf = torch.load(sample_path / SHAPENET_CAR_FILEMAP.volume_distance_to_surface, weights_only=True)
        volume_normals = torch.load(sample_path / SHAPENET_CAR_FILEMAP.volume_normals, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessed sample {sample_path}: {e}") from e

    if not (surface_position.shape[0] == surface_pressure.shape[0] == surface_normals.shape[0]):
        raise ValueError(
            f"Surface data shape mismatch in {simulation_relative_path}: "
            f"position={surface_position.shape[0]}, pressure={surface_pressure.shape[0]}, "
            f"normals={surface_normals.shape[0]}"
        )
    if not (
        volume_position.shape[0] == volume_velocity.shape[0] == volume_sdf.shape[0] == volume_normals.shape[0]
    ):
        raise ValueError(
            f"Volume data shape mismatch in {simulation_relative_path}: "
            f"position={volume_position.shape[0]}, velocity={volume_velocity.shape[0]}, "
            f"sdf={volume_sdf.shape[0]}, normals={volume_normals.shape[0]}"
        )

    return (
        surface_pressure,
        surface_position,
        surface_normals,
        volume_position,
        volume_velocity,
        volume_sdf,
        volume_normals,
    )


def compute_wake_mask(
    surface_position: torch.Tensor,
    volume_position: torch.Tensor,
    wake_box_lwh: tuple[float, float, float] = DEFAULT_WAKE_BOX_LWH,
    wake_axes: tuple[int, int, int] = DEFAULT_WAKE_AXES,
) -> torch.Tensor:
    """
    Compute the Aultman & Duan wake-box mask in raw ShapeNet-Car coordinates.

    Args:
        surface_position: Surface point positions
        volume_position: Volume point positions
        wake_box_lwh: Wake-box dimensions normalized by body length
        wake_axes: Coordinate axes in streamwise, spanwise, and road-normal order

    Returns:
        Boolean mask selecting volume points inside the wake box
    """
    if surface_position.ndim != 2 or volume_position.ndim != 2:
        raise ValueError("surface_position and volume_position must have shape (N, D).")
    if surface_position.shape[-1] != volume_position.shape[-1]:
        raise ValueError("surface_position and volume_position must have the same coordinate dimension.")
    if len(wake_axes) != 3 or len(set(wake_axes)) != 3:
        raise ValueError(f"wake_axes must contain three unique coordinate indices, got {wake_axes}.")
    if min(wake_axes) < 0 or max(wake_axes) >= volume_position.shape[-1]:
        raise ValueError(f"wake_axes {wake_axes} are invalid for position dimension {volume_position.shape[-1]}.")
    if any(length <= 0.0 for length in wake_box_lwh):
        raise ValueError(f"wake_box_lwh values must be positive, got {wake_box_lwh}.")

    stream_axis, span_axis, vertical_axis = wake_axes
    surface_min = surface_position.amin(dim=0)
    surface_max = surface_position.amax(dim=0)
    body_length = surface_max[stream_axis] - surface_min[stream_axis]
    if body_length <= 0.0:
        return torch.zeros(volume_position.shape[0], dtype=torch.bool, device=volume_position.device)

    rear = surface_max[stream_axis]
    span_center = (surface_min[span_axis] + surface_max[span_axis]) / 2.0
    ground = surface_min[vertical_axis]

    wake_length, wake_width, wake_height = wake_box_lwh
    stream = volume_position[:, stream_axis]
    span = volume_position[:, span_axis]
    vertical = volume_position[:, vertical_axis]

    return (
        (stream >= rear)
        & (stream <= rear + wake_length * body_length)
        & (span >= span_center - 0.5 * wake_width * body_length)
        & (span <= span_center + 0.5 * wake_width * body_length)
        & (vertical >= ground)
        & (vertical <= ground + wake_height * body_length)
    )


def oversample_wake_data(
    surface_position: torch.Tensor,
    volume_position: torch.Tensor,
    volume_velocity: torch.Tensor,
    volume_sdf: torch.Tensor,
    volume_normals: torch.Tensor,
    wake_oversampling_factor: int = DEFAULT_WAKE_OVERSAMPLING_FACTOR,
    wake_box_lwh: tuple[float, float, float] = DEFAULT_WAKE_BOX_LWH,
    wake_axes: tuple[int, int, int] = DEFAULT_WAKE_AXES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Append copies of wake-region volume points and matching fields.

    Args:
        surface_position: Surface point positions used to define the body bbox
        volume_position: Volume point positions
        volume_velocity: Volume velocity values
        volume_sdf: Signed distance field values
        volume_normals: Volume normal vectors
        wake_oversampling_factor: Target multiplicity for wake points
        wake_box_lwh: Wake-box dimensions normalized by body length
        wake_axes: Coordinate axes in streamwise, spanwise, and road-normal order

    Returns:
        Tuple of oversampled (volume_position, volume_velocity, volume_sdf, volume_normals, wake_mask)
    """
    if wake_oversampling_factor < 1:
        raise ValueError("wake_oversampling_factor must be >= 1.")

    wake_mask = compute_wake_mask(
        surface_position=surface_position,
        volume_position=volume_position,
        wake_box_lwh=wake_box_lwh,
        wake_axes=wake_axes,
    )
    if wake_oversampling_factor == 1 or not wake_mask.any():
        return volume_position, volume_velocity, volume_sdf, volume_normals, wake_mask

    wake_indices = torch.nonzero(wake_mask, as_tuple=False).flatten()
    repeat_indices = wake_indices.repeat(wake_oversampling_factor - 1)
    return (
        torch.cat([volume_position, volume_position[repeat_indices]], dim=0),
        torch.cat([volume_velocity, volume_velocity[repeat_indices]], dim=0),
        torch.cat([volume_sdf, volume_sdf[repeat_indices]], dim=0),
        torch.cat([volume_normals, volume_normals[repeat_indices]], dim=0),
        wake_mask,
    )


def save_preprocessed_data(
    save_path: Path,
    surface_pressure: torch.Tensor,
    surface_position: torch.Tensor,
    surface_normals: torch.Tensor,
    volume_position: torch.Tensor,
    volume_velocity: torch.Tensor,
    volume_sdf: torch.Tensor,
    volume_normals: torch.Tensor,
) -> None:
    """
    Save wake-oversampled preprocessed simulation data to disk.

    Args:
        save_path: Directory where preprocessed data will be saved
        surface_pressure: Surface pressure values
        surface_position: Surface point positions
        surface_normals: Surface normal vectors
        volume_position: Oversampled volume point positions
        volume_velocity: Oversampled velocity values
        volume_sdf: Oversampled signed distance field values
        volume_normals: Oversampled normal vectors
    """
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(surface_pressure, save_path / SHAPENET_CAR_FILEMAP.surface_pressure)  # type: ignore[operator]
    torch.save(surface_position, save_path / SHAPENET_CAR_FILEMAP.surface_position)  # type: ignore[operator]
    torch.save(surface_normals, save_path / SHAPENET_CAR_FILEMAP.surface_normals)  # type: ignore[operator]
    torch.save(volume_velocity, save_path / SHAPENET_CAR_FILEMAP.volume_velocity)  # type: ignore[operator]
    torch.save(volume_position, save_path / SHAPENET_CAR_FILEMAP.volume_position)  # type: ignore[operator]
    torch.save(volume_normals, save_path / SHAPENET_CAR_FILEMAP.volume_normals)  # type: ignore[operator]
    torch.save(volume_sdf, save_path / SHAPENET_CAR_FILEMAP.volume_distance_to_surface)  # type: ignore[operator]


def process_single_simulation(
    root: Path,
    simulation_relative_path: Path,
    output_dir: Path,
    wake_oversampling_factor: int = DEFAULT_WAKE_OVERSAMPLING_FACTOR,
    wake_box_lwh: tuple[float, float, float] = DEFAULT_WAKE_BOX_LWH,
    wake_axes: tuple[int, int, int] = DEFAULT_WAKE_AXES,
) -> dict[str, int]:
    """
    Process a single simulation: load, wake-oversample, validate, and save.

    Args:
        root: Root directory containing preprocessed data
        simulation_relative_path: Relative path to the simulation
        output_dir: Output directory for the new preprocessed data
        wake_oversampling_factor: Target multiplicity for wake points
        wake_box_lwh: Wake-box dimensions normalized by body length
        wake_axes: Coordinate axes in streamwise, spanwise, and road-normal order

    Returns:
        Dictionary with point-count statistics for the processed sample
    """
    (
        surface_pressure,
        surface_position,
        surface_normals,
        volume_position,
        volume_velocity,
        volume_sdf,
        volume_normals,
    ) = load_preprocessed_data(root, simulation_relative_path)

    original_volume_points = volume_position.shape[0]
    (
        volume_position,
        volume_velocity,
        volume_sdf,
        volume_normals,
        wake_mask,
    ) = oversample_wake_data(
        surface_position=surface_position,
        volume_position=volume_position,
        volume_velocity=volume_velocity,
        volume_sdf=volume_sdf,
        volume_normals=volume_normals,
        wake_oversampling_factor=wake_oversampling_factor,
        wake_box_lwh=wake_box_lwh,
        wake_axes=wake_axes,
    )

    save_path = output_dir / PREPROCESSED_FOLDER_NAME / simulation_relative_path
    save_preprocessed_data(
        save_path,
        surface_pressure,
        surface_position,
        surface_normals,
        volume_position,
        volume_velocity,
        volume_sdf,
        volume_normals,
    )

    return {
        "original_volume_points": original_volume_points,
        "wake_points": int(wake_mask.sum().item()),
        "oversampled_volume_points": volume_position.shape[0],
    }


def main(
    root: Path,
    output_dir: Path,
    continue_on_error: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
    wake_oversampling_factor: int = DEFAULT_WAKE_OVERSAMPLING_FACTOR,
    wake_box_lwh: tuple[float, float, float] = DEFAULT_WAKE_BOX_LWH,
    wake_axes: tuple[int, int, int] = DEFAULT_WAKE_AXES,
) -> dict[str, int | float]:
    """
    Create a wake-oversampled ShapeNet car dataset.

    Args:
        root: Path to the source ShapeNet-Car dataset or its preprocessed folder
        output_dir: Path to the output directory where preprocessed data will be saved
        continue_on_error: If True, continue processing on errors instead of stopping
        dry_run: If True, only validate and count data without saving
        overwrite: If True, allow overwriting existing output directory
        wake_oversampling_factor: Target multiplicity for wake points
        wake_box_lwh: Wake-box dimensions normalized by body length
        wake_axes: Coordinate axes in streamwise, spanwise, and road-normal order

    Returns:
        Dictionary with processing and point-count statistics

    Raises:
        FileNotFoundError: If root directory does not exist
        FileExistsError: If output directory exists and overwrite is False
    """
    if wake_oversampling_factor < 1:
        raise ValueError("wake_oversampling_factor must be >= 1.")
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if output_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    preprocessed_root = resolve_preprocessed_root(root)

    logger.info(f"Source: {preprocessed_root.as_posix()}")
    logger.info(f"Destination: {output_dir.as_posix()}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Continue on error: {continue_on_error}")
    logger.info(f"Wake oversampling factor: {wake_oversampling_factor}")
    logger.info(f"Wake box LWH: {wake_box_lwh}")
    logger.info(f"Wake axes: {wake_axes}")

    simulation_relative_paths = get_simulation_relative_paths(preprocessed_root)
    logger.info(f"Found {len(simulation_relative_paths)} simulations")

    stats: dict[str, int | float] = {
        "total": len(simulation_relative_paths),
        "success": 0,
        "failed": 0,
        "original_volume_points": 0,
        "wake_points": 0,
        "oversampled_volume_points": 0,
    }
    failed_simulations = []

    for simulation_relative_path in tqdm(simulation_relative_paths, desc="Processing simulations", unit="sim"):
        try:
            if dry_run:
                (
                    _surface_pressure,
                    surface_position,
                    _surface_normals,
                    volume_position,
                    volume_velocity,
                    volume_sdf,
                    volume_normals,
                ) = load_preprocessed_data(preprocessed_root, simulation_relative_path)
                (
                    oversampled_position,
                    _oversampled_velocity,
                    _oversampled_sdf,
                    _oversampled_normals,
                    wake_mask,
                ) = oversample_wake_data(
                    surface_position=surface_position,
                    volume_position=volume_position,
                    volume_velocity=volume_velocity,
                    volume_sdf=volume_sdf,
                    volume_normals=volume_normals,
                    wake_oversampling_factor=wake_oversampling_factor,
                    wake_box_lwh=wake_box_lwh,
                    wake_axes=wake_axes,
                )
                sample_stats = {
                    "original_volume_points": volume_position.shape[0],
                    "wake_points": int(wake_mask.sum().item()),
                    "oversampled_volume_points": oversampled_position.shape[0],
                }
            else:
                sample_stats = process_single_simulation(
                    preprocessed_root,
                    simulation_relative_path,
                    output_dir,
                    wake_oversampling_factor=wake_oversampling_factor,
                    wake_box_lwh=wake_box_lwh,
                    wake_axes=wake_axes,
                )

            stats["success"] += 1
            stats["original_volume_points"] += sample_stats["original_volume_points"]
            stats["wake_points"] += sample_stats["wake_points"]
            stats["oversampled_volume_points"] += sample_stats["oversampled_volume_points"]
            logger.debug(f"Successfully processed: {simulation_relative_path}")
        except Exception as e:
            stats["failed"] += 1
            failed_simulations.append((simulation_relative_path, str(e)))
            logger.error(f"Failed to process {simulation_relative_path}: {e}")
            if not continue_on_error:
                logger.error("Stopping due to error. Use --continue-on-error to skip failed simulations.")
                raise

    if stats["original_volume_points"] > 0:
        stats["wake_fraction"] = stats["wake_points"] / stats["original_volume_points"]
        stats["volume_point_factor"] = stats["oversampled_volume_points"] / stats["original_volume_points"]

    logger.info("=" * 60)
    logger.info("Processing Summary:")
    logger.info(f"  Total simulations: {stats['total']}")
    logger.info(f"  Successfully processed: {stats['success']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Original volume points: {stats['original_volume_points']}")
    logger.info(f"  Wake points: {stats['wake_points']}")
    logger.info(f"  Oversampled volume points: {stats['oversampled_volume_points']}")
    logger.info(f"  Wake fraction: {stats.get('wake_fraction', 0.0):.6f}")
    logger.info(f"  Volume point factor: {stats.get('volume_point_factor', 0.0):.6f}")

    if failed_simulations:
        logger.warning(f"\nFailed simulations ({len(failed_simulations)}):")
        for sim_path, error in failed_simulations:
            logger.warning(f"  - {sim_path}: {error}")

    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a wake-oversampled ShapeNet car CFD dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Path to the source ShapeNet-Car dataset or its preprocessed folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory where the new preprocessed data will be saved",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining simulations if one fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count data without actually saving preprocessed files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output directory",
    )
    parser.add_argument(
        "--wake-oversampling-factor",
        type=int,
        default=DEFAULT_WAKE_OVERSAMPLING_FACTOR,
        help="Target multiplicity for points inside the wake box",
    )
    parser.add_argument(
        "--wake-box-lwh",
        type=float,
        nargs=3,
        default=DEFAULT_WAKE_BOX_LWH,
        metavar=("LENGTH", "WIDTH", "HEIGHT"),
        help="Wake-box dimensions normalized by body length",
    )
    parser.add_argument(
        "--wake-axes",
        type=int,
        nargs=3,
        default=DEFAULT_WAKE_AXES,
        metavar=("STREAMWISE", "SPANWISE", "ROAD_NORMAL"),
        help="Coordinate axes in streamwise, spanwise, and road-normal order",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    root = Path(args.root)
    output_dir = Path(args.output_dir)

    try:
        stats = main(
            root=root,
            output_dir=output_dir,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            wake_oversampling_factor=args.wake_oversampling_factor,
            wake_box_lwh=tuple(args.wake_box_lwh),
            wake_axes=tuple(args.wake_axes),
        )

        if stats["failed"] > 0:
            logger.warning(f"Completed with {stats['failed']} failures")
            sys.exit(1)
        logger.info("All simulations processed successfully")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error occurred")
        sys.exit(1)
