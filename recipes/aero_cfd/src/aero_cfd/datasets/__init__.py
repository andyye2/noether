# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Dataset adapters used by the aerodynamic CFD recipe."""

from .transfer_drivaerml import (
    TransferDrivAerMLDataset,
    TransferDrivAerMLDatasetConfig,
    drivaer_to_shapenet_frame,
    transform_loaded_drivaerml_field,
)

__all__ = [
    "TransferDrivAerMLDataset",
    "TransferDrivAerMLDatasetConfig",
    "drivaer_to_shapenet_frame",
    "transform_loaded_drivaerml_field",
]
