#  Copyright (c) 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.data.datasets.cfd.shapenet_car.preprocessing_wake_oversampling import (
    compute_wake_mask,
    oversample_wake_data,
)


def test_compute_wake_mask_uses_shapenet_streamwise_z_axis() -> None:
    surface_position = torch.tensor(
        [
            [-1.0, 0.0, -2.0],
            [1.0, 1.0, 2.0],
        ]
    )
    volume_position = torch.tensor(
        [
            [0.0, 0.5, 2.5],  # downstream in z
            [0.0, 0.5, 1.9],  # inside body length, not wake
            [1.5, 0.5, 2.5],  # spanwise outside
            [0.0, 2.0, 2.5],  # road-normal outside
            [0.0, 0.5, 4.5],  # beyond 0.47L downstream
        ]
    )

    mask = compute_wake_mask(surface_position, volume_position)

    assert mask.tolist() == [True, False, False, False, False]


def test_oversample_wake_data_preserves_original_and_appends_aligned_wake_fields() -> None:
    surface_position = torch.tensor(
        [
            [-1.0, 0.0, -2.0],
            [1.0, 1.0, 2.0],
        ]
    )
    volume_position = torch.tensor(
        [
            [0.0, 0.5, 2.5],
            [0.5, 0.5, 2.7],
            [0.0, 0.5, 0.0],
            [1.5, 0.5, 2.5],
        ]
    )
    volume_velocity = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    volume_sdf = torch.arange(4, dtype=torch.float32)
    volume_normals = -volume_velocity

    (
        oversampled_position,
        oversampled_velocity,
        oversampled_sdf,
        oversampled_normals,
        wake_mask,
    ) = oversample_wake_data(
        surface_position=surface_position,
        volume_position=volume_position,
        volume_velocity=volume_velocity,
        volume_sdf=volume_sdf,
        volume_normals=volume_normals,
        wake_oversampling_factor=3,
    )

    assert wake_mask.tolist() == [True, True, False, False]
    assert oversampled_position.shape[0] == 8
    assert torch.equal(oversampled_position[:4], volume_position)
    assert torch.equal(oversampled_velocity[:4], volume_velocity)
    assert torch.equal(oversampled_sdf[:4], volume_sdf)
    assert torch.equal(oversampled_normals[:4], volume_normals)
    assert torch.equal(oversampled_position[4:], volume_position[[0, 1, 0, 1]])
    assert torch.equal(oversampled_velocity[4:], volume_velocity[[0, 1, 0, 1]])
    assert torch.equal(oversampled_sdf[4:], volume_sdf[[0, 1, 0, 1]])
    assert torch.equal(oversampled_normals[4:], volume_normals[[0, 1, 0, 1]])


def test_oversample_wake_data_factor_one_keeps_tensors_unchanged() -> None:
    surface_position = torch.tensor([[-1.0, 0.0, -2.0], [1.0, 1.0, 2.0]])
    volume_position = torch.tensor([[0.0, 0.5, 2.5], [0.0, 0.5, 0.0]])
    volume_velocity = torch.ones(2, 3)
    volume_sdf = torch.ones(2)
    volume_normals = torch.zeros(2, 3)

    outputs = oversample_wake_data(
        surface_position=surface_position,
        volume_position=volume_position,
        volume_velocity=volume_velocity,
        volume_sdf=volume_sdf,
        volume_normals=volume_normals,
        wake_oversampling_factor=1,
    )

    assert torch.equal(outputs[0], volume_position)
    assert torch.equal(outputs[1], volume_velocity)
    assert torch.equal(outputs[2], volume_sdf)
    assert torch.equal(outputs[3], volume_normals)
    assert outputs[4].tolist() == [True, False]
