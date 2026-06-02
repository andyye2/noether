#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.sample_processors import DropOutliersSampleProcessor


def test_processing_noaffecteditems():
    sample = {
        "surface_position": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "surface_pressure": torch.tensor([[19.0], [21.0], [24.0]]),
    }
    precollator = DropOutliersSampleProcessor(
        item="surface_pressure",
        max_value=21,
    )
    new_sample = precollator(sample)
    assert len(new_sample) == 2
    assert torch.all(new_sample["surface_position"] == torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert torch.all(new_sample["surface_pressure"] == torch.tensor([[19.0], [21.0]]))


def test_processing_affecteditems():
    sample = {
        "surface_position": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "surface_pressure": torch.tensor([[19.0], [21.0], [24.0]]),
    }
    precollator = DropOutliersSampleProcessor(
        item="surface_pressure",
        affected_items={"surface_position"},
        max_value=21,
    )
    new_sample = precollator(sample)
    assert len(new_sample) == 2
    assert torch.all(new_sample["surface_position"] == torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.all(new_sample["surface_pressure"] == torch.tensor([[19.0], [21.0]]))


def test_processing_affecteditems_minvalue_and_maxvalue():
    sample = {
        "surface_position": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "surface_pressure": torch.tensor([[19.0], [21.0], [24.0]]),
    }
    precollator = DropOutliersSampleProcessor(
        item="surface_pressure",
        affected_items={"surface_position"},
        min_value=20.0,
        max_value=21.0,
    )
    new_sample = precollator(sample)
    assert len(new_sample) == 2
    assert torch.all(new_sample["surface_position"] == torch.tensor([[3.0, 4.0]]))
    assert torch.all(new_sample["surface_pressure"] == torch.tensor([[21.0]]))


def test_processing_maxquantile():
    sample = {
        "surface_pressure": torch.tensor([[19.0], [21.0], [22.0], [24.0]]),
    }
    precollator = DropOutliersSampleProcessor(
        item="surface_pressure",
        max_quantile=0.5,
    )
    new_sample = precollator(sample)
    assert len(new_sample) == 1
    assert torch.all(new_sample["surface_pressure"] == torch.tensor([[19.0], [21.0]]))


def test_processing_minquantile():
    sample = {
        "surface_pressure": torch.tensor([[19.0], [21.0], [22.0], [24.0]]),
    }
    precollator = DropOutliersSampleProcessor(
        item="surface_pressure",
        min_quantile=0.5,
    )
    new_sample = precollator(sample)
    assert len(new_sample) == 1
    assert torch.all(new_sample["surface_pressure"] == torch.tensor([[22.0], [24.0]]))


def test_denormalize():
    precollator = DropOutliersSampleProcessor(
        item="surface_pressure",
        max_value=21,
    )
    value = torch.tensor([1, 2, 3])
    with pytest.raises(NotImplementedError):
        new_key, same_value = precollator.inverse(key="input_position", value=value)
