#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.sample_processors import DuplicateKeysSampleProcessor


@pytest.fixture
def sample_data():
    return [
        {
            "surface_position": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "unchanged_field": torch.tensor([[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]),
        },
        {
            "surface_position": torch.tensor([[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]),
            "unchanged_field": torch.tensor([[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]),
        },
    ]


def test_processing(sample_data):
    precollator = DuplicateKeysSampleProcessor(key_map={"surface_position": "input_position"})
    output = []
    for sample in sample_data:
        output.append(precollator(sample))
    assert len(output) == len(sample_data)
    for sample in output:
        assert len(sample) == 3
        assert "input_position" in sample
        assert "surface_position" in sample
        assert "unchanged_field" in sample
        assert torch.all(sample["input_position"] == sample["surface_position"])


def test_denormalize_remap():
    precollator = DuplicateKeysSampleProcessor(key_map={"surface_position": "input_position"})
    value = torch.tensor([1, 2, 3])
    with pytest.raises(NotImplementedError):
        new_key, same_value = precollator.inverse(key="input_position", value=value)
