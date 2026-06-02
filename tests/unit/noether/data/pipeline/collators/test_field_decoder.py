#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.collators.field_decoder import FieldDecoderCollator


@pytest.fixture
def collator():
    return FieldDecoderCollator(position_item="positions", target_items=["targets"])


def test_collator_no_padding(collator):
    samples = [
        {"positions": torch.tensor([1.0, 2.0]), "targets": torch.tensor([0, 1])},
        {"positions": torch.tensor([3.0, 4.0]), "targets": torch.tensor([1, 0])},
    ]

    result = collator(samples)

    assert torch.equal(result["positions"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(result["targets"], torch.tensor([0, 1, 1, 0]))
    assert result["unbatch_mask_positions"] is None


def test_collator_with_padding(collator):
    samples = [
        {"positions": torch.tensor([1.0, 2.0]), "targets": torch.tensor([0, 1])},
        {"positions": torch.tensor([3.0, 4.0, 5.0]), "targets": torch.tensor([1, 0, 1])},
    ]

    result = collator(samples)

    assert torch.equal(result["positions"], torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]))
    assert torch.equal(result["targets"], torch.tensor([0, 1, 1, 0, 1]))
    assert torch.equal(
        result["unbatch_mask_positions"],
        torch.tensor([True, True, False, True, True, True]),
    )


def test_collator_mismatched_lengths(collator):
    samples = [
        {"positions": torch.tensor([1.0, 2.0]), "targets": torch.tensor([0, 1])},
        {"positions": torch.tensor([3.0]), "targets": torch.tensor([1])},
    ]

    result = collator(samples)

    assert torch.equal(result["positions"], torch.tensor([[1.0, 2.0], [3.0, 0.0]]))
    assert torch.equal(result["targets"], torch.tensor([0, 1, 1]))
    assert torch.equal(
        result["unbatch_mask_positions"],
        torch.tensor([True, True, True, False]),
    )


def test_collator_empty_samples(collator):
    samples = []

    with pytest.raises(AssertionError):
        collator(samples)
