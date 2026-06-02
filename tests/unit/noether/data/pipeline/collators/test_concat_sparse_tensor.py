#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.collators.concat_sparse_tensor import ConcatSparseTensorCollator


@pytest.fixture
def sample_data():
    return [
        {"feature": torch.tensor([1, 2, 3]), "label": torch.tensor([0])},
        {"feature": torch.tensor([4, 5]), "label": torch.tensor([1])},
    ]


def test_concat_sparse_tensor_collator_without_batch_idx(sample_data):
    collator = ConcatSparseTensorCollator(items=["feature", "label"], create_batch_idx=False)
    result = collator(sample_data)

    assert "feature" in result
    assert "label" in result
    assert "batch_idx" not in result

    expected_feature = torch.tensor([1, 2, 3, 4, 5])
    expected_label = torch.tensor([0, 1])

    assert torch.equal(result["feature"], expected_feature)
    assert torch.equal(result["label"], expected_label)


def test_concat_sparse_tensor_collator_with_batch_idx(sample_data):
    collator = ConcatSparseTensorCollator(items=["feature", "label"], create_batch_idx=True)
    result = collator(sample_data)

    assert "feature" in result
    assert "label" in result
    assert "batch_idx" in result

    expected_feature = torch.tensor([1, 2, 3, 4, 5])
    expected_label = torch.tensor([0, 1])
    expected_batch_idx = torch.tensor([0, 0, 0, 1, 1])

    assert torch.equal(result["feature"], expected_feature)
    assert torch.equal(result["label"], expected_label)
    assert torch.equal(result["batch_idx"], expected_batch_idx)


def test_concat_sparse_tensor_collator_with_custom_batch_idx(sample_data):
    collator = ConcatSparseTensorCollator(
        items=["feature", "label"],
        create_batch_idx=True,
        batch_idx_key="custom_batch_idx",
    )
    result = collator(sample_data)

    assert "feature" in result
    assert "label" in result
    assert "batch_idx" not in result
    assert "custom_batch_idx" in result

    expected_feature = torch.tensor([1, 2, 3, 4, 5])
    expected_label = torch.tensor([0, 1])
    expected_batch_idx = torch.tensor([0, 0, 0, 1, 1])

    assert torch.equal(result["feature"], expected_feature)
    assert torch.equal(result["label"], expected_label)
    assert torch.equal(result["custom_batch_idx"], expected_batch_idx)


def test_concat_sparse_tensor_collator_empty_samples():
    collator = ConcatSparseTensorCollator(items=["feature"], create_batch_idx=True)
    with pytest.raises(AssertionError):
        collator([])
