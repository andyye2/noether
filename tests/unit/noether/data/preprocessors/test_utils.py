#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.preprocessors.utils import to_tensor


def test_to_tensor_with_none():
    """
    Test that to_tensor returns None when the input is None.
    """
    with pytest.raises(TypeError):
        to_tensor(None)


def test_to_tensor_with_tensor_input():
    """
    Test that to_tensor returns the original tensor if the input is already a tensor.
    """
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output_tensor = to_tensor(input_tensor)
    assert output_tensor is input_tensor


def test_to_tensor_with_list_input():
    """
    Test that to_tensor correctly converts a list of floats to a tensor.
    """
    input_list = [1.0, 2.0, 3.0]
    expected_tensor = torch.tensor(input_list)
    output_tensor = to_tensor(input_list)
    assert isinstance(output_tensor, torch.Tensor)
    assert torch.equal(output_tensor, expected_tensor)


def test_to_tensor_with_tuple_input():
    """
    Test that to_tensor correctly converts a tuple of floats to a tensor.
    """
    input_tuple = (1.0, 2.0, 3.0)
    expected_tensor = torch.tensor(input_tuple)
    output_tensor = to_tensor(input_tuple)
    assert isinstance(output_tensor, torch.Tensor)
    assert torch.equal(output_tensor, expected_tensor)


@pytest.mark.parametrize("unsupported_data", ["not a sequence", {"key": "value"}])
def test_to_tensor_with_unsupported_type_raises_type_error(unsupported_data):
    """
    Test that to_tensor raises a TypeError for unsupported data types.
    """
    with pytest.raises(TypeError):
        to_tensor(unsupported_data)


def test_to_tensor_with_multiple_dim():
    """
    Test that to_tensor correctly converts a tuple of floats to a tensor.
    """
    input_tuple = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    expected_tensor = torch.tensor(input_tuple)
    output_tensor = to_tensor(input_tuple)
    assert isinstance(output_tensor, torch.Tensor)
    assert torch.equal(output_tensor, expected_tensor)


def test_to_tensor_with_multiple_dim_unmatching():
    """
    Test that to_tensor correctly converts a tuple of floats to a tensor.
    """
    input_tuple = [[1.0, 2.0, 3.0], [4.0, 5.0]]
    with pytest.raises(ValueError):
        to_tensor(input_tuple)
