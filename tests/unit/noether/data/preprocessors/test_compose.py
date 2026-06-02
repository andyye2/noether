#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.preprocessors import ComposePreProcess, PreProcessor


class MockPreProcessor(PreProcessor):
    """A mock preprocessor for testing purposes."""

    def __init__(self, operation, inverse_operation, normalization_key="test"):
        super().__init__(normalization_key)
        self.operation = operation
        self.inverse_operation = inverse_operation

    def __call__(self, x):
        return self.operation(x)

    def denormalize(self, x):
        # Not needed for these tests, but required by the abstract base class
        return self.inverse_operation(x)


class MockPreProcessorWithoutDenorm(PreProcessor):
    """A mock preprocessor for testing purposes."""

    def __init__(self, operation, normalization_key="test"):
        super().__init__(normalization_key)
        self.operation = operation

    def __call__(self, x):
        return self.operation(x)

    # No denormalize method, to test behavior when not all preprocessors can be denormalized


def test_call_applies_preprocessors_in_sequence():
    """
    Tests that the __call__ method applies a sequence of preprocessors in the correct order.
    """
    # Arrange
    preprocessors = [
        MockPreProcessor(lambda x: x + 5, lambda x: x - 5),  # 10 -> 15// 15 -> 10
        MockPreProcessor(lambda x: x * 2, lambda x: x / 2),  # 15 -> 30 // 30 -> 15
        MockPreProcessor(lambda x: x - 1, lambda x: x + 1),  # 30 -> 29// 29 -> 30
    ]
    composer = ComposePreProcess(normalization_key="test", preprocessors=preprocessors)
    initial_data = 10
    expected_data = 29

    # Act
    result = composer(initial_data)

    # Assert
    assert result == expected_data

    assert composer.inverse(result) == initial_data


def test_call_with_torch_tensor():
    """
    Tests that the __call__ method works correctly with torch tensors.
    """
    # Arrange
    preprocessors = [
        MockPreProcessor(lambda x: x + torch.tensor([1.0, 2.0]), lambda x: x - torch.tensor([1.0, 2.0])),
        MockPreProcessor(lambda x: x * torch.tensor([3.0, 0.5]), lambda x: x / torch.tensor([3.0, 0.5])),
    ]
    composer = ComposePreProcess(normalization_key="test", preprocessors=preprocessors)
    initial_tensor = torch.tensor([10.0, 20.0])
    # Step 1: [10.0, 20.0] + [1.0, 2.0] = [11.0, 22.0]
    # Step 2: [11.0, 22.0] * [3.0, 0.5] = [33.0, 11.0]
    expected_tensor = torch.tensor([33.0, 11.0])

    # Act
    result_tensor = composer(initial_tensor)

    # Assert
    assert torch.equal(result_tensor, expected_tensor)

    assert torch.equal(composer.inverse(result_tensor), initial_tensor)


def test_call_with_empty_preprocessors_list():
    """
    Tests that calling a composer with an empty list of preprocessors returns the input unchanged.
    """
    # Arrange
    with pytest.raises(ValueError):
        ComposePreProcess(normalization_key="test", preprocessors=[])


def test_call_with_non_denormalizablefunction_items():
    preprocessors = [
        MockPreProcessor(lambda x: x + torch.tensor([1.0, 2.0]), lambda x: x - torch.tensor([1.0, 2.0])),
        MockPreProcessorWithoutDenorm(lambda x: x * torch.tensor([3.0, 0.5])),
    ]
    composer = ComposePreProcess(normalization_key="test", preprocessors=preprocessors)
    initial_tensor = torch.tensor([10.0, 20.0])
    with pytest.raises(NotImplementedError):
        composer.inverse(initial_tensor)  # Should not raise an error
