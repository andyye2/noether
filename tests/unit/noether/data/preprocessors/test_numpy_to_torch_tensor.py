#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import numpy as np
import pytest
import torch

from noether.data.preprocessors.numpy_to_torch_tensor import (
    NumpyToTorchTensorPreProcessor,
)


class TestNumpyToTorchTensorPreprocessor:
    @pytest.fixture
    def preprocessor(self) -> NumpyToTorchTensorPreProcessor:
        """Fixture for the NumpyToTorchTensorPreprocessor."""
        return NumpyToTorchTensorPreProcessor(normalization_key="numpy_to_tensor")

    def test_call_with_numpy_array(self, preprocessor: NumpyToTorchTensorPreProcessor):
        """Test that __call__ correctly converts a numpy array to a PyTorch tensor."""
        numpy_array = np.array([1, 2, 3], dtype=np.float32)
        tensor = preprocessor(numpy_array)

        assert isinstance(tensor, torch.Tensor)
        assert torch.equal(tensor, torch.from_numpy(numpy_array))
        assert tensor.dtype == torch.float32

    def test_call_with_non_numpy_array_raises_type_error(self, preprocessor: NumpyToTorchTensorPreProcessor):
        """Test that __call__ raises a TypeError for non-numpy array inputs."""
        non_numpy_input = [1, 2, 3]
        with pytest.raises(TypeError, match="Input must be a numpy array."):
            preprocessor(non_numpy_input)

    def test_call_with_different_dtypes(self, preprocessor: NumpyToTorchTensorPreProcessor):
        """Test conversion with various numpy dtypes."""
        test_cases = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([[1, 0], [0, 1]], dtype=np.bool_),
        ]

        for numpy_array in test_cases:
            tensor = preprocessor(numpy_array)
            assert isinstance(tensor, torch.Tensor)
            assert torch.equal(tensor, torch.from_numpy(numpy_array))

    def test_denormalize(self, preprocessor: NumpyToTorchTensorPreProcessor):
        """Test that denormalize converts a PyTorch tensor back to a numpy array."""
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = preprocessor(tensor)
        assert isinstance(out, torch.Tensor)
        numpy_array = preprocessor.denormalize(out)
        assert isinstance(numpy_array, np.ndarray)
