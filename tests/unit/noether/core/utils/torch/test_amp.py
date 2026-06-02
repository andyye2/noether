#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from noether.core.utils.torch import amp

MODULE_PATH = "noether.core.utils.torch.amp"


def test_get_supported_precision_fp32():
    """Float32 should always be supported and returned directly."""
    device = torch.device("cpu")
    assert amp.get_supported_precision("float32", device) == torch.float32
    assert amp.get_supported_precision("fp32", device) == torch.float32


@patch(f"{MODULE_PATH}.is_bfloat16_compatible")
def test_get_supported_precision_bf16(mock_check):
    """Test BF16 selection logic."""
    device = torch.device("cuda")

    # Device supports BF16
    mock_check.return_value = True
    assert amp.get_supported_precision("bfloat16", device) == torch.bfloat16
    assert amp.get_supported_precision("bf16", device) == torch.bfloat16  # alias

    # Device does NOT support BF16
    mock_check.return_value = False
    with pytest.raises(RuntimeError, match="bfloat16 not supported"):
        amp.get_supported_precision("bfloat16", device)


@patch(f"{MODULE_PATH}.is_float16_compatible")
def test_get_supported_precision_fp16(mock_check):
    """Test FP16 selection logic."""
    device = torch.device("cuda")

    # Device supports FP16
    mock_check.return_value = True
    assert amp.get_supported_precision("float16", device) == torch.float16
    assert amp.get_supported_precision("fp16", device) == torch.float16

    # Device does NOT support FP16
    mock_check.return_value = False
    with pytest.raises(RuntimeError, match="float16 not supported"):
        amp.get_supported_precision("float16", device)


def test_get_supported_precision_invalid():
    """Test invalid string input."""
    device = torch.device("cpu")
    with pytest.raises(AssertionError):
        amp.get_supported_precision("int8", device)


def test_is_compatible_success():
    """Test _is_compatible returns True when autocast succeeds."""
    device = torch.device("cpu")
    dtype = torch.float32

    # Mock torch.autocast to verify it enters context without error:
    with patch("torch.autocast") as mock_autocast:
        mock_autocast.return_value.__enter__.return_value = None
        assert amp.is_compatible(device, dtype) is True


def test_is_compatible_failure():
    """Test _is_compatible returns False when autocast raises RuntimeError."""
    device = torch.device("cpu")
    dtype = torch.float32

    with patch("torch.autocast") as mock_autocast:
        # Simulate hardware failure (e.g., bf16 on old CPU)
        mock_autocast.side_effect = RuntimeError("Not supported")
        assert amp.is_compatible(device, dtype) is False


def test_get_scaler_and_context_fp32():
    """FP32 should return NoopScaler and NoopContext."""
    device = torch.device("cpu")
    scaler, ctx = amp.get_grad_scaler_and_autocast_context(torch.float32, device)

    assert isinstance(scaler, amp.NoopGradScaler)
    assert isinstance(ctx, amp.NoopContext)


@patch("torch.autocast")
def test_get_scaler_and_context_bf16(mock_autocast):
    """BF16 should return NoopScaler (native support) and real Autocast."""
    device = torch.device("cpu")

    scaler, ctx = amp.get_grad_scaler_and_autocast_context(torch.bfloat16, device)

    # PyTorch documentation says BF16 doesn't need GradScaler:
    assert isinstance(scaler, amp.NoopGradScaler)

    # Verify torch.autocast was initialized correctly:
    mock_autocast.assert_called_with("cpu", dtype=torch.bfloat16)
    assert ctx == mock_autocast.return_value


@patch(f"{MODULE_PATH}.GradScaler")
@patch("torch.autocast")
def test_get_scaler_and_context_fp16_cuda(mock_autocast, MockGradScaler):
    """FP16 on CUDA should return Real Scaler and Real Autocast."""
    device = torch.device("cuda")

    scaler, ctx = amp.get_grad_scaler_and_autocast_context(torch.float16, device)

    # Check we got the Mock of the real scaler:
    assert isinstance(scaler, MagicMock)

    # Verify torch.autocast was initialized correctly:
    mock_autocast.assert_called_with("cuda", dtype=torch.float16)
    assert ctx == mock_autocast.return_value


@patch("torch.autocast")
def test_get_scaler_and_context_fp16_cpu(mock_autocast):
    """FP16 on CPU should return NoopScaler (CPU FP16 doesn't use scaler usually) and Autocast."""
    device = torch.device("cpu")

    scaler, ctx = amp.get_grad_scaler_and_autocast_context(torch.float16, device)

    # Code specifically handles str(device) == "cpu" -> NoopGradScaler
    assert isinstance(scaler, amp.NoopGradScaler)

    # Verify torch.autocast was initialized correctly:
    mock_autocast.assert_called_with("cpu", dtype=torch.float16)
    assert ctx == mock_autocast.return_value


def test_get_scaler_and_context_unsupported():
    """Unknown dtype should raise error."""
    device = torch.device("cpu")
    with pytest.raises(NotImplementedError):
        amp.get_grad_scaler_and_autocast_context(torch.int8, device)


def test_noop_classes_coverage():
    """Ensure Noop classes don't crash when methods are called."""
    with amp.NoopContext():
        pass

    scaler = amp.NoopGradScaler()
    mock_optim = MagicMock()

    # scale should return input unchanged:
    assert scaler.scale("tensor") == "tensor"

    # update/unscale should do nothing:
    scaler.update()
    scaler.unscale_(mock_optim)

    # step should call optimizer.step:
    scaler.step(mock_optim)
    mock_optim.step.assert_called_once()
