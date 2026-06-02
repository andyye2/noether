#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.extractors.forward_hook import ForwardHook, StopForwardException


def test_hook_stores_output_generated_key():
    """Test that output is stored using a generated UUID key."""
    outputs = dict()
    hook = ForwardHook(outputs=outputs)

    # Simulate a forward hook call (module, input, output):
    fake_output = "tensor_output"
    hook(None, None, fake_output)

    assert len(outputs) == 1
    # We don't know the UUID, so we check values:
    assert list(outputs.values())[0] == fake_output  # noqa: RUF015
    # Ensure the key stored in the hook matches the dict key:
    assert outputs[hook.key] == fake_output


def test_hook_stores_output_custom_key():
    """Test that output is stored using a provided custom key."""
    outputs = dict()
    custom_key = "layer_1"
    hook = ForwardHook(outputs=outputs, key=custom_key)

    hook(None, None, "data")

    assert outputs["layer_1"] == "data"
    assert hook.key == "layer_1"


def test_hook_raises_exception():
    """Test that the hook raises StopForwardException when configured."""
    outputs = dict()
    hook = ForwardHook(outputs=outputs, raise_exception=True)

    with pytest.raises(StopForwardException):
        hook(None, None, "data")

    # Even if it crashes, it should have stored the data first:
    assert len(outputs) == 1
    assert list(outputs.values())[0] == "data"  # noqa: RUF015


def test_hook_disabled():
    """Test that a disabled hook does nothing."""
    outputs = dict()
    hook = ForwardHook(outputs=outputs, raise_exception=True)
    hook.enabled = False

    # Should NOT raise exception because it returns early:
    hook(None, None, "data")

    # Should NOT store data
    assert len(outputs) == 0


def test_multiple_hooks_shared_dict():
    """Test multiple hooks writing to the same dictionary."""
    outputs = dict()
    h1 = ForwardHook(outputs, key="h1")
    h2 = ForwardHook(outputs, key="h2")

    h1(None, None, 1)
    h2(None, None, 2)

    assert outputs == {"h1": 1, "h2": 2}
