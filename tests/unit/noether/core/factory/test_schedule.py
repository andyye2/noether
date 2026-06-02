#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from noether.core.factory.schedule import ScheduleFactory


@pytest.fixture
def mock_dependencies():
    """Patches the external ScheduleWrapper dependency."""
    with patch("noether.core.factory.schedule.ScheduleWrapper") as mock_wrapper:
        yield mock_wrapper


@pytest.fixture
def factory():
    return ScheduleFactory()


def test_create_none_returns_none(factory):
    """Test that passing None returns None."""
    assert factory.create(None) is None


def test_create_instantiates_and_wraps(factory, mock_dependencies):
    """
    Test the standard creation flow:
    1. Extract 'update_counter' from kwargs.
    2. Call instantiate() to get the raw schedule.
    3. Wrap it in ScheduleWrapper.
    """
    MockScheduleWrapper = mock_dependencies

    # Mock config object (e.g. AnyScheduleConfig):
    mock_config = SimpleNamespace(interval="step", kind="torch.optim.lr_scheduler.StepLR")

    # Mock internal instantiate to return a dummy schedule object:
    with patch.object(factory, "instantiate") as mock_instantiate:
        dummy_schedule = "raw_schedule_obj"
        mock_instantiate.return_value = dummy_schedule

        update_counter = MagicMock()
        result = factory.create(mock_config, update_counter=update_counter, extra_arg=123)

        # Verify instantiate was called with config and extra kwargs (update_counter removed):
        mock_instantiate.assert_called_once_with(mock_config, extra_arg=123)

        # Verify ScheduleWrapper was initialized correctly:
        MockScheduleWrapper.assert_called_once_with(
            schedule=dummy_schedule, update_counter=update_counter, interval="step"
        )

        assert result == MockScheduleWrapper.return_value


def test_create_without_update_counter(factory, mock_dependencies):
    """Test that update_counter defaults to None if not provided."""
    MockScheduleWrapper = mock_dependencies
    mock_config = SimpleNamespace(interval="epoch")

    with patch.object(factory, "instantiate") as mock_instantiate:
        mock_instantiate.return_value = "sched"

        factory.create(mock_config)

        # Check wrapper call
        call_args = MockScheduleWrapper.call_args[1]
        assert call_args["update_counter"] is None
        assert call_args["interval"] == "epoch"
