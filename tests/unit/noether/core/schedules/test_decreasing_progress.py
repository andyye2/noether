#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest

# We assume the environment is set up correctly to import the class.
# If this import fails, it's a path/installation issue, not a logic issue.
from noether.core.schedules.base import DecreasingProgressSchedule

# ==========================================
# 1. Local Config Mock
# ==========================================
# We don't need the real Pydantic config for testing the Schedule logic.
# We just need an object that has the attributes expected by __init__.


class MockConfig:
    def __init__(
        self,
        max_value=1.0,
        end_value=0.0,
        overhang_percent=None,
        overhang_steps=None,
        exclude_first=False,
        exclude_last=False,
    ) -> None:
        self.max_value = max_value
        self.end_value = end_value
        self.overhang_percent = overhang_percent
        self.overhang_steps = overhang_steps
        self.exclude_first = exclude_first
        self.exclude_last = exclude_last


# ==========================================
# 2. Concrete Implementation
# ==========================================


class LinearDecreasingSchedule(DecreasingProgressSchedule):
    """
    Concrete implementation for testing abstract DecreasingProgressSchedule.
    Implements simple linear progress: step / total_steps.
    """

    def _get_progress(self, step: int, total_steps: int) -> float:
        if total_steps == 0:
            return 1.0
        return step / total_steps


# ==========================================
# 3. Tests
# ==========================================


def test_init_delta_calculation():
    """Test that delta is correctly calculated as (end - start)."""
    # Config: Start=1.0, End=0.2
    config = MockConfig(max_value=1.0, end_value=0.2)
    schedule = LinearDecreasingSchedule(config)

    # ProgressSchedule sets start_value = max_value
    assert schedule.start_value == 1.0
    # Delta should be negative (0.2 - 1.0 = -0.8)
    assert schedule.delta == pytest.approx(-0.8)


def test_init_validation_error():
    """Test validation: end_value must be <= max_value."""
    # Invalid: Start=0.5, End=0.6 (Increasing, not decreasing)
    config = MockConfig(max_value=0.5, end_value=0.6)

    with pytest.raises(ValueError, match="end_value must be less than or equal"):
        LinearDecreasingSchedule(config)


def test_get_value_logic():
    """
    Test full calculation logic inherited from ProgressSchedule.
    Formula: start + progress * delta
    """
    # Config: Start=1.0, End=0.0 -> Delta = -1.0
    config = MockConfig(max_value=1.0, end_value=0.0)
    schedule = LinearDecreasingSchedule(config)

    # Step 0: progress 0.0 -> 1.0 + (0.0 * -1.0) = 1.0
    assert schedule.get_value(0, 10) == 1.0

    # Step 5: progress 0.5 -> 1.0 + (0.5 * -1.0) = 0.5
    assert schedule.get_value(5, 10) == 0.5

    # Step 10: progress 1.0 -> 1.0 + (1.0 * -1.0) = 0.0
    assert schedule.get_value(10, 10) == 0.0


def test_exclude_first():
    """Test 'exclude_first' logic from ProgressSchedule."""
    # If exclude_first=True, step becomes step+1
    config = MockConfig(max_value=1.0, end_value=0.0, exclude_first=True)
    schedule = LinearDecreasingSchedule(config)

    # Asking for step 0/10 becomes 1/11 internally
    # Calculation: 1 / 11 * -1.0 + 1.0 = 0.909...
    expected = 1.0 + (1 / 11 * -1.0)
    assert schedule.get_value(0, 10) == pytest.approx(expected)


def test_exclude_last():
    """Test 'exclude_last' logic from ProgressSchedule."""
    # If exclude_last=True, total becomes total+1
    config = MockConfig(max_value=1.0, end_value=0.0, exclude_last=True)
    schedule = LinearDecreasingSchedule(config)

    # Asking for step 10/10 becomes 10/11 internally
    # Calculation: 10 / 11 * -1.0 + 1.0 = 0.0909...
    expected = 1.0 + (10 / 11 * -1.0)
    assert schedule.get_value(10, 10) == pytest.approx(expected)


def test_overhang_logic():
    """Test 'overhang_steps' logic from ScheduleBase."""
    # Add 10 steps of overhang
    config = MockConfig(max_value=1.0, end_value=0.0, overhang_steps=10)
    schedule = LinearDecreasingSchedule(config)

    # Ask for step 10 out of 10 (normally 100% progress)
    # With overhang, total becomes 20.
    # Progress = 10 / 20 = 0.5
    # Value = 1.0 + (0.5 * -1.0) = 0.5
    assert schedule.get_value(10, 10) == 0.5


def test_conflicting_overhangs():
    """
    Test that ScheduleBase validation logic works correctly via inheritance.
    (Setting both overhang_percent and overhang_steps should raise ValueError).
    """
    config = MockConfig(max_value=1.0, overhang_percent=0.1, overhang_steps=5)

    with pytest.raises(ValueError, match="only one of overhang_percent and overhang_steps"):
        LinearDecreasingSchedule(config)
