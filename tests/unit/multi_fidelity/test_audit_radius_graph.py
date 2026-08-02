from __future__ import annotations

import pytest
import torch

from research.multi_fidelity.tools.audit_radius_graph import summarize_degrees


def test_summarize_degrees_reports_zero_and_cap_fractions() -> None:
    summary = summarize_degrees(torch.tensor([0, 1, 32, 32]), max_degree=32)
    assert summary["supernodes"] == 4
    assert summary["zero_fraction"] == pytest.approx(0.25)
    assert summary["cap_fraction"] == pytest.approx(0.5)
    assert summary["histogram"][32] == 2


def test_summarize_degrees_rejects_values_above_model_cap() -> None:
    with pytest.raises(ValueError, match="must lie"):
        summarize_degrees(torch.tensor([33]), max_degree=32)
