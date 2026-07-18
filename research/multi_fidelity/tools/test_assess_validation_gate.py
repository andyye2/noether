# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for the preregistered final-raw validation futility gate."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from .assess_validation_gate import assess_validation_gate


def _write_gate_csv(
    path: Path,
    *,
    one_better_cell: bool = False,
    replicate_ids: tuple[int, ...] = (0, 1, 2),
    n_values: tuple[int, ...] = (50, 100),
) -> None:
    """Write a complete 3-replicate, 2-N paired validation grid."""
    factors = {
        50: (1.20, 1.15, 1.25),
        100: (1.12, 1.18, 1.14),
    }
    rows: list[dict[str, str | int | float]] = []
    for method in ("S", "P-FT"):
        for replicate_index, replicate in enumerate(replicate_ids):
            for n_value in n_values:
                for design_id in range(4):
                    for field in ("surface_pressure", "volume_velocity"):
                        scratch_error = 0.4 * n_value**-0.2
                        scratch_error *= 1.0 + 0.01 * design_id
                        if field == "volume_velocity":
                            scratch_error *= 1.3
                        relative_l2 = scratch_error
                        if method == "P-FT":
                            factor = factors.get(n_value, (1.2,) * len(replicate_ids))[replicate_index]
                            if one_better_cell and replicate == 0 and n_value == 50:
                                factor = 0.95
                            relative_l2 *= factor
                        rows.append(
                            {
                                "method": method,
                                "replicate": replicate,
                                "n": n_value,
                                "design_id": design_id,
                                "field": field,
                                "relative_l2": relative_l2,
                            }
                        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_exact_stop_gate_and_provenance_contract(tmp_path: Path) -> None:
    """Stop for two high medians and exactly six of six worse cells."""
    csv_path = tmp_path / "stop.csv"
    _write_gate_csv(csv_path)

    report = assess_validation_gate(csv_path, confirm_final_raw_val=True)

    assert report["decision"] == "stop"
    assert report["gate"]["worse_than_scratch_cell_count"] == 6
    assert report["gate"]["all_six_paired_cells_worse_than_scratch"] is True
    assert report["gate"]["per_n"]["50"]["median_paired_ratio_transfer_over_scratch"] == pytest.approx(1.20)
    assert report["gate"]["per_n"]["100"]["median_paired_ratio_transfer_over_scratch"] == pytest.approx(1.14)
    assert report["endpoint_contract"]["split"] == "validation_only"
    assert report["endpoint_contract"]["checkpoint"] == "final"
    assert report["endpoint_contract"]["weight_stream"] == "raw_non_ema"
    assert report["endpoint_contract"]["test_data_allowed"] is False


def test_one_nonworse_cell_continues(tmp_path: Path) -> None:
    """Continue when the six-of-six condition fails, even with high medians."""
    csv_path = tmp_path / "continue.csv"
    _write_gate_csv(csv_path, one_better_cell=True)

    report = assess_validation_gate(csv_path, confirm_final_raw_val=True)

    assert report["gate"]["both_n_medians_meet_or_exceed_threshold"] is True
    assert report["gate"]["worse_than_scratch_cell_count"] == 5
    assert report["decision"] == "continue"


def test_unconfirmed_or_incomplete_gate_input_is_rejected(tmp_path: Path) -> None:
    """Require provenance affirmation and every paired CSV cell."""
    csv_path = tmp_path / "incomplete.csv"
    _write_gate_csv(csv_path)
    with pytest.raises(ValueError, match="provenance is unconfirmed"):
        assess_validation_gate(csv_path, confirm_final_raw_val=False)

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows.pop()
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="complete paired rectangular grid"):
        assess_validation_gate(csv_path, confirm_final_raw_val=True)


@pytest.mark.parametrize(
    ("replicate_ids", "n_values", "message"),
    [
        ((1, 2, 3), (50, 100), "replicate IDs"),
        ((0, 1, 2), (50, 100, 200), "exactly N"),
    ],
)
def test_gate_rejects_wrong_labels_or_extra_sample_sizes(
    tmp_path: Path,
    replicate_ids: tuple[int, ...],
    n_values: tuple[int, ...],
    message: str,
) -> None:
    """The gate accepts only the preregistered first-phase cells."""
