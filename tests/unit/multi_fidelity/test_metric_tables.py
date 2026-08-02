# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests for merging frozen metric exports and reporting the comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from research.multi_fidelity.tools.compare_arms import ArmKey, build_report
from research.multi_fidelity.tools.merge_metric_csvs import COLUMNS, merge_metric_csvs

FROZEN = "ps1000-sr9"
MATCHED = "ps1000-sr0.1"


def _row(method: str, rendering: str, design_id: int, field: str, relative_l2: float) -> dict[str, str]:
    """Return one metric row in the exported schema."""
    return {
        "method": method,
        "rendering": rendering,
        "replicate": "0",
        "n": "100",
        "design_id": str(design_id),
        "field": field,
        "relative_l2": repr(relative_l2),
        "mae": "0.5",
    }


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write one metric CSV in the exported schema."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_merge_is_deterministic_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Merging is order-independent and refuses to score a cell twice."""
    scratch = _write(tmp_path / "s.csv", [_row("S", FROZEN, 10, "surface_pressure", 0.2)])
    transfer = _write(tmp_path / "p.csv", [_row("P-FT", FROZEN, 10, "surface_pressure", 0.1)])

    result = merge_metric_csvs([scratch, transfer], tmp_path / "merged.csv")
    assert result["rows"] == 2
    assert (tmp_path / "merged.csv").read_text(encoding="utf-8").splitlines()[1].startswith("P-FT,")

    duplicate = _write(tmp_path / "dup.csv", [_row("S", FROZEN, 10, "surface_pressure", 0.2)])
    with pytest.raises(ValueError, match="duplicate metric key"):
        merge_metric_csvs([scratch, duplicate], tmp_path / "invalid.csv")


def test_two_renderings_of_one_method_are_distinct_rows(tmp_path: Path) -> None:
    """The rendering is part of the row identity, so the 2x2 design merges."""
    frozen = _write(tmp_path / "frozen.csv", [_row("S", FROZEN, 10, "surface_pressure", 0.2)])
    matched = _write(tmp_path / "matched.csv", [_row("S", MATCHED, 10, "surface_pressure", 0.15)])

    result = merge_metric_csvs([frozen, matched], tmp_path / "merged.csv")
    assert result["rows"] == 2


def test_report_aggregates_fields_geometrically_and_pairs_by_design(tmp_path: Path) -> None:
    """Field aggregation and contrasts follow the preregistered definitions."""
    rows: list[dict[str, str]] = []
    for design, (pressure, velocity) in enumerate([(0.2, 0.8), (0.4, 0.1)], start=10):
        rows.append(_row("S", FROZEN, design, "surface_pressure", pressure))
        rows.append(_row("S", FROZEN, design, "volume_velocity", velocity))
        rows.append(_row("P-FT", FROZEN, design, "surface_pressure", pressure / 2))
        rows.append(_row("P-FT", FROZEN, design, "volume_velocity", velocity / 2))
    csv_path = _write(tmp_path / "metrics.csv", rows)

    report = build_report([csv_path], baseline=ArmKey(method="S", rendering=FROZEN))
    scratch = report["arms"]["S@ps1000-sr9"]
    assert scratch["relative_l2"]["surface_pressure"] == pytest.approx((0.2 * 0.4) ** 0.5)
    assert scratch["macro_relative_l2"] == pytest.approx((0.2 * 0.8 * 0.4 * 0.1) ** 0.25)

    contrast = report["contrasts"]["P-FT@ps1000-sr9"]
    assert contrast["macro_error_ratio"] == pytest.approx(0.5)
    assert contrast["per_field_error_ratio"]["volume_velocity"] == pytest.approx(0.5)
    assert contrast["designs_improved"] == 2
    assert contrast["designs_total"] == 2


def test_report_refuses_an_incomplete_arm(tmp_path: Path) -> None:
    """A missing design or field would silently bias an aggregate."""
    rows = [
        _row("S", FROZEN, 10, "surface_pressure", 0.2),
        _row("S", FROZEN, 10, "volume_velocity", 0.3),
        _row("S", FROZEN, 11, "surface_pressure", 0.2),
    ]
    csv_path = _write(tmp_path / "incomplete.csv", rows)
    with pytest.raises(ValueError, match="missing 1 design/field cells"):
        build_report([csv_path], baseline=None)


def test_report_refuses_contrasts_across_different_designs(tmp_path: Path) -> None:
    """Pairing is only defined when both arms scored the same designs."""
    rows = [
        _row("S", FROZEN, 10, "surface_pressure", 0.2),
        _row("P-FT", FROZEN, 11, "surface_pressure", 0.2),
    ]
    csv_path = _write(tmp_path / "unpaired.csv", rows)
    with pytest.raises(ValueError, match="scored different designs"):
        build_report([csv_path], baseline=ArmKey(method="S", rendering=FROZEN))


def test_report_records_input_hashes(tmp_path: Path) -> None:
    """The report pins the exact metric files it summarized."""
    csv_path = _write(
        tmp_path / "metrics.csv",
        [
            _row("S", FROZEN, 10, "surface_pressure", 0.2),
            _row("S", FROZEN, 10, "volume_velocity", 0.3),
        ],
    )
    report = build_report([csv_path], baseline=None)
    assert report["inputs"][0]["path"] == str(csv_path.resolve())
    assert len(report["inputs"][0]["sha256"]) == 64
    assert json.dumps(report)
