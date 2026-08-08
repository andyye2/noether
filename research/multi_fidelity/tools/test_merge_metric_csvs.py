from __future__ import annotations

import csv
from pathlib import Path

import pytest

from research.multi_fidelity.tools.merge_metric_csvs import merge_metric_csvs


def _write(path: Path, rows: list[dict[str, str]], *, with_mse: bool = False) -> None:
    fieldnames = ["method", "replicate", "n", "design_id", "field", "relative_l2"]
    if with_mse:
        fieldnames.append("mse")
    fieldnames.append("mae")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_merge_metric_csvs_is_deterministic_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "merged.csv"
    row_s = {
        "method": "S",
        "replicate": "0",
        "n": "25",
        "design_id": "10",
        "field": "surface_pressure",
        "relative_l2": "0.2",
        "mae": "0.1",
    }
    row_pft = {**row_s, "method": "P-FT", "relative_l2": "0.1"}
    _write(first, [row_s])
    _write(second, [row_pft])

    result = merge_metric_csvs([first, second], output)
    assert result["rows"] == 2
    assert output.read_text(encoding="utf-8").splitlines()[1].startswith("P-FT,")

    duplicate = tmp_path / "duplicate.csv"
    _write(duplicate, [row_s])
    with pytest.raises(ValueError, match="duplicate metric key"):
        merge_metric_csvs([first, duplicate], tmp_path / "invalid.csv")


def test_mse_is_carried_only_when_every_input_reports_it(tmp_path: Path) -> None:
    """Tables exported before MSE existed still merge, without a ragged column."""
    row_s = {
        "method": "S",
        "replicate": "0",
        "n": "25",
        "design_id": "10",
        "field": "surface_pressure",
        "relative_l2": "0.2",
        "mse": "4.0",
        "mae": "0.1",
    }
    row_pft = {**row_s, "method": "P-FT", "relative_l2": "0.1", "mse": "2.0"}

    with_mse = tmp_path / "with_mse.csv"
    also_with_mse = tmp_path / "also_with_mse.csv"
    _write(with_mse, [row_s], with_mse=True)
    _write(also_with_mse, [row_pft], with_mse=True)
    both = tmp_path / "both.csv"
    merge_metric_csvs([with_mse, also_with_mse], both)
    header, first_row = both.read_text(encoding="utf-8").splitlines()[:2]
    # Optional metrics are appended, so the required column order never moves.
    assert header == "method,replicate,n,design_id,field,relative_l2,mae,mse"
    assert first_row == "P-FT,0,25,10,surface_pressure,0.1,0.1,2.0"

    legacy = tmp_path / "legacy.csv"
    _write(legacy, [{key: value for key, value in row_pft.items() if key != "mse"}])
    mixed = tmp_path / "mixed.csv"
    merge_metric_csvs([with_mse, legacy], mixed)
    assert mixed.read_text(encoding="utf-8").splitlines()[0].endswith("relative_l2,mae")
