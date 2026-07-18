from __future__ import annotations

import csv
from pathlib import Path

import pytest

from research.multi_fidelity.tools.merge_metric_csvs import merge_metric_csvs


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method", "replicate", "n", "design_id", "field", "relative_l2", "mae"],
        )
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
