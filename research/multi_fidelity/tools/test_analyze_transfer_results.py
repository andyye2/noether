# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Synthetic validation for ``analyze_transfer_results.py``."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs

from .analyze_transfer_results import (
    _exact_paired_sign_flip_pvalue,
    _holm_adjust,
    _isotonic_nonincreasing,
    _sample_efficiency,
    analyze_transfer_csv,
)


def _write_known_two_x_curve(path: Path) -> None:
    """Write a paired curve where transfer at N matches scratch at 2N."""
    fields = ("surface_pressure", "volume_velocity")
    n_values = (25, 50, 100, 200, 400)
    rows = []
    for method in ("S", "P-FT"):
        for replicate in range(8):
            for n_value in n_values:
                for design_id in DrivAerMLDefaultSplitIDs().test:
                    for field in fields:
                        relative_l2 = 0.8 * n_value**-0.3
                        if method == "P-FT":
                            relative_l2 *= 2**-0.3
                        relative_l2 *= 1.0 + 0.01 * replicate
                        relative_l2 *= 1.0 + 0.002 * design_id
                        if field == "volume_velocity":
                            relative_l2 *= 1.1
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


def test_known_two_x_sample_efficiency(tmp_path: Path) -> None:
    """Recover the paired ratio and 2x sample efficiency from synthetic data."""
    csv_path = tmp_path / "known_curve.csv"
    _write_known_two_x_curve(csv_path)

    report = analyze_transfer_csv(csv_path, confirm_frozen_test=True, bootstrap_samples=500, bootstrap_seed=7)
    ratio = report["analysis"]["per_n"]["100"]["paired_geometric_error_ratio_transfer_over_scratch"]
    efficiency = report["analysis"]["sample_efficiency"]

    assert math.isclose(ratio, 2**-0.3, rel_tol=1e-12)
    assert math.isclose(efficiency["sample_efficiency_ratio_reference_n_over_transfer_n"], 2.0, rel_tol=1e-10)
    assert efficiency["status"] == "interpolated_within_observed_n"
    aulc_test = report["analysis"]["aulc_log2_n"]["paired_exact_sign_flip"]
    per_n_test = report["analysis"]["per_n"]["100"]["paired_exact_sign_flip"]
    assert aulc_test["exact_sign_assignments"] == 256
    assert aulc_test["p_value"] == 1.0 / 256.0
    assert per_n_test["raw_p_value"] == 1.0 / 256.0
    assert per_n_test["holm_adjusted_p_value"] == 5.0 / 256.0
    assert report["analysis"]["inference"]["confidence_interval_multiplicity"] == "none"
    json.dumps(report, allow_nan=False)


def test_isotonic_fit_and_no_extrapolation() -> None:
    """Pool violations and refuse a threshold beyond the observed N range."""
    fitted = _isotonic_nonincreasing(np.asarray([3.0, 1.0, 2.0]))
    np.testing.assert_allclose(fitted, [3.0, 1.5, 1.5])

    n_values = np.asarray([25, 50, 100, 200, 400])
    scratch = np.log(np.asarray([0.5, 0.4, 0.3, 0.2, 0.15]))
    transfer = np.log(np.asarray([0.8, 0.7, 0.6, 0.5, 0.4]))
    estimate = _sample_efficiency(scratch, transfer, n_values, threshold_n=200)

    assert estimate.status == "not_reached_within_observed_n"
    assert estimate.transfer_n is None
    assert estimate.ratio is None


def test_exact_sign_flip_and_holm_known_values() -> None:
    """Enumerate all 256 signs and recover a textbook Holm adjustment."""
    assert _exact_paired_sign_flip_pvalue(np.ones(8)) == 1.0 / 256.0
    assert _holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_incomplete_paired_grid_is_rejected(tmp_path: Path) -> None:
    """Reject a missing method/replicate/N/design/field cell."""
    csv_path = tmp_path / "missing_cell.csv"
    _write_known_two_x_curve(csv_path)
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
        analyze_transfer_csv(csv_path, confirm_frozen_test=True, bootstrap_samples=1)


def test_final_analysis_requires_explicit_test_provenance_confirmation(tmp_path: Path) -> None:
    """A complete grid cannot be analyzed as final output without test provenance confirmation."""
    csv_path = tmp_path / "unconfirmed.csv"
    _write_known_two_x_curve(csv_path)

    with pytest.raises(ValueError, match="explicit confirmation"):
        analyze_transfer_csv(csv_path, confirm_frozen_test=False, bootstrap_samples=1)
