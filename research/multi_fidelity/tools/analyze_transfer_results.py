# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Analyze paired ShapeNet-to-DrivAerML transfer-learning results.

The input is a long-table CSV with, at minimum, the columns ``method``,
``replicate``, ``n``, ``design_id``, ``field``, and ``relative_l2``.  The
analysis is deliberately strict: every requested field and test design must be
present for both methods at every replicate and high-fidelity sample count.

The common-task endpoint is the equal-field mean of log relative-L2 errors.
The script reports paired P-FT/S geometric error ratios at each ``n``, a
normalized AULC over log2(n), and the high-fidelity sample efficiency at the
error reached by scratch at n=200.  Confidence intervals use a two-way paired
crossed bootstrap that resamples training replicates and test designs
while retaining the same indices across methods, sample counts, and fields.

Example:

    .. code-block:: bash

        uv run python research/multi_fidelity/tools/analyze_transfer_results.py results.csv \
            --confirm-frozen-test \
            --output transfer_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any
from collections.abc import Sequence

import numpy as np
from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs

REQUIRED_COLUMNS = {"method", "replicate", "n", "design_id", "field", "relative_l2"}
DEFAULT_COMMON_FIELDS = ("surface_pressure", "volume_velocity")
MAX_EXACT_SIGN_FLIP_REPLICATES = 16
EXPECTED_REPLICATES = tuple(str(index) for index in range(8))
EXPECTED_N_VALUES = (25, 50, 100, 200, 400)


@dataclass(frozen=True)
class CommonGrid:
    """Complete paired common-field results on a rectangular analysis grid."""

    scratch_method: str
    transfer_method: str
    common_fields: tuple[str, ...]
    replicates: tuple[str, ...]
    n_values: np.ndarray
    design_ids: tuple[str, ...]
    scratch_log_macro: np.ndarray
    transfer_log_macro: np.ndarray
    total_csv_rows: int
    relevant_csv_rows: int
    methods_seen: tuple[str, ...]


@dataclass(frozen=True)
class EfficiencyEstimate:
    """Isotonic sample-efficiency estimate without curve extrapolation."""

    threshold_log_error: float
    scratch_isotonic_log_curve: np.ndarray
    transfer_isotonic_log_curve: np.ndarray
    transfer_n: float | None
    ratio: float | None
    samples_saved_fraction: float | None
    status: str
    interpretation: str


def _identifier_sort_key(value: str) -> tuple[int, int | str]:
    """Sort integer-like identifiers numerically and other identifiers lexically."""
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _parse_positive_int(value: str, *, column: str, line_number: int) -> int:
    """Parse a strictly positive integer CSV value."""
    try:
        parsed_float = float(value)
        parsed = int(parsed_float)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: column '{column}' must be an integer, got {value!r}") from exc
    if not math.isfinite(parsed_float) or parsed_float != parsed or parsed <= 0:
        raise ValueError(f"Line {line_number}: column '{column}' must be a positive integer, got {value!r}")
    return parsed


def _parse_positive_float(value: str, *, column: str, line_number: int) -> float:
    """Parse a finite, strictly positive floating-point CSV value."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: column '{column}' must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"Line {line_number}: column '{column}' must be finite and > 0, got {value!r}")
    return parsed


def load_common_grid(
    csv_path: Path,
    *,
    scratch_method: str = "S",
    transfer_method: str = "P-FT",
    common_fields: Sequence[str] = DEFAULT_COMMON_FIELDS,
) -> CommonGrid:
    """Load and validate a complete paired common-field result grid.

    Args:
        csv_path: Long-table result CSV.
        scratch_method: Method label for the from-scratch baseline.
        transfer_method: Method label for pretrained full fine-tuning.
        common_fields: Fields included with equal weight in the common endpoint.

    Returns:
        A dense paired grid containing per-design log-macro errors.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If columns, values, keys, or the paired grid are invalid.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if scratch_method == transfer_method:
        raise ValueError("Scratch and transfer method labels must differ")

    fields = tuple(common_fields)
    if not fields or any(not field for field in fields):
        raise ValueError("At least one non-empty common field is required")
    if len(set(fields)) != len(fields):
        raise ValueError(f"Common fields must be unique, got {fields}")

    requested_methods = {scratch_method, transfer_method}
    requested_fields = set(fields)
    records: dict[tuple[str, str, int, str, str], float] = {}
    methods_seen: set[str] = set()
    total_rows = 0
    relevant_rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

        for line_number, row in enumerate(reader, start=2):
            total_rows += 1
            method = row["method"].strip()
            methods_seen.add(method)
            field = row["field"].strip()
            if method not in requested_methods or field not in requested_fields:
                continue

            replicate = row["replicate"].strip()
            design_id = row["design_id"].strip()
            if not method or not replicate or not design_id or not field:
                raise ValueError(f"Line {line_number}: method, replicate, design_id, and field must be non-empty")
            n_value = _parse_positive_int(row["n"].strip(), column="n", line_number=line_number)
            relative_l2 = _parse_positive_float(
                row["relative_l2"].strip(), column="relative_l2", line_number=line_number
            )
            key = (method, replicate, n_value, design_id, field)
            if key in records:
                raise ValueError(f"Line {line_number}: duplicate result key {key}")
            records[key] = math.log(relative_l2)
            relevant_rows += 1

    if not records:
        raise ValueError(f"No rows matched methods {sorted(requested_methods)} and fields {sorted(requested_fields)}")
    missing_methods = requested_methods - methods_seen
    if missing_methods:
        raise ValueError(f"CSV does not contain requested methods: {sorted(missing_methods)}")

    replicates = tuple(sorted({key[1] for key in records}, key=_identifier_sort_key))
    n_values = np.asarray(sorted({key[2] for key in records}), dtype=np.int64)
    design_ids = tuple(sorted({key[3] for key in records}, key=_identifier_sort_key))
    method_order = (scratch_method, transfer_method)
    method_arrays: dict[str, np.ndarray] = {
        method: np.empty((len(replicates), len(n_values), len(design_ids)), dtype=np.float64) for method in method_order
    }

    missing_keys: list[tuple[str, str, int, str, str]] = []
    for method in method_order:
        output = method_arrays[method]
        for replicate_idx, replicate in enumerate(replicates):
            for n_idx, n_value in enumerate(n_values.tolist()):
                for design_idx, design_id in enumerate(design_ids):
                    log_values: list[float] = []
                    for field in fields:
                        key = (method, replicate, n_value, design_id, field)
                        value = records.get(key)
                        if value is None:
                            if len(missing_keys) < 20:
                                missing_keys.append(key)
                        else:
                            log_values.append(value)
                    if len(log_values) == len(fields):
                        output[replicate_idx, n_idx, design_idx] = float(np.mean(log_values))

    if missing_keys:
        expected_rows = len(method_order) * len(replicates) * len(n_values) * len(design_ids) * len(fields)
        missing_count = expected_rows - len(records)
        raise ValueError(
            "Requested methods/fields do not form a complete paired rectangular grid; "
            f"{missing_count} keys are missing. First missing keys: {missing_keys}"
        )

    expected_rows = len(method_order) * len(replicates) * len(n_values) * len(design_ids) * len(fields)
    if len(records) != expected_rows:
        raise ValueError(
            "Internal grid validation failed: the relevant CSV contains unexpected keys "
            f"({len(records)} rows, expected {expected_rows})"
        )

    return CommonGrid(
        scratch_method=scratch_method,
        transfer_method=transfer_method,
        common_fields=fields,
        replicates=replicates,
        n_values=n_values,
        design_ids=design_ids,
        scratch_log_macro=method_arrays[scratch_method],
        transfer_log_macro=method_arrays[transfer_method],
        total_csv_rows=total_rows,
        relevant_csv_rows=relevant_rows,
        methods_seen=tuple(sorted(methods_seen)),
    )


def _isotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Return the least-squares non-increasing fit using NumPy-only PAVA."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("Isotonic input must be a non-empty finite one-dimensional array")

    # Fit -y as a non-decreasing sequence. Each N has the same number of
    # replicate/design observations because load_common_grid requires a full grid.
    levels: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, value in enumerate((-y).tolist()):
        levels.append(value)
        weights.append(1.0)
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_weight = weights[-2] + weights[-1]
            merged_level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / merged_weight
            merged_start = starts[-2]
            merged_end = ends[-1]
            del levels[-2:]
            del weights[-2:]
            del starts[-2:]
            del ends[-2:]
            levels.append(merged_level)
            weights.append(merged_weight)
            starts.append(merged_start)
            ends.append(merged_end)

    fitted = np.empty_like(y)
    for level, start, end in zip(levels, starts, ends, strict=True):
        fitted[start:end] = -level
    return fitted


def _crossing_without_extrapolation(
    n_values: np.ndarray,
    nonincreasing_log_curve: np.ndarray,
    threshold_log_error: float,
) -> tuple[float | None, str]:
    """Find the first threshold crossing by interpolation inside observed N."""
    x = np.log2(np.asarray(n_values, dtype=np.float64))
    y = np.asarray(nonincreasing_log_curve, dtype=np.float64)
    tolerance = 1e-12

    if threshold_log_error >= y[0] - tolerance:
        return float(n_values[0]), "at_or_below_minimum_n"
    if threshold_log_error < y[-1] - tolerance:
        return None, "not_reached_within_observed_n"

    crossing_indices = np.flatnonzero(y <= threshold_log_error + tolerance)
    if crossing_indices.size == 0:
        return None, "not_reached_within_observed_n"
    right = int(crossing_indices[0])
    if right == 0:
        return float(n_values[0]), "at_or_below_minimum_n"
    left = right - 1

    y_left = float(y[left])
    y_right = float(y[right])
    if abs(y_left - y_right) <= tolerance:
        # A flat isotonic block equal to the threshold has no identifiable
        # within-block crossing; selecting its first observed N is conservative.
        return float(n_values[right]), "isotonic_plateau_boundary"
    fraction = (y_left - threshold_log_error) / (y_left - y_right)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    crossing_log2_n = float(x[left] + fraction * (x[right] - x[left]))
    return float(2.0**crossing_log2_n), "interpolated_within_observed_n"


def _sample_efficiency(
    scratch_log_curve: np.ndarray,
    transfer_log_curve: np.ndarray,
    n_values: np.ndarray,
    *,
    threshold_n: int,
) -> EfficiencyEstimate:
    """Estimate transfer N needed to match scratch at ``threshold_n``."""
    threshold_matches = np.flatnonzero(n_values == threshold_n)
    if threshold_matches.size != 1:
        raise ValueError(f"Sample-efficiency threshold n={threshold_n} must appear exactly once in n values")

    scratch_isotonic = _isotonic_nonincreasing(scratch_log_curve)
    transfer_isotonic = _isotonic_nonincreasing(transfer_log_curve)
    threshold_log_error = float(scratch_isotonic[int(threshold_matches[0])])
    transfer_n, status = _crossing_without_extrapolation(n_values, transfer_isotonic, threshold_log_error)

    if transfer_n is None:
        return EfficiencyEstimate(
            threshold_log_error=threshold_log_error,
            scratch_isotonic_log_curve=scratch_isotonic,
            transfer_isotonic_log_curve=transfer_isotonic,
            transfer_n=None,
            ratio=None,
            samples_saved_fraction=None,
            status=status,
            interpretation="not_identifiable_without_extrapolation",
        )

    ratio = threshold_n / transfer_n
    samples_saved_fraction = 1.0 - transfer_n / threshold_n
    interpretation = "point_estimate"
    if status == "at_or_below_minimum_n":
        interpretation = "conservative_lower_bound_due_to_minimum_n"
    elif status == "isotonic_plateau_boundary":
        interpretation = "conservative_plateau_boundary"
    return EfficiencyEstimate(
        threshold_log_error=threshold_log_error,
        scratch_isotonic_log_curve=scratch_isotonic,
        transfer_isotonic_log_curve=transfer_isotonic,
        transfer_n=transfer_n,
        ratio=ratio,
        samples_saved_fraction=samples_saved_fraction,
        status=status,
        interpretation=interpretation,
    )


def _normalized_aulc(log_curve: np.ndarray, n_values: np.ndarray) -> float:
    """Integrate a log-error curve over log2(N), normalized by its span."""
    if n_values.size < 2:
        raise ValueError("AULC requires at least two distinct n values")
    x = np.log2(np.asarray(n_values, dtype=np.float64))
    span = float(x[-1] - x[0])
    if span <= 0:
        raise ValueError("AULC requires strictly increasing positive n values")
    return float(np.trapezoid(log_curve, x=x) / span)


def _percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    """Return a two-sided percentile confidence interval."""
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return [float(lower), float(upper)]


def _wilson_interval(successes: int, trials: int, confidence: float) -> list[float]:
    """Return a Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
    half_width /= denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _paired_hedges_g(benefits: np.ndarray) -> float | None:
    """Return small-sample-corrected paired standardized benefit."""
    if benefits.size < 2:
        return None
    standard_deviation = float(np.std(benefits, ddof=1))
    if standard_deviation <= 0:
        return None
    d_z = float(np.mean(benefits) / standard_deviation)
    correction = 1.0 - 3.0 / (4.0 * benefits.size - 5.0)
    return correction * d_z


def _exact_paired_sign_flip_pvalue(benefits: np.ndarray) -> float:
    """Return an exact one-sided paired sign-flip p-value.

    Positive values are benefits (scratch log error minus transfer log error),
    so the alternative is that transfer has lower error. Every one of the
    ``2**R`` sign assignments is enumerated, including the observed all-positive
    assignment. The implementation intentionally refuses large ``R`` instead
    of silently switching the preregistered exact test to a Monte Carlo test.
    """
    values = np.asarray(benefits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Sign-flip benefits must be a non-empty finite one-dimensional array")
    if values.size > MAX_EXACT_SIGN_FLIP_REPLICATES:
        raise ValueError(
            "Exact paired sign-flip enumeration supports at most "
            f"{MAX_EXACT_SIGN_FLIP_REPLICATES} replicates, got {values.size}"
        )

    assignment_count = 1 << values.size
    assignments = np.arange(assignment_count, dtype=np.uint64)[:, None]
    bit_positions = np.arange(values.size, dtype=np.uint64)[None, :]
    signs = np.where(((assignments >> bit_positions) & 1) == 1, 1.0, -1.0)
    permuted_statistics = (signs @ values) / values.size
    observed_statistic = float(np.mean(values))
    # nextafter admits values equal up to the final floating-point rounding of
    # the matrix product without materially relaxing the exact ordering.
    comparison_boundary = np.nextafter(observed_statistic, -math.inf)
    return float(np.count_nonzero(permuted_statistics >= comparison_boundary) / assignment_count)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Holm adjustment requires a non-empty finite one-dimensional sequence")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Holm adjustment requires p-values between 0 and 1")

    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running_maximum = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, float((values.size - rank) * values[original_index]))
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted[rank] = running_maximum
    adjusted = np.empty(values.size, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def _bootstrap_statistics(
    grid: CommonGrid,
    *,
    samples: int,
    seed: int,
    threshold_n: int,
) -> dict[str, Any]:
    """Run the paired two-way replicate/design crossed bootstrap."""
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    replicate_count, n_count, design_count = grid.scratch_log_macro.shape
    ratio_samples = np.empty((samples, n_count), dtype=np.float64)
    aulc_ratio_samples = np.empty(samples, dtype=np.float64)
    aulc_difference_samples = np.empty(samples, dtype=np.float64)
    efficiency_ratio_samples = np.full(samples, np.nan, dtype=np.float64)
    efficiency_statuses: Counter[str] = Counter()

    for bootstrap_idx in range(samples):
        replicate_indices = rng.integers(0, replicate_count, size=replicate_count)
        design_indices = rng.integers(0, design_count, size=design_count)

        scratch_sample = grid.scratch_log_macro[replicate_indices][:, :, design_indices]
        transfer_sample = grid.transfer_log_macro[replicate_indices][:, :, design_indices]
        scratch_curve = scratch_sample.mean(axis=(0, 2))
        transfer_curve = transfer_sample.mean(axis=(0, 2))
        log_ratio_curve = transfer_curve - scratch_curve
        ratio_samples[bootstrap_idx] = np.exp(log_ratio_curve)

        scratch_aulc = _normalized_aulc(scratch_curve, grid.n_values)
        transfer_aulc = _normalized_aulc(transfer_curve, grid.n_values)
        aulc_difference = transfer_aulc - scratch_aulc
        aulc_difference_samples[bootstrap_idx] = aulc_difference
        aulc_ratio_samples[bootstrap_idx] = math.exp(aulc_difference)

        efficiency = _sample_efficiency(
            scratch_curve,
            transfer_curve,
            grid.n_values,
            threshold_n=threshold_n,
        )
        efficiency_statuses[efficiency.status] += 1
        if efficiency.ratio is not None:
            efficiency_ratio_samples[bootstrap_idx] = efficiency.ratio

    return {
        "ratio_samples": ratio_samples,
        "aulc_ratio_samples": aulc_ratio_samples,
        "aulc_difference_samples": aulc_difference_samples,
        "efficiency_ratio_samples": efficiency_ratio_samples,
        "efficiency_statuses": efficiency_statuses,
    }


def analyze_common_grid(
    grid: CommonGrid,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_716,
    confidence: float = 0.95,
    threshold_n: int = 200,
    negative_transfer_margin: float = 1.05,
) -> dict[str, Any]:
    """Compute paired transfer metrics and crossed-bootstrap intervals.

    Args:
        grid: Complete paired common-task analysis grid.
        bootstrap_samples: Number of two-way paired bootstrap samples.
        bootstrap_seed: NumPy RNG seed for reproducibility.
        confidence: Two-sided confidence level.
        threshold_n: Scratch sample count defining the sample-efficiency target.
        negative_transfer_margin: Material negative-transfer error-ratio margin.

    Returns:
        JSON-serializable analysis results.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between 0 and 1")
    if negative_transfer_margin <= 0:
        raise ValueError("Negative-transfer margin must be positive")
    if threshold_n not in grid.n_values:
        raise ValueError(f"Threshold n={threshold_n} is absent; available n values are {grid.n_values.tolist()}")

    scratch_curve = grid.scratch_log_macro.mean(axis=(0, 2))
    transfer_curve = grid.transfer_log_macro.mean(axis=(0, 2))
    log_ratio_curve = transfer_curve - scratch_curve
    point_ratios = np.exp(log_ratio_curve)
    bootstrap = _bootstrap_statistics(
        grid,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        threshold_n=threshold_n,
    )
    replicate_log_ratios_by_n = (grid.transfer_log_macro - grid.scratch_log_macro).mean(axis=2)
    per_n_raw_p_values = [_exact_paired_sign_flip_pvalue(-values) for values in replicate_log_ratios_by_n.T]
    per_n_holm_p_values = _holm_adjust(per_n_raw_p_values)

    per_n: dict[str, Any] = {}
    for n_idx, n_value in enumerate(grid.n_values.tolist()):
        ratio_interval = _percentile_interval(bootstrap["ratio_samples"][:, n_idx], confidence)
        replicate_log_ratios = replicate_log_ratios_by_n[:, n_idx]
        replicate_ratios = np.exp(replicate_log_ratios)
        materially_worse_count = int(np.count_nonzero(replicate_ratios > negative_transfer_margin))
        benefit = -replicate_log_ratios
        per_n[str(n_value)] = {
            "scratch_geometric_macro_relative_l2": float(math.exp(scratch_curve[n_idx])),
            "transfer_geometric_macro_relative_l2": float(math.exp(transfer_curve[n_idx])),
            "paired_geometric_error_ratio_transfer_over_scratch": float(point_ratios[n_idx]),
            "paired_geometric_error_ratio_ci": ratio_interval,
            "percent_error_change": float(100.0 * (point_ratios[n_idx] - 1.0)),
            "paired_hedges_g_benefit": _paired_hedges_g(benefit),
            "paired_exact_sign_flip": {
                "alternative": "transfer_error_lower_than_scratch",
                "statistic": "mean replicate scratch-minus-transfer log error",
                "observed_mean_log_benefit": float(np.mean(benefit)),
                "raw_p_value": per_n_raw_p_values[n_idx],
                "holm_adjusted_p_value": per_n_holm_p_values[n_idx],
                "holm_family": "all reported common-task N values",
                "exact_sign_assignments": 1 << len(grid.replicates),
            },
            "replicate_ratios": {
                replicate: float(replicate_ratios[idx]) for idx, replicate in enumerate(grid.replicates)
            },
            "negative_transfer": {
                "material_margin_ratio": negative_transfer_margin,
                "point_ratio_exceeds_margin": bool(point_ratios[n_idx] > negative_transfer_margin),
                "material_negative_transfer_ci_lower_exceeds_margin": bool(
                    ratio_interval[0] > negative_transfer_margin
                ),
                "noninferior_ci_upper_below_margin": bool(ratio_interval[1] < negative_transfer_margin),
                "materially_worse_replicates": materially_worse_count,
                "replicate_count": len(grid.replicates),
                "materially_worse_replicate_fraction": materially_worse_count / len(grid.replicates),
                "materially_worse_replicate_fraction_wilson_ci": _wilson_interval(
                    materially_worse_count, len(grid.replicates), confidence
                ),
            },
        }

    scratch_aulc = _normalized_aulc(scratch_curve, grid.n_values)
    transfer_aulc = _normalized_aulc(transfer_curve, grid.n_values)
    aulc_difference = transfer_aulc - scratch_aulc
    aulc_ratio = math.exp(aulc_difference)
    replicate_scratch_curves = grid.scratch_log_macro.mean(axis=2)
    replicate_transfer_curves = grid.transfer_log_macro.mean(axis=2)
    replicate_aulc_benefits = np.asarray(
        [
            _normalized_aulc(scratch_rep, grid.n_values) - _normalized_aulc(transfer_rep, grid.n_values)
            for scratch_rep, transfer_rep in zip(replicate_scratch_curves, replicate_transfer_curves, strict=True)
        ],
        dtype=np.float64,
    )
    aulc_sign_flip_p_value = _exact_paired_sign_flip_pvalue(replicate_aulc_benefits)
    efficiency = _sample_efficiency(
        scratch_curve,
        transfer_curve,
        grid.n_values,
        threshold_n=threshold_n,
    )
    efficiency_bootstrap = bootstrap["efficiency_ratio_samples"]
    identifiable_mask = np.isfinite(efficiency_bootstrap)
    identifiable_count = int(np.count_nonzero(identifiable_mask))
    efficiency_interval: list[float] | None = None
    efficiency_interval_note = "all bootstrap draws identifiable without extrapolation"
    if identifiable_count == bootstrap_samples:
        efficiency_interval = _percentile_interval(efficiency_bootstrap, confidence)
    else:
        efficiency_interval_note = (
            "CI withheld because at least one bootstrap draw did not reach the scratch threshold within observed N"
        )

    return {
        "endpoint": {
            "name": "common_equal_field_log_macro_relative_l2",
            "fields": list(grid.common_fields),
            "field_weights": {field: 1.0 / len(grid.common_fields) for field in grid.common_fields},
            "definition": "exp(mean over fields, paired replicates, and test designs of log(relative_l2))",
        },
        "inference": {
            "sign_flip_test": {
                "kind": "exact paired replicate-level enumeration",
                "alternative": "transfer_error_lower_than_scratch",
                "maximum_supported_replicates": MAX_EXACT_SIGN_FLIP_REPLICATES,
            },
            "per_n_p_value_multiplicity": "Holm family-wise-error adjustment across all common-task N values",
            "confidence_interval_multiplicity": "none",
            "confidence_interval_note": (
                "All crossed-bootstrap confidence intervals are pointwise; no simultaneous or multiplicity-adjusted CI is claimed."
            ),
        },
        "per_n": per_n,
        "aulc_log2_n": {
            "normalization": "trapezoidal integral of log macro error divided by log2(N_max)-log2(N_min)",
            "scratch_log_aulc": scratch_aulc,
            "transfer_log_aulc": transfer_aulc,
            "transfer_minus_scratch_log_aulc": aulc_difference,
            "geometric_aulc_error_ratio_transfer_over_scratch": aulc_ratio,
            "geometric_aulc_error_ratio_ci": _percentile_interval(bootstrap["aulc_ratio_samples"], confidence),
            "log_aulc_difference_ci": _percentile_interval(bootstrap["aulc_difference_samples"], confidence),
            "paired_exact_sign_flip": {
                "alternative": "transfer_AULC_lower_than_scratch",
                "statistic": "mean replicate scratch-minus-transfer normalized log AULC",
                "observed_mean_log_aulc_benefit": float(np.mean(replicate_aulc_benefits)),
                "p_value": aulc_sign_flip_p_value,
                "replicate_log_aulc_benefits": {
                    replicate: float(replicate_aulc_benefits[idx]) for idx, replicate in enumerate(grid.replicates)
                },
                "exact_sign_assignments": 1 << len(grid.replicates),
            },
        },
        "sample_efficiency": {
            "threshold_source": f"scratch isotonic common-task error at n={threshold_n}",
            "threshold_n": threshold_n,
            "threshold_macro_relative_l2": math.exp(efficiency.threshold_log_error),
            "isotonic_backend": "numpy_pool_adjacent_violators",
            "no_extrapolation": True,
            "scratch_isotonic_macro_relative_l2": {
                str(n): float(math.exp(value))
                for n, value in zip(grid.n_values.tolist(), efficiency.scratch_isotonic_log_curve, strict=True)
            },
            "transfer_isotonic_macro_relative_l2": {
                str(n): float(math.exp(value))
                for n, value in zip(grid.n_values.tolist(), efficiency.transfer_isotonic_log_curve, strict=True)
            },
            "transfer_n_to_reach_threshold": efficiency.transfer_n,
            "sample_efficiency_ratio_reference_n_over_transfer_n": efficiency.ratio,
            "samples_saved_fraction": efficiency.samples_saved_fraction,
            "status": efficiency.status,
            "interpretation": efficiency.interpretation,
            "ratio_ci": efficiency_interval,
            "ratio_ci_note": efficiency_interval_note,
            "bootstrap_identifiable_draws": identifiable_count,
            "bootstrap_identifiable_fraction": identifiable_count / bootstrap_samples,
            "bootstrap_status_counts": dict(sorted(bootstrap["efficiency_statuses"].items())),
        },
        "bootstrap": {
            "kind": "two_way_paired_crossed",
            "resampled_units": ["replicate", "design_id"],
            "pairing_preserved_across": ["method", "n", "field"],
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": confidence,
        },
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_transfer_csv(
    csv_path: Path,
    *,
    confirm_frozen_test: bool,
    scratch_method: str = "S",
    transfer_method: str = "P-FT",
    common_fields: Sequence[str] = DEFAULT_COMMON_FIELDS,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_716,
    confidence: float = 0.95,
    threshold_n: int = 200,
    negative_transfer_margin: float = 1.05,
) -> dict[str, Any]:
    """Load confirmed frozen-test results and return the complete analysis."""
    if not confirm_frozen_test:
        raise ValueError("final analysis requires explicit confirmation of frozen test-only metrics")
    grid = load_common_grid(
        csv_path,
        scratch_method=scratch_method,
        transfer_method=transfer_method,
        common_fields=common_fields,
    )
    if grid.replicates != EXPECTED_REPLICATES:
        raise ValueError(
            f"final analysis requires replicate IDs {list(EXPECTED_REPLICATES)}, got {list(grid.replicates)}"
        )
    if tuple(grid.n_values.tolist()) != EXPECTED_N_VALUES:
        raise ValueError(f"final analysis requires exactly N={list(EXPECTED_N_VALUES)}")
    expected_design_ids = tuple(
        sorted((str(value) for value in DrivAerMLDefaultSplitIDs().test), key=_identifier_sort_key)
    )
    if grid.design_ids != expected_design_ids:
        raise ValueError("final analysis design IDs must exactly match the official DrivAerML test split")
    analysis = analyze_common_grid(
        grid,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
        threshold_n=threshold_n,
        negative_transfer_margin=negative_transfer_margin,
    )
    return {
        "schema_version": 1,
        "input": {
            "evaluation_split": "test",
            "caller_confirmed_frozen_test": True,
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": _sha256(csv_path),
            "total_csv_rows": grid.total_csv_rows,
            "relevant_csv_rows": grid.relevant_csv_rows,
            "methods_seen": list(grid.methods_seen),
            "scratch_method": scratch_method,
            "transfer_method": transfer_method,
            "replicates": list(grid.replicates),
            "n_values": grid.n_values.tolist(),
            "design_ids": list(grid.design_ids),
            "replicate_count": len(grid.replicates),
            "design_count": len(grid.design_ids),
        },
        "analysis": analysis,
    }


def main() -> None:
    """Parse CLI arguments, analyze a long-table CSV, and emit JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Long-table result CSV")
    parser.add_argument("--output", type=Path, help="Optional JSON output path; defaults to stdout")
    parser.add_argument(
        "--confirm-frozen-test",
        action="store_true",
        help="Confirm the CSV contains only frozen test metrics from the coordinated release",
    )
    parser.add_argument("--scratch-method", default="S", help="CSV method label for scratch (default: S)")
    parser.add_argument("--transfer-method", default="P-FT", help="CSV method label for transfer (default: P-FT)")
    parser.add_argument(
        "--common-fields",
        nargs="+",
        default=list(DEFAULT_COMMON_FIELDS),
        help="Equally weighted common fields (default: surface_pressure volume_velocity)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_716)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--threshold-n", type=int, default=200)
    parser.add_argument("--negative-transfer-margin", type=float, default=1.05)
    args = parser.parse_args()

    try:
        report = analyze_transfer_csv(
            args.csv,
            confirm_frozen_test=args.confirm_frozen_test,
            scratch_method=args.scratch_method,
            transfer_method=args.transfer_method,
            common_fields=args.common_fields,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence,
            threshold_n=args.threshold_n,
            negative_transfer_margin=args.negative_transfer_margin,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
