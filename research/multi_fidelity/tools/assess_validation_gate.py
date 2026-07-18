# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Assess the preregistered validation-only P-FT futility gate.

The input uses the same long-table schema as ``analyze_transfer_results`` but
must contain metrics from the *final raw* (non-EMA, non-best-selected)
checkpoint on the validation split.  The CLI requires an explicit confirmation
of that provenance and never reads test results.

For each of the first-phase three paired replicates at N=50 and N=100, the
endpoint is the common equal-field log macro: log(relative-L2) is averaged over
the two common fields and all validation designs before exponentiating the
paired P-FT-minus-scratch difference.  The raw P-FT route stops only when both
per-N median ratios are at least 1.10 and every one of the six replicate-by-N
ratios is greater than 1.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .analyze_transfer_results import DEFAULT_COMMON_FIELDS, _sha256, load_common_grid
else:  # Support direct execution by file path as well as module execution.
    from analyze_transfer_results import DEFAULT_COMMON_FIELDS, _sha256, load_common_grid

EXPECTED_REPLICATE_IDS = ("0", "1", "2")
GATE_N_VALUES = (50, 100)
EXPECTED_REPLICATES = 3
MEDIAN_RATIO_THRESHOLD = 1.10


def assess_validation_gate(
    csv_path: Path,
    *,
    confirm_final_raw_val: bool,
    scratch_method: str = "S",
    transfer_method: str = "P-FT",
    common_fields: tuple[str, ...] = DEFAULT_COMMON_FIELDS,
) -> dict[str, Any]:
    """Return a JSON-ready preregistered continue/stop decision.

    ``confirm_final_raw_val`` is deliberately mandatory: the CSV schema cannot
    itself prove checkpoint or split provenance, so callers must affirm that
    the rows came from final raw validation evaluation.
    """
    if not confirm_final_raw_val:
        raise ValueError(
            "Gate input provenance is unconfirmed; affirm that the CSV contains only final raw validation metrics"
        )

    grid = load_common_grid(
        csv_path,
        scratch_method=scratch_method,
        transfer_method=transfer_method,
        common_fields=common_fields,
    )
    actual_n_values = tuple(grid.n_values.tolist())
    if actual_n_values != GATE_N_VALUES:
        raise ValueError(f"Validation gate requires exactly N={list(GATE_N_VALUES)}, got {list(actual_n_values)}")
    if grid.replicates != EXPECTED_REPLICATE_IDS:
        raise ValueError(
            f"Validation gate requires paired first-phase replicate IDs {list(EXPECTED_REPLICATE_IDS)}, "
            f"got {list(grid.replicates)}"
        )

    n_indices = [int(np.flatnonzero(grid.n_values == n_value)[0]) for n_value in GATE_N_VALUES]
    paired_log_ratios = (grid.transfer_log_macro[:, n_indices, :] - grid.scratch_log_macro[:, n_indices, :]).mean(
        axis=2
    )
    paired_ratios = np.exp(paired_log_ratios)

    per_n: dict[str, Any] = {}
    median_conditions: list[bool] = []
    for gate_idx, n_value in enumerate(GATE_N_VALUES):
        ratios = paired_ratios[:, gate_idx]
        median_ratio = float(np.median(ratios))
        median_meets_threshold = bool(median_ratio >= MEDIAN_RATIO_THRESHOLD)
        median_conditions.append(median_meets_threshold)
        per_n[str(n_value)] = {
            "median_paired_ratio_transfer_over_scratch": median_ratio,
            "median_ratio_threshold": MEDIAN_RATIO_THRESHOLD,
            "median_meets_or_exceeds_threshold": median_meets_threshold,
            "all_replicates_worse_than_scratch": bool(np.all(ratios > 1.0)),
            "replicate_ratios": {
                replicate: float(ratios[replicate_idx]) for replicate_idx, replicate in enumerate(grid.replicates)
            },
        }

    worse_cell_count = int(np.count_nonzero(paired_ratios > 1.0))
    expected_cell_count = EXPECTED_REPLICATES * len(GATE_N_VALUES)
    all_six_worse = worse_cell_count == expected_cell_count
    should_stop = all(median_conditions) and all_six_worse

    return {
        "schema_version": 1,
        "input": {
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": _sha256(csv_path),
            "scratch_method": scratch_method,
            "transfer_method": transfer_method,
            "replicates": list(grid.replicates),
            "validation_design_ids": list(grid.design_ids),
            "common_fields": list(grid.common_fields),
        },
        "endpoint_contract": {
            "split": "validation_only",
            "checkpoint": "final",
            "weight_stream": "raw_non_ema",
            "selection": "no_best_validation_checkpoint_selection",
            "name": "common_equal_field_validation_log_macro_relative_l2",
            "aggregation": (
                "within each replicate-by-N-method cell, mean log(relative_l2) over equally weighted common "
                "fields and all validation designs; paired ratio is exp(P-FT minus scratch)"
            ),
            "test_data_allowed": False,
            "caller_confirmed_final_raw_validation": True,
        },
        "gate": {
            "n_values": list(GATE_N_VALUES),
            "expected_paired_replicates": EXPECTED_REPLICATES,
            "expected_paired_cells": expected_cell_count,
            "median_ratio_threshold": MEDIAN_RATIO_THRESHOLD,
            "per_n": per_n,
            "both_n_medians_meet_or_exceed_threshold": all(median_conditions),
            "worse_than_scratch_cell_count": worse_cell_count,
            "all_six_paired_cells_worse_than_scratch": all_six_worse,
            "rule": ("stop iff the N=50 and N=100 median ratios are each >=1.10 and all 3x2 paired ratios are >1.0"),
        },
        "decision": "stop" if should_stop else "continue",
        "route": "raw_P-FT",
        "action": (
            "stop raw P-FT and audit coordinates/normalization before adapters"
            if should_stop
            else "continue the preregistered raw P-FT route"
        ),
    }


def main() -> None:
    """Parse arguments and write the validation-gate JSON decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Final-raw validation long-table CSV")
    parser.add_argument("--output", type=Path, help="Optional JSON output path; defaults to stdout")
    parser.add_argument("--scratch-method", default="S")
    parser.add_argument("--transfer-method", default="P-FT")
    parser.add_argument(
        "--common-fields",
        nargs="+",
        default=list(DEFAULT_COMMON_FIELDS),
        help="Equally weighted common validation fields",
    )
    parser.add_argument(
        "--confirm-final-raw-val",
        action="store_true",
        help="Confirm the CSV contains only final raw validation metrics and no test rows",
    )
    args = parser.parse_args()

    try:
        report = assess_validation_gate(
            args.csv,
            confirm_final_raw_val=args.confirm_final_raw_val,
            scratch_method=args.scratch_method,
            transfer_method=args.transfer_method,
            common_fields=tuple(args.common_fields),
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
