# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Turn the frozen per-design metric table into the reported comparison.

The study reports one paired data cell, so every arm scores the same official
designs with the same evaluator.  Contrasts are therefore formed per design
and only then aggregated, which is what makes a ratio meaningful without
replicate-level resampling.

Aggregation follows the preregistration: the per-field endpoint is the
geometric mean of the per-design relative L2, and the macro endpoint is the
equal-field mean of the log relative L2.

Example:

    .. code-block:: bash

        uv run python -m research.multi_fidelity.tools.compare_arms \\
            <artifacts>/metrics/val/common/*.csv \\
            --baseline ps1000-sr9 --method S \\
            --output-json <artifacts>/reports/comparison.json \\
            --output-markdown <artifacts>/reports/comparison.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
from typing import Any

from aero_cfd.multi_fidelity.integrity import atomic_write_json, sha256_file

from .merge_metric_csvs import load_metric_rows

ARM_COLUMNS = ("method", "rendering")


@dataclass(frozen=True)
class ArmKey:
    """Identity of one arm inside the metric table.

    Attributes:
        method: Preregistered method code, ``S`` or ``P-FT``.
        rendering: Geometry rendering label, ``ps<scale>-sr<radius>``.
    """

    method: str
    rendering: str

    def label(self) -> str:
        """Return the short label used in reports and JSON keys."""
        return f"{self.method}@{self.rendering}"


@dataclass(frozen=True)
class ArmMetrics:
    """Aggregated endpoints of one arm.

    Attributes:
        key: Arm identity.
        designs: Official design IDs scored by the arm.
        relative_l2: Geometric-mean relative L2 per field.
        mae: Arithmetic-mean MAE per field.
        macro_relative_l2: Equal-field geometric aggregate over designs.
        per_design_macro: Equal-field macro per design.
    """

    key: ArmKey
    designs: tuple[int, ...]
    relative_l2: dict[str, float]
    mae: dict[str, float]
    macro_relative_l2: float
    per_design_macro: dict[int, float]


def _geometric_mean(values: list[float]) -> float:
    """Return the geometric mean of strictly positive values."""
    return math.exp(statistics.fmean(math.log(value) for value in values))


def group_rows(rows: list[dict[str, str]]) -> dict[ArmKey, dict[tuple[int, str], dict[str, float]]]:
    """Group validated metric rows by arm, design, and field.

    Args:
        rows: Rows produced by :func:`~.merge_metric_csvs.load_metric_rows`.

    Returns:
        Nested mapping ``arm -> (design_id, field) -> {relative_l2, mae}``.

    Raises:
        ValueError: If a row lacks the arm columns, which means it came from
            an evaluator older than the geometry-rendering export.
    """
    grouped: dict[ArmKey, dict[tuple[int, str], dict[str, float]]] = defaultdict(dict)
    for row in rows:
        missing = [column for column in ARM_COLUMNS if not row.get(column)]
        if missing:
            raise ValueError(f"metric row is missing {missing}; regenerate the evaluation exports")
        key = ArmKey(method=row["method"], rendering=row["rendering"])
        grouped[key][(int(row["design_id"]), row["field"])] = {
            "relative_l2": float(row["relative_l2"]),
            "mae": float(row["mae"]),
        }
    return dict(grouped)


def summarize_arm(key: ArmKey, cells: dict[tuple[int, str], dict[str, float]]) -> ArmMetrics:
    """Aggregate one arm's per-design, per-field metrics.

    Args:
        key: Arm identity.
        cells: Metric cells of that arm.

    Returns:
        The aggregated :class:`ArmMetrics`.

    Raises:
        ValueError: If the arm does not score every field on every design.
    """
    designs = sorted({design for design, _ in cells})
    fields = sorted({field for _, field in cells})
    missing = [(design, field) for design in designs for field in fields if (design, field) not in cells]
    if missing:
        raise ValueError(f"arm {key.label()} is missing {len(missing)} design/field cells, first={missing[0]}")

    per_design_macro = {
        design: _geometric_mean([cells[(design, field)]["relative_l2"] for field in fields]) for design in designs
    }
    return ArmMetrics(
        key=key,
        designs=tuple(designs),
        relative_l2={
            field: _geometric_mean([cells[(design, field)]["relative_l2"] for design in designs]) for field in fields
        },
        mae={field: statistics.fmean([cells[(design, field)]["mae"] for design in designs]) for field in fields},
        macro_relative_l2=_geometric_mean(list(per_design_macro.values())),
        per_design_macro=per_design_macro,
    )


def paired_contrast(arm: ArmMetrics, baseline: ArmMetrics) -> dict[str, Any]:
    """Contrast one arm against a baseline on the shared designs.

    Args:
        arm: Arm under test.
        baseline: Reference arm.

    Returns:
        Paired ratios and the per-design differences behind them.

    Raises:
        ValueError: If the two arms did not score the same designs and fields.
    """
    if arm.designs != baseline.designs:
        raise ValueError(f"{arm.key.label()} and {baseline.key.label()} scored different designs")
    if sorted(arm.relative_l2) != sorted(baseline.relative_l2):
        raise ValueError(f"{arm.key.label()} and {baseline.key.label()} scored different fields")

    log_differences = [
        math.log(arm.per_design_macro[design]) - math.log(baseline.per_design_macro[design]) for design in arm.designs
    ]
    wins = sum(1 for difference in log_differences if difference < 0)
    return {
        "baseline": baseline.key.label(),
        "macro_error_ratio": math.exp(statistics.fmean(log_differences)),
        "per_field_error_ratio": {
            field: arm.relative_l2[field] / baseline.relative_l2[field] for field in sorted(arm.relative_l2)
        },
        "designs_improved": wins,
        "designs_total": len(arm.designs),
        "per_design_log_ratio": {
            str(design): difference for design, difference in zip(arm.designs, log_differences, strict=True)
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    """Render the reported tables as Markdown.

    Args:
        report: Report payload produced by :func:`build_report`.

    Returns:
        A Markdown document with one endpoint table and one contrast table.
    """
    fields = sorted(next(iter(report["arms"].values()))["relative_l2"])
    lines = [
        "# Single-cell paired comparison",
        "",
        f"Designs scored: {report['design_count']} | fields: {', '.join(fields)}",
        "",
        "| arm | " + " | ".join(f"relL2 {field}" for field in fields) + " | macro relL2 |",
        "| --- | " + " | ".join(["---"] * (len(fields) + 1)) + " |",
    ]
    for label in sorted(report["arms"]):
        arm = report["arms"][label]
        cells = " | ".join(f"{arm['relative_l2'][field]:.6f}" for field in fields)
        lines.append(f"| {label} | {cells} | {arm['macro_relative_l2']:.6f} |")

    if report["contrasts"]:
        lines += [
            "",
            "| arm | baseline | macro ratio | " + " | ".join(f"ratio {field}" for field in fields) + " | improved |",
            "| --- | --- | --- | " + " | ".join(["---"] * (len(fields) + 1)) + " |",
        ]
        for label in sorted(report["contrasts"]):
            contrast = report["contrasts"][label]
            ratios = " | ".join(f"{contrast['per_field_error_ratio'][field]:.4f}" for field in fields)
            lines.append(
                f"| {label} | {contrast['baseline']} | {contrast['macro_error_ratio']:.4f} | {ratios} | "
                f"{contrast['designs_improved']}/{contrast['designs_total']} |"
            )
    return "\n".join(lines) + "\n"


def build_report(inputs: list[Path], *, baseline: ArmKey | None) -> dict[str, Any]:
    """Build the complete comparison payload from frozen metric CSVs.

    Args:
        inputs: Per-design metric CSVs written by the frozen evaluator.
        baseline: Arm every other arm is contrasted against, or ``None`` to
            report endpoints only.

    Returns:
        JSON-compatible report with endpoints, contrasts, and input hashes.

    Raises:
        ValueError: If the requested baseline arm is absent.
    """
    rows, provenance = load_metric_rows(inputs)
    arms = {key: summarize_arm(key, cells) for key, cells in group_rows(rows).items()}
    if baseline is not None and baseline not in arms:
        raise ValueError(f"baseline {baseline.label()} not among {sorted(key.label() for key in arms)}")

    design_counts = {len(arm.designs) for arm in arms.values()}
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "drivaerml_transfer_single_cell_comparison",
        "design_count": sorted(design_counts)[0] if len(design_counts) == 1 else sorted(design_counts),
        "arms": {
            key.label(): {
                "method": key.method,
                "rendering": key.rendering,
                "designs": list(arm.designs),
                "relative_l2": arm.relative_l2,
                "mae": arm.mae,
                "macro_relative_l2": arm.macro_relative_l2,
            }
            for key, arm in arms.items()
        },
        "contrasts": {},
        "inputs": provenance,
    }
    if baseline is not None:
        report["contrasts"] = {
            key.label(): paired_contrast(arm, arms[baseline]) for key, arm in arms.items() if key != baseline
        }
    return report


def main(argv: list[str] | None = None) -> None:
    """Write the reported comparison as JSON and Markdown.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--baseline-method", help="Method code of the reference arm, for example S")
    parser.add_argument("--baseline-rendering", help="Rendering label of the reference arm, for example ps1000-sr9")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)

    baseline = None
    if bool(args.baseline_method) != bool(args.baseline_rendering):
        parser.error("--baseline-method and --baseline-rendering must be given together")
    if args.baseline_method:
        baseline = ArmKey(method=args.baseline_method, rendering=args.baseline_rendering)

    report = build_report(args.inputs, baseline=baseline)
    atomic_write_json(args.output_json, report)
    print(f"report={args.output_json} sha256={sha256_file(args.output_json)}")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(_markdown(report), encoding="utf-8")
        print(f"markdown={args.output_markdown} sha256={sha256_file(args.output_markdown)}")


if __name__ == "__main__":
    main()
