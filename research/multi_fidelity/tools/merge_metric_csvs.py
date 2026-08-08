# Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Merge frozen-evaluation metric CSVs into one deterministic analysis table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

COLUMNS = (
    "method",
    "replicate",
    "n",
    "design_id",
    "field",
    "relative_l2",
    "mae",
)
#: Carried through when every input reports it. Exports written before the
#: metric was added to the callback do not have it, and those tables still merge.
OPTIONAL_COLUMNS = ("mse",)
#: Column order of the merged table, matching the order the callback exports.
MERGED_COLUMN_ORDER = (
    "method",
    "replicate",
    "n",
    "design_id",
    "field",
    "relative_l2",
    "mse",
    "mae",
)
KEY_COLUMNS = COLUMNS[:5]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_identifier(value: str, *, column: str, source: Path, line: int) -> str:
    """Validate and canonicalize a positive integer identifier."""
    try:
        parsed_float = float(value)
        parsed = int(parsed_float)
    except ValueError as exc:
        raise ValueError(f"{source}:{line}: {column} must be an integer") from exc
    if not math.isfinite(parsed_float) or parsed_float != parsed or parsed <= 0:
        raise ValueError(f"{source}:{line}: {column} must be a positive integer")
    return str(parsed)


def _finite_metric(
    value: str,
    *,
    column: str,
    source: Path,
    line: int,
    strictly_positive: bool,
) -> str:
    """Validate a metric and return a stable round-trip representation."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{source}:{line}: {column} must be numeric") from exc
    if not math.isfinite(parsed) or (parsed <= 0 if strictly_positive else parsed < 0):
        relation = "> 0" if strictly_positive else ">= 0"
        raise ValueError(f"{source}:{line}: {column} must be finite and {relation}")
    return repr(parsed)


def load_metric_rows(inputs: list[Path]) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[str]]:
    """Load, validate, and deduplicate frozen metric CSV rows.

    Args:
        inputs: Metric CSVs written by the paired export callback.

    Returns:
        The sorted rows, one provenance record per input, and the column order
        of the merged table, which carries an optional metric only when every
        input reports it.
    """
    if not inputs:
        raise ValueError("at least one input CSV is required")
    resolved = [path.resolve() for path in inputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input CSV paths must be unique")

    headers: dict[Path, list[str]] = {}
    for source in resolved:
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            fieldnames = csv.DictReader(file).fieldnames
        if fieldnames is None:
            raise ValueError(f"{source}: CSV has no header")
        headers[source] = list(fieldnames)
    # An optional column only survives when every table has it; a merged table
    # with holes in a metric column would silently compare unequal populations.
    present_optional = tuple(
        column for column in OPTIONAL_COLUMNS if all(column in header for header in headers.values())
    )

    rows: list[dict[str, str]] = []
    seen: dict[tuple[str, ...], tuple[Path, int]] = {}
    provenance: list[dict[str, Any]] = []
    for source in resolved:
        source_rows = 0
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"{source}: CSV has no header")
            missing = set(COLUMNS) - set(reader.fieldnames)
            if missing:
                raise ValueError(f"{source}: missing columns {sorted(missing)}")
            for line, raw in enumerate(reader, start=2):
                row = {column: raw[column].strip() for column in COLUMNS}
                for column in ("method", "replicate", "field"):
                    if not row[column]:
                        raise ValueError(f"{source}:{line}: {column} must be non-empty")
                row["n"] = _integer_identifier(row["n"], column="n", source=source, line=line)
                row["design_id"] = _integer_identifier(
                    row["design_id"],
                    column="design_id",
                    source=source,
                    line=line,
                )
                row["relative_l2"] = _finite_metric(
                    row["relative_l2"],
                    column="relative_l2",
                    source=source,
                    line=line,
                    strictly_positive=True,
                )
                for column in OPTIONAL_COLUMNS:
                    if column not in present_optional:
                        continue
                    row[column] = _finite_metric(
                        raw[column].strip(),
                        column=column,
                        source=source,
                        line=line,
                        strictly_positive=False,
                    )
                row["mae"] = _finite_metric(
                    row["mae"],
                    column="mae",
                    source=source,
                    line=line,
                    strictly_positive=False,
                )
                key = tuple(row[column] for column in KEY_COLUMNS)
                if key in seen:
                    previous_source, previous_line = seen[key]
                    raise ValueError(
                        f"duplicate metric key {key}: {previous_source}:{previous_line} and {source}:{line}"
                    )
                seen[key] = (source, line)
                rows.append(row)
                source_rows += 1
        if source_rows == 0:
            raise ValueError(f"{source}: CSV contains no metric rows")
        provenance.append(
            {
                "path": str(source),
                "sha256": _sha256(source),
                "rows": source_rows,
            }
        )

    rows.sort(
        key=lambda row: (
            row["method"],
            row["replicate"],
            int(row["n"]),
            int(row["design_id"]),
            row["field"],
        )
    )
    carried = {*COLUMNS, *present_optional}
    return rows, provenance, [column for column in MERGED_COLUMN_ORDER if column in carried]


def merge_metric_csvs(inputs: list[Path], output: Path) -> dict[str, Any]:
    """Write one atomic deterministic CSV and return provenance metadata."""
    resolved_output = output.resolve()
    if resolved_output in {path.resolve() for path in inputs}:
        raise ValueError("output CSV must not also be an input")
    rows, provenance, fieldnames = load_metric_rows(inputs)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(f".{resolved_output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, resolved_output)
    return {
        "output": str(resolved_output),
        "output_sha256": _sha256(resolved_output),
        "rows": len(rows),
        "inputs": provenance,
    }


def main() -> None:
    """Parse CLI arguments, merge CSVs, and print auditable provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    try:
        result = merge_metric_csvs(args.inputs, args.output)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.provenance is not None:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
