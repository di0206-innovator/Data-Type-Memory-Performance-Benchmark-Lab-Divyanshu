"""Export helpers for DataType Lab benchmark results.

This module provides functions that serialise raw benchmark result dicts
(as returned by :mod:`benchmark_lab.engine`) into portable file formats
suitable for reproducible experiment records.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .engine import system_info


def results_bundle_csv(
    memory_rows: list[dict[str, Any]],
    speed_rows: list[dict[str, Any]],
    precision_rows: list[dict[str, Any]],
) -> bytes:
    """Combine memory, speed and precision results into a single UTF-8 CSV.

    A ``section`` column is prepended to each row so the three result sets
    can be distinguished after concatenation.  Columns that exist in only
    one section are filled with NaN in the others.

    Args:
        memory_rows: Output of :func:`~benchmark_lab.engine.memory_benchmark`.
        speed_rows: Output of :func:`~benchmark_lab.engine.speed_benchmark`.
        precision_rows: Output of :func:`~benchmark_lab.engine.precision_benchmark`.

    Returns:
        UTF-8 encoded CSV bytes, or an empty ``b""`` when all inputs are empty.
    """
    frames = []
    for section, rows in (
        ("memory", memory_rows),
        ("speed", speed_rows),
        ("precision", precision_rows),
    ):
        if rows:
            frame = pd.DataFrame(rows)
            frame.insert(0, "section", section)
            frames.append(frame)

    if not frames:
        return b""

    output = io.StringIO()
    pd.concat(frames, ignore_index=True, sort=False).to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def metadata_json(config: dict[str, Any] | None = None) -> bytes:
    """Serialise experiment configuration and environment metadata to JSON.

    Args:
        config: Optional dict of experiment parameters (e.g. dataset size,
            chosen dtypes, timing settings) to embed alongside environment info.

    Returns:
        Pretty-printed UTF-8 JSON bytes containing the UTC timestamp,
        environment metadata, and the supplied configuration.
    """
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": system_info(),
        "experiment_config": config or {},
    }
    return json.dumps(payload, indent=2).encode("utf-8")
