"""CSV and report helpers."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .engine import system_info


def results_bundle_csv(memory_rows: list[dict[str, Any]], speed_rows: list[dict[str, Any]], precision_rows: list[dict[str, Any]]) -> bytes:
    """Combine unlike result tables into one CSV with a section column."""
    frames = []
    for section, rows in (("memory", memory_rows), ("speed", speed_rows), ("precision", precision_rows)):
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
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": system_info(),
        "experiment_config": config or {},
    }
    return json.dumps(payload, indent=2).encode("utf-8")
