"""CSV output for per-round metric rows (schema: fl.metrics.ROW_KEYS). None is written as an empty cell."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path


def write_rows_csv(rows: list[dict], path, overwrite: bool = False) -> Path:
    """Write rows to ``path`` (parents created).

    Default: append; the header is written once, and rows whose keys differ from an existing file's header raise
    ValueError instead of silently misaligning columns. ``overwrite=True`` replaces the file atomically (temp file
    + rename), so an interrupted run never leaves a partial file that looks complete.
    """
    path = Path(path)
    if not rows:
        return path
    keys = list(rows[0])
    if any(list(r) != keys for r in rows):
        raise ValueError("all rows must have the same keys in the same order")
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
        return path
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != keys:
            raise ValueError(f"{path} has columns {header}, rows have {keys}")
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    return path


def _parse(v: str):
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() and not math.isinf(f) else f


def read_rows_csv(path) -> list[dict]:
    """Inverse of write_rows_csv: numbers parsed back to int/float, empty cells to None."""
    with Path(path).open(newline="") as f:
        return [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]
