"""Small, explicit result writers; no Excel mutation or implicit global paths."""
import csv
import json
from pathlib import Path
from mars_ipp.utils.config import validate_output_path
import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path, value):
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path, rows):
    """Append a pending batch, checking the existing CSV schema first."""
    if not rows:
        return
    path = validate_output_path(path)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f'Inconsistent CSV fields: {path}')
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline='') as stream:
            if next(csv.reader(stream)) != fields:
                raise ValueError(f'CSV header differs: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
