from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .config import RunConfig


def read_sample_records(config: RunConfig) -> list[dict[str, Any]]:
    table = pq.read_table(config.input_parquet).slice(0, config.sample_size)
    rows = table.to_pylist()
    parsed_rows = []
    for row in rows:
        parsed_rows.append(_normalize_row(row, config))
    return parsed_rows


def _normalize_row(row: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    normalized = dict(row)
    normalized["sequence_available"] = bool((row.get("sequence") or "").strip())
    normalized["backbone_available"] = bool((row.get("backbone") or "").strip())
    for field in config.summary_fields:
        raw = row.get(field) or ""
        if raw:
            try:
                normalized[field] = json.loads(raw)
            except json.JSONDecodeError:
                normalized[field] = {"summary": "", "details": [], "raw_text": raw}
        else:
            normalized[field] = {"summary": "", "details": []}
    return normalized


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
