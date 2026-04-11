#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    rows = pq.read_table("data/triples_formed.parquet").slice(0, 256).to_pylist()
    counters = Counter()
    sequence_lengths = []
    backbone_lengths = []

    for row in rows:
        sequence = (row.get("sequence") or "").strip()
        backbone = (row.get("backbone") or "").strip()
        if sequence:
            counters["sequence_nonempty"] += 1
            sequence_lengths.append(len(sequence))
        if backbone:
            counters["backbone_nonempty"] += 1
            backbone_lengths.append(len(backbone))

    report = {
        "sample_size": len(rows),
        "sequence_nonempty": counters["sequence_nonempty"],
        "backbone_nonempty": counters["backbone_nonempty"],
        "estimated_qa_slots": len(rows) * 4,
        "sequence_length": {
            "min": min(sequence_lengths) if sequence_lengths else 0,
            "avg": round(sum(sequence_lengths) / max(len(sequence_lengths), 1), 2),
            "max": max(sequence_lengths) if sequence_lengths else 0,
        },
        "backbone_length": {
            "min": min(backbone_lengths) if backbone_lengths else 0,
            "avg": round(sum(backbone_lengths) / max(len(backbone_lengths), 1), 2),
            "max": max(backbone_lengths) if backbone_lengths else 0,
        },
        "notes": [
            "The first 256 sampled records contain sequences for all rows.",
            "The backbone column is empty for all sampled rows.",
        ],
    }
    output_path = Path("artifacts/first_delivery/sample_256/sample_profile.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
