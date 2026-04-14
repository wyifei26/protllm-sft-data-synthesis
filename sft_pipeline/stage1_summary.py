from __future__ import annotations

from typing import Any

from tqdm import tqdm

from .config import RunConfig
from .io_utils import append_jsonl, load_jsonl
from .llm_client import JSONClient
from .prompts import build_stage1_messages, dry_run_stage1


def run_stage1(
    records: list[dict[str, Any]],
    config: RunConfig,
    client: JSONClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if config.dry_run:
        outputs = [dry_run_stage1(record, config) for record in records]
        usage = [
            {
                "primary_accession": record["primary_accession"],
                "stage": "stage1",
                "latency_sec": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            for record in records
        ]
        return outputs, usage

    outputs: list[dict[str, Any]] = load_jsonl(config.stage1_path) if config.stage1_path.exists() else []
    usage_rows: list[dict[str, Any]] = []
    if client is None:
        raise ValueError("run_stage1 requires a JSON client when dry_run is False")

    completed = len(outputs)
    for start in tqdm(range(completed, len(records), config.stage1_batch_size), desc="stage1"):
        batch_records = records[start : start + config.stage1_batch_size]
        batch_messages = [build_stage1_messages(record, config) for record in batch_records]
        batch_outputs, batch_usage = client.generate_json_batch(batch_messages, max_tokens=config.stage1_max_tokens)
        append_jsonl(config.stage1_path, batch_outputs)
        for record, result, usage in zip(batch_records, batch_outputs, batch_usage):
            usage["primary_accession"] = record["primary_accession"]
            usage["stage"] = "stage1"
            outputs.append(result)
            usage_rows.append(usage)
    return outputs, usage_rows
