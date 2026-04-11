from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from .config import RunConfig
from .llm_client import LocalChatClient
from .prompts import build_stage1_messages, dry_run_stage1


def run_stage1(records: list[dict[str, Any]], config: RunConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

    client = LocalChatClient(config)
    outputs: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        future_map = {
            executor.submit(
                _run_one_stage1,
                client,
                record,
                config,
            ): record["primary_accession"]
            for record in records
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="stage1"):
            result, usage = future.result()
            outputs.append(result)
            usage_rows.append(usage)
    outputs.sort(key=lambda item: item["primary_accession"])
    usage_rows.sort(key=lambda item: item["primary_accession"])
    return outputs, usage_rows


def _run_one_stage1(
    client: LocalChatClient,
    record: dict[str, Any],
    config: RunConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result, usage = client.generate_json(
        build_stage1_messages(record, config),
        max_tokens=config.stage1_max_tokens,
    )
    usage["primary_accession"] = record["primary_accession"]
    usage["stage"] = "stage1"
    return result, usage
