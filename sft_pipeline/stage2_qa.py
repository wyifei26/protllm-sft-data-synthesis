from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from .config import RunConfig
from .llm_client import LocalChatClient
from .prompts import build_stage2_messages, dry_run_stage2


def run_stage2(
    records: list[dict[str, Any]],
    stage1_records: list[dict[str, Any]],
    config: RunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage1_map = {row["primary_accession"]: row for row in stage1_records}
    if config.dry_run:
        outputs = [
            _materialize_modalities(record, dry_run_stage2(record, stage1_map[record["primary_accession"]], config), config)
            for record in records
        ]
        usage = [
            {
                "primary_accession": record["primary_accession"],
                "stage": "stage2",
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
                _run_one_stage2,
                client,
                record,
                stage1_map[record["primary_accession"]],
                config,
            ): record["primary_accession"]
            for record in records
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="stage2"):
            result, usage = future.result()
            outputs.append(result)
            usage_rows.append(usage)
    outputs.sort(key=lambda item: item["primary_accession"])
    usage_rows.sort(key=lambda item: item["primary_accession"])
    return outputs, usage_rows


def _run_one_stage2(
    client: LocalChatClient,
    record: dict[str, Any],
    summary_record: dict[str, Any],
    config: RunConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result, usage = client.generate_json(
        build_stage2_messages(record, summary_record, config),
        max_tokens=config.stage2_max_tokens,
    )
    usage["primary_accession"] = record["primary_accession"]
    usage["stage"] = "stage2"
    return _materialize_modalities(record, result, config), usage


def _materialize_modalities(record: dict[str, Any], result: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    sequence = record.get("sequence") or ""
    backbone = record.get("backbone") or ""
    sequence_token_text = _format_sequence_tokens(sequence, config)
    structure_token_text = _format_structure_tokens(backbone, config)
    for task in result.get("tasks", []):
        question = (task.get("question") or "")
        answer = (task.get("answer") or "")
        question = question.replace(config.sequence_placeholder, sequence_token_text)
        answer = answer.replace(config.sequence_placeholder, sequence_token_text)
        question = question.replace(config.structure_placeholder, structure_token_text)
        answer = answer.replace(config.structure_placeholder, structure_token_text)
        task_type = task.get("task_type") or ""
        if task_type == "bb2t" and structure_token_text not in question:
            question = (
                "Given the following protein structure tokens, describe the protein's likely "
                f"architecture, domains, and biological characteristics in English: {structure_token_text}"
            )
        if task_type == "s2t" and sequence_token_text not in question:
            question = (
                "Given the following protein sequence tokens, describe the protein's likely "
                f"function, domains, and relevant properties in English: {sequence_token_text}"
            )
        task["question"] = question
        task["answer"] = answer
    return result


def _format_sequence_tokens(sequence: str, config: RunConfig) -> str:
    token_body = "".join(f"[{residue}]" for residue in sequence) if sequence else "{SEQUENCE}"
    return config.sequence_placeholder.replace("{SEQUENCE}", token_body)


def _format_structure_tokens(backbone: str, config: RunConfig) -> str:
    token_body = backbone if backbone else "{BACKBONE}"
    return config.structure_placeholder.replace("{BACKBONE}", token_body)
