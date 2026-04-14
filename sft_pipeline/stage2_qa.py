from __future__ import annotations

from typing import Any

from tqdm import tqdm

from .config import RunConfig
from .io_utils import append_jsonl, extract_primary_accession, load_jsonl
from .llm_client import JSONClient
from .prompts import build_stage2_messages, dry_run_stage2


def run_stage2(
    records: list[dict[str, Any]],
    stage1_records: list[dict[str, Any]],
    config: RunConfig,
    client: JSONClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage1_map: dict[str, dict[str, Any]] = {}
    for row in stage1_records:
        accession = extract_primary_accession(row)
        if accession:
            stage1_map[accession] = row
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

    outputs: list[dict[str, Any]] = load_jsonl(config.stage2_path) if config.stage2_path.exists() else []
    usage_rows: list[dict[str, Any]] = []
    if client is None:
        raise ValueError("run_stage2 requires a JSON client when dry_run is False")

    completed_accessions = {
        accession
        for row in outputs
        if (accession := extract_primary_accession(row)) is not None
    }
    pending_records: list[dict[str, Any]] = []
    seen_pending_accessions: set[str] = set()
    for record in records:
        accession = record["primary_accession"]
        if accession in seen_pending_accessions:
            continue
        if accession not in stage1_map or accession in completed_accessions:
            continue
        seen_pending_accessions.add(accession)
        pending_records.append(record)

    for start in tqdm(range(0, len(pending_records), config.stage2_batch_size), desc="stage2"):
        batch_records = pending_records[start : start + config.stage2_batch_size]
        batch_messages = [
            build_stage2_messages(record, stage1_map[record["primary_accession"]], config)
            for record in batch_records
        ]
        batch_outputs, batch_usage = client.generate_json_batch(batch_messages, max_tokens=config.stage2_max_tokens)
        materialized_outputs = [
            _materialize_modalities(record, result, config)
            for record, result in zip(batch_records, batch_outputs)
        ]
        append_jsonl(config.stage2_path, materialized_outputs)
        for record, result, usage in zip(batch_records, materialized_outputs, batch_usage):
            usage["primary_accession"] = record["primary_accession"]
            usage["stage"] = "stage2"
            outputs.append(result)
            usage_rows.append(usage)
    return outputs, usage_rows


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
