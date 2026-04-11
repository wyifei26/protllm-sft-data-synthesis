from __future__ import annotations

from collections import Counter
from typing import Any

from .config import RunConfig


def build_qc_report(
    records: list[dict[str, Any]],
    stage2_records: list[dict[str, Any]],
    timings: list[dict[str, Any]],
    config: RunConfig,
) -> dict[str, Any]:
    expected_accessions = {row["primary_accession"] for row in records}
    seen_accessions = {row["primary_accession"] for row in stage2_records}
    task_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    issues: list[str] = []
    total_nonempty_questions = 0
    total_nonempty_answers = 0

    for row in stage2_records:
        tasks = row.get("tasks", [])
        if len(tasks) != 4:
            issues.append(f"{row['primary_accession']}: expected 4 tasks, got {len(tasks)}")
        present_types = {task.get("task_type") for task in tasks}
        missing = set(config.stage2_tasks) - present_types
        if missing:
            issues.append(f"{row['primary_accession']}: missing task types {sorted(missing)}")
        for task in tasks:
            task_type = task.get("task_type", "unknown")
            status = task.get("status", "unknown")
            task_counter[task_type] += 1
            status_counter[f"{task_type}:{status}"] += 1
            question = (task.get("question") or "").strip()
            answer = (task.get("answer") or "").strip()
            if question:
                total_nonempty_questions += 1
            if answer:
                total_nonempty_answers += 1
            if status == "ok" and (not question or not answer):
                issues.append(f"{row['primary_accession']}:{task_type}: status ok but question/answer empty")

    timing_by_stage = _summarize_timings(timings)
    return {
        "sample_size": len(records),
        "expected_accessions": len(expected_accessions),
        "generated_accessions": len(seen_accessions),
        "missing_accessions": sorted(expected_accessions - seen_accessions)[:20],
        "task_counts": dict(task_counter),
        "status_counts": dict(status_counter),
        "nonempty_questions": total_nonempty_questions,
        "nonempty_answers": total_nonempty_answers,
        "issues": issues,
        "timing_summary": timing_by_stage,
        "notes": [
            "backbone is empty for all sampled records, so t2bb and bb2t retain <STRUCTURE_TOKENS> placeholders for later backfill.",
            "t2s and s2t can be generated because sequence is present in the sample.",
        ],
    }


def _summarize_timings(timings: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in timings:
        by_stage.setdefault(row["stage"], []).append(row)
    summary: dict[str, Any] = {}
    for stage, rows in by_stage.items():
        latencies = [float(row.get("latency_sec") or 0.0) for row in rows]
        total_tokens = [int(row.get("total_tokens") or 0) for row in rows]
        completion_tokens = [int(row.get("completion_tokens") or 0) for row in rows]
        summary[stage] = {
            "requests": len(rows),
            "total_latency_sec": round(sum(latencies), 3),
            "avg_latency_sec": round(sum(latencies) / max(len(latencies), 1), 3),
            "total_tokens": sum(total_tokens),
            "completion_tokens": sum(completion_tokens),
            "avg_output_tokens": round(sum(completion_tokens) / max(len(completion_tokens), 1), 2),
        }
    return summary


def estimate_full_runtime(
    sample_size: int,
    corpus_size: int,
    qc_report: dict[str, Any],
    wall_clock_sec: float | None = None,
) -> dict[str, Any]:
    timing = qc_report.get("timing_summary", {})
    summed_request_latency = sum(stage["total_latency_sec"] for stage in timing.values())
    result = {
        "sample_size": sample_size,
        "corpus_size": corpus_size,
        "summed_request_latency_sec": round(summed_request_latency, 3),
    }
    if wall_clock_sec is not None:
        per_record = wall_clock_sec / max(sample_size, 1)
        estimated = per_record * corpus_size
        result.update(
            {
                "measured_total_wall_clock_sec": round(wall_clock_sec, 3),
                "measured_wall_clock_per_record_sec": round(per_record, 4),
                "estimated_full_wall_clock_sec": round(estimated, 3),
                "estimated_full_wall_clock_hours": round(estimated / 3600, 3),
            }
        )
    return result
