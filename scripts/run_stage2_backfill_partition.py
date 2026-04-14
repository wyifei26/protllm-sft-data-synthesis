#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sft_pipeline.config import RunConfig
from sft_pipeline.io_utils import dump_json, dump_jsonl, extract_primary_accession, load_jsonl, read_sample_records
from sft_pipeline.llm_client import build_json_client
from sft_pipeline.quality import build_qc_report, estimate_full_runtime
from sft_pipeline.stage2_qa import run_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill remaining stage2 records in deterministic partitions.")
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--input-parquet", default="data/triples_formed.parquet")
    parser.add_argument("--output-root", default="artifacts/first_delivery")
    parser.add_argument("--model", default="/GenSIvePFS/users/model/gemma-4-31B-it")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "openai_api"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--stage2-batch-size", type=int, default=32)
    parser.add_argument("--stage2-max-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--num-partitions", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_partitions <= 0:
        raise ValueError("--num-partitions must be positive")
    if not 0 <= args.partition_index < args.num_partitions:
        raise ValueError("--partition-index must be in [0, num-partitions)")

    partition_tag = f"part{args.partition_index}of{args.num_partitions}"
    config = RunConfig(
        input_parquet=args.input_parquet,
        sample_size=args.sample_size,
        start_offset=args.start_offset,
        output_root=args.output_root,
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        stage2_batch_size=args.stage2_batch_size,
        stage2_max_tokens=args.stage2_max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        stage2_output_name=f"stage2_qa.{partition_tag}.jsonl",
        qc_output_name=f"qc_report.{partition_tag}.json",
        timing_output_name=f"timing_report.{partition_tag}.json",
    )
    config.ensure_dirs()

    sample_config = RunConfig(
        input_parquet=args.input_parquet,
        sample_size=args.sample_size,
        start_offset=args.start_offset,
        output_root=args.output_root,
    )
    if sample_config.sample_path.exists():
        records = load_jsonl(sample_config.sample_path)
    else:
        records = read_sample_records(sample_config)
        dump_jsonl(sample_config.sample_path, records)

    records_by_accession = {record["primary_accession"]: record for record in records}
    raw_stage1_rows = load_jsonl(sample_config.stage1_path)
    valid_stage1_rows = []
    for row in raw_stage1_rows:
        accession = extract_primary_accession(row)
        if accession and accession in records_by_accession:
            normalized_row = dict(row)
            normalized_row["primary_accession"] = accession
            valid_stage1_rows.append(normalized_row)

    completed_accessions = set()
    if sample_config.stage2_path.exists():
        for row in load_jsonl(sample_config.stage2_path):
            accession = extract_primary_accession(row)
            if accession:
                completed_accessions.add(accession)

    candidate_accessions = []
    seen_candidate_accessions = set()
    for row in valid_stage1_rows:
        accession = row["primary_accession"]
        if accession in completed_accessions or accession in seen_candidate_accessions:
            continue
        seen_candidate_accessions.add(accession)
        candidate_accessions.append(accession)
    partition_accessions = candidate_accessions[args.partition_index :: args.num_partitions]
    partition_accession_set = set(partition_accessions)
    partition_records = [records_by_accession[accession] for accession in partition_accessions]
    partition_stage1_rows = [
        row
        for row in valid_stage1_rows
        if row["primary_accession"] in partition_accession_set
    ]

    print(
        f"STAGE2_BACKFILL partition={partition_tag} total_candidates={len(candidate_accessions)} "
        f"partition_records={len(partition_records)} completed_main={len(completed_accessions)}",
        flush=True,
    )

    client = build_json_client(config)
    try:
        stage2_rows, stage2_usage = run_stage2(partition_records, partition_stage1_rows, config, client=client)
    finally:
        client.close()

    qc_report = build_qc_report(partition_records, stage2_rows, stage2_usage, config)
    dump_json(config.qc_path, qc_report)
    dump_json(
        config.timing_path,
        estimate_full_runtime(len(partition_records), len(partition_records), qc_report, wall_clock_sec=0.0),
    )


if __name__ == "__main__":
    main()
