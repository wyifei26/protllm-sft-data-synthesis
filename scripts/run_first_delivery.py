#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from sft_pipeline.config import RunConfig
from sft_pipeline.io_utils import dump_json, dump_jsonl, read_sample_records
from sft_pipeline.quality import build_qc_report, estimate_full_runtime
from sft_pipeline.stage1_summary import run_stage1
from sft_pipeline.stage2_qa import run_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first SFT-QA delivery pipeline.")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--input-parquet", default="data/triples_formed.parquet")
    parser.add_argument("--output-root", default="artifacts/first_delivery")
    parser.add_argument("--model", default="/data/cloudroot/minimax-m2.5-deploy/models/MiniMax-M2.5")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--stage1-max-tokens", type=int, default=1200)
    parser.add_argument("--stage2-max-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = RunConfig(
        input_parquet=args.input_parquet,
        sample_size=args.sample_size,
        output_root=args.output_root,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        stage1_max_tokens=args.stage1_max_tokens,
        stage2_max_tokens=args.stage2_max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    config.ensure_dirs()

    records = read_sample_records(config)
    dump_jsonl(config.sample_path, records)

    stage1_rows, stage1_usage = run_stage1(records, config)
    dump_jsonl(config.stage1_path, stage1_rows)

    stage2_rows, stage2_usage = run_stage2(records, stage1_rows, config)
    dump_jsonl(config.stage2_path, stage2_rows)

    timings = stage1_usage + stage2_usage
    qc_report = build_qc_report(records, stage2_rows, timings, config)
    dump_json(config.qc_path, qc_report)
    wall_clock_sec = time.perf_counter() - started
    dump_json(config.timing_path, estimate_full_runtime(len(records), 573661, qc_report, wall_clock_sec=wall_clock_sec))


if __name__ == "__main__":
    main()
