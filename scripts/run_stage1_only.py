#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sft_pipeline.config import RunConfig
from sft_pipeline.io_utils import dump_jsonl, read_sample_records
from sft_pipeline.llm_client import build_json_client
from sft_pipeline.stage1_summary import run_stage1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only stage1 summary synthesis.")
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--input-parquet", default="data/triples_formed.parquet")
    parser.add_argument("--output-root", default="artifacts/first_delivery")
    parser.add_argument("--model", default="/GenSIvePFS/users/model/gemma-4-31B-it")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "openai_api"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--stage1-batch-size", type=int, default=32)
    parser.add_argument("--stage1-max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        stage1_batch_size=args.stage1_batch_size,
        stage1_max_tokens=args.stage1_max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
    )
    config.ensure_dirs()

    records = read_sample_records(config)
    if not config.sample_path.exists():
        dump_jsonl(config.sample_path, records)

    client = build_json_client(config)
    try:
        run_stage1(records, config, client=client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
