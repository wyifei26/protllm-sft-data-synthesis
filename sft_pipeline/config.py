from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    input_parquet: Path = Path("data/triples_formed.parquet")
    sample_size: int = 256
    start_offset: int = 0
    output_root: Path = Path("artifacts/first_delivery")
    model: str = "/GenSIvePFS/users/model/gemma-4-31B-it"
    backend: str = "vllm"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    concurrency: int = 16
    stage1_batch_size: int = 32
    stage2_batch_size: int = 32
    stage1_max_tokens: int = 1200
    stage2_max_tokens: int = 1600
    temperature: float = 0.2
    timeout: float = 300.0
    tensor_parallel_size: int = 8
    gpu_memory_utilization: float = 0.92
    max_model_len: int = 32768
    max_num_seqs: int = 64
    enable_chunked_prefill: bool = True
    enforce_eager: bool = False
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    stop_strings: tuple[str, ...] = ("<turn|>", "<eos>", "<|tool_response|>")
    summary_fields: tuple[str, ...] = (
        "secondary",
        "interpro",
        "cath",
        "pfam",
        "go",
        "disease",
        "function",
        "domain",
        "subunit",
        "catalytic",
    )
    stage2_tasks: tuple[str, ...] = ("t2bb", "t2s", "bb2t", "s2t")
    sequence_placeholder: str = "<sequence><|seq_start|>{SEQUENCE}<|seq_end|></sequence>"
    structure_placeholder: str = "<structure><|struct_start|>{BACKBONE}<|struct_end|></structure>"
    metadata_version: str = "v1"
    dry_run: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    stage2_output_name: str = "stage2_qa.jsonl"
    qc_output_name: str = "qc_report.json"
    timing_output_name: str = "timing_report.json"

    def __post_init__(self) -> None:
        self.input_parquet = Path(self.input_parquet)
        self.output_root = Path(self.output_root)

    @property
    def run_dir(self) -> Path:
        return self.output_root / f"offset_{self.start_offset}_sample_{self.sample_size}"

    @property
    def stage1_path(self) -> Path:
        return self.run_dir / "stage1_summaries.jsonl"

    @property
    def stage2_path(self) -> Path:
        return self.run_dir / self.stage2_output_name

    @property
    def qc_path(self) -> Path:
        return self.run_dir / self.qc_output_name

    @property
    def timing_path(self) -> Path:
        return self.run_dir / self.timing_output_name

    @property
    def sample_path(self) -> Path:
        return self.run_dir / "sample_records.jsonl"

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
