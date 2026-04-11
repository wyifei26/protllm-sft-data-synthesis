from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    input_parquet: Path = Path("data/triples_formed.parquet")
    sample_size: int = 256
    output_root: Path = Path("artifacts/first_delivery")
    model: str = "/data/cloudroot/minimax-m2.5-deploy/models/MiniMax-M2.5"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    concurrency: int = 16
    stage1_max_tokens: int = 1200
    stage2_max_tokens: int = 1600
    temperature: float = 0.2
    timeout: float = 300.0
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

    def __post_init__(self) -> None:
        self.input_parquet = Path(self.input_parquet)
        self.output_root = Path(self.output_root)

    @property
    def run_dir(self) -> Path:
        return self.output_root / f"sample_{self.sample_size}"

    @property
    def stage1_path(self) -> Path:
        return self.run_dir / "stage1_summaries.jsonl"

    @property
    def stage2_path(self) -> Path:
        return self.run_dir / "stage2_qa.jsonl"

    @property
    def qc_path(self) -> Path:
        return self.run_dir / "qc_report.json"

    @property
    def timing_path(self) -> Path:
        return self.run_dir / "timing_report.json"

    @property
    def sample_path(self) -> Path:
        return self.run_dir / "sample_records.jsonl"

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
