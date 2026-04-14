# ProtLLM SFT Data Synthesis

这个仓库用于从 `data/triples_formed.parquet` 合成 ProtLLM 的第一版 SFT-QA 数据。当前默认模型是 `/GenSIvePFS/users/model/gemma-4-31B-it`，默认通过进程内 `vllm.LLM` 在 8 卡上运行。

当前维护的主流程是两阶段：

1. `stage1`：把结构化元数据字段转写成英文 summary。
2. `stage2`：基于 `stage1` summary 生成四类 QA 任务 `t2bb / t2s / bb2t / s2t`。

仓库已经支持三种运行方式：

- `stage1 only`：先批量产出 summary。
- `stage2 streaming`：流式追读不断增长的 `stage1_summaries.jsonl`。
- `stage2 backfill`：把尚未消费完的 `stage1` backlog 切成多个分区并行补齐。

## 数据与产物

- 输入数据：`data/triples_formed.parquet`
- 核心代码：`sft_pipeline/`
- 运行脚本：`scripts/`
- 默认输出根目录：`artifacts/full_delivery_gemma4`

每次运行都会落到一个按切片命名的目录：

```text
artifacts/full_delivery_gemma4/offset_{start_offset}_sample_{sample_size}/
```

常见产物：

- `sample_records.jsonl`：本次切片对应的标准化输入记录
- `stage1_summaries.jsonl`：stage1 summary 结果，支持断点续跑
- `stage2_qa.jsonl`：主 stage2 结果
- `stage2_qa.part{K}of{N}.jsonl`：stage2 分区回填结果
- `qc_report*.json`：QA 质量检查结果
- `timing_report*.json`：吞吐和耗时估算

## 当前实现约束

- 默认使用英文合成。
- `stage1` 和 `stage2` 都支持续跑，已有结果会被复用。
- `stage2` 现在对脏行更稳健：
  - 兼容 `primary_accession` 和历史遗留的 `primary_access`
  - 会跳过无法识别 accession 的坏行
  - 会避免重复 accession 被重复处理
- 当前数据里 `backbone` 基本为空，所以 `t2bb` 和 `bb2t` 仍会保留结构占位符，`t2s` 和 `s2t` 是主要可用产出。

## 环境准备

推荐直接使用仓库内置环境脚本：

```bash
bash scripts/setup_env.sh
```

默认环境：

- conda env: `/GenSIvePFS/users/yfwang/miniconda3/envs/protllm-sft-vllm`
- Python: `3.12`
- 关键依赖：`vllm`、`transformers`、`pyarrow`、`tqdm`

`setup_env.sh` 已经做了并发保护，多个 Volc 任务同时启动时不会再互相破坏同一个环境。

## 目录说明

当前保留的脚本只覆盖仍在使用的流程：

- `scripts/setup_env.sh`：准备或复用运行环境
- `scripts/run_stage1_only.py`：stage1 Python 入口
- `scripts/run_stage1_vllm_8gpu.sh`：8 卡 stage1 本地启动
- `scripts/run_stage2_streaming.py`：流式消费 stage1 的 stage2 入口
- `scripts/run_stage2_streaming_vllm_8gpu.sh`：8 卡 stage2 streaming 本地启动
- `scripts/run_stage2_backfill_partition.py`：stage2 分区回填入口
- `scripts/run_stage2_backfill_vllm_8gpu.sh`：8 卡 stage2 backfill 本地启动
- `scripts/volc_launch_gemma4_stage1.sh`：Volc stage1 任务入口
- `scripts/volc_launch_gemma4_stage2_streaming.sh`：Volc stage2 streaming 任务入口
- `scripts/volc_launch_gemma4_stage2_backfill.sh`：Volc stage2 backfill 任务入口

## 本地运行

先激活环境：

```bash
source /GenSIvePFS/users/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate /GenSIvePFS/users/yfwang/miniconda3/envs/protllm-sft-vllm
export PYTHONPATH=.
```

### 1. 运行 stage1

```bash
SAMPLE_SIZE=256 \
START_OFFSET=0 \
bash scripts/run_stage1_vllm_8gpu.sh
```

常用覆盖参数：

```bash
SAMPLE_SIZE=286831 \
START_OFFSET=286830 \
STAGE1_BATCH_SIZE=32 \
MAX_NUM_SEQS=64 \
GPU_MEMORY_UTILIZATION=0.92 \
bash scripts/run_stage1_vllm_8gpu.sh
```

### 2. 运行 stage2 streaming

适合一个 8 卡任务持续追读另一个正在写入的 `stage1_summaries.jsonl`：

```bash
SAMPLE_SIZE=286831 \
START_OFFSET=286830 \
POLL_SECONDS=60 \
bash scripts/run_stage2_streaming_vllm_8gpu.sh
```

### 3. 运行 stage2 backfill

适合 `stage1` 已经跑出大量 backlog，想把剩余未消费部分切成多个 8 卡任务并行补齐。

例如两路并行时：

```bash
SAMPLE_SIZE=286831 \
START_OFFSET=286830 \
PARTITION_INDEX=0 \
NUM_PARTITIONS=2 \
bash scripts/run_stage2_backfill_vllm_8gpu.sh
```

```bash
SAMPLE_SIZE=286831 \
START_OFFSET=286830 \
PARTITION_INDEX=1 \
NUM_PARTITIONS=2 \
bash scripts/run_stage2_backfill_vllm_8gpu.sh
```

每个分区会写自己的 `stage2_qa.part{K}of{N}.jsonl`，避免并发写同一个输出文件。

## Volc 提交

常用任务配置放在：

- `/GenSIvePFS/users/yfwang/volc/task-configs/protllm/`

当前实际在用的是这几类配置：

- `gemma4_stage1_only_8gpu_*.yaml`
- `gemma4_stage2_stream_8gpu_*.yaml`
- `gemma4_stage2_backfill_8gpu_*.yaml`

仓库里的 Volc launch 脚本只负责：

1. 打印启动标记
2. 调用 `setup_env.sh`
3. 运行对应的 8 卡本地脚本

## 代码结构

- `sft_pipeline/config.py`：运行配置与输出路径
- `sft_pipeline/io_utils.py`：parquet 读取、jsonl 读写、accession 兼容逻辑
- `sft_pipeline/llm_client.py`：vLLM / API 客户端与 JSON 容错解析
- `sft_pipeline/stage1_summary.py`：stage1 合成与续跑
- `sft_pipeline/stage2_qa.py`：stage2 合成、去重与续跑
- `sft_pipeline/quality.py`：QA 质检与耗时估算
- `sft_pipeline/prompts.py`：stage1 / stage2 提示词

## 维护建议

- 不要让多个任务同时写同一个 `stage2_qa.jsonl`。
- 多任务并行补数时，优先使用 `run_stage2_backfill_partition.py` 的分区方式。
- 如果更换模型或 `transformers` / `vllm` 版本，优先检查 `scripts/setup_env.sh` 里的版本钉死逻辑。
