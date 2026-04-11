# SFT数据构造

- 数据目录：`data/`
- 主输入文件：`data/triples_formed.parquet`
- 可选预览文件：`data/triples_formed.preview.json`

## 项目背景

ProtLLM项目的总目标是：以 qwen3.5-9b-base 为基模，系统推进大规模 CPT、SFT、RL 训练，最终得到一个能够在多种蛋白质任务上取得 SOTA 级表现，并具备 text-structure-sequence 三模态理解及 reasoning 能力的模型。核心特征如下：
1. 除了普通的文字token以外，还引入了结构token和序列token，在输入输出中能够实现三个模态的交错。
2. 具有reasoning能力，在输出最终答案前，会先输出思考过程，这一部分范式需要在SFT训练过程中被塑造。
3. 能够完成多种蛋白质任务：t2bb(text to backbone，即从文本描述生成structure tokens), t2s(text to sequence), bb2t(backbone to text), s2t(sequence to text)。

为此，我们需要构建两版SFT数据，初版仅构造QA风格的SFT-QA数据，后续则构造推理风格的SFT-reasoning数据。


## SFT-QA数据构建

SFT-QA仅包含question-answer风格的数据，输入是question，输出是answer。

对于不同的蛋白质任务，输入输出的内容不同：
- t2bb
    - 任务：蛋白质结构设计
    - input: 需要生成的结构描述（结构域、二级结构等）
    - output: structure tokens
- t2s
    - 任务：蛋白质序列设计
    - input: 需要生成的序列描述（蛋白质类型、功能、interpro等）
    - output: sequence tokens
- bb2t
    - 任务：根据蛋白质结构tokens，生成文本描述
    - input: structure tokens
    - output: 输入结构tokens对应的文本描述（结构域、二级结构、功能等）
- s2t
    - 任务：根据蛋白质序列tokens，预测蛋白性质
    - input: sequence tokens
    - output: 输入序列tokens对应的文本描述（蛋白质类型、功能、性质等）

### 数据构建细节

总体的数据构建为对数据的两次扫描：
1. 第一次扫描：为每个字段转写总结自然语言的summary（即数据文件的"TODO LLM"部分）
    - 可以将多个需要转写总结的字段合并，让模型一次完成转写总结
2. 第二次扫描：对于四种不同的任务形式，给模型完整的json元数据summary，让模型根据任务要求自行抽取相关信息并生成QA对
    - 对于每个任务，都要精心设计数据合成的prompt，让模型能够准确理解任务要求，并通过其自身的知识来抽取信息合成数据。

部署模型`/data/cloudroot/minimax-m2.5-deploy/models/MiniMax-M2.5`来进行数据合成。

由于LLM看不懂sequence token和structure token，因此在数据合成的prompt中，用占位符来表示这两种token。

用英文来进行数据合成。

### TODO 第一版交付
你需要充分理解上述数据构建的背景和细节，完成以下任务：
1. 以前256条数据为样本进行实验
2. 本地vllm部署模型来合成数据，配好环境。在合成的时候需要注意提速，合理设置batch size和并发数
3. 搭建第一次扫描的代码和prompt，完成对需要转写总结的字段的转写总结
4. 搭建第二次扫描的数据合成pipeline，每条蛋白质数据需要生成四条QA对（t2bb, t2s, bb2t, s2t），完成prompt设计和代码实现
5. 对合成的数据进行质量检查
6. 汇报合成速度，估算全量数据合成需要的时间，在估算的时候忽略模型部署时间，只考虑纯前向推理所需时间

适当使用subagent来完成上述任务。

## 第一版交付实现

当前仓库已经补齐第一版交付所需的可运行脚手架。建议把代码纳入 Git，把数据单独放在仓库根目录下的 `data/` 中，不提交到远端：

- 主入口：`scripts/run_first_delivery.py`
- vLLM 启动脚本：`scripts/start_vllm_minimax.sh`
- 配置样例：`config.sample.yaml`
- 核心代码：`sft_pipeline/`

### 实现范围

1. 从 `data/triples_formed.parquet` 中读取前 256 条样本。
2. 第一次扫描：把 `secondary/interpro/cath/pfam/go/disease/function/domain/subunit/catalytic` 的 `details` 转写为英文 summary。
3. 第二次扫描：为每条蛋白质生成四个任务槽位 `t2bb/t2s/bb2t/s2t`。
4. 质检：检查样本覆盖率、四任务完整性、状态分布、空字段和推理耗时聚合。
5. 速度估算：根据样本实测延迟线性外推全量 573661 条数据的纯推理耗时。

### 当前数据事实

对当前数据集的检查结果表明：

- `sequence` 在前 256 条样本中全量存在。
- `backbone` 在全量 573661 条数据中全部为空。

这意味着：

- `t2s` 和 `s2t` 可以真实生成。
- `t2bb` 和 `bb2t` 当前会保留结构占位符，等后续回填真实 backbone tokens。

### 环境准备

建议使用 Python 3.12 环境。当前仓库已在本地用 conda 环境验证基础依赖安装：

```bash
conda create -y -p /data/yfwang/sft-data/.conda-env python=3.12 pip
source /data/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate /data/yfwang/sft-data/.conda-env
pip install -r requirements.txt
```

把数据放到本仓库的 `data/` 目录后再运行，例如：

```bash
mkdir -p data
# 然后把 triples_formed.parquet 等数据文件拷贝到 data/
```

### 启动 vLLM

模型路径使用 README 中指定的本地模型：

```bash
source /data/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate /data/yfwang/sft-data/.conda-env
bash scripts/start_vllm_minimax.sh
```

默认参数使用 8 卡 expert parallel，并设置：

- `TP_SIZE=8`
- `MAX_MODEL_LEN=32768`
- `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`

这些设置的目标是先稳定打通服务，再逐步调高并发和上下文长度。

### 运行第一版 pipeline

先做结构验证：

```bash
source /data/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate /data/yfwang/sft-data/.conda-env
PYTHONPATH=. python scripts/run_first_delivery.py --dry-run
```

接 vLLM 真跑：

```bash
source /data/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate /data/yfwang/sft-data/.conda-env
PYTHONPATH=. python scripts/run_first_delivery.py \
  --sample-size 256 \
  --concurrency 16 \
  --input-parquet data/triples_formed.parquet \
  --base-url http://127.0.0.1:8000/v1 \
  --model /data/cloudroot/minimax-m2.5-deploy/models/MiniMax-M2.5
```

### 输出产物

默认输出目录：`artifacts/first_delivery/sample_256/`

- `sample_records.jsonl`：抽样后的标准化输入
- `stage1_summaries.jsonl`：第一次扫描 summary 结果
- `stage2_qa.jsonl`：第二次扫描四任务 QA 结果
- `qc_report.json`：质检报告
- `timing_report.json`：全量耗时估算
- `sample_profile.json`：样本模态覆盖统计
