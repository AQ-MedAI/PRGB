# PRGB - 通过Placeholders实现的RAG评估工具

## 项目背景

PRGB (Placeholder RAG Benchmark) 是一个专注于评估检索增强生成（RAG）系统中文档忠实性与外部知识利用效率的基准工具。它通过多级过滤、跨实体推理等渐进维度，以及配备噪声文档的数据集，使用placeholders全面评测模型性能，助力研究人员和开发者分析主流RAG模型在复杂场景中的表现。

![benchmark](png/benchmark.png)

### 样例数据

![data](png/examples.jpg)

### 即将更新

1. **基于API的推理**：提供基于API的推理以及检验功能
2. **数据质量提升**: 数据集正在经过人工校验中，很快会有更准确的版本，提供改进的数据质量和一致性。
3. **检索测试版本**: 专门用于测试检索系统的版本正在开发中，即将发布。

## 主要特性

- 🎯 **多模型支持**: 支持多种大语言模型本地vllm推理
- 📊 **标准化评估**: 提供统一的评估指标和流程
- 🔧 **灵活配置**: 支持噪声配置、placeholder的配置等参数调整
- 🌍 **多语言支持**: 支持中英文数据集评估
- 📈 **详细报告**: 生成详细的评估结果和分数报告

## 实验结果

### 中文数据集性能对比

下表展示了各种最先进模型在中文数据集上的性能表现，按Overall分数从高到低排序。**粗体**值表示最佳实验结果，***斜体粗体***值表示第二佳实验结果。

| Models                        | Overall | Multi-Level Filter | Composition   | Reasoning     |
| ----------------------------- | ------- | ------------------ | ------------- | ------------- |
| `Gemini-2.5-pro-preview`    | 87.33   | **97.92**         | **94.20**    | ***70.18*** |
| `Claude-3.7-sonnet`         | 85.74   | ***97.62***      | ***90.59*** | **70.39**    |
| `Gemini-2.5-flash-preview`  | 81.85   | 93.92              | 88.54         | 63.86         |
| `Qwen3-235B-A22B`           | 80.76   | 94.92              | 88.18         | 60.23         |
| `Qwen3-30B-A3B`             | 80.45   | 95.87              | 86.11         | 61.42         |
| `Deepseek-V3(241226)`       | 77.54   | 94.58              | 81.00         | 60.32         |
| `Qwen3-235B-A22B w/o think` | 75.20   | 91.50              | 79.67         | 57.14         |
| `Qwen-2.5-MAX`              | 74.43   | 93.25              | 78.28         | 55.37         |
| `Qwen3-30B-A3B w/o think`   | 71.05   | 91.08              | 72.22         | 54.76         |
| `Gemma3_27b`                | 70.24   | 73.09              | 92.21         | 50.24         |
| `Qwen3_32B`                 | 69.69   | 89.75              | 75.74         | 46.70         |
| `GPT4.1`                    | 66.26   | 89.75              | 71.95         | 41.27         |
| `Qwen2.5_72B`               | 64.87   | 92.92              | 64.99         | 44.14         |
| `GPT4o-1120`                | 64.58   | 88.50              | 70.21         | 39.35         |
| `Gemma3_12b`                | 64.10   | 60.20              | 89.92         | 50.52         |
| `Qwen3_8B`                  | 63.04   | 86.87              | 67.49         | 39.47         |
| `Qwen3_32B w/o think`       | 60.73   | 59.53              | 89.50         | 41.30         |
| `Qwen2.5_32B`               | 58.76   | 92.00              | 51.33         | 44.60         |
| `Qwen2.5_14B`               | 55.94   | 89.42              | 52.69         | 35.87         |
| `Qwen2.5_7B`                | 49.31   | 83.29              | 47.47         | 26.92         |
| `Qwen3_8B w/o think`        | 50.02   | 47.83              | 83.96         | 28.17         |
| `Gemma3_4b`                 | 47.67   | 37.41              | 78.33         | 39.26         |

### 英文数据集性能对比

下表展示了各种最先进模型在英文数据集上的性能表现，按Overall分数从高到低排序。**粗体**值表示最佳实验结果，***斜体粗体***值表示第二佳实验结果。

| Models                       | Overall             | Multi-Level Filter  | Composition         | Reasoning           |
| ---------------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| `Gemini-2.5-pro-preview`   | **84.89**     | **94.89**     | ***85.32*** | ***76.09*** |
| `Claude-3.7-sonnet`        | ***82.96*** | ***93.18*** | 82.13               | **76.51**     |
| `Gemini-2.5-flash-preview` | 79.20               | 90.69               | 80.30               | 67.90               |
| `Gemma3_27b`               | 79.18               | 92.03               | 78.00               | 71.33               |
| `Qwen3-30B-A3B`            | 79.09               | 78.01               | **91.01**     | 71.78               |
| `Deepseek-V3(241226)`      | 79.02               | 89.91               | 77.18               | 74.03               |
| `Qwen-3-235B-A22B`         | 78.68               | 90.56               | 78.32               | 69.97               |
| `Qwen-2.5-MAX`             | 78.45               | 89.32               | 75.83               | 65.89               |
| `Qwen3_32B`                | 78.05               | 90.69               | 77.23               | 69.65               |
| `Qwen3_8B`                 | 76.80               | 88.36               | 76.27               | 68.71               |
| `Gemma3_12b`               | 72.35               | 87.42               | 68.46               | 68.12               |
| `Qwen2.5_72B`              | 68.90               | 87.01               | 64.30               | 63.69               |
| `Qwen3_32B w/o think`      | 68.30               | 84.35               | 63.74               | 64.59               |
| `Qwen2.5_32B`              | 66.70               | 85.66               | 63.04               | 58.92               |
| `Qwen2.5_14B`              | 63.29               | 84.40               | 57.35               | 58.34               |
| `Qwen2.5_7B`               | 63.16               | 81.90               | 56.76               | 61.00               |
| `Qwen3_8B w/o think`       | 64.71               | 83.21               | 58.93               | 61.52               |
| `GPT4o-1120`               | 60.89               | 81.62               | 60.69               | 44.83               |
| `GPT4.1`                   | 60.79               | 84.76               | 64.02               | 35.37               |
| `Gemma3_4b`                | 57.58               | 77.98               | 48.50               | 59.41               |

## 安装方法

### 环境要求

- Python 3.7+
- CUDA (如果使用GPU推理)

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/Alipay-Med/PRGB.git
cd PRGB
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 安装开发依赖（可选）

```bash
pip install -e .
```

4. 验证安装

```bash
python test_imports.py
```

## 使用方法

### 验证导入

在运行评估之前，建议先验证导入是否正常：

```bash
python test_imports.py
```

### 运行评估的三种方式

#### 方式1：使用 Makefile(推荐)

如果只需要修改模型路径，推荐使用 Makefile

```bash
# 查看所有可用命令
make help

# 设置环境变量并运行评估
export EVAL_MODEL_PATH=/path/to/your/model
make eval

# 或者在一行中设置环境变量
EVAL_MODEL_PATH=/path/to/your/model make eval

# 中文评估（使用data/zh.jsonl）
EVAL_MODEL_PATH=/path/to/your/model make eval-ch

# 中文推理模式评估（使用data/zh.jsonl）
EVAL_MODEL_PATH=/path/to/your/model make eval-ch-infer

# 英文评估（使用data/en.jsonl）
EVAL_MODEL_PATH=/path/to/your/model make eval-en

# 英文推理模式评估（使用data/en.jsonl）
EVAL_MODEL_PATH=/path/to/your/model make eval-en-infer

# 导出错误样本（需要提供评估结果文件路径）
EVAL_RESULT_FILE=results/model_eval_result.jsonl make export-errors
```

#### 方式2：使用 Shell 脚本

如果需要修改其他参数，推荐使用shell

```bash
# 使用默认参数运行（需要提供模型路径）
./run_eval.sh /path/to/your/model

# 传递所有参数
./run_eval.sh /path/to/your/model data/zh.jsonl Qwen3_infer ./results
```

#### 方式3：使用 Python 命令

```bash
# 基本用法
python eval.py \
    --model-name "Qwen3" \
    --model-path "/path/to/your/model" \
    --data-path "tests/test.jsonl" \
    --output-path "./results"

# 完整参数示例
python eval.py \
    --model-name "Qwen3" \
    --model-path "/path/to/your/model" \
    --data-path "your_data.jsonl" \
    --output-path "./results" \
    --batch-size 16 \
    --temperature 0.7 \
    --noise-config '{"noise_doc_level1":4,"noise_doc_level2":4,"noise_doc_level3":1}' \
    --custom_config "config/default_prompt_config.json" \
    --shuffle True \
    --num-iterations 3 \
    --verbose
```

### 参数说明

#### 必需参数

- `--model-path`: 模型路径或API密钥
  - 本地模型：`/path/to/your/model`
  - API密钥：`sk-xxxxxxxxxxxxxxxxxxxxxxxx`

#### 可选参数

##### 模型配置参数

- `--model-name`: 模型名称（默认：Qwen3）
  - 支持的模型类型：Qwen3, Qwen2.5, Gemma3, Claude, GPT4, Deepseek等
  - 用于指定模型的具体类型和版本

##### 数据配置参数

- `--data-path`: 数据文件路径（默认：tests/test.jsonl）
  - 支持JSONL格式的数据文件
  - 中文数据：`data/zh.jsonl`
  - 英文数据：`data/en.jsonl`
- `--num-iterations`: 评估迭代次数（默认：3）
  - 对每个查询，随机选择n个不同的占位符进行评估
  - 每个占位符代表同一查询的不同版本，具有不同的变量替换
  - 用于多次评估取平均分数
  - 建议值：1-5次
- `--shuffle`: 是否打乱数据（默认：True）
  - 控制是否随机打乱评估数据顺序
  - 有助于减少顺序偏差

##### 输出配置参数

- `--output-path`: 输出目录（默认：./results）
  - 评估结果和分数文件的保存路径
  - 会自动创建目录（如果不存在）

##### 评估参数

- `--batch-size`: 批处理大小（默认：16）
  - 控制GPU内存使用和推理速度
  - 根据GPU内存大小调整：8-32
- `--temperature`: 生成温度（默认：0.7）
  - 控制生成文本的随机性
  - 范围：0.0-1.0，0.0为确定性生成
- `--noise-config`: 噪声配置JSON字符串（默认：'{"noise_doc_level1":4,"noise_doc_level2":4,"noise_doc_level3":1}'）
  - 控制不同级别噪声文档的数量
  - `noise_doc_level1`: 一级噪声文档数
  - `noise_doc_level2`: 二级噪声文档数
  - `noise_doc_level3`: 三级噪声文档数
- `--custom_config`: 自定义提示配置文件路径（默认：None）
  - 用于指定自定义的prompt配置文件
  - 支持JSON格式的配置文件
  - 默认使用语言相关的配置文件（中文/英文）

##### 调试参数

- `--verbose`: 启用详细日志（默认：False）
  - 输出详细的评估过程和调试信息
  - 有助于问题排查和性能分析

- `--inference-mode`: 是否启用推理模式（默认：False）
  - 设为 True 时，启用模型的推理模式
  - 适用于如 Qwen3_infer 等推理优化模型
  - 通常与 eval-ch-infer 和 eval-en-infer 命令配合使用

#### Shell脚本参数

`run_eval.sh` 脚本的参数顺序：

1. `MODEL_NAME` (默认: "Qwen3_infer")
2. `MODEL_PATH` (必需)
3. `DATA_PATH` (默认: "data/zh.jsonl")
4. `OUTPUT_PATH` (默认: "./results")

### 使用示例脚本

```bash
# 运行基本示例
python examples/basic_evaluation.py

# 运行自定义示例
python examples/basic_evaluation.py --mode custom

# 导出错误样本
python examples/export_errors.py
```

## 项目结构

```
PRGB/
├── README.md                 # Project documentation
├── README_zh.md             # Chinese documentation
├── USAGE.md                 # Usage guide
├── pyproject.toml           # Project configuration
├── requirements.txt         # Python dependencies
├── eval.py                 # Main evaluation script
├── example_usage.py        # Example usage script
├── run_eval.sh             # Run script
├── run_http_eval.sh        # HTTP evaluation script
├── Makefile                # Build and development commands
├── CONTRIBUTING.md         # Contribution guidelines
├── CHANGELOG.md            # Change log
├── LEGAL.md                # Legal notice
├── .gitignore              # Git ignore file
├── .flake8                 # Code style configuration
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── .gitattributes          # Git attributes
│
├── core/                   # Core functionality modules
│   ├── __init__.py        # Module initialization
│   ├── eval.py            # Main evaluation logic
│   ├── data.py            # Data processing
│   ├── eval_types.py      # Evaluation type definitions
│   ├── logger.py          # Logging functionality
│   └── models/            # Model implementations
│       ├── __init__.py    # Models module initialization
│       ├── api_models.py  # API-based model implementations
│       └── vllm_models.py # VLLM-based model implementations
│
├── config/                 # Configuration files
│   ├── api_prompt_config_ch.json  # Chinese API prompt configuration
│   ├── api_prompt_config_en.json  # English API prompt configuration
│   └── default_prompt_config.json # Default prompt configuration
│
├── utils/                  # Utility functions
│   ├── __init__.py        # Module initialization
│   └── transfer_csv_to_jsonl.py  # CSV to JSONL conversion tool
│
├── examples/               # Example scripts
│   ├── basic_evaluation.py # Basic usage example
│   └── export_errors.py    # Error sample export example
│
├── tests/                  # Test files
│   ├── test_imports.py     # Import test script
│   ├── test_data_process.py    # Data processing tests
│   ├── test_checkanswer.py     # Answer checking tests
│   ├── test_eval.py            # Evaluation tests
│   └── test_import_models.py   # Model import tests
│
├── data/                   # Data files
│   ├── zh.jsonl           # Chinese evaluation data
│   └── en.jsonl           # English evaluation data
│
├── results/                # Evaluation results (generated)
│
└── pic/                    # Images and figures
    ├── benchmark.jpg       # Benchmark figure
    └── examples.jpg        # Examples figure
```

## 开发指南

### 代码风格

项目使用以下工具确保代码质量：

- `black`: 代码格式化
- `flake8`: 代码风格检查
- `pre-commit`: 预提交钩子

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 使用Makefile
make test
```

### 代码格式化

```bash
# 格式化代码
make format

# 检查代码风格
make lint
```

### 许可证

请查看 [LEGAL.md](LEGAL.md) 文件了解详细的许可证信息。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至: tanzhehao.tzh@antgroup.com 或 jiaoyihan.yh@antgroup.com

---

**注意**: 本项目仅供研究和评估使用，请确保遵守相关法律法规和模型使用条款。
