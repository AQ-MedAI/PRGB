# PRGB 使用指南

## 快速开始

### 1. 验证环境

在开始使用之前，请确保导入正常工作：

```bash
python test_imports.py
```

如果看到 "🎉 All imports working correctly!" 消息，说明环境配置正确。

### 2. 运行评估的三种方式

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

# 英文评估（使用data/en.jsonl）
EVAL_MODEL_PATH=/path/to/your/model make eval-en

# 测试评估（不需要真实模型）
make eval-test

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
    --num-iterations 1 \
    --verbose
```

## 参数详解

### 必需参数

- `--model-path`: 模型路径或API密钥
  - 本地模型：`/path/to/your/model`
  - API密钥：`sk-xxxxxxxxxxxxxxxxxxxxxxxx`

### 可选参数

#### 模型配置参数

- `--model-name`: 模型名称（默认：Qwen3）
  - 支持的模型类型：Qwen3, Qwen2.5, Gemma3, Claude, GPT4, Deepseek等
  - 用于指定模型的具体类型和版本

#### 数据配置参数

- `--data-path`: 数据文件路径（默认：tests/test.jsonl）
  - 支持JSONL格式的数据文件
  - 中文数据：`data/zh.jsonl`
  - 英文数据：`data/en.jsonl`
- `--num-iterations`: 评估迭代次数（默认：1）
  - 用于多次评估取平均分数
  - 建议值：1-5次
- `--shuffle`: 是否打乱数据（默认：True）
  - 控制是否随机打乱评估数据顺序
  - 有助于减少顺序偏差

#### 输出配置参数

- `--output-path`: 输出目录（默认：./results）
  - 评估结果和分数文件的保存路径
  - 会自动创建目录（如果不存在）

#### 评估参数

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

#### 调试参数

- `--verbose`: 启用详细日志（默认：False）
  - 输出详细的评估过程和调试信息
  - 有助于问题排查和性能分析

### Shell脚本参数

`run_eval.sh` 脚本的参数顺序：

1. `MODEL_NAME` (默认: "Qwen3_infer")
2. `MODEL_PATH` (必需)
3. `DATA_PATH` (默认: "data/zh.jsonl")
4. `OUTPUT_PATH` (默认: "./results")

## 数据格式

### 输入数据格式（JSONL）

每行一个JSON对象，包含以下字段：

```json
{
    "id": "sample_001",
    "query": "用户查询",
    "context": "相关文档内容",
    "answer": "标准答案"
}
```

### 输出结果

评估完成后，会在输出目录生成以下文件：

1. `{model_name}_eval_result_{noise_config}.jsonl`: 详细评估结果
2. `{model_name}_eval_scores.jsonl`: 评估分数汇总

## 高级用法

### 导出推理错误数据

在评估完成后，可以导出推理错误的样本进行进一步分析：

```python
from core.eval_types import EvalResults

# 加载评估结果
results = EvalResults.load_from_jsonl(
    "/path/to/your/eval_result.jsonl"
)

# 获取错误样本
errors = results.get_incorrect_results()

# 导出错误数据到新文件
results.save_to_jsonl(
    "/path/to/error_samples.jsonl",
    error_only=True,
)

print(f"导出了 {len(errors)} 个错误样本")
```

或者使用提供的示例脚本：

```bash
# 运行错误样本导出示例
python examples/export_errors.py

# 自定义输入输出路径
python -c "
from examples.export_errors import export_error_samples
export_error_samples('results/my_model_eval_result.jsonl', 'results/my_model_errors.jsonl')
"
```

### 批量评估

```bash
# 评估多个模型
for model in model1 model2 model3; do
    python eval.py \
        --model-name "$model" \
        --model-path "/path/to/$model" \
        --output-path "./results/$model"
done
```

### 自定义评估指标

修改 `core/eval.py` 中的 `checkanswer_acc` 函数来自定义评估逻辑。

### 集成到其他项目

```python
from core import get_eval

# 创建参数对象
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

args = Args(
    model_name="Qwen3",
    model_path="/path/to/model",
    data_path="data/test.jsonl",
    output_path="./results"
)

# 运行评估
get_eval(args)
```

## 配置管理

### 使用配置文件

创建配置文件 `config/my_config.json`：

```json
{
  "model": {
    "name": "Qwen3",
    "path": "/path/to/model",
    "temperature": 0.7,
    "batch_size": 16
  },
  "data": {
    "path": "data/test.jsonl",
    "shuffle": true
  },
  "output": {
    "path": "./results"
  }
}
```

### 环境变量

可以设置以下环境变量：

```bash
export PRGB_MODEL_PATH="/path/to/model"
export PRGB_DATA_PATH="data/test.jsonl"
export PRGB_OUTPUT_PATH="./results"
```

## 故障排除

### 调试模式

启用详细日志：

```bash
python eval.py --verbose --model-path "/path/to/model"
```

### 测试模式

使用测试数据验证功能：

```bash
make eval-test
```

## 性能优化

### GPU优化

- 使用适当的 `batch_size`
- 启用混合精度训练
- 使用梯度累积

### 内存优化

- 减少 `batch_size`
- 使用数据流式处理
- 启用内存清理

## 获取帮助

- 查看 `README.md` 了解项目概述
- 查看 `CONTRIBUTING.md` 了解如何贡献
- 提交 Issue 报告问题
- 查看 `examples/` 目录中的示例
