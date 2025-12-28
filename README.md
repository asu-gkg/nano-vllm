# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* 🎯 **MoE Support** - Mixtral 8x7B with dynamic expert loading
* 🔧 **Multi-GPU Architecture Support** - Flash Attention 1.x for Turing (2080 Ti), 2.x for Ampere+

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Manual Download

If you prefer to download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

---

## 🎯 在 RTX 2080 Ti 上运行 Mixtral 8x7B

### 1. 环境准备

```bash
cd /home/asu/Desktop/nano-vllm

# 创建2080 Ti专用环境
cd envs/2080ti
uv sync

# 安装 Flash Attention 1.x (支持Turing架构)
uv pip install flash-attn==1.0.9 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载 Mixtral 模型

```bash
# 使用 huggingface-cli
huggingface-cli download --resume-download mistralai/Mixtral-8x7B-v0.1 \
  --local-dir ./Mixtral-8x7B-v0.1/ \
  --local-dir-use-symlinks False

# 或使用 hfd.sh (推荐，支持断点续传)
./hfd.sh mistralai/Mixtral-8x7B-v0.1 --local-dir ./Mixtral-8x7B-v0.1/
```

### 3. 运行 Mixtral

```bash
# 指定使用 2080 Ti (PyTorch GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python example_mixtral.py
```

或者使用Python API：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 选择2080 Ti

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner

# 配置
config = Config(
    model="/path/to/Mixtral-8x7B-v0.1",
    tensor_parallel_size=1,
    max_num_batched_tokens=2048,
    max_model_len=1024,
    gpu_memory_utilization=0.85,
    enforce_eager=True,  # 2080 Ti建议开启
)

# 初始化
runner = ModelRunner(config, rank=0, event=None)

# 生成...
```

### 4. 性能优化建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_model_len` | 512-1024 | 减少KV cache占用 |
| `max_num_batched_tokens` | 1024-2048 | 控制显存峰值 |
| `gpu_memory_utilization` | 0.85 | 留出空间给激活值 |
| `enforce_eager` | True | 2080 Ti建议禁用CUDA Graph |

### 5. 验证 Flash Attention

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -c "
from nanovllm.layers.attention import print_attention_backend_info
print_attention_backend_info()
"
```

预期输出：
```
GPU: NVIDIA GeForce RTX 2080 Ti (compute 7.5)
GPU架构: Turing (RTX 20系列, 支持Flash Attn 1.x)
Flash Attention 1.x: ✓ 可用
```

### 6. Attention 性能对比

在 RTX 2080 Ti 上的 Prefill 性能：

| seq_len | PyTorch SDPA | Flash Attn 1.x | 提升 |
|---------|-------------|----------------|------|
| 128 | 0.77ms | 0.16ms | **4.8x** |
| 256 | 0.73ms | 0.17ms | **4.3x** |
| 512 | 0.81ms | 0.35ms | **2.3x** |
| 1024 | 1.10ms | 0.80ms | **1.4x** |

---

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)


## SVD Expert Decomposition

Nano-vLLM 支持两种专家分解方法：

### 方法 1: PCA 分解（快速但可能产生乱码）

使用 PCA 计算共享 U 矩阵，然后通过 `V = U^T @ W^T` 计算 V：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/decompose_experts.py \
    --model-path ./Mixtral-8x7B-v0.1 \
    --rank 256 \
    --dtype float16
```

**参数说明：**
- `--model-path`: Mixtral 模型路径
- `--rank`: 分解的秩（默认 256，越小压缩率越高但精度越低）
- `--dtype`: 输出数据类型（float16/bfloat16/float32）
- `--device`: 计算设备（cuda/cpu）

### 方法 2: Activation-Aware 蒸馏（推荐，解决乱码问题）

使用真实激活数据进行蒸馏回归拟合 V，解决 PCA 方法可能导致的输出乱码问题。

#### 步骤 1: 采集校准激活数据

运行 teacher 模型（禁用 SVD），收集每层每专家的输入激活：

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/collect_moe_calib.py \
    --model-path ./Mixtral-8x7B-v0.1 \
    --out ./calib_mixtral.pt \
    --cap-per-group 1024 \
    --num-prompts 200 \
    --max-new-tokens 64 \
    --temperature 0.7
```

**参数说明：**
- `--model-path`: Mixtral 模型路径
- `--out`: 校准数据保存路径（默认 `calib_mixtral.pt`）
- `--cap-per-group`: 每个 (layer, expert) 组保留的最大样本数（默认 1024）
- `--num-prompts`: 用于校准的 prompt 数量（默认 200）
- `--max-new-tokens`: 每个 prompt 生成的最大 token 数（默认 64）
- `--temperature`: 采样温度（默认 0.7）

**注意事项：**
- 此步骤会自动设置 `NANOVLLM_DISABLE_SVD=1`，强制使用原始 expert manager
- 采集过程会占用 GPU 显存，建议在推理空闲时进行
- 样本数量越多，蒸馏效果越好，但采集时间更长

#### 步骤 2: 蒸馏生成 SVD Experts

使用校准数据对每个 expert 进行 activation-aware ridge regression 拟合 V：

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/distill_experts_activation_aware.py \
    --model-path ./Mixtral-8x7B-v0.1 \
    --calib-path ./calib_mixtral.pt \
    --rank 256 \
    --dtype float16 \
    --ridge 1e-4 \
    --device cuda
```

**参数说明：**
- `--model-path`: Mixtral 模型路径
- `--calib-path`: 步骤 1 生成的校准数据路径
- `--rank`: 分解的秩（默认 256，必须与推理时一致）
- `--dtype`: 输出数据类型（默认 float16）
- `--ridge`: Ridge regression 正则化系数（默认 1e-4）
- `--device`: 计算设备（cuda/cpu，推荐 cuda 加速）
- `--chunk`: w1/w3 的批处理大小（默认 64）
- `--chunk-w2`: w2 的批处理大小（默认 16，w2 计算更复杂）
- `--pca-oversample`: w2 的 PCA 过采样参数（默认 32）
- `--output-dir`: 输出目录（默认 `{model_path}/svd_experts`）

**输出结构：**
```
{model_path}/svd_experts/
├── U_matrices.safetensors          # 共享 U 矩阵（每层每权重类型）
├── V_experts/
│   ├── layer_0_expert_0.safetensors
│   ├── layer_0_expert_1.safetensors
│   └── ...
└── metadata.json                   # 元数据（rank, dtype, ridge 等）
```

**性能提示：**
- 使用 `--device cuda` 可以显著加速蒸馏过程（校准数据会加载到 GPU）
- 如果 GPU 显存不足，可以使用 `--device cpu` 或减少 `--cap-per-group`

#### 步骤 3: 使用 SVD Experts 进行推理

蒸馏完成后，推理时会自动检测并使用 SVD experts：

```bash
CUDA_VISIBLE_DEVICES=1 uv run python example_mixtral.py
```

**验证 SVD 是否生效：**

运行时会看到日志：
```
[ModelRunner] Using SVD Expert Manager from ./Mixtral-8x7B-v0.1/svd_experts
```

### 两种方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **PCA 分解** | 快速，无需校准数据 | 可能产生乱码/随机符号 | 快速测试，对精度要求不高 |
| **Activation-Aware 蒸馏** | 精度高，解决乱码问题 | 需要校准数据，耗时更长 | 生产环境，要求高质量输出 |

### 常见问题

**Q: 为什么蒸馏后还是出现乱码？**
- 检查 `--rank` 是否太小（建议 ≥ 256）
- 增加 `--cap-per-group` 和 `--num-prompts` 收集更多校准数据
- 调整 `--ridge` 参数（尝试 1e-5 到 1e-3）

**Q: 蒸馏过程很慢怎么办？**
- 使用 `--device cuda` 加速
- 减少 `--cap-per-group`（但可能影响精度）
- 减少 `--num-prompts`（但可能影响精度）

**Q: GPU 显存不足怎么办？**
- 使用 `--device cpu` 进行蒸馏
- 减少 `--cap-per-group` 参数
- 在采集校准数据时减少 `--max-new-tokens`

**Q: 如何验证蒸馏效果？**
- 对比蒸馏前后的生成文本质量
- 检查是否还有乱码/随机符号
- 观察生成文本的语义连贯性