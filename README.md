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


## SVD

```
CUDA_VISIBLE_DEVICES=1 uv run python scripts/decompose_experts.py \
    --model-path /home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1 \
    --rank 256

```


```

CUDA_VISIBLE_DEVICES=1 uv run python example_mixtral.py

```