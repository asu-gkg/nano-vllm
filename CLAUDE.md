# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight implementation of vLLM built from scratch in ~1,200 lines of Python. It provides fast offline inference with a clean, readable codebase.

## Development Commands

### Running Examples
```bash
# Basic example with Qwen3 (requires model weights)
python example.py

# Mixtral example (requires Mixtral model weights)
python example_mixtral.py

# Benchmark performance  
python bench.py

# Interactive chat
python chat.py
```

### Model Download
```bash
# Download model weights using the provided script
./hfd.sh Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/

# Or use huggingface-cli
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Architecture Overview

### Core Components

1. **LLM Interface** (`nanovllm/llm.py`)
   - Main entry point, inherits from LLMEngine
   - API mirrors vLLM with minor differences in generate method

2. **Engine Layer** (`nanovllm/engine/`)
   - `llm_engine.py`: Orchestrates the inference process, manages tokenization and generation loop
   - `model_runner.py`: Handles model execution, tensor parallelism, and CUDA graph optimization
   - `scheduler.py`: Manages sequence scheduling for batched inference
   - `sequence.py`: Represents individual generation sequences
   - `block_manager.py`: Manages KV-cache memory blocks

3. **Model Implementation** (`nanovllm/models/`)
   - Supports Qwen3 model (`qwen3.py`)
   - Supports Mixtral MoE models (`mixtral.py`)
   - Uses custom layers from `nanovllm/layers/`
   
4. **Layers** (`nanovllm/layers/`)
   - `moe.py`: Mixture of Experts implementation for Mixtral
   - `linear.py`: Various parallel linear layers for tensor parallelism
   - `attention.py`: Multi-head attention implementation
   - Other supporting layers (activation, normalization, embeddings)

### Optimization Features
   - Prefix caching via block manager
   - Tensor parallelism support (up to 8 GPUs)
   - CUDA graph compilation (disable with `enforce_eager=True`)
   - Configurable KV-cache block size
   - Efficient MoE implementation for Mixtral models

### Key Configuration

The `Config` class (`nanovllm/config.py`) controls:
- `max_num_batched_tokens`: Maximum tokens in a batch (default: 16384)
- `max_num_seqs`: Maximum concurrent sequences (default: 512)
- `max_model_len`: Maximum model context length (default: 4096)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `kvcache_block_size`: Size of KV-cache blocks (must be divisible by 256)

### Multi-Process Architecture

For tensor parallelism, the system spawns multiple processes:
- Rank 0 process handles scheduling and coordination
- Additional processes (ranks 1+) handle distributed model execution
- Communication via NCCL and shared memory

## Important Notes

- Model path must be absolute and point to a directory containing model weights
- The system uses multiprocessing spawn context for CUDA compatibility
- Default tensor parallel communication uses `tcp://localhost:2333`
- Model type is automatically detected from the config file (`model_type` field)
- Mixtral models require more GPU memory due to the MoE architecture