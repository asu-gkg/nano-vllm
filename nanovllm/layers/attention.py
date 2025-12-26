"""
Attention layer implementation for nano-vllm

默认使用优化版本 (attention_optimized.py)，支持:
- Flash Attention 1.x (适用于2080 Ti等Turing架构)
- xFormers memory_efficient_attention (备选)
- 向量化的格式转换
- Triton paged attention

如需使用原始版本，设置环境变量: NANOVLLM_USE_LEGACY_ATTN=1
"""

import os

# 检查是否使用legacy版本
USE_LEGACY = os.environ.get("NANOVLLM_USE_LEGACY_ATTN", "0") == "1"

if USE_LEGACY:
    # 使用原始版本
    from nanovllm.layers.attention_legacy import (
        Attention,
        store_kvcache,
        attn_varlen_func,
        attn_with_kvcache,
        print_attention_backend_info,
        FLASH_ATTN_AVAILABLE,
        XFORMERS_AVAILABLE,
    )
else:
    # 使用优化版本
    from nanovllm.layers.attention_optimized import (
        Attention,
        store_kvcache,
        attn_varlen_func,
        attn_with_kvcache,
        print_attention_backend_info,
        FLASH_ATTN_AVAILABLE,
        XFORMERS_AVAILABLE,
    )

# 导出所有需要的符号
__all__ = [
    'Attention',
    'store_kvcache',
    'attn_varlen_func',
    'attn_with_kvcache',
    'print_attention_backend_info',
    'FLASH_ATTN_AVAILABLE',
    'XFORMERS_AVAILABLE',
]
