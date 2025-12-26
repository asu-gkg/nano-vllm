#!/usr/bin/env python3
"""
Attention性能测试脚本

测试不同attention后端的性能:
1. Flash Attention 1.x (推荐用于2080 Ti)
2. xFormers memory_efficient_attention
3. PyTorch SDPA (fallback)
"""

import os
import sys
import time
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def benchmark_attention():
    """性能测试"""
    print("=" * 60)
    print("Attention Performance Benchmark")
    print("=" * 60)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    device = torch.device("cuda")
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    
    # 导入并显示后端信息
    from nanovllm.layers.attention import print_attention_backend_info
    print_attention_backend_info()
    
    # 测试配置 (模拟Mixtral 8x7B的attention配置)
    batch_size = 1
    num_heads = 32
    num_kv_heads = 8  # GQA
    head_dim = 128
    
    # 测试不同序列长度
    seq_lens = [128, 256, 512, 1024]
    
    print("\n" + "-" * 60)
    print("Prefill Performance Test (单次前向传播)")
    print("-" * 60)
    
    from nanovllm.layers.attention import attn_varlen_func
    
    for seq_len in seq_lens:
        # 准备数据
        q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        
        scale = head_dim ** -0.5
        
        # 预热
        for _ in range(3):
            _ = attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, 
                                seq_len, seq_len, scale, causal=True)
        torch.cuda.synchronize()
        
        # 计时
        num_runs = 20
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, 
                                seq_len, seq_len, scale, causal=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs * 1000  # ms
        
        print(f"  seq_len={seq_len:4d}: {elapsed:.2f} ms")
    
    print("\n" + "-" * 60)
    print("Decode Performance Test (单token生成)")
    print("-" * 60)
    
    from nanovllm.layers.attention import attn_with_kvcache
    from nanovllm.utils.context import set_context, reset_context
    
    block_size = 256
    
    for cache_len in [128, 256, 512, 1024]:
        # 计算需要的块数
        num_blocks = (cache_len + block_size - 1) // block_size
        
        # 准备数据
        q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.float16, device=device)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, 
                             dtype=torch.float16, device=device)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, 
                             dtype=torch.float16, device=device)
        
        cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
        block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
        
        scale = head_dim ** -0.5
        
        # 设置context
        slot_mapping = torch.tensor([0], dtype=torch.int32, device=device)
        set_context(False, slot_mapping=slot_mapping, context_lens=cache_seqlens, block_tables=block_tables)
        
        # 预热
        for _ in range(3):
            _ = attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, scale, causal=True)
        torch.cuda.synchronize()
        
        # 计时
        num_runs = 50
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, scale, causal=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs * 1000  # ms
        
        reset_context()
        
        print(f"  cache_len={cache_len:4d}: {elapsed:.3f} ms")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


def compare_backends():
    """比较不同后端的性能"""
    print("\n" + "=" * 60)
    print("Backend Comparison")
    print("=" * 60)
    
    # 测试legacy vs optimized
    print("\n比较legacy和optimized版本...")
    
    seq_len = 512
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    device = torch.device("cuda")
    
    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    scale = head_dim ** -0.5
    
    # 测试legacy
    from nanovllm.layers.attention_legacy import attn_varlen_func as legacy_attn
    
    for _ in range(3):
        _ = legacy_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, seq_len, seq_len, scale)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(20):
        _ = legacy_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, seq_len, seq_len, scale)
    torch.cuda.synchronize()
    legacy_time = (time.perf_counter() - start) / 20 * 1000
    
    # 测试optimized
    from nanovllm.layers.attention_optimized import attn_varlen_func as opt_attn
    
    for _ in range(3):
        _ = opt_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, seq_len, seq_len, scale)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(20):
        _ = opt_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, seq_len, seq_len, scale)
    torch.cuda.synchronize()
    opt_time = (time.perf_counter() - start) / 20 * 1000
    
    print(f"\nPrefill (seq_len=512):")
    print(f"  Legacy:    {legacy_time:.2f} ms")
    print(f"  Optimized: {opt_time:.2f} ms")
    print(f"  Speedup:   {legacy_time/opt_time:.2f}x")


if __name__ == "__main__":
    benchmark_attention()
    compare_backends()

