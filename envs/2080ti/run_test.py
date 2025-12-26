#!/usr/bin/env python3
"""
2080 Ti Attention 性能测试

运行方式:
  cd /home/asu/Desktop/nano-vllm/envs/2080ti
  CUDA_VISIBLE_DEVICES=1 uv run python run_test.py
"""

import os
import sys

# 添加主项目路径
sys.path.insert(0, "/home/asu/Desktop/nano-vllm")

import torch
import time

def main():
    print("=" * 60)
    print("2080 Ti Attention Performance Test")
    print("=" * 60)
    
    # GPU信息
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {major}.{minor}")
    
    if major != 7:
        print(f"⚠ 警告: 当前GPU不是Turing架构 (期望7.x, 实际{major}.{minor})")
    
    # 检查attention后端
    from nanovllm.layers.attention import print_attention_backend_info
    print_attention_backend_info()
    
    # 运行性能测试
    from nanovllm.layers.attention import attn_varlen_func
    
    device = torch.device("cuda")
    
    # Mixtral配置
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    
    print("\n" + "-" * 60)
    print("Prefill Performance Test")
    print("-" * 60)
    
    for seq_len in [128, 256, 512, 1024]:
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
        elapsed = (time.perf_counter() - start) / num_runs * 1000
        
        print(f"  seq_len={seq_len:4d}: {elapsed:.2f} ms")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

