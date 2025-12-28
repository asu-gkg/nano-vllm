#!/usr/bin/env python3
"""
测试ExpertManager的文件句柄缓存优化效果
"""

import os
import sys
import time
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import MixtralConfig
from nanovllm.engine.expert_manager import ExpertManager


def test_expert_loading(model_path: str):
    """测试专家加载性能"""
    print("=" * 70)
    print("ExpertManager 文件句柄缓存优化测试")
    print("=" * 70)
    
    # 初始化配置
    config = MixtralConfig.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n设备: {device}")
    print(f"模型路径: {model_path}")
    print()
    
    # 创建ExpertManager
    manager = ExpertManager(
        model_path=model_path,
        config=config,
        device=device,
        max_gpu_experts=8,
    )
    
    print("测试1: 首次加载（需要打开文件）")
    print("-" * 70)
    
    # 测试首次加载
    start_time = time.time()
    expert1 = manager.get_expert(0, 0)
    first_load_time = time.time() - start_time
    
    stats1 = manager.get_stats()
    print(f"加载专家 (0, 0): {first_load_time:.3f}s")
    print(f"文件打开次数: {stats1['file_opens']}")
    print(f"文件句柄数: {stats1['file_handles']}")
    print()
    
    print("测试2: 缓存命中（无需重新加载）")
    print("-" * 70)
    
    start_time = time.time()
    expert2 = manager.get_expert(0, 0)
    cached_time = time.time() - start_time
    
    stats2 = manager.get_stats()
    print(f"加载专家 (0, 0) [缓存]: {cached_time:.6f}s")
    print(f"缓存命中率: {stats2['hit_rate']:.1%}")
    print()
    
    print("测试3: 同一文件的不同专家（复用文件句柄）")
    print("-" * 70)
    
    # 加载同一层的不同专家（应该复用文件句柄）
    start_time = time.time()
    expert3 = manager.get_expert(0, 1)
    second_expert_time = time.time() - start_time
    
    stats3 = manager.get_stats()
    print(f"加载专家 (0, 1): {second_expert_time:.3f}s")
    print(f"文件打开次数: {stats3['file_opens']} (应该和之前相同)")
    print(f"文件句柄数: {stats3['file_handles']}")
    print()
    
    print("测试4: 不同层的专家（可能需要新文件）")
    print("-" * 70)
    
    start_time = time.time()
    expert4 = manager.get_expert(1, 0)
    different_layer_time = time.time() - start_time
    
    stats4 = manager.get_stats()
    print(f"加载专家 (1, 0): {different_layer_time:.3f}s")
    print(f"文件打开次数: {stats4['file_opens']}")
    print(f"文件句柄数: {stats4['file_handles']}")
    print()
    
    print("性能对比:")
    print("-" * 70)
    print(f"首次加载: {first_load_time:.3f}s")
    print(f"缓存命中: {cached_time:.6f}s ({first_load_time/cached_time:.0f}x 加速)")
    print(f"复用句柄: {second_expert_time:.3f}s")
    print()
    
    print("统计信息:")
    print("-" * 70)
    final_stats = manager.get_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'rate' in key else f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # 清理
    manager.close_file_handles()
    print("✅ 测试完成")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试ExpertManager优化")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1",
        help="模型路径"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    test_expert_loading(args.model_path)

