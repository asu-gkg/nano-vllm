#!/usr/bin/env python3
"""
SVD Expert Decomposition Script

将Mixtral专家权重分解为 W ≈ U @ V 的形式：
- U: 共享基矩阵，所有专家共用，常驻GPU
- V: 专家特定矩阵，按需从mmap加载

使用方法:
    python scripts/decompose_experts.py --model-path ./Mixtral-8x7B-v0.1 --rank 256

输出:
    {model_path}/svd_experts/
    ├── U_matrices.safetensors  # 共享U矩阵 (~350MB)
    └── V_experts/
        ├── layer_0_expert_0.safetensors
        ├── layer_0_expert_1.safetensors
        └── ...  # 256个文件，每个~22MB
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="SVD decomposition for Mixtral experts")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Mixtral model directory")
    parser.add_argument("--rank", type=int, default=256,
                        help="SVD rank (default: 256)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {model_path}/svd_experts)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Output dtype (default: float16)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for SVD computation (default: cuda)")
    parser.add_argument("--compute-shared-u", action="store_true",
                        help="Compute shared U from average of all experts (slower but potentially better)")
    return parser.parse_args()


def get_expert_weight_files(model_path: str) -> Dict[Tuple[int, int], Dict[str, Tuple[str, str]]]:
    """获取专家权重的文件位置映射"""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    
    expert_info = {}
    for weight_name, filename in weight_map.items():
        if "block_sparse_moe.experts" in weight_name:
            # Parse: model.layers.{layer}.block_sparse_moe.experts.{expert}.{w1/w2/w3}.weight
            parts = weight_name.split(".")
            layer_idx = int(parts[2])
            expert_idx = int(parts[5])
            weight_type = parts[6]  # w1, w2, or w3
            
            key = (layer_idx, expert_idx)
            if key not in expert_info:
                expert_info[key] = {}
            expert_info[key][weight_type] = (filename, weight_name)
    
    return expert_info


def load_expert_weight(model_path: str, filename: str, weight_name: str) -> torch.Tensor:
    """加载单个专家权重"""
    file_path = os.path.join(model_path, filename)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        return f.get_tensor(weight_name)


def svd_decompose(W: torch.Tensor, rank: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对权重矩阵进行截断SVD分解（使用GPU加速）
    
    注意：PyTorch Linear的weight是[out_features, in_features]，但forward时使用weight.T
    所以我们分解W.T，使得 x @ W.T ≈ x @ (U @ V) = (x @ U) @ V
    
    其中:
    - W: [out_features, in_features] (PyTorch Linear weight)
    - W_actual = W.T: [in_features, out_features] (实际使用的矩阵)
    - U: [in_features, rank]
    - V: [rank, out_features]
    
    返回 (U, V) - 在CPU上
    """
    # PyTorch Linear的weight需要转置才是实际使用的矩阵
    W_actual = W.T  # [in_features, out_features]
    
    # 移动到指定设备
    if device == "cuda" and torch.cuda.is_available():
        W_device = W_actual.to(device).float()
    else:
        W_device = W_actual.float()
        device = "cpu"
    
    # SVD: W_actual = U @ S @ Vh
    # 使用float32计算以保证精度
    # 截断SVD
    U, S, Vh = torch.linalg.svd(W_device, full_matrices=False)
    
    # 截取前rank个
    U_r = U[:, :rank]  # [in_features, rank]
    S_r = S[:rank]     # [rank]
    Vh_r = Vh[:rank, :]  # [rank, out_features]
    
    # 将奇异值合并到V中: V = diag(S) @ Vh
    V_r = torch.diag(S_r) @ Vh_r  # [rank, out_features]
    
    # 移回CPU
    return U_r.cpu(), V_r.cpu()


def compute_reconstruction_error(W: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> float:
    """
    计算重建误差
    
    注意：U和V是分解W.T得到的，所以需要比较W.T和U@V
    """
    W_actual = W.T.float()  # [in_features, out_features]
    W_reconstructed = (U @ V).float()  # [in_features, out_features]
    error = torch.norm(W_actual - W_reconstructed) / torch.norm(W_actual)
    return error.item()


def main():
    args = parse_args()
    
    model_path = args.model_path
    rank = args.rank
    output_dir = args.output_dir or os.path.join(model_path, "svd_experts")
    device = args.device
    
    # 检查GPU可用性
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    # 设置dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    output_dtype = dtype_map[args.dtype]
    
    print("=" * 60)
    print("SVD Expert Decomposition")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Rank: {rank}")
    print(f"Output dir: {output_dir}")
    print(f"Output dtype: {args.dtype}")
    print(f"Compute device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    v_dir = os.path.join(output_dir, "V_experts")
    os.makedirs(v_dir, exist_ok=True)
    
    # 获取专家权重信息
    print("Loading expert weight info...")
    expert_info = get_expert_weight_files(model_path)
    
    num_layers = max(key[0] for key in expert_info.keys()) + 1
    num_experts = max(key[1] for key in expert_info.keys()) + 1
    
    print(f"Found {num_layers} layers, {num_experts} experts per layer")
    print(f"Total experts: {num_layers * num_experts}")
    print()
    
    # 存储所有U矩阵
    all_U_matrices = {}
    
    # 存储重建误差统计
    errors = {"w1": [], "w2": [], "w3": []}
    
    # 处理每一层
    total_experts = num_layers * num_experts
    pbar = tqdm(total=total_experts, desc="Processing experts")
    
    for layer_idx in range(num_layers):
        layer_U = {}  # 这一层的U矩阵
        
        for expert_idx in range(num_experts):
            key = (layer_idx, expert_idx)
            expert_weights = expert_info[key]
            
            V_tensors = {}
            
            for weight_type in ["w1", "w2", "w3"]:
                filename, weight_name = expert_weights[weight_type]
                
                # 加载权重（在CPU上）
                W = load_expert_weight(model_path, filename, weight_name)
                
                # SVD分解（在GPU上计算，返回CPU）
                U, V = svd_decompose(W, rank, device=device)
                
                # 计算重建误差（在CPU上）
                error = compute_reconstruction_error(W, U, V)
                errors[weight_type].append(error)
                
                # 存储U (只存第一个专家的U作为共享U)
                if expert_idx == 0:
                    layer_U[f"layer_{layer_idx}_{weight_type}_U"] = U.to(output_dtype).contiguous()
                
                # 存储V
                V_tensors[f"{weight_type}_V"] = V.to(output_dtype).contiguous()
            
            # 保存这个专家的V矩阵
            v_file = os.path.join(v_dir, f"layer_{layer_idx}_expert_{expert_idx}.safetensors")
            save_file(V_tensors, v_file)
            
            pbar.update(1)
        
        # 合并这一层的U到总的U字典
        all_U_matrices.update(layer_U)
    
    pbar.close()
    
    # 保存所有U矩阵
    print("\nSaving shared U matrices...")
    u_file = os.path.join(output_dir, "U_matrices.safetensors")
    # 确保所有tensor都是连续的
    all_U_matrices_contiguous = {
        k: v.contiguous() if not v.is_contiguous() else v
        for k, v in all_U_matrices.items()
    }
    save_file(all_U_matrices_contiguous, u_file)
    
    # 保存元数据
    metadata = {
        "rank": rank,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "dtype": args.dtype,
        "model_path": os.path.basename(model_path),
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Decomposition Complete!")
    print("=" * 60)
    
    print("\nReconstruction Error Statistics:")
    for weight_type, err_list in errors.items():
        mean_err = np.mean(err_list)
        max_err = np.max(err_list)
        min_err = np.min(err_list)
        print(f"  {weight_type}: mean={mean_err:.4f}, min={min_err:.4f}, max={max_err:.4f}")
    
    # 计算文件大小
    u_size = os.path.getsize(u_file) / 1024 / 1024
    v_total_size = sum(
        os.path.getsize(os.path.join(v_dir, f)) 
        for f in os.listdir(v_dir)
    ) / 1024 / 1024
    
    print(f"\nFile Sizes:")
    print(f"  U matrices: {u_size:.1f} MB")
    print(f"  V experts (total): {v_total_size:.1f} MB")
    print(f"  V per expert: {v_total_size / total_experts:.1f} MB")
    
    # 计算压缩比
    # 原始大小: 每专家 ~350MB
    original_size = total_experts * 350
    compressed_size = u_size + v_total_size
    compression_ratio = original_size / compressed_size
    
    print(f"\nCompression:")
    print(f"  Original (estimated): {original_size:.0f} MB")
    print(f"  Compressed: {compressed_size:.1f} MB")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()



