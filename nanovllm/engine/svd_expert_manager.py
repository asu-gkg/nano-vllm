"""
SVD Expert Manager for Mixtral

使用SVD分解的专家权重进行高效推理：
- U矩阵（共享）常驻GPU
- V矩阵（专家特定）通过mmap按需加载

数学原理：
    Expert(x) = x @ W ≈ x @ (U @ V) = (x @ U) @ V
"""

import os
import json
import mmap
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file


class SVDExpert(nn.Module):
    """
    SVD分解的专家模块
    
    计算: output = act(x @ U_w1 @ V_w1) * (x @ U_w3 @ V_w3)
          output = output @ U_w2 @ V_w2
    """
    
    def __init__(
        self,
        U_w1: torch.Tensor,
        U_w2: torch.Tensor,
        U_w3: torch.Tensor,
        V_w1: torch.Tensor,
        V_w2: torch.Tensor,
        V_w3: torch.Tensor,
    ):
        super().__init__()
        # U矩阵是共享的引用（不复制）
        self.U_w1 = U_w1
        self.U_w2 = U_w2
        self.U_w3 = U_w3
        
        # V矩阵是专家特定的
        self.register_buffer("V_w1", V_w1)
        self.register_buffer("V_w2", V_w2)
        self.register_buffer("V_w3", V_w3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, hidden_size]
            
        Returns:
            output: [batch_size, hidden_size]
        """
        # w1: gate projection
        # x @ W1 ≈ x @ U_w1 @ V_w1
        h1 = x @ self.U_w1  # [batch, rank]
        gate = h1 @ self.V_w1  # [batch, intermediate_size]
        
        # w3: up projection
        # x @ W3 ≈ x @ U_w3 @ V_w3
        h3 = x @ self.U_w3  # [batch, rank]
        up = h3 @ self.V_w3  # [batch, intermediate_size]
        
        # SiLU activation and element-wise multiply
        hidden = torch.nn.functional.silu(gate) * up
        
        # w2: down projection
        # hidden @ W2 ≈ hidden @ U_w2 @ V_w2
        h2 = hidden @ self.U_w2  # [batch, rank]
        output = h2 @ self.V_w2  # [batch, hidden_size]
        
        return output


class SVDExpertManager:
    """
    SVD专家管理器
    
    - 启动时加载所有U矩阵到GPU（~350MB）
    - 运行时按需从mmap加载V矩阵（~22MB/专家）
    """
    
    def __init__(
        self,
        svd_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        preload_v_to_cpu: bool = False,
    ):
        """
        初始化SVD专家管理器
        
        Args:
            svd_path: SVD分解文件目录（包含U_matrices.safetensors和V_experts/）
            device: GPU设备
            dtype: 数据类型
            preload_v_to_cpu: 是否预加载所有V到CPU内存（更快但占用更多内存）
        """
        self.svd_path = svd_path
        self.device = device
        self.dtype = dtype
        self.preload_v_to_cpu = preload_v_to_cpu
        
        # 加载元数据
        metadata_file = os.path.join(svd_path, "metadata.json")
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        
        self.rank = self.metadata["rank"]
        self.num_layers = self.metadata["num_layers"]
        self.num_experts = self.metadata["num_experts"]
        
        print(f"[SVDExpertManager] Loading SVD experts: rank={self.rank}, "
              f"layers={self.num_layers}, experts={self.num_experts}")
        
        # 加载所有U矩阵到GPU
        self._load_u_matrices()
        
        # 初始化V矩阵存储
        self.v_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        
        if preload_v_to_cpu:
            self._preload_all_v()
        
        # 统计
        self.loads = 0
        self.v_hits = 0  # V矩阵缓存命中
        self.v_misses = 0  # V矩阵缓存未命中
    
    def _load_u_matrices(self):
        """加载所有U矩阵到GPU"""
        u_file = os.path.join(self.svd_path, "U_matrices.safetensors")
        
        print(f"[SVDExpertManager] Loading U matrices from {u_file}...")
        
        # 加载到CPU再移动到GPU
        u_tensors = load_file(u_file)
        
        # 按层组织U矩阵
        self.U_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        
        for key, tensor in u_tensors.items():
            # key format: layer_{l}_{w1/w2/w3}_U
            parts = key.split("_")
            layer_idx = int(parts[1])
            weight_type = parts[2]  # w1, w2, w3
            
            if layer_idx not in self.U_matrices:
                self.U_matrices[layer_idx] = {}
            
            # 移动到GPU
            self.U_matrices[layer_idx][weight_type] = tensor.to(
                device=self.device, dtype=self.dtype
            )
        
        # 计算GPU内存使用
        total_bytes = sum(
            t.numel() * t.element_size()
            for layer_dict in self.U_matrices.values()
            for t in layer_dict.values()
        )
        print(f"[SVDExpertManager] U matrices loaded: {total_bytes / 1024**2:.1f} MB on GPU")
    
    def _preload_all_v(self):
        """预加载所有V矩阵到CPU内存"""
        print("[SVDExpertManager] Preloading all V matrices to CPU...")
        
        v_dir = os.path.join(self.svd_path, "V_experts")
        
        for layer_idx in range(self.num_layers):
            for expert_idx in range(self.num_experts):
                v_file = os.path.join(v_dir, f"layer_{layer_idx}_expert_{expert_idx}.safetensors")
                v_tensors = load_file(v_file)
                
                self.v_cache[(layer_idx, expert_idx)] = {
                    k: v.to(dtype=self.dtype) for k, v in v_tensors.items()
                }
        
        total_bytes = sum(
            t.numel() * t.element_size()
            for expert_dict in self.v_cache.values()
            for t in expert_dict.values()
        )
        print(f"[SVDExpertManager] V matrices preloaded: {total_bytes / 1024**2:.1f} MB on CPU")
    
    def _load_v_from_file(self, layer_idx: int, expert_idx: int) -> Dict[str, torch.Tensor]:
        """从文件加载V矩阵"""
        v_dir = os.path.join(self.svd_path, "V_experts")
        v_file = os.path.join(v_dir, f"layer_{layer_idx}_expert_{expert_idx}.safetensors")
        
        v_tensors = load_file(v_file)
        return {k: v.to(dtype=self.dtype) for k, v in v_tensors.items()}
    
    def get_expert(self, layer_idx: int, expert_idx: int) -> SVDExpert:
        """
        获取SVD专家
        
        Args:
            layer_idx: 层索引
            expert_idx: 专家索引
            
        Returns:
            SVDExpert实例
        """
        self.loads += 1
        
        # 获取U矩阵（已在GPU上）
        U = self.U_matrices[layer_idx]
        
        # 获取V矩阵
        key = (layer_idx, expert_idx)
        if key in self.v_cache:
            V = self.v_cache[key]
            self.v_hits += 1
        else:
            V = self._load_v_from_file(layer_idx, expert_idx)
            self.v_misses += 1
            # 可选：缓存到内存（如果内存充足）
            # self.v_cache[key] = V
        
        # 创建SVDExpert
        expert = SVDExpert(
            U_w1=U["w1"],
            U_w2=U["w2"],
            U_w3=U["w3"],
            V_w1=V["w1_V"].to(self.device),
            V_w2=V["w2_V"].to(self.device),
            V_w3=V["w3_V"].to(self.device),
        )
        
        return expert
    
    def get_stats(self) -> Dict:
        """
        获取统计信息（兼容原始ExpertManager接口）
        """
        total = self.v_hits + self.v_misses
        hit_rate = self.v_hits / total if total > 0 else 0.0
        
        return {
            "hits": self.v_hits,
            "misses": self.v_misses,
            "total": total,
            "hit_rate": hit_rate,
            "cached_experts": len(self.v_cache),
            "max_experts": self.num_layers * self.num_experts,  # 理论上可以缓存所有专家
            # SVD特定信息
            "rank": self.rank,
            "loads": self.loads,
            "preloaded": self.preload_v_to_cpu,
        }


class SVDMixtralExpert(nn.Module):
    """
    用于替换原始MixtralExpert的SVD版本
    
    与原始MixtralExpert保持相同的接口
    """
    
    def __init__(self, U_w1, U_w2, U_w3, V_w1, V_w2, V_w3):
        super().__init__()
        # U矩阵（共享引用）
        self.U_w1 = U_w1
        self.U_w2 = U_w2
        self.U_w3 = U_w3
        
        # V矩阵（专家特定）
        self.V_w1 = V_w1
        self.V_w2 = V_w2
        self.V_w3 = V_w3
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播，与原始MixtralExpert接口一致
        
        Args:
            hidden_states: [batch_size, hidden_size] 或 [seq_len, hidden_size]
            
        Returns:
            output: 相同形状
        """
        # Gate projection: x @ W1 ≈ (x @ U_w1) @ V_w1
        gate = (hidden_states @ self.U_w1) @ self.V_w1
        
        # Up projection: x @ W3 ≈ (x @ U_w3) @ V_w3
        up = (hidden_states @ self.U_w3) @ self.V_w3
        
        # SiLU and multiply
        hidden = torch.nn.functional.silu(gate) * up
        
        # Down projection: hidden @ W2 ≈ (hidden @ U_w2) @ V_w2
        output = (hidden @ self.U_w2) @ self.V_w2
        
        return output

