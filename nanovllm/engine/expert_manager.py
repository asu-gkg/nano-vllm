"""
Expert Manager for Mixtral dynamic loading

This module manages the dynamic loading and caching of MoE experts.
It maintains an LRU cache of experts in GPU memory and loads/offloads
experts as needed.
"""

import os
import threading
from collections import OrderedDict
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from transformers import MixtralConfig

from nanovllm.models.mixtral import MixtralExpert
from nanovllm.utils.loader import load_expert_weights, get_expert_weight_info


class ExpertManager:
    """Manages dynamic loading of Mixtral experts"""
    
    def __init__(
        self,
        model_path: str,
        config: MixtralConfig,
        device: str = "cuda",
        max_gpu_experts: int = 42,
    ):
        """
        Initialize ExpertManager
        
        Args:
            model_path: Path to model directory
            config: Mixtral configuration
            device: Device to load experts to
            max_gpu_experts: Maximum number of experts to keep in GPU
        """
        self.model_path = model_path
        self.config = config
        self.device = device
        self.max_gpu_experts = max_gpu_experts
        
        # Get expert weight info
        self.expert_info = get_expert_weight_info(model_path)
        
        # LRU cache for loaded experts
        self.expert_cache: OrderedDict[Tuple[int, int], MixtralExpert] = OrderedDict()
        
        # Thread lock for cache operations
        self.lock = threading.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def get_expert(self, layer_idx: int, expert_idx: int) -> MixtralExpert:
        """
        Get an expert, loading it if necessary
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index within the layer
            
        Returns:
            The requested expert module
        """
        key = (layer_idx, expert_idx)
        
        # Fast path: check without lock first
        if key in self.expert_cache:
            with self.lock:
                # Double-check inside lock
                if key in self.expert_cache:
                    self.expert_cache.move_to_end(key)
                    self.hits += 1
                    return self.expert_cache[key]
        
        # Slow path: need to load
        print(f"[ExpertManager] Loading expert {key}...")
        import time
        start_time = time.time()
        
        with self.lock:
            # Check again in case another thread loaded it
            if key in self.expert_cache:
                self.expert_cache.move_to_end(key)
                self.hits += 1
                print(f"[ExpertManager] Expert {key} found in cache after lock")
                return self.expert_cache[key]
            
            # Expert not in cache, need to load
            self.misses += 1
            
            # Check if we need to evict
            if len(self.expert_cache) >= self.max_gpu_experts:
                # Evict least recently used expert
                evict_key, evicted_expert = self.expert_cache.popitem(last=False)
                print(f"[ExpertManager] Evicting expert {evict_key}")
                # Move expert to CPU to free GPU memory
                evicted_expert.cpu()
                del evicted_expert
                torch.cuda.empty_cache()
            
            # Load the expert
            expert = self._load_expert(layer_idx, expert_idx)
            self.expert_cache[key] = expert
            
            load_time = time.time() - start_time
            print(f"[ExpertManager] Expert {key} loaded in {load_time:.2f}s")
            
            return expert
    
    def _load_expert(self, layer_idx: int, expert_idx: int) -> MixtralExpert:
        """
        Load an expert from disk
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            Loaded expert module
        """
        # Create expert module
        expert = MixtralExpert(self.config)
        
        # Load weights from disk
        weights = load_expert_weights(self.model_path, layer_idx, expert_idx)
        
        # Create state dict for the expert
        state_dict = {}
        for name, tensor in weights.items():
            # Map weight names to module parameter names
            if name == "w1":
                state_dict["w1.weight"] = tensor
            elif name == "w2":
                state_dict["w2.weight"] = tensor
            elif name == "w3":
                state_dict["w3.weight"] = tensor
        
        # Load state dict
        expert.load_state_dict(state_dict)
        
        # Move to device and convert to correct dtype
        # Get dtype from config (usually bfloat16 for Mixtral)
        dtype = getattr(self.config, 'torch_dtype', torch.float16)
        expert = expert.to(device=self.device, dtype=dtype)
        expert.eval()
        
        return expert
    
    def preload_experts(self, layer_idx: int, expert_indices: list[int]):
        """
        Preload multiple experts for a layer
        
        This can be used to preload experts that are likely to be used soon.
        
        Args:
            layer_idx: Layer index
            expert_indices: List of expert indices to preload
        """
        for expert_idx in expert_indices:
            if (layer_idx, expert_idx) not in self.expert_cache:
                self.get_expert(layer_idx, expert_idx)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total": total,
                "hit_rate": hit_rate,
                "cached_experts": len(self.expert_cache),
                "max_experts": self.max_gpu_experts,
            }
    
    def clear_cache(self):
        """Clear the expert cache and free GPU memory"""
        with self.lock:
            for expert in self.expert_cache.values():
                expert.cpu()
                del expert
            
            self.expert_cache.clear()
            torch.cuda.empty_cache()
            
            # Reset statistics
            self.hits = 0
            self.misses = 0