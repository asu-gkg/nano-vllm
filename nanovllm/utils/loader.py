"""
Model loader utilities for nano-vllm

Supports standard models and Mixtral with dynamic expert loading
"""

import os
import json
import torch
from torch import nn
from glob import glob
from typing import Dict, Optional, Set
import logging
from safetensors import safe_open

logger = logging.getLogger(__name__)

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Default weight loading function"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, model_name_or_path: str):
    """
    Universal model loading function
    
    Args:
        model: The model to load weights into
        model_name_or_path: Path to model directory or HuggingFace model name
    """
    # Check if this is a Mixtral model
    if hasattr(model, 'is_mixtral') and model.is_mixtral:
        # Mixtral special handling: only load non-expert weights
        load_mixtral_non_expert_weights(model, model_name_or_path)
    else:
        # Standard loading logic for other models
        load_standard_model(model, model_name_or_path)


def load_standard_model(model: nn.Module, path: str):
    """Standard model loading with packed modules support"""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Check packed modules (q/k/v proj)
                handled = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        handled = True
                        break
                
                # Default loading
                if not handled:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_mixtral_non_expert_weights(model: nn.Module, model_path: str):
    """
    Load Mixtral non-expert weights only
    
    This function:
    1. Reads the weight index to find all weights
    2. Groups non-expert weights by file for efficient loading
    3. Loads only non-expert weights and maps names correctly
    4. Applies weights to the model with strict=False
    
    Args:
        model: Mixtral model instance
        model_path: Path to Mixtral model directory
    """
    # Read weight index
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        # Fallback to PyTorch format
        index_file = os.path.join(model_path, "pytorch_model.bin.index.json")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"No weight index found in {model_path}")
    
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    
    logger.info(f"Loading Mixtral non-expert weights from {model_path}")
    
    # Group non-expert weights by file for efficient loading
    files_to_load: Dict[str, list] = {}
    expert_count = 0
    
    for weight_name, filename in weight_map.items():
        if "block_sparse_moe.experts" in weight_name:
            expert_count += 1
            continue
        
        if filename not in files_to_load:
            files_to_load[filename] = []
        files_to_load[filename].append(weight_name)
    
    logger.info(f"Skipping {expert_count} expert weights")
    logger.info(f"Loading non-expert weights from {len(files_to_load)} files")
    
    # Load weights
    state_dict = {}
    loaded_count = 0
    
    for filename, weight_names in sorted(files_to_load.items()):
        file_path = os.path.join(model_path, filename)
        
        try:
            if filename.endswith('.safetensors'):
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    # Only load the specific non-expert weights from this file
                    for weight_name in weight_names:
                        tensor = f.get_tensor(weight_name)
                        state_dict[weight_name] = tensor
                        loaded_count += 1
            else:
                # PyTorch format
                file_weights = torch.load(file_path, map_location='cpu')
                for weight_name in weight_names:
                    if weight_name in file_weights:
                        state_dict[weight_name] = file_weights[weight_name]
                        loaded_count += 1
            
            logger.debug(f"Loaded {len(weight_names)} weights from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            raise
    
    logger.info(f"Loaded {loaded_count} non-expert weights")
    
    # Map weight names (remove "model." prefix to match nano-vllm convention)
    mapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "", 1)
        mapped_state_dict[new_key] = value
    
    # Apply to model
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    
    # Log missing non-expert keys (expert keys are expected to be missing)
    non_expert_missing = [k for k in missing_keys if "experts" not in k]
    if non_expert_missing:
        logger.warning(f"Missing non-expert keys: {len(non_expert_missing)}")
        for key in non_expert_missing[:10]:
            logger.warning(f"  - {key}")
        if len(non_expert_missing) > 10:
            logger.warning(f"  ... and {len(non_expert_missing) - 10} more")
    
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        for key in unexpected_keys[:10]:
            logger.warning(f"  - {key}")
    
    logger.info("Mixtral non-expert weights loaded successfully")
    
    # Verify critical weights were loaded
    critical_prefixes = ["embed_tokens", "norm", "lm_head"]
    loaded_critical = []
    for prefix in critical_prefixes:
        if any(prefix in k for k in mapped_state_dict):
            loaded_critical.append(prefix)
    
    if len(loaded_critical) < len(critical_prefixes):
        missing_critical = set(critical_prefixes) - set(loaded_critical)
        logger.error(f"Critical weights missing: {missing_critical}")
        raise RuntimeError(f"Failed to load critical weights: {missing_critical}")


# Utility functions for expert loading (used by ExpertManager)

def get_expert_weight_info(model_path: str) -> Dict[tuple, Dict[str, tuple]]:
    """
    Get mapping of expert weights to their file locations
    
    Returns:
        Dict mapping (layer_idx, expert_idx) to weight info
    """
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        index_file = os.path.join(model_path, "pytorch_model.bin.index.json")
    
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


def load_expert_weights(
    model_path: str, 
    layer_idx: int, 
    expert_idx: int,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load weights for a specific expert
    
    Args:
        model_path: Path to model directory
        layer_idx: Layer index (0-31 for Mixtral)
        expert_idx: Expert index (0-7 for Mixtral)
        device: Device to load weights to
        
    Returns:
        Dict with keys 'w1', 'w2', 'w3' containing weight tensors
    """
    expert_info = get_expert_weight_info(model_path)
    key = (layer_idx, expert_idx)
    
    if key not in expert_info:
        raise ValueError(f"Expert ({layer_idx}, {expert_idx}) not found")
    
    weights = {}
    loaded_files = {}
    
    for weight_type, (filename, full_name) in expert_info[key].items():
        file_path = os.path.join(model_path, filename)
        
        # Cache file handles
        if filename not in loaded_files:
            if filename.endswith('.safetensors'):
                loaded_files[filename] = safe_open(file_path, framework="pt", device=device)
            else:
                loaded_files[filename] = torch.load(file_path, map_location=device)
        
        # Extract weight
        if filename.endswith('.safetensors'):
            weights[weight_type] = loaded_files[filename].get_tensor(full_name)
        else:
            weights[weight_type] = loaded_files[filename][full_name]
    
    # Close safetensors files
    for f in loaded_files.values():
        if hasattr(f, '__exit__'):
            f.__exit__(None, None, None)
    
    return weights