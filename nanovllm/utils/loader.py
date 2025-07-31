import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    expert_params_mapping = getattr(model, "expert_params_mapping", [])
    
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
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        handled = True
                        break
                
                # Check expert weights
                if not handled and expert_params_mapping:
                    # Expert weights in Mixtral have format: model.layers.{layer}.block_sparse_moe.experts.{expert_id}.{w1/w2/w3}
                    for layer_idx in range(100):  # Assume max 100 layers
                        for expert_id in range(32):  # Assume max 32 experts
                            for weight_type in ["w1", "w2", "w3"]:
                                expert_pattern = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.{weight_type}"
                                if expert_pattern in weight_name:
                                    # Find the right parameter
                                    param_name = f"model.layers.{layer_idx}.block_sparse_moe.experts.{weight_type}"
                                    try:
                                        param = model.get_parameter(param_name)
                                        weight_loader = getattr(param, "weight_loader")
                                        weight_loader(param, f.get_tensor(weight_name), expert_id, weight_type)
                                        handled = True
                                        break
                                    except AttributeError:
                                        continue
                            if handled:
                                break
                        if handled:
                            break
                
                # Default loading
                if not handled:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
