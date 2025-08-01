#!/usr/bin/env python3
"""Test nano-vllm Mixtral loader integration"""

import os
import sys
import torch
import time
from unittest.mock import MagicMock

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.utils.loader import (
    load_mixtral_non_expert_weights,
    get_expert_weight_info,
    load_expert_weights
)

class MockMixtralModel:
    """Mock Mixtral model for testing"""
    
    def __init__(self):
        self.is_mixtral = True
        self.state_dict_loaded = {}
        self.missing_keys = []
        self.unexpected_keys = []
        
    def load_state_dict(self, state_dict, strict=True):
        """Mock load_state_dict"""
        self.state_dict_loaded = state_dict
        
        # Simulate some missing keys (expert weights)
        self.missing_keys = [
            f"layers.{i}.block_sparse_moe.experts.{j}.{w}.weight" 
            for i in range(32) 
            for j in range(8) 
            for w in ['w1', 'w2', 'w3']
        ]
        
        return self.missing_keys, self.unexpected_keys


def test_non_expert_loading():
    """Test loading non-expert weights"""
    print("=" * 60)
    print("Testing non-expert weight loading")
    print("=" * 60)
    
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    # Create mock model
    model = MockMixtralModel()
    
    # Test loading
    start_time = time.time()
    
    try:
        load_mixtral_non_expert_weights(model, model_path)
        load_time = time.time() - start_time
        
        print(f"\n✓ Loading completed in {load_time:.2f} seconds")
        print(f"✓ Loaded {len(model.state_dict_loaded)} weights")
        
        # Check weight mapping
        unmapped = sum(1 for k in model.state_dict_loaded if k.startswith("model."))
        mapped = sum(1 for k in model.state_dict_loaded if not k.startswith("model."))
        
        print(f"\nWeight mapping:")
        print(f"  - Correctly mapped: {mapped}")
        print(f"  - Still has prefix: {unmapped}")
        
        # Check critical weights
        critical_weights = {
            "embed_tokens": False,
            "norm": False,
            "lm_head": False
        }
        
        for weight_name in model.state_dict_loaded:
            for critical in critical_weights:
                if critical in weight_name:
                    critical_weights[critical] = True
        
        print("\nCritical weights:")
        for name, loaded in critical_weights.items():
            status = "✓" if loaded else "❌"
            print(f"  {status} {name}")
        
        # Sample weights
        print("\nSample loaded weights:")
        for i, (name, tensor) in enumerate(model.state_dict_loaded.items()):
            if i >= 5:
                break
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_expert_info():
    """Test expert weight information extraction"""
    print("\n" + "=" * 60)
    print("Testing expert weight info")
    print("=" * 60)
    
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    try:
        expert_info = get_expert_weight_info(model_path)
        
        print(f"\n✓ Found {len(expert_info)} experts")
        
        # Check structure
        layers = set()
        experts_per_layer = {}
        
        for (layer_idx, expert_idx) in expert_info:
            layers.add(layer_idx)
            if layer_idx not in experts_per_layer:
                experts_per_layer[layer_idx] = set()
            experts_per_layer[layer_idx].add(expert_idx)
        
        print(f"✓ {len(layers)} layers with experts")
        print(f"✓ {len(experts_per_layer[0])} experts per layer")
        
        # Check completeness
        incomplete = []
        for key, weights in expert_info.items():
            if set(weights.keys()) != {'w1', 'w2', 'w3'}:
                incomplete.append(key)
        
        if incomplete:
            print(f"❌ {len(incomplete)} experts have incomplete weights")
        else:
            print(f"✓ All experts have complete weights (w1, w2, w3)")
        
        # Sample expert info
        print("\nSample expert info:")
        for i, (key, weights) in enumerate(expert_info.items()):
            if i >= 3:
                break
            layer_idx, expert_idx = key
            print(f"  Layer {layer_idx}, Expert {expert_idx}:")
            for w_type, (filename, _) in weights.items():
                print(f"    {w_type}: {filename}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_expert_loading():
    """Test loading a single expert"""
    print("\n" + "=" * 60)
    print("Testing single expert loading")
    print("=" * 60)
    
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    # Test loading expert from layer 0, expert 0
    layer_idx = 0
    expert_idx = 0
    
    print(f"\nLoading expert {expert_idx} from layer {layer_idx}...")
    
    try:
        start_time = time.time()
        weights = load_expert_weights(model_path, layer_idx, expert_idx, device="cpu")
        load_time = time.time() - start_time
        
        print(f"✓ Loaded in {load_time:.3f} seconds")
        
        # Check weights
        if set(weights.keys()) == {'w1', 'w2', 'w3'}:
            print("✓ All expert weights present")
            
            total_params = 0
            for name, tensor in weights.items():
                params = tensor.numel()
                total_params += params
                print(f"  {name}: shape={tensor.shape}, params={params:,}")
            
            print(f"\nTotal expert parameters: {total_params:,}")
            print(f"Expert size: {total_params * 2 / 1024**3:.2f} GB (bfloat16)")
        else:
            print(f"❌ Missing weights: expected w1,w2,w3, got {list(weights.keys())}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test loading performance"""
    print("\n" + "=" * 60)
    print("Performance test")
    print("=" * 60)
    
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    # Test loading multiple experts
    print("\nLoading 10 experts sequentially...")
    
    load_times = []
    
    for i in range(10):
        layer = i % 32
        expert = i % 8
        
        start = time.time()

        weights = load_expert_weights(model_path, layer, expert, device="cpu")
        load_time = time.time() - start
        load_times.append(load_time)
        print(f"  Expert ({layer}, {expert}): {load_time:.3f}s")

    
    if load_times:
        avg_time = sum(load_times) / len(load_times)
        print(f"\nAverage load time: {avg_time:.3f}s")
        print(f"Min: {min(load_times):.3f}s, Max: {max(load_times):.3f}s")
    return True

def main():
    """Run all tests"""
    print("Testing nano-vllm Mixtral Loader")
    print("================================\n")
    
    # Check if running in the right environment
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("This script should be run on the machine with the model")
        return 1
    
    tests = [
        ("Non-expert loading", test_non_expert_loading),
        ("Expert info extraction", test_expert_info),
        ("Single expert loading", test_single_expert_loading),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())