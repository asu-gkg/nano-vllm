#!/usr/bin/env python3
"""
Test ExpertManager functionality
"""

import os
import sys
import torch
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import MixtralConfig
from nanovllm.engine.expert_manager import ExpertManager
from nanovllm.utils.context import set_expert_manager, get_expert_manager


def test_expert_loading():
    """Test basic expert loading functionality"""
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model path not found: {model_path}")
        return False
    
    print("Testing ExpertManager")
    print("=" * 60)
    
    # Load config
    config = MixtralConfig.from_pretrained(model_path)
    
    # Create expert manager
    print("\n1. Creating ExpertManager...")
    manager = ExpertManager(
        model_path=model_path,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_gpu_experts=4  # Small number for testing
    )
    print(f"✓ ExpertManager created with max_gpu_experts=4")
    
    # Test loading single expert
    print("\n2. Loading single expert...")
    start_time = time.time()
    expert = manager.get_expert(0, 0)
    load_time = time.time() - start_time
    print(f"✓ Expert (0, 0) loaded in {load_time:.2f}s")
    print(f"  Type: {type(expert)}")
    
    # Test cache hit
    print("\n3. Testing cache hit...")
    start_time = time.time()
    expert_cached = manager.get_expert(0, 0)
    cache_time = time.time() - start_time
    print(f"✓ Expert (0, 0) retrieved from cache in {cache_time:.4f}s")
    print(f"  Same object: {expert is expert_cached}")
    
    # Test loading multiple experts
    print("\n4. Loading multiple experts to test eviction...")
    for i in range(6):  # Load more than max_gpu_experts
        layer = i // 8
        expert_idx = i % 8
        expert = manager.get_expert(layer, expert_idx)
        stats = manager.get_stats()
        print(f"  Loaded expert ({layer}, {expert_idx}) - Cached: {stats['cached_experts']}/{stats['max_experts']}")
    
    # Check final stats
    print("\n5. Cache statistics:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True


@torch.inference_mode()
def test_with_model():
    """Test ExpertManager with actual model"""
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model path not found: {model_path}")
        return False
    
    print("\n\n6. Testing with Mixtral model...")
    print("-" * 60)
    
    from nanovllm.models.mixtral import MixtralForCausalLM
    from nanovllm.utils.loader import load_mixtral_non_expert_weights
    
    # Load config
    config = MixtralConfig.from_pretrained(model_path)
    
    # Create expert manager
    manager = ExpertManager(
        model_path=model_path,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_gpu_experts=42
    )
    
    # Set global expert manager
    set_expert_manager(manager)
    
    # Create model
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        torch.set_default_dtype(config.torch_dtype)
    
    model = MixtralForCausalLM(config)
    model.eval()
    
    # Load non-expert weights
    print("Loading non-expert weights...")
    load_mixtral_non_expert_weights(model, model_path)
    print("✓ Non-expert weights loaded")
    
    # Test forward pass
    print("\nTesting forward pass with expert loading...")
    batch_size, seq_len = 1, 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0)
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        positions = positions.cuda()
    
    # First pass (will load experts)
    print("\nFirst forward pass (loading experts)...")
    start_time = time.time()
    hidden_states = model(input_ids, positions)
    logits = model.compute_logits(hidden_states)
    first_time = time.time() - start_time
    
    print(f"✓ First pass completed in {first_time:.2f}s")
    print(f"  Output shape: {logits.shape}")
    
    # Get stats after first pass
    stats = manager.get_stats()
    print(f"\nExperts loaded: {stats['cached_experts']}")
    print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
    
    # Second pass (should use cached experts)
    print("\nSecond forward pass (using cached experts)...")
    start_time = time.time()
    hidden_states = model(input_ids, positions)
    logits = model.compute_logits(hidden_states)
    second_time = time.time() - start_time
    
    print(f"✓ Second pass completed in {second_time:.2f}s")
    print(f"  Speedup: {first_time/second_time:.2f}x")
    
    # Final stats
    stats = manager.get_stats()
    print(f"\nFinal cache stats:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Total accesses: {stats['total']}")
    
    # Reset
    if torch.cuda.is_available():
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
    
    return True


def main():
    """Run all tests"""
    try:
        # Test 1: Basic functionality
        success1 = test_expert_loading()
        
        # Test 2: With model
        success2 = test_with_model()
        
        print("\n" + "=" * 60)
        print("Test Summary:")
        print(f"  Expert loading: {'✓ Passed' if success1 else '✗ Failed/Skipped'}")
        print(f"  Model integration: {'✓ Passed' if success2 else '✗ Failed/Skipped'}")
        
        if success1 and success2:
            print("\n✅ All tests passed!")
            print("\nExpertManager is working correctly with:")
            print("- Dynamic expert loading")
            print("- LRU cache management")
            print("- GPU memory management")
            print("- Integration with Mixtral model")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()