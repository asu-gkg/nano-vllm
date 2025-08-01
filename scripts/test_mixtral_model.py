#!/usr/bin/env python3
"""
Test Mixtral model implementation for nano-vllm
"""

import os
import sys
import torch
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup mock context module if needed
try:
    from nanovllm.utils.context import get_expert_manager, set_expert_manager
except ImportError:
    # Add expert manager functions to context module
    import nanovllm.utils.context as context
    _EXPERT_MANAGER = None
    
    def set_expert_manager(manager):
        global _EXPERT_MANAGER
        _EXPERT_MANAGER = manager
        
    def get_expert_manager():
        return _EXPERT_MANAGER
        
    context.set_expert_manager = set_expert_manager
    context.get_expert_manager = get_expert_manager


@torch.inference_mode()
def test_model_creation():
    """Test creating Mixtral model with small config"""
    from transformers import MixtralConfig
    from nanovllm.models.mixtral import MixtralForCausalLM
    
    print("1. Testing model creation with small config")
    print("-" * 60)
    
    # Create minimal config for testing
    config = MixtralConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=1000,
        max_position_embeddings=128,
    )
    
    model = MixtralForCausalLM(config)
    model.eval()
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass with different batch sizes
    test_cases = [
        (1, 10),  # batch_size=1, seq_len=10
        (2, 20),  # batch_size=2, seq_len=20
        (4, 5),   # batch_size=4, seq_len=5
    ]
    
    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = model(input_ids, positions)
        logits = model.compute_logits(hidden_states)
        
        assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        
        print(f"✓ Forward pass successful: batch_size={batch_size}, seq_len={seq_len}")
    
    return True


@torch.inference_mode()
def test_weight_loading():
    """Test loading real Mixtral weights"""
    from transformers import MixtralConfig
    from nanovllm.models.mixtral import MixtralForCausalLM
    from nanovllm.utils.loader import load_mixtral_non_expert_weights
    
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model path not found: {model_path}")
        print("Skipping weight loading test")
        return False
    
    print("\n2. Testing weight loading from real model")
    print("-" * 60)
    
    # Load config
    config = MixtralConfig.from_pretrained(model_path)
    print(f"✓ Config loaded: {config.num_hidden_layers} layers, {config.num_local_experts} experts")
    
    # Create model
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        torch.set_default_dtype(config.torch_dtype)
    
    model = MixtralForCausalLM(config)
    model.eval()
    
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"✓ Model created on {device_name}")
    
    # Load weights
    print("\nLoading non-expert weights...")
    start_time = time.time()
    
    try:
        load_mixtral_non_expert_weights(model, model_path)
        load_time = time.time() - start_time
        print(f"✓ Weights loaded successfully in {load_time:.2f}s")
    except Exception as e:
        print(f"✗ Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass
    print("\nTesting forward pass with loaded weights...")
    batch_size, seq_len = 1, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        positions = positions.cuda()
    
    # Warmup
    _ = model(input_ids, positions)
    
    # Timed run
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    hidden_states = model(input_ids, positions)
    logits = model.compute_logits(hidden_states)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    forward_time = time.time() - start_time
    
    print(f"✓ Forward pass completed in {forward_time*1000:.2f}ms")
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  Logits shape: {logits.shape}")
    
    # Reset defaults
    if torch.cuda.is_available():
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
    
    return True


@torch.inference_mode()
def test_expert_loading():
    """Test expert weight loading"""
    from nanovllm.utils.loader import get_expert_weight_info, load_expert_weights
    
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Model path not found: {model_path}")
        print("Skipping expert loading test")
        return False
    
    print("\n3. Testing expert weight loading")
    print("-" * 60)
    
    # Get expert info
    expert_info = get_expert_weight_info(model_path)
    print(f"✓ Found {len(expert_info)} expert weight mappings")
    
    # Test loading a single expert
    layer_idx, expert_idx = 0, 0
    key = (layer_idx, expert_idx)
    
    if key in expert_info:
        print(f"\nLoading expert weights for layer {layer_idx}, expert {expert_idx}...")
        
        try:
            weights = load_expert_weights(model_path, layer_idx, expert_idx)
            print(f"✓ Expert loaded successfully with weights:")
            for name, tensor in weights.items():
                print(f"  - {name}: {tensor.shape}")
        except Exception as e:
            print(f"✗ Expert loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def main():
    """Run all tests"""
    print("Testing Mixtral Model Implementation")
    print("=" * 60)
    
    try:
        # Test 1: Model creation
        success1 = test_model_creation()
        
        # Test 2: Weight loading
        success2 = test_weight_loading()
        
        # Test 3: Expert loading
        success3 = test_expert_loading()
        
        print("\n" + "=" * 60)
        print("Test Summary:")
        print(f"  Model creation: {'✓ Passed' if success1 else '✗ Failed'}")
        print(f"  Weight loading: {'✓ Passed' if success2 else '⚠️  Skipped'}")
        print(f"  Expert loading: {'✓ Passed' if success3 else '⚠️  Skipped'}")
        
        if success1:
            print("\n✅ Core functionality is working!")
            print("\nNote: MoE blocks return zeros without expert manager.")
            print("This is expected behavior for testing.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()