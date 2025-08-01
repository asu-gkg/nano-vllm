#!/usr/bin/env python3
"""
Debug attention layer behavior
"""

import os
import sys
import torch

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monkey patch the attention layer to add debug output
original_forward = None

def debug_attention_forward(self, q, k, v):
    """Debug version of attention forward"""
    from nanovllm.utils.context import get_context
    
    print(f"\n[Attention Debug]")
    print(f"  q shape: {q.shape}")
    print(f"  k shape: {k.shape}")
    print(f"  v shape: {v.shape}")
    
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    
    print(f"  After reshape - q: {q.shape}, k: {k.shape}, v: {v.shape}")
    
    context = get_context()
    print(f"  Context is_prefill: {context.is_prefill}")
    
    k_cache, v_cache = self.k_cache, self.v_cache
    print(f"  k_cache shape: {k_cache.shape if k_cache.numel() else 'empty'}")
    print(f"  v_cache shape: {v_cache.shape if v_cache.numel() else 'empty'}")
    print(f"  k_cache numel: {k_cache.numel()}")
    print(f"  v_cache numel: {v_cache.numel()}")
    
    if context.slot_mapping is not None:
        print(f"  slot_mapping shape: {context.slot_mapping.shape}")
        print(f"  slot_mapping numel: {context.slot_mapping.numel()}")
        print(f"  slot_mapping: {context.slot_mapping}")
    else:
        print(f"  slot_mapping: None")
    
    print(f"  N (num tokens): {q.shape[0]}")
    
    # Call original forward
    return original_forward(self, q, k, v)


# Patch the attention layer
from nanovllm.layers.attention import Attention
original_forward = Attention.forward
Attention.forward = debug_attention_forward


def test_with_debug():
    """Run test with debug output"""
    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence
    
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model path not found: {model_path}")
        return
    
    print("Testing Mixtral with debug attention")
    print("=" * 60)
    
    try:
        # Create config with minimal settings
        config = Config(
            model=model_path,
            tensor_parallel_size=1,
            max_num_batched_tokens=256,  # Very small for debugging
            max_model_len=128,  # Very small for debugging
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )
        
        print("\nCreating ModelRunner...")
        print("This will run warmup first, then our test")
        print("-" * 60)
        
        runner = ModelRunner(config, rank=0, event=None)
        
        print("\n" + "-" * 60)
        print("Warmup completed! Now running test inference...")
        print("-" * 60)
        
        # Create test sequences
        test_sequences = [
            Sequence([1, 2, 3]),
            Sequence([100, 200]),
        ]
        
        # Allocate blocks
        block_id = 0
        for seq in test_sequences:
            num_blocks = seq.num_blocks
            seq.block_table = list(range(block_id, block_id + num_blocks))
            block_id += num_blocks
            print(f"Sequence {seq.seq_id}: {len(seq)} tokens, block_table={seq.block_table}")
        
        # Run inference
        with torch.inference_mode():
            token_ids = runner.run(test_sequences, is_prefill=True)
            print(f"\n✓ Success! Generated tokens: {token_ids}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original forward
        Attention.forward = original_forward


def main():
    """Run test"""
    test_with_debug()


if __name__ == "__main__":
    main()