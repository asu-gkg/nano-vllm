#!/usr/bin/env python3
"""
Test Mixtral integration with ModelRunner
"""

import os
import sys
import torch
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import MixtralConfig
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence


def test_mixtral_runner():
    """Test Mixtral model with ModelRunner"""
    model_path = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model path not found: {model_path}")
        print("Please ensure Mixtral model is downloaded to the correct location")
        return False
    
    print("Testing Mixtral with ModelRunner")
    print("=" * 60)
    
    try:
        # Create config
        print("\n1. Creating configuration...")
        config = Config(
            model=model_path,
            tensor_parallel_size=1,  # Single GPU for dynamic expert loading
            max_num_batched_tokens=2048,  # Reduced for memory
            max_model_len=512,  # Reduced for memory
            gpu_memory_utilization=0.85,  # More conservative
            enforce_eager=True,  # Disable CUDA graphs for initial testing
        )
        
        print(f"✓ Config created:")
        print(f"  Model type: {config.hf_config.model_type}")
        print(f"  Hidden size: {config.hf_config.hidden_size}")
        print(f"  Num layers: {config.hf_config.num_hidden_layers}")
        print(f"  Num experts: {config.hf_config.num_local_experts}")
        
        # Create ModelRunner
        print("\n2. Creating ModelRunner...")
        print("  This will load non-expert weights and initialize expert manager...")
        
        runner = ModelRunner(config, rank=0, event=None)
        
        print("✓ ModelRunner created successfully")
        
        if hasattr(runner, 'expert_manager') and runner.expert_manager:
            print(f"✓ ExpertManager initialized with max_gpu_experts={runner.expert_manager.max_gpu_experts}")
        else:
            print("✗ ExpertManager not initialized (check model type detection)")
            return False
        
        # Test inference
        print("\n3. Testing inference...")
        
        # Create test sequences
        test_sequences = [
            Sequence([1, 2, 3, 4, 5]),  # Simple test sequence
            Sequence([100, 200, 300]),   # Another test
        ]
        
        # Manually allocate blocks for testing (normally done by scheduler)
        # Each sequence needs blocks allocated from the available pool
        block_id = 0
        for seq in test_sequences:
            num_blocks_needed = seq.num_blocks
            seq.block_table = list(range(block_id, block_id + num_blocks_needed))
            block_id += num_blocks_needed
            print(f"  Sequence {seq.seq_id}: {len(seq)} tokens, {num_blocks_needed} blocks, table={seq.block_table}")
        
        # Prefill pass
        print("\nRunning prefill pass...")
        print(f"  Test sequences: {len(test_sequences)}")
        for i, seq in enumerate(test_sequences):
            print(f"  Seq {i}: {len(seq)} tokens, block_table={seq.block_table}")
        
        start_time = time.time()
        
        with torch.inference_mode():
            token_ids = runner.run(test_sequences, is_prefill=True)
        
        prefill_time = time.time() - start_time
        print(f"✓ Prefill completed in {prefill_time:.2f}s")
        print(f"  Generated tokens: {token_ids}")
        
        # Check expert manager stats
        if runner.expert_manager:
            stats = runner.expert_manager.get_stats()
            print(f"\nExpert Manager Statistics:")
            print(f"  Cached experts: {stats['cached_experts']}/{stats['max_experts']}")
            print(f"  Cache hits: {stats['hits']}")
            print(f"  Cache misses: {stats['misses']}")
            print(f"  Hit rate: {stats['hit_rate']:.2%}")
        
        # Decode pass (if tokens were generated)
        if token_ids:
            print("\n4. Testing decode pass...")
            
            # Update sequences with generated tokens
            for seq, token_id in zip(test_sequences, token_ids):
                seq.append_token(token_id)
            
            start_time = time.time()
            
            with torch.inference_mode():
                new_tokens = runner.run(test_sequences, is_prefill=False)
            
            decode_time = time.time() - start_time
            print(f"✓ Decode completed in {decode_time:.3f}s")
            print(f"  Generated tokens: {new_tokens}")
            
            # Final stats
            if runner.expert_manager:
                stats = runner.expert_manager.get_stats()
                print(f"\nFinal Expert Manager Statistics:")
                print(f"  Hit rate: {stats['hit_rate']:.2%}")
                print(f"  Total accesses: {stats['total']}")
        
        # Cleanup
        print("\n5. Cleaning up...")
        runner.exit()
        print("✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test"""
    print("Mixtral ModelRunner Integration Test")
    print("=" * 60)
    print("\nNote: This test requires:")
    print("- Mixtral model downloaded to /home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1")
    print("- At least 22GB GPU memory")
    print("- Sufficient CPU RAM for model files")
    print()
    
    success = test_mixtral_runner()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Mixtral integration test passed!")
        print("\nThe ModelRunner successfully:")
        print("- Detected Mixtral model type")
        print("- Created ExpertManager for dynamic loading")
        print("- Loaded non-expert weights")
        print("- Performed inference with expert loading")
    else:
        print("❌ Mixtral integration test failed")
        print("\nPlease check:")
        print("- Model path is correct")
        print("- Sufficient GPU/CPU memory")
        print("- All dependencies are installed")


if __name__ == "__main__":
    main()