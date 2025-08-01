#!/usr/bin/env python3
"""
Mixtral 8x7B Example with Dynamic Expert Loading

This example demonstrates how to run the full Mixtral 8x7B model (95GB) 
on a single GPU with only 22GB VRAM using dynamic expert loading.

Requirements:
- GPU with at least 22GB VRAM
- Mixtral model downloaded locally
- Sufficient CPU RAM or fast SSD for model files
"""

import os
import sys
import time
import torch
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"GPU Memory: {used/1024**3:.1f}GB / {total/1024**3:.1f}GB")


def generate_text(
    runner: ModelRunner,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """Generate text from a prompt"""
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
    print(f"\nPrompt: {prompt}")
    print(f"Input tokens ({len(input_ids)}): {input_ids[:10]}..." if len(input_ids) > 10 else f"Input tokens: {input_ids}")
    
    # Create sequence with sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    sequence = Sequence(input_ids, sampling_params)
    
    # Allocate blocks for the sequence (required for KV cache)
    # In a real system, this would be done by the scheduler
    sequence.block_table = list(range(sequence.num_blocks))
    
    # Generate tokens
    generated_tokens = []
    
    # Prefill phase
    print("\nGenerating...")
    start_time = time.time()
    
    with torch.inference_mode():
        # Run prefill
        token_ids = runner.run([sequence], is_prefill=True)
        if token_ids and token_ids[0] is not None:
            generated_tokens.append(token_ids[0])
            sequence.append_token(token_ids[0])
            
            # Decode phase - generate remaining tokens
            for i in range(max_new_tokens - 1):
                # Check if we hit EOS token
                if token_ids[0] == tokenizer.eos_token_id:
                    break
                    
                # Allocate new blocks if needed
                if sequence.num_blocks > len(sequence.block_table):
                    new_blocks_needed = sequence.num_blocks - len(sequence.block_table)
                    last_block = sequence.block_table[-1] if sequence.block_table else -1
                    sequence.block_table.extend(range(last_block + 1, last_block + 1 + new_blocks_needed))
                
                # Run decode
                token_ids = runner.run([sequence], is_prefill=False)
                if token_ids and token_ids[0] is not None:
                    generated_tokens.append(token_ids[0])
                    sequence.append_token(token_ids[0])
                else:
                    break
    
    generation_time = time.time() - start_time
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(input_ids + generated_tokens, skip_special_tokens=True)
    
    print(f"\nGenerated {len(generated_tokens)} tokens in {generation_time:.1f}s")
    print(f"Speed: {len(generated_tokens)/generation_time:.1f} tokens/s")
    
    # Print expert manager stats if available
    if hasattr(runner, 'expert_manager') and runner.expert_manager:
        stats = runner.expert_manager.get_stats()
        print(f"\nExpert Manager Stats:")
        print(f"  Cache hit rate: {stats['hit_rate']:.1%}")
        print(f"  Cached experts: {stats['cached_experts']}/{stats['max_experts']}")
        print(f"  Total accesses: {stats['total']}")
    
    return full_text


def main():
    """Main example function"""
    # Configuration
    MODEL_PATH = "/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1"  # Update this path
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your Mixtral model directory")
        return
    
    print("Mixtral 8x7B Dynamic Expert Loading Example")
    print("=" * 60)
    
    # Print initial memory
    print("\nInitial state:")
    print_memory_usage()
    
    # Create configuration
    print("\n1. Creating configuration...")
    config = Config(
        model=MODEL_PATH,
        tensor_parallel_size=1,  # Must be 1 for dynamic expert loading
        max_num_batched_tokens=2048,
        max_model_len=1024,
        gpu_memory_utilization=0.85,  # Conservative to leave room for activations
        enforce_eager=True,  # Disable CUDA graphs for now
    )
    
    print(f"   Model type: {config.hf_config.model_type}")
    print(f"   Hidden size: {config.hf_config.hidden_size}")
    print(f"   Num experts: {config.hf_config.num_local_experts}")
    print(f"   Experts per token: {config.hf_config.num_experts_per_tok}")
    
    # Create ModelRunner
    print("\n2. Initializing ModelRunner...")
    print("   This will load non-expert weights and run warmup...")
    init_start = time.time()
    
    runner = ModelRunner(config, rank=0, event=None)
    
    init_time = time.time() - init_start
    print(f"   ✓ Initialization completed in {init_time:.1f}s")
    print_memory_usage()
    
    # Load tokenizer
    print("\n3. Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Example prompts
    prompts = [
        "The key to artificial intelligence is",
        "In the future, robots will",
        "The most important scientific discovery of the 21st century is",
    ]
    
    print("\n4. Generating text...")
    print("-" * 60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nExample {i}:")
        result = generate_text(
            runner,
            prompt,
            tokenizer,
            max_new_tokens=50,
            temperature=0.7,
        )
        print(f"\nResult: {result}")
        print("-" * 60)
        print_memory_usage()
    
    # Cleanup
    print("\n5. Cleaning up...")
    runner.exit()
    print("✓ Done!")


if __name__ == "__main__":
    main()