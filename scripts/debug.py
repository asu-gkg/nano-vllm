#!/usr/bin/env python3
"""
Compare tokenization and decoding between Qwen3 and Mixtral
"""

import torch
from transformers import AutoTokenizer

print("Testing tokenization differences between Qwen3 and Mixtral\n")

# Test tokens
test_tokens = [29007, 325]  # The tokens our Mixtral model generated

print("1. Decoding the generated tokens:")
print(f"   Tokens: {test_tokens}")

# Try to decode with different tokenizers
try:
    # Qwen3 tokenizer (if available)
    qwen_tokenizer = AutoTokenizer.from_pretrained("/home/asu/qwen3-0.6b", trust_remote_code=True)
    qwen_decoded = qwen_tokenizer.decode(test_tokens, skip_special_tokens=True)
    print(f"   Qwen3 decodes as: '{qwen_decoded}'")
except Exception as e:
    print(f"   Qwen3 tokenizer not available: {e}")

try:
    # Mixtral tokenizer
    mixtral_tokenizer = AutoTokenizer.from_pretrained("/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1")
    mixtral_decoded = mixtral_tokenizer.decode(test_tokens, skip_special_tokens=True)
    print(f"   Mixtral decodes as: '{mixtral_decoded}'")
except Exception as e:
    print(f"   Mixtral tokenizer error: {e}")

print("\n2. Testing common prompt encoding:")
test_prompt = "Hello, world!"
print(f"   Prompt: '{test_prompt}'")

try:
    qwen_tokens = qwen_tokenizer.encode(test_prompt)
    print(f"   Qwen3 encodes as: {qwen_tokens}")
except:
    pass

try:
    mixtral_tokens = mixtral_tokenizer.encode(test_prompt)
    print(f"   Mixtral encodes as: {mixtral_tokens}")
except:
    pass

print("\n3. Vocabulary comparison:")
try:
    print(f"   Qwen3 vocab size: {qwen_tokenizer.vocab_size}")
except:
    pass

try:
    print(f"   Mixtral vocab size: {mixtral_tokenizer.vocab_size}")
except:
    pass

print("\n4. Special tokens comparison:")
try:
    print(f"   Qwen3 BOS token: {qwen_tokenizer.bos_token} (id: {qwen_tokenizer.bos_token_id})")
    print(f"   Qwen3 EOS token: {qwen_tokenizer.eos_token} (id: {qwen_tokenizer.eos_token_id})")
except:
    pass

try:
    print(f"   Mixtral BOS token: {mixtral_tokenizer.bos_token} (id: {mixtral_tokenizer.bos_token_id})")
    print(f"   Mixtral EOS token: {mixtral_tokenizer.eos_token} (id: {mixtral_tokenizer.eos_token_id})")
except:
    pass