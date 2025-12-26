"""
Legacy Attention implementation (原始版本)

使用PyTorch SDPA，没有Flash Attention或xFormers优化
"""

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

FLASH_ATTN_AVAILABLE = False
XFORMERS_AVAILABLE = False

def print_attention_backend_info():
    print("Using legacy attention (no Flash Attention or xFormers)")


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _varlen_to_padded(tensor: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
    """Convert varlen format to padded format for PyTorch SDPA"""
    batch_size = cu_seqlens.shape[0] - 1
    num_heads, head_dim = tensor.shape[1], tensor.shape[2]
    padded = torch.zeros(batch_size, max_seqlen, num_heads, head_dim, 
                        dtype=tensor.dtype, device=tensor.device)
    
    for i in range(batch_size):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]
        seq_len = end_idx - start_idx
        padded[i, :seq_len] = tensor[start_idx:end_idx]
    
    return padded


def _padded_to_varlen(padded: torch.Tensor, cu_seqlens: torch.Tensor):
    """Convert padded format back to varlen format"""
    batch_size = cu_seqlens.shape[0] - 1
    total_tokens = cu_seqlens[-1]
    
    num_heads, head_dim = padded.shape[2], padded.shape[3]
    varlen = torch.zeros(total_tokens, num_heads, head_dim,
                        dtype=padded.dtype, device=padded.device)
    
    for i in range(batch_size):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]
        seq_len = end_idx - start_idx
        varlen[start_idx:end_idx] = padded[i, :seq_len]
    
    return varlen


def attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, 
                     max_seqlen_q, max_seqlen_k, softmax_scale, 
                     causal=True, **kwargs):
    assert q.dim() == 3, f"期望q是3维 [total_tokens, num_heads, head_dim], 实际: {q.shape}"
    assert k.dim() == 3, f"期望k是3维 [total_tokens, num_kv_heads, head_dim], 实际: {k.shape}"
    assert v.dim() == 3, f"期望v是3维 [total_tokens, num_kv_heads, head_dim], 实际: {v.shape}"
    
    total_tokens_q, num_heads, head_dim = q.shape
    total_tokens_k, num_kv_heads, _ = k.shape
    
    batch_size = cu_seqlens_q.shape[0] - 1
    
    q_padded = _varlen_to_padded(q, cu_seqlens_q, max_seqlen_q)
    k_padded = _varlen_to_padded(k, cu_seqlens_k, max_seqlen_k)
    v_padded = _varlen_to_padded(v, cu_seqlens_k, max_seqlen_k)
    
    if num_kv_heads != num_heads:
        assert num_heads % num_kv_heads == 0
        repeat_factor = num_heads // num_kv_heads
        k_padded = k_padded.repeat_interleave(repeat_factor, dim=2)
        v_padded = v_padded.repeat_interleave(repeat_factor, dim=2)
    
    q_sdpa = q_padded.transpose(1, 2)
    k_sdpa = k_padded.transpose(1, 2)
    v_sdpa = v_padded.transpose(1, 2)
    
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        out_sdpa = scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, 
            scale=softmax_scale,
            is_causal=causal
        )
    
    out_padded = out_sdpa.transpose(1, 2)
    output = _padded_to_varlen(out_padded, cu_seqlens_q)
    
    return output


def attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, 
                      softmax_scale, causal=True, **kwargs):
    assert q.dim() == 4, f"期望q是4维 [batch_size, 1, num_heads, head_dim], 实际: {q.shape}"
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    assert seq_len_q == 1, f"decode阶段q的seq_len应该是1, 实际: {seq_len_q}"
    
    if k_cache.numel() == 0 or v_cache.numel() == 0:
        return torch.zeros_like(q)
    
    num_blocks, block_size, num_kv_heads, cache_head_dim = k_cache.shape
    
    context = get_context()
    
    if context.block_tables is None:
        k_cache_flat = k_cache.view(-1, num_kv_heads, head_dim)
        v_cache_flat = v_cache.view(-1, num_kv_heads, head_dim)
        
        outputs = []
        for batch_idx in range(batch_size):
            curr_cache_len = cache_seqlens[batch_idx]
            if curr_cache_len == 0:
                batch_output = torch.zeros(1, 1, num_heads, head_dim, 
                                         dtype=q.dtype, device=q.device)
                outputs.append(batch_output)
                continue
                
            batch_k = k_cache_flat[:curr_cache_len]
            batch_v = v_cache_flat[:curr_cache_len]
            
            if num_kv_heads != num_heads:
                repeat_factor = num_heads // num_kv_heads
                batch_k = batch_k.repeat_interleave(repeat_factor, dim=1)
                batch_v = batch_v.repeat_interleave(repeat_factor, dim=1)
            
            batch_q = q[batch_idx:batch_idx+1].transpose(1, 2)
            batch_k = batch_k.unsqueeze(0).transpose(1, 2)
            batch_v = batch_v.unsqueeze(0).transpose(1, 2)
            
            batch_out = scaled_dot_product_attention(
                batch_q, batch_k, batch_v,
                scale=softmax_scale,
                is_causal=False
            )
            
            batch_out = batch_out.transpose(1, 2)
            outputs.append(batch_out)
        
        output = torch.cat(outputs, dim=0)
        
    else:
        block_tables = context.block_tables
        
        outputs = []
        for batch_idx in range(batch_size):
            curr_cache_len = cache_seqlens[batch_idx]
            if curr_cache_len == 0:
                batch_output = torch.zeros(1, 1, num_heads, head_dim, 
                                         dtype=q.dtype, device=q.device)
                outputs.append(batch_output)
                continue
                
            num_complete_blocks = curr_cache_len // block_size
            last_block_len = curr_cache_len % block_size
            
            batch_block_table = block_tables[batch_idx]
            
            batch_k_parts = []
            batch_v_parts = []
            
            for block_idx in range(num_complete_blocks):
                physical_block_id = batch_block_table[block_idx]
                block_k = k_cache[physical_block_id]
                block_v = v_cache[physical_block_id]
                batch_k_parts.append(block_k)
                batch_v_parts.append(block_v)
            
            if last_block_len > 0:
                physical_block_id = batch_block_table[num_complete_blocks]
                block_k = k_cache[physical_block_id, :last_block_len]
                block_v = v_cache[physical_block_id, :last_block_len]
                batch_k_parts.append(block_k)
                batch_v_parts.append(block_v)
            
            if batch_k_parts:
                batch_k = torch.cat(batch_k_parts, dim=0)
                batch_v = torch.cat(batch_v_parts, dim=0)
            else:
                batch_k = torch.zeros(curr_cache_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
                batch_v = torch.zeros(curr_cache_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
            
            if num_kv_heads != num_heads:
                repeat_factor = num_heads // num_kv_heads
                batch_k = batch_k.repeat_interleave(repeat_factor, dim=1)
                batch_v = batch_v.repeat_interleave(repeat_factor, dim=1)
            
            batch_q = q[batch_idx:batch_idx+1].transpose(1, 2)
            batch_k = batch_k.unsqueeze(0).transpose(1, 2)
            batch_v = batch_v.unsqueeze(0).transpose(1, 2)
            
            batch_out = scaled_dot_product_attention(
                batch_q, batch_k, batch_v,
                scale=softmax_scale,
                is_causal=False
            )
            
            batch_out = batch_out.transpose(1, 2)
            outputs.append(batch_out)
        
        output = torch.cat(outputs, dim=0)
    
    return output


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            
            o = attn_varlen_func(q, k, v,
                                cu_seqlens_q=context.cu_seqlens_q,
                                cu_seqlens_k=context.cu_seqlens_k,
                                max_seqlen_q=context.max_seqlen_q,
                                max_seqlen_k=context.max_seqlen_k,
                                softmax_scale=self.scale, causal=True)
        else:
            o = attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                  cache_seqlens=context.context_lens,
                                  softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o

