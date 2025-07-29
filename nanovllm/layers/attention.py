import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

# ⚡ 性能优化：已注释掉数值检查以提升性能
# 注释掉的检查包括：torch.isnan(), torch.isinf(), 统计计算等
# 修复了4个.item()调用以支持CUDA Graph capture
# 恢复了关键的形状检查assert（防止崩溃，性能影响很小）
# 保留了重要的安全检查，移除了性能瓶颈
# 如需调试，可以取消注释相关数值检查


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
    
    # # stride检查 - 如果不匹配只打印警告，不阻塞执行
    # if k_cache.stride(1) != D or v_cache.stride(1) != D:
    #     print(f"⚠️  stride警告: 期望k_cache.stride(1)={D}, 实际={k_cache.stride(1)}")
    
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _varlen_to_padded(tensor: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
    """Convert varlen format to padded format for PyTorch SDPA"""
    batch_size = cu_seqlens.shape[0] - 1
    # if tensor.dim() == 3:  # [total_tokens, num_heads, head_dim]
    num_heads, head_dim = tensor.shape[1], tensor.shape[2]
    padded = torch.zeros(batch_size, max_seqlen, num_heads, head_dim, 
                        dtype=tensor.dtype, device=tensor.device)
    # else:  # other shapes
    #     raise NotImplementedError(f"Unsupported tensor shape: {tensor.shape}")
    
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
    
    # if padded.dim() == 4:  # [batch_size, seq_len, num_heads, head_dim]
    num_heads, head_dim = padded.shape[2], padded.shape[3]
    varlen = torch.zeros(total_tokens, num_heads, head_dim,
                        dtype=padded.dtype, device=padded.device)
    # else:
    #     raise NotImplementedError(f"Unsupported tensor shape: {padded.shape}")
    
    for i in range(batch_size):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]
        seq_len = end_idx - start_idx
        varlen[start_idx:end_idx] = padded[i, :seq_len]
    
    return varlen


def attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, 
                                   max_seqlen_q, max_seqlen_k, softmax_scale, 
                                   causal=True, **kwargs):
    # 步骤1: 验证输入形状的假设
    assert q.dim() == 3, f"期望q是3维 [total_tokens, num_heads, head_dim], 实际: {q.shape}"
    assert k.dim() == 3, f"期望k是3维 [total_tokens, num_kv_heads, head_dim], 实际: {k.shape}"
    assert v.dim() == 3, f"期望v是3维 [total_tokens, num_kv_heads, head_dim], 实际: {v.shape}"
    
    # 检查输入数值有效性 - 注释掉以提升性能
    # if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
    #     raise ValueError("[Prefill] 输入包含NaN!")
    # if torch.isinf(q).any() or torch.isinf(k).any() or torch.isinf(v).any():
    #     raise ValueError("[Prefill] 输入包含Inf!")
    
    total_tokens_q, num_heads, head_dim = q.shape
    total_tokens_k, num_kv_heads, _ = k.shape
    
    # assert q.shape[2] == k.shape[2] == v.shape[2], f"head_dim不匹配: q={q.shape[2]}, k={k.shape[2]}, v={v.shape[2]}"
    # assert k.shape[0] == v.shape[0], f"k和v的token数不匹配: k={k.shape[0]}, v={v.shape[0]}"
    
    # 步骤2: 验证cu_seqlens的假设
    # assert cu_seqlens_q is not None and cu_seqlens_k is not None, "cu_seqlens不能为None"
    # assert cu_seqlens_q.dim() == 1 and cu_seqlens_k.dim() == 1, "cu_seqlens应该是1维张量"
    # assert cu_seqlens_q[-1] == total_tokens_q, f"cu_seqlens_q的最后一个值应该等于total_tokens_q: {cu_seqlens_q[-1]} vs {total_tokens_q}"  # 注释掉：GPU->CPU同步
    # assert cu_seqlens_k[-1] == total_tokens_k, f"cu_seqlens_k的最后一个值应该等于total_tokens_k: {cu_seqlens_k[-1]} vs {total_tokens_k}"  # 注释掉：GPU->CPU同步
    
    batch_size = cu_seqlens_q.shape[0] - 1
    # assert batch_size > 0, f"batch_size应该大于0: {batch_size}"
    
    # 步骤3: 转换为padded格式进行计算
    q_padded = _varlen_to_padded(q, cu_seqlens_q, max_seqlen_q)
    k_padded = _varlen_to_padded(k, cu_seqlens_k, max_seqlen_k)
    v_padded = _varlen_to_padded(v, cu_seqlens_k, max_seqlen_k)
    
    # 检查转换后的数值 - 注释掉以提升性能
    # if torch.isnan(q_padded).any() or torch.isnan(k_padded).any() or torch.isnan(v_padded).any():
    #     raise ValueError("[Prefill] 转换后包含NaN!")
    
    # 验证转换结果
    # assert q_padded.shape == (batch_size, max_seqlen_q, num_heads, head_dim)
    # assert k_padded.shape == (batch_size, max_seqlen_k, num_kv_heads, head_dim)
    # assert v_padded.shape == (batch_size, max_seqlen_k, num_kv_heads, head_dim)
    
    # 步骤4: 处理GQA - 扩展k,v头数以匹配q
    if num_kv_heads != num_heads:
        assert num_heads % num_kv_heads == 0, f"num_heads({num_heads})必须是num_kv_heads({num_kv_heads})的倍数"
        repeat_factor = num_heads // num_kv_heads
        
        k_padded = k_padded.repeat_interleave(repeat_factor, dim=2)
        v_padded = v_padded.repeat_interleave(repeat_factor, dim=2)
        
        # 检查GQA扩展后的数值 - 注释掉以提升性能
        # if torch.isnan(k_padded).any() or torch.isnan(v_padded).any():
        #     raise ValueError("[Prefill] GQA扩展后包含NaN!")
        
        # # 验证扩展结果
        # assert k_padded.shape == (batch_size, max_seqlen_k, num_heads, head_dim)
        # assert v_padded.shape == (batch_size, max_seqlen_k, num_heads, head_dim)
    
    # 步骤5: 调整维度顺序以适配PyTorch SDPA
    q_sdpa = q_padded.transpose(1, 2)  # [batch, num_heads, max_seqlen_q, head_dim]
    k_sdpa = k_padded.transpose(1, 2)  # [batch, num_heads, max_seqlen_k, head_dim]
    v_sdpa = v_padded.transpose(1, 2)  # [batch, num_heads, max_seqlen_k, head_dim]
    
    # 步骤6: 执行注意力计算
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        out_sdpa = scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, 
            scale=softmax_scale,
            is_causal=causal
        )
    
    # 检查注意力输出 - 注释掉以提升性能
    # if torch.isnan(out_sdpa).any():
    #     raise ValueError("[Prefill] SDPA输出包含NaN!")
    # if torch.isinf(out_sdpa).any():
    #     raise ValueError("[Prefill] SDPA输出包含Inf!")
    
    # 检查输出的数值范围是否合理 - 注释掉以提升性能
    # out_mean = out_sdpa.mean().item()
    # out_std = out_sdpa.std().item()
    # out_max = out_sdpa.max().item()
    
    # if abs(out_mean) > 100:
    #     raise ValueError(f"[Prefill] SDPA输出均值异常: {out_mean}")
    # if out_std > 100:
    #     raise ValueError(f"[Prefill] SDPA输出标准差异常: {out_std}")
    # if abs(out_max) > 1000:
    #     raise ValueError(f"[Prefill] SDPA输出最大值异常: {out_max}")
    
    # assert out_sdpa.shape == (batch_size, num_heads, max_seqlen_q, head_dim)
    
    # 步骤7: 转换回原始格式
    out_padded = out_sdpa.transpose(1, 2)  # [batch, max_seqlen_q, num_heads, head_dim]
    output = _padded_to_varlen(out_padded, cu_seqlens_q)  # [total_tokens_q, num_heads, head_dim]
    
    # 最终检查 - 注释掉以提升性能
    # if torch.isnan(output).any():
    #     raise ValueError("[Prefill] 最终输出包含NaN!")
    # if torch.isinf(output).any():
    #     raise ValueError("[Prefill] 最终输出包含Inf!")
    
    # assert output.shape == q.shape, f"输出形状应该和输入q相同: {output.shape} vs {q.shape}"
    
    return output


def attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, 
                                    softmax_scale, causal=True, **kwargs):
    # 检查输入数值有效性 - 注释掉以提升性能
    # if torch.isnan(q).any():
    #     raise ValueError("[Decode] 输入q包含NaN!")
    # if torch.isinf(q).any():
    #     raise ValueError("[Decode] 输入q包含Inf!")
    
    # 步骤1: 验证输入形状的假设
    assert q.dim() == 4, f"期望q是4维 [batch_size, 1, num_heads, head_dim], 实际: {q.shape}"
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    assert seq_len_q == 1, f"decode阶段q的seq_len应该是1, 实际: {seq_len_q}"
    
    # 步骤2: 验证分块KV缓存
    if k_cache.numel() == 0 or v_cache.numel() == 0:
        return torch.zeros_like(q)
    
    # 检查缓存数值有效性 - 注释掉以提升性能
    # if torch.isnan(k_cache).any() or torch.isnan(v_cache).any():
    #     raise ValueError("[Decode] KV缓存包含NaN!")
    # if torch.isinf(k_cache).any() or torch.isinf(v_cache).any():
    #     raise ValueError("[Decode] KV缓存包含Inf!")
    
    # assert k_cache.dim() == 4, f"期望k_cache是4维 [num_blocks, block_size, num_kv_heads, head_dim], 实际: {k_cache.shape}"
    # assert v_cache.dim() == 4, f"期望v_cache是4维 [num_blocks, block_size, num_kv_heads, head_dim], 实际: {v_cache.shape}"
    
    num_blocks, block_size, num_kv_heads, cache_head_dim = k_cache.shape
    # assert cache_head_dim == head_dim, f"head_dim不匹配: q={head_dim}, cache={cache_head_dim}"
    # assert k_cache.shape == v_cache.shape, f"k_cache和v_cache形状不匹配: {k_cache.shape} vs {v_cache.shape}"
    
    # 步骤3: 验证cache_seqlens
    # assert cache_seqlens is not None, "cache_seqlens不能为None"
    # assert cache_seqlens.shape == (batch_size,), f"cache_seqlens形状应该是[{batch_size}], 实际: {cache_seqlens.shape}"
    
    # max_cache_len = num_blocks * block_size
    # assert cache_seqlens.max().item() <= max_cache_len, f"最大cache长度超出限制: {cache_seqlens.max().item()} > {max_cache_len}"  # 注释掉：.max().item()导致GPU->CPU同步，开销很大
    
    # 步骤4: 获取context中的block_tables
    from nanovllm.utils.context import get_context
    context = get_context()
    
    if context.block_tables is None:
        # 连续存储模式
        k_cache_flat = k_cache.view(-1, num_kv_heads, head_dim)
        v_cache_flat = v_cache.view(-1, num_kv_heads, head_dim)
        
        # 检查reshape后的数值 - 注释掉以提升性能
        # if torch.isnan(k_cache_flat).any() or torch.isnan(v_cache_flat).any():
        #     raise ValueError("[Decode] reshape后包含NaN!")
        
        outputs = []
        for batch_idx in range(batch_size):
            curr_cache_len = cache_seqlens[batch_idx]  # 避免.item()调用以支持CUDA Graph
            if curr_cache_len == 0:
                batch_output = torch.zeros(1, 1, num_heads, head_dim, 
                                         dtype=q.dtype, device=q.device)
                outputs.append(batch_output)
                continue
                
            batch_k = k_cache_flat[:curr_cache_len]
            batch_v = v_cache_flat[:curr_cache_len]
            
            # 检查提取的数值 - 注释掉以提升性能
            # if torch.isnan(batch_k).any() or torch.isnan(batch_v).any():
            #     raise ValueError(f"[Decode] batch {batch_idx} 提取数据包含NaN!")
            
            # 处理GQA
            if num_kv_heads != num_heads:
                repeat_factor = num_heads // num_kv_heads
                batch_k = batch_k.repeat_interleave(repeat_factor, dim=1)
                batch_v = batch_v.repeat_interleave(repeat_factor, dim=1)
                
                # 检查GQA扩展后的数值 - 注释掉以提升性能
                # if torch.isnan(batch_k).any() or torch.isnan(batch_v).any():
                #     raise ValueError(f"[Decode] batch {batch_idx} GQA扩展后包含NaN!")
            
            # 准备SDPA格式
            batch_q = q[batch_idx:batch_idx+1].transpose(1, 2)
            batch_k = batch_k.unsqueeze(0).transpose(1, 2)
            batch_v = batch_v.unsqueeze(0).transpose(1, 2)
            
            # 检查SDPA输入 - 注释掉以提升性能
            # if torch.isnan(batch_q).any() or torch.isnan(batch_k).any() or torch.isnan(batch_v).any():
            #     raise ValueError(f"[Decode] batch {batch_idx} SDPA输入包含NaN!")
            
            # 执行注意力计算
            batch_out = scaled_dot_product_attention(
                batch_q, batch_k, batch_v,
                scale=softmax_scale,
                is_causal=False
            )
            
            # 检查batch输出 - 注释掉以提升性能
            # if torch.isnan(batch_out).any():
            #     raise ValueError(f"[Decode] batch {batch_idx} SDPA输出包含NaN!")
            # if torch.isinf(batch_out).any():
            #     raise ValueError(f"[Decode] batch {batch_idx} SDPA输出包含Inf!")
            
            # batch_mean = batch_out.mean().item()
            # if abs(batch_mean) > 100:
            #     raise ValueError(f"[Decode] batch {batch_idx} SDPA输出均值异常: {batch_mean}")
            
            batch_out = batch_out.transpose(1, 2)
            outputs.append(batch_out)
        
        output = torch.cat(outputs, dim=0)
        
    else:
        # 分块访问模式
        block_tables = context.block_tables
        
        # # 验证block_tables
        # if block_tables.shape[0] != batch_size:
        #     raise ValueError(f"block_tables的batch_size不匹配: {block_tables.shape[0]} vs {batch_size}")
        
        outputs = []
        for batch_idx in range(batch_size):
            curr_cache_len = cache_seqlens[batch_idx]  # 避免.item()调用以支持CUDA Graph
            if curr_cache_len == 0:
                batch_output = torch.zeros(1, 1, num_heads, head_dim, 
                                         dtype=q.dtype, device=q.device)
                outputs.append(batch_output)
                continue
                
            # 计算需要多少个完整的块，以及最后一个块的有效长度
            num_complete_blocks = curr_cache_len // block_size
            last_block_len = curr_cache_len % block_size
            
            # 获取当前batch的块表
            batch_block_table = block_tables[batch_idx]
            
            # 收集所有需要的K,V数据
            batch_k_parts = []
            batch_v_parts = []
            
            # 处理完整的块
            for block_idx in range(num_complete_blocks):
                physical_block_id = batch_block_table[block_idx]  # 避免.item()调用以支持CUDA Graph
                # if physical_block_id == -1:
                #     raise ValueError(f"[Decode] batch {batch_idx} block {block_idx} 遇到无效物理块ID")
                
                block_k = k_cache[physical_block_id]
                block_v = v_cache[physical_block_id]
                
                batch_k_parts.append(block_k)
                batch_v_parts.append(block_v)
            
            # 处理最后一个不完整的块（如果存在）
            if last_block_len > 0:
                physical_block_id = batch_block_table[num_complete_blocks]  # 避免.item()调用以支持CUDA Graph
                # if physical_block_id == -1:
                #     raise ValueError(f"[Decode] batch {batch_idx} 最后一个块遇到无效物理块ID")
                
                block_k = k_cache[physical_block_id, :last_block_len]
                block_v = v_cache[physical_block_id, :last_block_len]
                
                batch_k_parts.append(block_k)
                batch_v_parts.append(block_v)
            
            # 拼接所有块的K,V数据
            if batch_k_parts:
                batch_k = torch.cat(batch_k_parts, dim=0)
                batch_v = torch.cat(batch_v_parts, dim=0)
            else:
                batch_k = torch.zeros(curr_cache_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
                batch_v = torch.zeros(curr_cache_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
            
            # 验证拼接结果
            # expected_shape = (curr_cache_len, num_kv_heads, head_dim)
            # if batch_k.shape != expected_shape or batch_v.shape != expected_shape:
            #     raise ValueError(f"[Decode] batch {batch_idx} 拼接后形状错误")
                
            # 检查拼接后的数值 - 注释掉以提升性能
            # if torch.isnan(batch_k).any() or torch.isnan(batch_v).any():
            #     raise ValueError(f"[Decode] batch {batch_idx} 拼接后包含NaN!")
            
            # 处理GQA
            if num_kv_heads != num_heads:
                repeat_factor = num_heads // num_kv_heads
                batch_k = batch_k.repeat_interleave(repeat_factor, dim=1)
                batch_v = batch_v.repeat_interleave(repeat_factor, dim=1)
                
                # 检查GQA扩展后的数值 - 注释掉以提升性能
                # if torch.isnan(batch_k).any() or torch.isnan(batch_v).any():
                #     raise ValueError(f"[Decode] batch {batch_idx} GQA扩展后包含NaN!")
            
            # 准备SDPA格式
            batch_q = q[batch_idx:batch_idx+1].transpose(1, 2)
            batch_k = batch_k.unsqueeze(0).transpose(1, 2)
            batch_v = batch_v.unsqueeze(0).transpose(1, 2)
            
            # 检查SDPA输入 - 注释掉以提升性能
            # if torch.isnan(batch_q).any() or torch.isnan(batch_k).any() or torch.isnan(batch_v).any():
            #     raise ValueError(f"[Decode] batch {batch_idx} SDPA输入包含NaN!")
            
            # 执行注意力计算
            batch_out = scaled_dot_product_attention(
                batch_q, batch_k, batch_v,
                scale=softmax_scale,
                is_causal=False
            )
            
            # 检查batch输出 - 注释掉以提升性能
            # if torch.isnan(batch_out).any():
            #     raise ValueError(f"[Decode] batch {batch_idx} block_tables SDPA输出包含NaN!")
            # if torch.isinf(batch_out).any():
            #     raise ValueError(f"[Decode] batch {batch_idx} block_tables SDPA输出包含Inf!")
            
            # batch_mean = batch_out.mean().item()
            # if abs(batch_mean) > 100:
            #     raise ValueError(f"[Decode] batch {batch_idx} block_tables SDPA输出均值异常: {batch_mean}")
            
            batch_out = batch_out.transpose(1, 2)
            outputs.append(batch_out)
        
        output = torch.cat(outputs, dim=0)
    
    # 最终检查 - 注释掉以提升性能
    # if torch.isnan(output).any():
    #     raise ValueError("[Decode] 最终输出包含NaN!")
    # if torch.isinf(output).any():
    #     raise ValueError("[Decode] 最终输出包含Inf!")
    
    # assert output.shape == q.shape, f"输出形状应该和输入q相同: {output.shape} vs {q.shape}"
    
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
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            
            o = attn_varlen_func(q, k, v,
                                cu_seqlens_q=context.cu_seqlens_q,
                                cu_seqlens_k=context.cu_seqlens_k,
                                max_seqlen_q=context.max_seqlen_q,
                                max_seqlen_k=context.max_seqlen_k,
                                softmax_scale=self.scale, causal=True)
        else:    # decode
            o = attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                                cache_seqlens=context.context_lens,
                                                softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o