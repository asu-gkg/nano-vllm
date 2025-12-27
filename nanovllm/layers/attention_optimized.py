"""
Optimized Attention for 2080 Ti (Turing architecture)

支持:
1. Flash Attention 1.x (如果安装) - 推荐
2. xFormers memory_efficient_attention (备选)
3. PyTorch SDPA with 向量化转换 (fallback)
4. Triton paged attention for decode phase
"""

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

# 尝试导入可用的attention后端
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_V2 = False  # True表示Flash Attention 2.x (需要Ampere+)
XFORMERS_AVAILABLE = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_V2 = True
    print("✓ Flash Attention 2.x 已加载 (需要Ampere或更新架构)")
except ImportError:
    try:
        # 尝试导入flash_attn 1.x版本
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func
        FLASH_ATTN_AVAILABLE = True
        FLASH_ATTN_V2 = False
        print("✓ Flash Attention 1.x 已加载 (支持Turing架构)")
    except ImportError:
        pass

try:
    from xformers.ops import memory_efficient_attention, fmha
    XFORMERS_AVAILABLE = True
    print("✓ xFormers 已加载")
except ImportError:
    pass

# 检查GPU架构以确定是否可以使用Flash Attention
def _check_gpu_arch():
    """检查当前GPU是否支持Flash Attention 2.x"""
    try:
        import torch
        if torch.cuda.is_available():
            # 获取GPU计算能力
            major, minor = torch.cuda.get_device_capability()
            # Ampere及以上架构: 8.0+
            return major >= 8
    except:
        pass
    return False

GPU_SUPPORTS_FLASH_V2 = _check_gpu_arch()

if FLASH_ATTN_V2 and not GPU_SUPPORTS_FLASH_V2:
    print(f"⚠ 当前GPU不支持Flash Attention 2.x，将使用PyTorch SDPA")
    
if not FLASH_ATTN_AVAILABLE and not XFORMERS_AVAILABLE:
    print("⚠ 未检测到Flash Attention或xFormers，将使用PyTorch SDPA")


# ==================== Triton Kernels ====================

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


# ==================== Triton Paged Attention Kernel ====================

@triton.jit
def _paged_attention_kernel(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale,
    stride_output_batch, stride_output_head, stride_output_dim,
    stride_query_batch, stride_query_head, stride_query_dim,
    stride_kcache_block, stride_kcache_token, stride_kcache_head, stride_kcache_dim,
    stride_vcache_block, stride_vcache_token, stride_vcache_head, stride_vcache_dim,
    stride_bt_batch, stride_bt_block,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # GQA: 计算对应的kv head
    kv_head_idx = head_idx // (num_heads // num_kv_heads) if num_kv_heads < num_heads else head_idx
    
    # 获取当前序列的context长度
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    if context_len == 0:
        # 输出零
        output_offs = batch_idx * stride_output_batch + head_idx * stride_output_head + tl.arange(0, BLOCK_D)
        tl.store(output_ptr + output_offs, tl.zeros([BLOCK_D], dtype=tl.float16))
        return
    
    # 加载query [head_dim]
    q_offs = batch_idx * stride_query_batch + head_idx * stride_query_head + tl.arange(0, BLOCK_D)
    q = tl.load(query_ptr + q_offs).to(tl.float32)
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    max_score = float("-inf")
    sum_exp = 0.0
    
    # 计算需要的块数
    num_blocks_needed = (context_len + block_size - 1) // block_size
    
    # 遍历所有块
    for block_idx in range(num_blocks_needed):
        # 获取物理块ID
        physical_block_id = tl.load(block_tables_ptr + batch_idx * stride_bt_batch + block_idx * stride_bt_block)
        
        # 计算当前块中有效的token数
        block_start = block_idx * block_size
        block_end = tl.minimum(block_start + block_size, context_len)
        valid_tokens = block_end - block_start
        
        # 遍历块中的每个token
        for token_offset in range(block_size):
            if block_start + token_offset >= context_len:
                break
            
            # 加载key [head_dim]
            k_offs = (physical_block_id * stride_kcache_block + 
                     token_offset * stride_kcache_token + 
                     kv_head_idx * stride_kcache_head + 
                     tl.arange(0, BLOCK_D))
            k = tl.load(key_cache_ptr + k_offs).to(tl.float32)
            
            # 计算attention score
            score = tl.sum(q * k, axis=0) * scale
            
            # Online softmax
            new_max = tl.maximum(max_score, score)
            old_scale = tl.exp(max_score - new_max)
            new_scale = tl.exp(score - new_max)
            
            # 加载value [head_dim]
            v_offs = (physical_block_id * stride_vcache_block + 
                     token_offset * stride_vcache_token + 
                     kv_head_idx * stride_vcache_head + 
                     tl.arange(0, BLOCK_D))
            v = tl.load(value_cache_ptr + v_offs).to(tl.float32)
            
            # 更新累加器
            acc = acc * old_scale + v * new_scale
            sum_exp = sum_exp * old_scale + new_scale
            max_score = new_max
    
    # 最终normalize
    output = acc / sum_exp
    
    # 写回结果
    output_offs = batch_idx * stride_output_batch + head_idx * stride_output_head + tl.arange(0, BLOCK_D)
    tl.store(output_ptr + output_offs, output.to(tl.float16))


def paged_attention_triton(
    query: torch.Tensor,  # [batch, 1, num_heads, head_dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [batch, max_num_blocks]
    context_lens: torch.Tensor,  # [batch]
    scale: float,
) -> torch.Tensor:
    """Triton实现的paged attention，用于decode阶段"""
    batch_size, _, num_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    
    # 输出张量
    output = torch.empty(batch_size, num_heads, head_dim, dtype=query.dtype, device=query.device)
    
    # 调整query形状 [batch, num_heads, head_dim]
    query = query.squeeze(1)
    
    # 确定block维度
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    # 启动kernel
    grid = (batch_size, num_heads)
    
    _paged_attention_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        output.stride(0), output.stride(1), output.stride(2),
        query.stride(0), query.stride(1), query.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_D=BLOCK_D,
    )
    
    return output.unsqueeze(1)  # [batch, 1, num_heads, head_dim]


# ==================== 向量化的格式转换 ====================

def _varlen_to_padded_vectorized(tensor: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
    """向量化的varlen到padded转换"""
    batch_size = cu_seqlens.shape[0] - 1
    num_heads, head_dim = tensor.shape[1], tensor.shape[2]
    total_tokens = tensor.shape[0]
    
    # 预分配输出
    padded = torch.zeros(batch_size, max_seqlen, num_heads, head_dim, 
                        dtype=tensor.dtype, device=tensor.device)
    
    if batch_size == 1:
        # 单batch快速路径
        seq_len = cu_seqlens[1] - cu_seqlens[0]
        padded[0, :seq_len] = tensor[:seq_len]
        return padded
    
    # 创建索引映射 (向量化)
    # 为每个token计算其在padded tensor中的位置
    seq_indices = torch.zeros(total_tokens, dtype=torch.long, device=tensor.device)
    pos_indices = torch.zeros(total_tokens, dtype=torch.long, device=tensor.device)
    
    cu_seqlens_cpu = cu_seqlens.cpu()
    for i in range(batch_size):
        start, end = cu_seqlens_cpu[i].item(), cu_seqlens_cpu[i + 1].item()
        seq_len = end - start
        seq_indices[start:end] = i
        pos_indices[start:end] = torch.arange(seq_len, device=tensor.device)
    
    # 使用advanced indexing进行向量化赋值
    padded[seq_indices, pos_indices] = tensor
    
    return padded


def _padded_to_varlen_vectorized(padded: torch.Tensor, cu_seqlens: torch.Tensor):
    """向量化的padded到varlen转换"""
    batch_size = cu_seqlens.shape[0] - 1
    total_tokens = cu_seqlens[-1].item()
    num_heads, head_dim = padded.shape[2], padded.shape[3]
    
    # 预分配输出
    varlen = torch.empty(total_tokens, num_heads, head_dim, dtype=padded.dtype, device=padded.device)
    
    if batch_size == 1:
        # 单batch快速路径
        seq_len = total_tokens
        varlen[:] = padded[0, :seq_len]
        return varlen
    
    # 创建索引映射 (向量化)
    seq_indices = torch.zeros(total_tokens, dtype=torch.long, device=padded.device)
    pos_indices = torch.zeros(total_tokens, dtype=torch.long, device=padded.device)
    
    cu_seqlens_cpu = cu_seqlens.cpu()
    for i in range(batch_size):
        start, end = cu_seqlens_cpu[i].item(), cu_seqlens_cpu[i + 1].item()
        seq_len = end - start
        seq_indices[start:end] = i
        pos_indices[start:end] = torch.arange(seq_len, device=padded.device)
    
    # 使用advanced indexing进行向量化提取
    varlen = padded[seq_indices, pos_indices]
    
    return varlen


# ==================== Prefill Attention ====================

def attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, 
                     max_seqlen_q, max_seqlen_k, softmax_scale, 
                     causal=True, **kwargs):
    """Prefill阶段的attention计算"""
    total_tokens_q, num_heads, head_dim = q.shape
    total_tokens_k, num_kv_heads, _ = k.shape
    
    # 方案1: Flash Attention (如果GPU支持)
    if FLASH_ATTN_AVAILABLE and GPU_SUPPORTS_FLASH_V2:
        try:
            # Flash Attention 2.x API
            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            return output
        except (NameError, TypeError, RuntimeError) as e:
            # 如果Flash Attention失败，继续尝试其他方案
            pass
    
    # 方案1b: Flash Attention 1.x (支持Turing架构)
    if FLASH_ATTN_AVAILABLE and not FLASH_ATTN_V2:
        try:
            # Flash Attention 1.x 不支持GQA，需要先扩展kv heads
            k_fa = k
            v_fa = v
            if num_kv_heads != num_heads:
                repeat_factor = num_heads // num_kv_heads
                k_fa = k.repeat_interleave(repeat_factor, dim=1)
                v_fa = v.repeat_interleave(repeat_factor, dim=1)
            
            # Flash Attention 1.x on Turing only supports float16
            # Convert to float16 if needed
            import torch
            if torch.cuda.is_available():
                gpu_major, _ = torch.cuda.get_device_capability()
                if gpu_major == 7:  # Turing architecture
                    q_fa = q.to(torch.float16)
                    k_fa = k_fa.to(torch.float16)
                    v_fa = v_fa.to(torch.float16)
                else:
                    q_fa = q
            else:
                q_fa = q
            
            output = flash_attn_unpadded_func(
                q_fa, k_fa, v_fa,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            
            # Convert back to original dtype if needed
            if output.dtype != q.dtype:
                output = output.to(q.dtype)
            
            return output
        except (NameError, RuntimeError) as e:
            # 如果失败，继续尝试其他方案
            pass
    
    # 方案2: xFormers
    if XFORMERS_AVAILABLE:
        try:
            # 创建attention bias for causal
            if causal:
                attn_bias = fmha.attn_bias.LowerTriangularMask()
            else:
                attn_bias = None
            
            # xFormers支持varlen输入通过BlockDiagonalMask
            batch_size = cu_seqlens_q.shape[0] - 1
            
            # 转换为xFormers格式
            q_padded = _varlen_to_padded_vectorized(q, cu_seqlens_q, max_seqlen_q)
            k_padded = _varlen_to_padded_vectorized(k, cu_seqlens_k, max_seqlen_k)
            v_padded = _varlen_to_padded_vectorized(v, cu_seqlens_k, max_seqlen_k)
            
            # GQA处理
            if num_kv_heads != num_heads:
                repeat_factor = num_heads // num_kv_heads
                k_padded = k_padded.repeat_interleave(repeat_factor, dim=2)
                v_padded = v_padded.repeat_interleave(repeat_factor, dim=2)
            
            # xFormers期望 [batch, seq, heads, dim]
            output = memory_efficient_attention(
                q_padded, k_padded, v_padded,
                attn_bias=attn_bias,
                scale=softmax_scale,
            )
            
            return _padded_to_varlen_vectorized(output, cu_seqlens_q)
        except Exception as e:
            print(f"xFormers失败: {e}, 回退到SDPA")
    
    # 方案3: PyTorch SDPA with 向量化转换
    batch_size = cu_seqlens_q.shape[0] - 1
    
    q_padded = _varlen_to_padded_vectorized(q, cu_seqlens_q, max_seqlen_q)
    k_padded = _varlen_to_padded_vectorized(k, cu_seqlens_k, max_seqlen_k)
    v_padded = _varlen_to_padded_vectorized(v, cu_seqlens_k, max_seqlen_k)
    
    # GQA处理
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_padded = k_padded.repeat_interleave(repeat_factor, dim=2)
        v_padded = v_padded.repeat_interleave(repeat_factor, dim=2)
    
    # 调整维度顺序 [batch, heads, seq, dim]
    q_sdpa = q_padded.transpose(1, 2)
    k_sdpa = k_padded.transpose(1, 2)
    v_sdpa = v_padded.transpose(1, 2)
    
    # SDPA计算
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        out_sdpa = scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            scale=softmax_scale,
            is_causal=causal
        )
    
    out_padded = out_sdpa.transpose(1, 2)
    return _padded_to_varlen_vectorized(out_padded, cu_seqlens_q)


# ==================== Decode Attention ====================

def attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, 
                      softmax_scale, causal=True, **kwargs):
    """Decode阶段的attention计算，使用paged KV cache
    
    注意：此函数需要兼容CUDA Graph capture，避免使用:
    - .item() 调用
    - 基于tensor值的条件分支 (if cache_len == 0 等)
    """
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    
    if k_cache.numel() == 0 or v_cache.numel() == 0:
        return torch.zeros_like(q)
    
    num_kv_heads = k_cache.shape[2]
    num_blocks, block_size = k_cache.shape[0], k_cache.shape[1]
    
    # 获取block_tables
    context = get_context()
    block_tables = context.block_tables
    
    # 计算每个batch需要的最大块数
    max_blocks_per_batch = block_tables.shape[1]
    total_slots = max_blocks_per_batch * block_size
    
    # 使用index_select批量获取所有需要的KV块
    # block_tables: [batch, max_blocks] -> 展平后索引
    flat_block_indices = block_tables.view(-1)  # [batch * max_blocks]
    
    # 获取所有块的K,V: [batch * max_blocks, block_size, num_kv_heads, head_dim]
    all_k_blocks = k_cache[flat_block_indices]
    all_v_blocks = v_cache[flat_block_indices]
    
    # 重塑为 [batch, max_blocks * block_size, num_kv_heads, head_dim]
    all_k = all_k_blocks.view(batch_size, total_slots, num_kv_heads, head_dim)
    all_v = all_v_blocks.view(batch_size, total_slots, num_kv_heads, head_dim)
    
    # GQA处理
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        all_k = all_k.repeat_interleave(repeat_factor, dim=2)
        all_v = all_v.repeat_interleave(repeat_factor, dim=2)
    
    # 准备SDPA格式 [batch, num_heads, seq, head_dim]
    q_sdpa = q.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
    k_sdpa = all_k.transpose(1, 2)  # [batch, num_heads, total_slots, head_dim]
    v_sdpa = all_v.transpose(1, 2)  # [batch, num_heads, total_slots, head_dim]
    
    # 创建attention mask来屏蔽超出cache_len的位置
    # positions: [1, total_slots]
    positions = torch.arange(total_slots, device=q.device, dtype=cache_seqlens.dtype).unsqueeze(0)
    # cache_seqlens: [batch, 1]
    cache_lens_expanded = cache_seqlens.unsqueeze(1)
    # mask: [batch, total_slots], True表示有效位置
    valid_mask = positions < cache_lens_expanded
    
    # 转换为attention mask格式
    # SDPA的attn_mask添加到attention scores上，所以无效位置应该是-inf
    attn_mask = torch.where(
        valid_mask.unsqueeze(1).unsqueeze(2),  # [batch, 1, 1, total_slots]
        torch.zeros(1, device=q.device, dtype=q.dtype),
        torch.tensor(float('-inf'), device=q.device, dtype=q.dtype)
    )
    
    # SDPA计算
    output = scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        attn_mask=attn_mask,
        scale=softmax_scale,
        is_causal=False
    )
    
    # 转换回 [batch, 1, num_heads, head_dim]
    return output.transpose(1, 2)


# ==================== Attention Module ====================

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


# ==================== 性能信息打印 ====================
def print_attention_backend_info():
    """打印当前使用的attention后端信息"""
    import torch
    
    print("\n=== Attention Backend Info ===")
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        major, minor = torch.cuda.get_device_capability()
        arch_name = {
            6: "Pascal (GTX 10系列)",
            7: "Turing (RTX 20系列, 支持Flash Attn 1.x)",
            8: "Ampere (RTX 30系列, 支持Flash Attn 2.x)",
            9: "Hopper (H100, 支持Flash Attn 2.x)",
        }.get(major, f"Unknown (compute {major}.{minor})")
        print(f"GPU: {gpu_name} (compute {major}.{minor})")
        print(f"GPU架构: {arch_name}")
    
    if FLASH_ATTN_AVAILABLE:
        if FLASH_ATTN_V2:
            if GPU_SUPPORTS_FLASH_V2:
                print(f"Flash Attention 2.x: ✓ 可用且GPU支持")
            else:
                print(f"Flash Attention 2.x: ⚠ 已安装但GPU不支持 (需要Ampere+)")
        else:
            print(f"Flash Attention 1.x: ✓ 可用")
    else:
        print(f"Flash Attention: ✗ 未安装")
        
    print(f"xFormers: {'✓ 可用' if XFORMERS_AVAILABLE else '✗ 不可用'}")
    print(f"PyTorch SDPA: ✓ 可用 (fallback)")
    print(f"Triton Paged Attention: ✓ 可用")
    
    # 建议
    if not FLASH_ATTN_AVAILABLE:
        print("\n建议: 安装Flash Attention以获得最佳性能")
        if not GPU_SUPPORTS_FLASH_V2:
            print("  对于Turing架构 (1080 Ti, 2080 Ti等):")
            print("    pip install flash-attn==1.0.9 --no-build-isolation")
        else:
            print("    pip install flash-attn")
    elif FLASH_ATTN_V2 and not GPU_SUPPORTS_FLASH_V2:
        print("\n建议: 当前GPU是Turing架构，安装Flash Attention 1.x可获得更好性能:")
        print("  pip uninstall flash-attn")
        print("  pip install flash-attn==1.0.9 --no-build-isolation")
    
    if not XFORMERS_AVAILABLE:
        print("\n备选: 安装xFormers:")
        print("  pip install xformers")
    print("=" * 30)


if __name__ == "__main__":
    print_attention_backend_info()

