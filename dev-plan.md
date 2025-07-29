# Nano-vLLM 移除 Flash-Attention 依赖开发计划

## 项目背景

当前 nano-vllm 项目依赖 `flash-attn` 库。为了减少依赖和提高兼容性，需要完全移除 flash-attn 依赖，并参考 sglang 的实现方式进行优化。

## 当前状态分析

### 1. Flash-Attention 使用情况
- **依赖声明**: `pyproject.toml` 第18行声明了 `flash-attn` 依赖
- **导入语句**: `nanovllm/layers/attention.py` 第6行导入了 `flash_attn_varlen_func, flash_attn_with_kvcache`
- **实际使用**: 代码中第70行和第75行直接调用原库函数，**没有自定义替代实现**

### 2. 需要实现的替代方案
- **缺失**: 当前没有 `flash_attn_varlen_func` 的替代实现
- **缺失**: 当前没有 `flash_attn_with_kvcache` 的替代实现
- **需要**: 基于 PyTorch 的 `scaled_dot_product_attention` 实现替代方案

### 3. SGLang 参考实现
- **TorchNativeAttnBackend**: 使用 PyTorch 原生 `scaled_dot_product_attention`
- **TritonAttnBackend**: 使用 Triton 自定义 kernel
- 支持多种 backend 切换的架构设计

## 开发计划

### 阶段1: 实现替代方案并移除 Flash-Attention 依赖 (2-3天)

#### 1.1 移除依赖声明
- [ ] 从 `pyproject.toml` 中移除 `flash-attn` 依赖
- [ ] 测试安装过程确保无错误

#### 1.2 实现自定义函数
- [ ] 在 `nanovllm/layers/attention.py` 中实现 `attn_varlen_func` 函数
- [ ] 在 `nanovllm/layers/attention.py` 中实现 `attn_with_kvcache` 函数  
- [ ] 参考 SGLang 的 `TorchNativeAttnBackend` 实现
- [ ] 确保函数接口与原 flash_attn 函数兼容

#### 1.3 更新函数调用
- [ ] 将第70行的 `flash_attn_varlen_func` 改为 `attn_varlen_func`
- [ ] 将第75行的 `flash_attn_with_kvcache` 改为 `attn_with_kvcache`

#### 1.4 移除导入和依赖
- [ ] 移除 `nanovllm/layers/attention.py` 中的 `from flash_attn import` 导入语句
- [ ] 从 `pyproject.toml` 中移除 `flash-attn` 依赖

#### 1.5 基本功能测试
- [ ] 运行 `example.py` 确保基本推理功能正常
- [ ] 运行 `bench.py` 进行性能测试，确保性能无明显下降

### 阶段2: 优化和改进 (2-3天)

#### 2.1 参考 SGLang 的架构优化
- [ ] 研究 SGLang 的 `TorchNativeAttnBackend` 实现细节
- [ ] 优化现有的 `flash_attn_varlen_func` 实现：
  - 改进 GQA (Grouped Query Attention) 处理
  - 优化内存使用模式
  - 改进 causal mask 处理

#### 2.2 增强 KV Cache 处理
- [ ] 参考 SGLang 优化 `flash_attn_with_kvcache` 实现：
  - 改进 cache 格式处理
  - 优化 decode 阶段性能
  - 增强对不同输入格式的兼容性

#### 2.3 考虑 Triton Backend (可选)
- [ ] 评估是否需要添加 Triton backend
- [ ] 如果需要，参考 SGLang 的 `TritonAttnBackend` 实现 Triton 版本

### 阶段3: 测试和验证 (2-3天)

#### 3.1 单元测试
- [ ] **基础功能测试**
  - [ ] 测试不同 batch size (1, 4, 16, 32)
  - [ ] 测试不同序列长度 (128, 512, 2048, 4096)
  - [ ] 测试不同 head 配置 (num_heads: 8, 16, 32; head_dim: 64, 128)
  - [ ] 测试 GQA 配置 (num_kv_heads < num_heads)

- [ ] **边界条件测试**
  - [ ] 极短序列 (seq_len = 1)
  - [ ] 极长序列 (接近最大长度)
  - [ ] 空序列处理
  - [ ] 不规则 batch (变长序列)

- [ ] **数值精度测试**
  - [ ] 与原 flash_attn 输出对比 (差异 < 1e-3)
  - [ ] 梯度计算验证
  - [ ] 不同数据类型 (fp16, bf16, fp32)

#### 3.2 集成测试
- [ ] **完整模型测试**
  - [ ] Qwen 系列模型 (0.6B, 1.8B, 7B)
  - [ ] Llama 系列模型
  - [ ] 其他支持的模型架构

- [ ] **功能组合测试**
  - [ ] Tensor Parallelism + 自定义 attention
  - [ ] Prefix Caching + 自定义 attention  
  - [ ] CUDA Graph + 自定义 attention
  - [ ] Torch Compilation + 自定义 attention

#### 3.3 性能基准测试
- [ ] **吞吐量测试**
  ```bash
  # 测试配置
  - 硬件: RTX 4070 Laptop (8GB)
  - 模型: Qwen3-0.6B
  - 请求数: 256 sequences
  - 输入长度: 100-1024 tokens (随机)
  - 输出长度: 100-1024 tokens (随机)
  ```

- [ ] **延迟测试**
  - [ ] 首个 token 延迟 (TTFT)
  - [ ] 平均 token 间隔 (ITL)
  - [ ] 端到端延迟

- [ ] **内存使用测试**
  - [ ] 峰值内存使用
  - [ ] 内存泄漏检测
  - [ ] 不同序列长度下的内存scaling

#### 3.4 对比验证
- [ ] **与 Flash-Attention 对比**
  - [ ] 性能对比 (目标: 下降 < 10%)
  - [ ] 内存使用对比
  - [ ] 数值精度对比

- [ ] **与 vLLM 对比**
  - [ ] 相同配置下的性能对比
  - [ ] 资源使用对比

#### 3.5 压力测试
- [ ] **长时间运行测试**
  - [ ] 连续运行 24 小时
  - [ ] 监控内存泄漏
  - [ ] 性能稳定性检查

- [ ] **高负载测试**
  - [ ] 并发请求处理
  - [ ] 资源竞争场景
  - [ ] 异常恢复测试

### 阶段4: 文档和清理 (1-2天)

#### 4.1 更新文档
- [ ] **README.md 更新**
  - [ ] 移除 flash-attn 依赖说明
  - [ ] 更新安装说明 (更简单的依赖)
  - [ ] 添加性能对比数据表格
  - [ ] 说明兼容性改进

- [ ] **技术文档**
  - [ ] 添加 `docs/attention_implementation.md`
  - [ ] 记录实现细节和设计决策
  - [ ] 添加 API 文档
  - [ ] 性能调优指南

- [ ] **更新日志**
  - [ ] 详细记录变更内容
  - [ ] 标明破坏性变更 (如果有)
  - [ ] 迁移指南

#### 4.2 代码质量
- [ ] **代码清理**
  - [ ] 移除调试代码和注释
  - [ ] 统一代码风格 (black + isort)
  - [ ] 添加完整的类型注解
  - [ ] 添加 docstring

- [ ] **测试覆盖率**
  - [ ] 确保核心函数 100% 覆盖
  - [ ] 添加必要的单元测试
  - [ ] 集成测试覆盖所有功能

#### 4.3 发布准备
- [ ] **版本管理**
  - [ ] 更新版本号 (考虑使用语义化版本)
  - [ ] 准备发布说明
  - [ ] 标记重要的 git tag

- [ ] **CI/CD 更新**
  - [ ] 更新测试流水线
  - [ ] 添加性能回归测试
  - [ ] 更新构建配置

## 技术实现细节

### 核心修改点

1. **pyproject.toml**
```toml
# 移除这一行
"flash-attn",
```

2. **nanovllm/layers/attention.py**
```python
# 第一步：添加自定义实现函数（在导入部分之后）
def attn_varlen_func(q, k, v, max_seqlen_q=None, cu_seqlens_q=None,
                    max_seqlen_k=None, cu_seqlens_k=None, 
                    softmax_scale=1.0, causal=True, block_table=None):
    """
    PyTorch native implementation of variable-length attention
    使用 PyTorch scaled_dot_product_attention 实现，参考 SGLang TorchNativeAttnBackend
    """
    pass

def attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=None,
                     block_table=None, softmax_scale=1.0, causal=True):
    """
    PyTorch native implementation of attention with KV cache
    使用 PyTorch scaled_dot_product_attention 实现 KV cache 版本
    """
    pass

# 第二步：更新函数调用
# 第70行: flash_attn_varlen_func -> attn_varlen_func
# 第75行: flash_attn_with_kvcache -> attn_with_kvcache

# 第三步：移除导入语句
# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
```

### 关键技术实现细节

#### 函数签名分析
```python
# 原始 flash_attn 函数签名
flash_attn_varlen_func(
    q, k, v,                    # [total_tokens, num_heads, head_dim]
    cu_seqlens_q,              # [batch_size + 1] cumulative sequence lengths
    cu_seqlens_k,              # [batch_size + 1] 
    max_seqlen_q,              # int
    max_seqlen_k,              # int
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    return_attn_probs=False,
    block_table=None           # for prefix caching
)

flash_attn_with_kvcache(
    q,                         # [batch_size, seqlen_q, num_heads, head_dim]
    k_cache,                   # [batch_size, seqlen_k, num_kv_heads, head_dim]
    v_cache,                   # [batch_size, seqlen_k, num_kv_heads, head_dim]
    k=None,                    # new key (optional)
    v=None,                    # new value (optional)
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,        # [batch_size] current cache lengths
    block_table=None,          # for paged attention
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None
)
```

#### 基于 SGLang 的实现策略

##### 1. attn_varlen_func 实现
```python
def attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, 
                    max_seqlen_q, max_seqlen_k, 
                    softmax_scale=None, causal=False, **kwargs):
    """
    参考 SGLang TorchNativeAttnBackend._run_sdpa_forward_extend
    """
    # 1. 解析变长序列
    batch_size = cu_seqlens_q.numel() - 1
    
    # 2. 逐序列处理（类似 SGLang 的循环）
    outputs = []
    for seq_idx in range(batch_size):
        start_q = cu_seqlens_q[seq_idx]
        end_q = cu_seqlens_q[seq_idx + 1]
        start_k = cu_seqlens_k[seq_idx] 
        end_k = cu_seqlens_k[seq_idx + 1]
        
        q_seq = q[start_q:end_q]  # [seq_len_q, num_heads, head_dim]
        k_seq = k[start_k:end_k]  # [seq_len_k, num_kv_heads, head_dim]
        v_seq = v[start_k:end_k]  # [seq_len_k, num_kv_heads, head_dim]
        
        # 3. 使用 PyTorch SDPA
        output_seq = torch.nn.functional.scaled_dot_product_attention(
            q_seq.unsqueeze(0).transpose(1, 2),  # [1, num_heads, seq_len_q, head_dim]
            k_seq.unsqueeze(0).transpose(1, 2),  # [1, num_kv_heads, seq_len_k, head_dim]
            v_seq.unsqueeze(0).transpose(1, 2),  # [1, num_kv_heads, seq_len_k, head_dim]
            scale=softmax_scale,
            is_causal=causal
        ).transpose(1, 2).squeeze(0)  # [seq_len_q, num_heads, head_dim]
        
        outputs.append(output_seq)
    
    return torch.cat(outputs, dim=0)
```

##### 2. attn_with_kvcache 实现
```python
def attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=None,
                     softmax_scale=None, causal=False, **kwargs):
    """
    参考 SGLang TorchNativeAttnBackend._run_sdpa_forward_decode
    """
    # 1. 处理批次维度
    if q.dim() == 3:  # [batch, num_heads, head_dim] -> [batch, 1, num_heads, head_dim]
        q = q.unsqueeze(1)
        
    batch_size = q.shape[0]
    outputs = []
    
    # 2. 逐批次处理
    for batch_idx in range(batch_size):
        if cache_seqlens is not None:
            cache_len = cache_seqlens[batch_idx]
            k_seq = k_cache[batch_idx, :cache_len]  # [cache_len, num_kv_heads, head_dim]
            v_seq = v_cache[batch_idx, :cache_len]  # [cache_len, num_kv_heads, head_dim]
        else:
            k_seq = k_cache[batch_idx]
            v_seq = v_cache[batch_idx]
            
        q_seq = q[batch_idx]  # [1, num_heads, head_dim]
        
        # 3. 使用 PyTorch SDPA
        output_seq = torch.nn.functional.scaled_dot_product_attention(
            q_seq.unsqueeze(0).transpose(1, 2),  # [1, num_heads, 1, head_dim]
            k_seq.unsqueeze(0).transpose(1, 2),  # [1, num_kv_heads, cache_len, head_dim]
            v_seq.unsqueeze(0).transpose(1, 2),  # [1, num_kv_heads, cache_len, head_dim]
            scale=softmax_scale,
            is_causal=False  # decode 阶段通常不需要 causal
        ).transpose(1, 2).squeeze(0)  # [1, num_heads, head_dim]
        
        outputs.append(output_seq)
    
    return torch.stack(outputs, dim=0)
```

### 潜在优化方向

1. **参考 SGLang 的分批处理逻辑**
   - 改进序列并行处理
   - 优化内存访问模式
   - 支持 Grouped Query Attention (GQA)

2. **增强错误处理**
   - 添加更好的异常处理和回退机制
   - 提供清晰的错误信息
   - 支持多种输入格式的自动转换

3. **性能调优**
   - 调整分块大小 (chunk_size)
   - 优化 CUDA 内核选择策略
   - 支持 mixed precision

## 风险评估与缓解策略

### 高风险 🔴

#### 1. 性能下降风险
- **风险**: PyTorch SDPA 可能比 Flash-Attention 慢 10-30%
- **影响**: 用户体验下降，竞争力削弱
- **缓解策略**:
  - [ ] 分阶段发布，先在测试分支验证
  - [ ] 建立性能基准，设置 10% 下降的红线
  - [ ] 如超过红线，考虑 Triton 自定义实现
  - [ ] 保持 flash-attn 作为可选依赖 (环境变量控制)

#### 2. 数值精度差异
- **风险**: 不同 attention 实现可能产生不同的结果
- **影响**: 模型输出不一致，影响复现性
- **缓解策略**:
  - [ ] 设置严格的数值对比测试 (误差 < 1e-3)
  - [ ] 多模型验证，确保输出质量
  - [ ] 提供精度对比报告
  - [ ] 如发现显著差异，优先修复而非发布

### 中等风险 🟡

#### 3. 内存使用增加
- **风险**: 自定义实现可能内存效率较低
- **影响**: 在内存受限环境下性能下降
- **缓解策略**:
  - [ ] 实现分块计算，控制内存峰值
  - [ ] 内存使用监控和基准测试
  - [ ] 优化张量操作，减少临时内存分配

#### 4. 兼容性回归
- **风险**: 新实现可能在某些配置下失效
- **影响**: 部分用户无法正常使用
- **缓解策略**:
  - [ ] 全面的配置矩阵测试
  - [ ] 渐进式发布 (beta → stable)
  - [ ] 完善的错误处理和降级机制

#### 5. 维护负担增加
- **风险**: 自维护 attention 实现需要更多资源
- **影响**: 开发资源分散，bug 修复周期长
- **缓解策略**:
  - [ ] 完善的单元测试和文档
  - [ ] 简化实现，减少复杂度
  - [ ] 建立社区贡献机制

### 低风险 🟢

#### 6. 安装复杂性
- **风险**: 用户安装过程可能出现问题
- **影响**: 新用户体验下降
- **缓解策略**:
  - [ ] 依赖减少实际上降低安装难度
  - [ ] 更新安装文档和常见问题解答
  - [ ] 提供多种安装方式

## 成功标准

1. **功能性**: 所有现有功能正常工作
2. **性能**: 性能下降不超过 10%
3. **兼容性**: 支持所有现有的模型和配置
4. **稳定性**: 通过所有测试用例
5. **易用性**: 安装和使用更加简单

## 发布和监控策略

### 发布策略
#### Phase 1: 内部验证 (1-2天)
- [ ] 开发分支完成基本实现
- [ ] 内部功能测试通过
- [ ] 基础性能测试达标

#### Phase 2: Beta 测试 (3-5天)  
- [ ] 创建 `beta-no-flash-attn` 分支
- [ ] 邀请核心用户测试
- [ ] 收集反馈并修复问题
- [ ] 性能和稳定性验证

#### Phase 3: 正式发布 (1天)
- [ ] 合并到主分支
- [ ] 发布新版本 (如 v0.3.0)
- [ ] 更新文档和说明

### 监控指标
- [ ] **性能监控**
  - 吞吐量 (tokens/second)
  - 延迟指标 (TTFT, ITL)
  - 内存使用峰值

- [ ] **质量监控**  
  - 错误率和崩溃统计
  - 数值精度对比
  - 用户反馈评分

- [ ] **兼容性监控**
  - 不同模型的兼容性测试
  - 各种配置组合测试
  - 边界情况处理

### 回退策略
- [ ] **紧急回退机制**
  - 保留 flash-attn 版本的 git tag
  - 环境变量控制切换 (`USE_FLASH_ATTN=1`)
  - 快速版本回退能力

- [ ] **渐进回退**
  - 首先禁用新功能
  - 然后降级到安全版本
  - 最后分析和修复问题

## 时间估算

- **总计**: 7-12 个工作日
- **最小可行版本**: 3-4 天 (实现替代函数，移除依赖，基本测试)
- **完整优化版本**: 7-12 天 (包含所有优化、测试和发布)

### 详细时间分配
- **阶段1** (实现和基本测试): 2-3 天
- **阶段2** (优化改进): 2-3 天  
- **阶段3** (全面测试): 2-3 天
- **阶段4** (文档和发布): 1-2 天
- **缓冲时间** (问题修复): 1-2 天

## 下一步行动

### 即时行动 (今天)
1. **环境准备**: 设置开发分支，确保现有功能正常
2. **函数签名分析**: 详细分析现有调用，了解参数格式
3. **SGLang 代码研究**: 深入研究 TorchNativeAttnBackend 实现

### 第一周行动
1. **实现替代函数**: 完成 `attn_varlen_func` 和 `attn_with_kvcache`
2. **基础测试**: 确保数值正确性和基本功能
3. **性能初测**: 了解性能差距，决定是否需要优化

### 第二周行动  
1. **全面测试**: 执行完整的测试套件
2. **性能优化**: 根据测试结果进行必要优化
3. **文档更新**: 准备发布相关文档

## 长期考虑

### 社区建设
- [ ] 建立贡献者指南
- [ ] 设置 issue 模板和 PR 模板
- [ ] 创建性能回归测试的 CI/CD

### 技术演进
- [ ] **可选的 Flash-Attention 支持**
  - 环境变量控制: `USE_FLASH_ATTN=1`
  - 运行时检测和自动选择
  - 性能对比工具

- [ ] **未来优化方向**
  - Triton 自定义 kernel (如果性能不达标)
  - 更多 backend 支持 (CUTLASS, TensorRT)
  - 硬件特定优化 (不同 GPU 架构)

### 维护策略
- [ ] **版本策略**
  - 主版本: 重大架构变更
  - 次版本: 新功能和优化
  - 补丁版本: Bug 修复

- [ ] **向后兼容性**
  - API 稳定性保证
  - 废弃功能的迁移路径
  - 用户友好的错误信息

## 项目意义

这个项目的成功将为 nano-vLLM 带来：

1. **🔓 降低使用门槛**: 减少依赖，简化安装
2. **🎯 提高兼容性**: 支持更多环境和硬件
3. **🛠️ 增强可控性**: 自主掌控核心 attention 实现
4. **📈 技术积累**: 深入理解 attention 机制，为未来优化打基础
5. **🌟 社区价值**: 为开源社区提供一个高质量的 flash-attn 替代方案

---

**总结**: 这是一个技术挑战与机遇并存的项目。通过系统的计划、充分的测试和谨慎的发布策略，我们可以成功移除 flash-attn 依赖，同时保持甚至提升项目的质量和性能。 