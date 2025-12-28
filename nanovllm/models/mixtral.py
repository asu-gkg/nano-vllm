"""
Mixtral model implementation for nano-vllm

This implementation is designed for single-GPU execution with dynamic expert loading.
"""

import torch
from torch import nn
from transformers import MixtralConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention  
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import get_expert_manager
from nanovllm.utils.moe_calib import get_global_collector

# For single-GPU mode, we need non-parallel versions
class Embedding(nn.Module):
    """Non-parallel embedding for single GPU mode"""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        
    def forward(self, x: torch.Tensor):
        return nn.functional.embedding(x, self.weight)


class LMHead(nn.Module):
    """Non-parallel LM head for single GPU mode"""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        
    def forward(self, x: torch.Tensor):
        # In prefill mode, extract only the last token of each sequence
        from nanovllm.utils.context import get_context
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            # Extract last token indices for each sequence
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        return nn.functional.linear(x, self.weight)


class MixtralExpert(nn.Module):
    """Mixtral expert module"""
    
    def __init__(self, config: MixtralConfig):
        super().__init__()
        # Using standard nn.Linear
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = SiluAndMul()
        
    def forward(self, hidden_states):
        # [w1(x) * act(w3(x))] -> w2
        gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
        activated = self.act_fn(gate_up)
        return self.w2(activated)
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict to handle weight name mapping"""
        # Map weight names if needed
        mapped_dict = {}
        for key, value in state_dict.items():
            if key in ['w1', 'w2', 'w3']:
                mapped_dict[f"{key}.weight"] = value
            else:
                mapped_dict[key] = value
        
        return super().load_state_dict(mapped_dict, strict=strict)


class MixtralSparseMoeBlock(nn.Module):
    """Mixture of Experts block"""
    
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.layer_idx = layer_idx
        
        # Router gate
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
    def forward(self, hidden_states):
        # Handle both 2D and 3D inputs
        input_shape = hidden_states.shape
        if hidden_states.dim() == 2:
            # Already 2D: (num_tokens, hidden_dim)
            num_tokens, hidden_dim = hidden_states.shape
            is_3d = False
        else:
            # 3D: (batch_size, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            num_tokens = batch_size * seq_len
            hidden_states = hidden_states.view(num_tokens, hidden_dim)
            is_3d = True
        
        # Router computation
        router_logits = self.gate(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # Collect activation for calibration (if collector is active)
        collector = get_global_collector()
        if collector is not None:
            # hidden_states 必须是进入 MoE FFN 的输入（和 router 输入一致）
            # selected_experts 是每个 token 选中的 expert id（shape [num_tokens, K]，K=num_experts_per_tok）
            collector.observe(self.layer_idx, hidden_states, selected_experts)
        
        # Get expert manager
        expert_manager = get_expert_manager()
        if expert_manager is None:
            # Fallback for testing without expert manager
            # Just return zeros - this is only for testing!
            if is_3d:
                return torch.zeros(
                    batch_size, seq_len, hidden_dim,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
            else:
                return torch.zeros_like(hidden_states)
        
        # print(f"[MoE Layer {self.layer_idx}] Processing {num_tokens} tokens")
        
        # Execute expert computation
        final_hidden_states = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # Group tokens by expert for batch processing
        # import time
        # layer_start = time.time()
        
        for expert_idx in range(self.num_experts):
            # Find all tokens that need this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            num_tokens = expert_mask.sum().item()
            # print(f"[MoE Layer {self.layer_idx}] Expert {expert_idx} needed by {num_tokens} tokens")
            
            # Get the expert
            expert = expert_manager.get_expert(self.layer_idx, expert_idx)
            
            # Get tokens for this expert
            token_indices = torch.where(expert_mask)[0]
            expert_input = hidden_states[token_indices]
            
            # Compute expert output for all tokens at once
            if expert_input.shape[0] > 0:
                # compute_start = time.time()
                expert_output = expert(expert_input)
                # compute_time = time.time() - compute_start
                # print(f"[MoE Layer {self.layer_idx}] Expert {expert_idx} compute time: {compute_time:.3f}s")
                
                # Add weighted expert output to final result
                for i, (token_idx, output) in enumerate(zip(token_indices, expert_output)):
                    # Find which position(s) this expert is in for this token
                    expert_positions = (selected_experts[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                    for pos in expert_positions:
                        weight = routing_weights[token_idx, pos]
                        final_hidden_states[token_idx] += weight * output
        
        # layer_time = time.time() - layer_start
        # print(f"[MoE Layer {self.layer_idx}] Total time: {layer_time:.2f}s")
        
        # Restore original shape if needed
        if is_3d:
            final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        
        return final_hidden_states


class MixtralAttention(nn.Module):
    """Mixtral attention layer"""
    
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Using standard nn.Linear
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Attention
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim ** -0.5,
            self.num_kv_heads,
        )
        
    def forward(self, positions, hidden_states):
        # Handle both 2D and 3D inputs
        if hidden_states.dim() == 2:
            # 2D input: (num_tokens, hidden_size)
            num_tokens = hidden_states.size(0)
        else:
            # 3D input: (batch_size, seq_len, hidden_size)
            bsz, seqlen, _ = hidden_states.shape
            num_tokens = bsz * seqlen
            hidden_states = hidden_states.view(num_tokens, -1)
        
        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        
        # Flatten positions for RoPE (expects 1D tensor)
        positions = positions.reshape(-1)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Attention computation
        attn_output = self.attn(q, k, v)
        
        # Output projection
        return self.o_proj(attn_output)


class MixtralDecoderLayer(nn.Module):
    """Mixtral decoder layer"""
    
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MixtralAttention(config)
        self.block_sparse_moe = MixtralSparseMoeBlock(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, positions, hidden_states, residual):
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
        hidden_states = self.self_attn(positions, hidden_states)
        
        # MoE
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        
        return hidden_states, residual


class MixtralModel(nn.Module):
    """Mixtral base model"""
    
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Use non-parallel embedding for single GPU
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
            
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    """Mixtral for Causal Language Modeling"""
    
    # nano-vllm compatible mapping (not used in single GPU mode)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"), 
        "v_proj": ("qkv_proj", "v"),
    }
    
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.model = MixtralModel(config)
        # Use non-parallel LM head for single GPU
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        self.is_mixtral = True  # Marker for loader
        
        # Handle tied embeddings (critical for correct output!)
        if getattr(config, 'tie_word_embeddings', False):
            # Share weights between input embeddings and output head
            self.lm_head.weight = self.model.embed_tokens.weight
        
    def forward(self, input_ids, positions):
        hidden_states = self.model(input_ids, positions)
        return hidden_states
        
    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict to handle weight name mapping"""
        # Map weight names from loader format to model format
        mapped_dict = {}
        for key, value in state_dict.items():
            # The loader provides "embed_tokens.weight" but model expects "model.embed_tokens.weight"
            if not key.startswith("model.") and not key.startswith("lm_head."):
                if key.startswith("embed_tokens") or key.startswith("layers") or key.startswith("norm"):
                    mapped_dict[f"model.{key}"] = value
                else:
                    mapped_dict[key] = value
            else:
                mapped_dict[key] = value
        
        return super().load_state_dict(mapped_dict, strict=strict)