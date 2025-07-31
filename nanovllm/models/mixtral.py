import torch
from torch import nn
import torch.distributed as dist
from transformers import MixtralConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.moe import FusedMoE


class MixtralMoE(nn.Module):
    """Mixtral MoE module combining gating and expert computation."""
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        
        # Gate network - always replicated across ranks
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        
        # Fused MoE computation
        self.experts = FusedMoE(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=top_k,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Get router logits from gate network
        router_logits = self.gate(hidden_states)
        
        # Run MoE computation
        output = self.experts(hidden_states, router_logits)
        return output


class MixtralAttention(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 1000000,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class MixtralDecoderLayer(nn.Module):
    
    def __init__(
        self,
        config: MixtralConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = MixtralAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000),
        )
        
        self.block_sparse_moe = MixtralMoE(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            top_k=config.num_experts_per_tok,
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        
        # MoE block
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        
        return hidden_states, residual


class MixtralModel(nn.Module):
    
    def __init__(
        self,
        config: MixtralConfig,
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }
    
    # Expert weight mapping for MoE
    expert_params_mapping = []
    
    def __init__(
        self,
        config: MixtralConfig,
    ):
        super().__init__()
        self.config = config
        self.model = MixtralModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        
        # Generate expert weight mappings
        self._generate_expert_params_mapping()
    
    def _generate_expert_params_mapping(self):
        """Generate mapping for expert weights loading."""
        self.expert_params_mapping = []
        num_experts = self.config.num_local_experts
        
        for layer_idx in range(self.config.num_hidden_layers):
            layer_prefix = f"model.layers.{layer_idx}.block_sparse_moe"
            
            for expert_id in range(num_experts):
                # Map Mixtral weight names to our parameter names
                # In Mixtral: experts.{expert_id}.w1/w2/w3
                # In our model: block_sparse_moe.experts.w1/w2/w3 (with expert_id as index)
                mappings = [
                    (f"{layer_prefix}.experts.w1", f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w1", expert_id, "w1"),
                    (f"{layer_prefix}.experts.w2", f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w2", expert_id, "w2"),
                    (f"{layer_prefix}.experts.w3", f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}.w3", expert_id, "w3"),
                ]
                self.expert_params_mapping.extend(mappings)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits