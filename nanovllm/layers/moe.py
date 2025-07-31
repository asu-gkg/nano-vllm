import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.activation import SiluAndMul


class FusedMoE(nn.Module):
    """Fused Mixture of Experts layer for efficient expert computation.
    
    This implementation shards expert weights across tensor parallel ranks
    and performs efficient batched computation of expert outputs.
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        
        # Expert weights - each expert has gate_proj, up_proj, down_proj
        # We shard experts across TP ranks
        self.experts_per_rank = num_experts // self.tp_size
        assert num_experts % self.tp_size == 0, f"num_experts {num_experts} must be divisible by tp_size {self.tp_size}"
        
        # Create expert weights for this rank
        self.w1 = nn.Parameter(torch.empty(self.experts_per_rank, intermediate_size, hidden_size))  # gate_proj
        self.w2 = nn.Parameter(torch.empty(self.experts_per_rank, hidden_size, intermediate_size))  # down_proj
        self.w3 = nn.Parameter(torch.empty(self.experts_per_rank, intermediate_size, hidden_size))  # up_proj
        
        # Register weight loaders
        self.w1.weight_loader = self.expert_weight_loader
        self.w2.weight_loader = self.expert_weight_loader
        self.w3.weight_loader = self.expert_weight_loader
        
        self.act_fn = SiluAndMul()
    
    def expert_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, 
                        expert_id: int, shard_name: str):
        """Load expert weights for the experts assigned to this rank."""
        # Determine which experts belong to this rank
        expert_rank = expert_id // self.experts_per_rank
        if expert_rank != self.tp_rank:
            return
        
        # Local expert index on this rank
        local_expert_idx = expert_id % self.experts_per_rank
        param.data[local_expert_idx].copy_(loaded_weight)
    
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_tokens, hidden_size]
            router_logits: [num_tokens, num_experts]
        
        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.size(0)
        
        # Select top-k experts for each token
        topk_weights, topk_ids = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        # Initialize output tensor
        output = torch.zeros_like(hidden_states)
        
        # Group tokens by expert for efficient batch processing
        for expert_id in range(self.experts_per_rank):
            # Get global expert ID for this local expert
            global_expert_id = self.tp_rank * self.experts_per_rank + expert_id
            
            # Find all tokens and their positions that selected this expert
            expert_mask = (topk_ids == global_expert_id)
            token_indices, expert_positions = expert_mask.nonzero(as_tuple=True)
            
            if len(token_indices) == 0:
                continue
            
            # Get input tokens for this expert
            expert_input = hidden_states[token_indices]
            
            # Expert computation: gate_proj and up_proj -> act_fn -> down_proj
            gate = F.linear(expert_input, self.w1[expert_id])
            up = F.linear(expert_input, self.w3[expert_id])
            gate_up = torch.cat([gate, up], dim=-1)
            activated = self.act_fn(gate_up)
            expert_output = F.linear(activated, self.w2[expert_id])
            
            # Get weights for these tokens
            expert_weights = topk_weights[token_indices, expert_positions]
            
            # Accumulate weighted expert outputs
            output.index_add_(0, token_indices, expert_weights.unsqueeze(-1) * expert_output)
        
        # All-reduce across all ranks to combine expert outputs
        if self.tp_size > 1:
            dist.all_reduce(output)
        
        return output