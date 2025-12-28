# nanovllm/utils/moe_calib.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import random
import torch

_GLOBAL_COLLECTOR: Optional["MoECalibCollector"] = None

def set_global_collector(c: Optional["MoECalibCollector"]) -> None:
    global _GLOBAL_COLLECTOR
    _GLOBAL_COLLECTOR = c

def get_global_collector() -> Optional["MoECalibCollector"]:
    return _GLOBAL_COLLECTOR

class Reservoir:
    def __init__(self, capacity: int, dim: int, dtype: torch.dtype = torch.float16) -> None:
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.dtype = dtype
        self.n_seen = 0
        self.buf = torch.empty((self.capacity, self.dim), dtype=self.dtype, device="cpu")

    @torch.no_grad()
    def add(self, x_cpu: torch.Tensor) -> None:
        # x_cpu: [m, dim] on CPU
        if x_cpu.numel() == 0:
            return
        assert x_cpu.device.type == "cpu"
        x_cpu = x_cpu.to(dtype=self.dtype)
        m = x_cpu.shape[0]
        for i in range(m):
            self.n_seen += 1
            if self.n_seen <= self.capacity:
                self.buf[self.n_seen - 1].copy_(x_cpu[i])
            else:
                j = random.randint(1, self.n_seen)
                if j <= self.capacity:
                    self.buf[j - 1].copy_(x_cpu[i])

    def get(self) -> torch.Tensor:
        n = min(self.n_seen, self.capacity)
        return self.buf[:n].contiguous()

@dataclass
class MoECalibStats:
    cap_per_group: int
    total_added: int = 0

class MoECalibCollector:
    """
    Collect per-(layer, expert) inputs X for Mixtral MoE.
    We store X on CPU fp16 using reservoir sampling.
    """
    def __init__(self, num_layers: int, num_experts: int, hidden_size: int,
                 cap_per_group: int = 1024, seed: int = 0) -> None:
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.stats = MoECalibStats(cap_per_group=int(cap_per_group))
        random.seed(seed)

        self._res: Dict[Tuple[int, int], Reservoir] = {}
        for l in range(self.num_layers):
            for e in range(self.num_experts):
                self._res[(l, e)] = Reservoir(capacity=cap_per_group, dim=self.hidden_size)

    @torch.no_grad()
    def observe(self, layer_idx: int, hidden_states: torch.Tensor, topk_ids: torch.Tensor) -> None:
        """
        hidden_states: [T, H] or [B,S,H] on GPU/CPU
        topk_ids:      [T, K] or [B,S,K] on GPU/CPU
        """
        if hidden_states.numel() == 0:
            return
        # flatten tokens
        hs = hidden_states.detach()
        ids = topk_ids.detach()
        hs = hs.reshape(-1, hs.shape[-1])
        ids = ids.reshape(-1, ids.shape[-1])

        # move to CPU once
        hs_cpu = hs.to("cpu", dtype=torch.float16, non_blocking=True)
        ids_cpu = ids.to("cpu", non_blocking=True)

        K = ids_cpu.shape[-1]
        for k in range(K):
            col = ids_cpu[:, k]
            uniq = torch.unique(col)
            for e in uniq.tolist():
                if not (0 <= e < self.num_experts):
                    continue
                mask = (col == e)
                if mask.any():
                    self._res[(int(layer_idx), int(e))].add(hs_cpu[mask])
                    self.stats.total_added += int(mask.sum().item())

    def export(self) -> dict:
        out = {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "hidden_size": self.hidden_size,
            "cap_per_group": self.stats.cap_per_group,
            "total_added": self.stats.total_added,
            "samples": {},
        }
        for (l, e), r in self._res.items():
            out["samples"][f"l{l}_e{e}"] = r.get()
        return out

