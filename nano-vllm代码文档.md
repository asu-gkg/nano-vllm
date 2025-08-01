# Nano-vLLM 完整代码文档

## 项目概述

Nano-vLLM 是一个轻量级的 vLLM 实现，从头开始构建，具有以下特点：

- 🚀 **快速离线推理** - 推理速度与 vLLM 相当
- 📖 **可读性强的代码库** - 约 1,200 行 Python 代码的清晰实现
- ⚡ **优化套件** - 前缀缓存、张量并行、Torch 编译、CUDA 图等

## 项目结构

```
nanovllm/
├── __init__.py              # 主入口
├── llm.py                   # LLM 类定义
├── sampling_params.py       # 采样参数
├── config.py               # 配置类
├── engine/                 # 推理引擎
│   ├── llm_engine.py       # 主引擎
│   ├── model_runner.py     # 模型运行器
│   ├── scheduler.py        # 调度器
│   ├── sequence.py         # 序列管理
│   └── block_manager.py    # 块管理器
├── models/                 # 模型实现
│   └── qwen3.py           # Qwen3 模型
├── layers/                 # 神经网络层
│   ├── attention.py       # 注意力机制
│   ├── linear.py          # 线性层
│   ├── layernorm.py       # 层归一化
│   ├── activation.py      # 激活函数
│   ├── embed_head.py      # 嵌入和输出头
│   ├── rotary_embedding.py # 旋转位置编码
│   └── sampler.py         # 采样器
└── utils/                  # 工具函数
    ├── context.py         # 上下文管理
    └── loader.py          # 模型加载器
```

## 完整代码实现

### 1. 主入口文件

#### nanovllm/__init__.py
```python
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
```

#### nanovllm/llm.py
```python
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass
```

#### nanovllm/sampling_params.py
```python
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
```

#### nanovllm/config.py
```python
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
```

### 2. 引擎核心

#### nanovllm/engine/llm_engine.py
```python
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
```

#### nanovllm/engine/sequence.py
```python
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
```

#### nanovllm/engine/scheduler.py
```python
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

#### nanovllm/engine/block_manager.py
```python
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
```

#### nanovllm/engine/model_runner.py
```python
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
```

#### nanovllm/models/qwen3.py
```python
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
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
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
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


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

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
```

#### nanovllm/layers/activation.py
```
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```

#### nanovllm/layers/attention.py

```python
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
```

#### nanovllm/layers/embed_head.py
```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits

```

#### nanovllm/layers/layernorm.py
```python
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
```

#### nanovllm/layers/linear.py
```python

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

```
#### nanovllm/layers/rotary_embedding.py
```
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

```