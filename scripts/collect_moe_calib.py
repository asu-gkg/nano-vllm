#!/usr/bin/env python3
import os, time, argparse, torch
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

from nanovllm.utils.moe_calib import MoECalibCollector, set_global_collector

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect calibration activations for MoE expert distillation. "
                    "More diverse prompts and more samples lead to better distillation quality."
    )
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--out", type=str, default="calib_mixtral.pt")
    p.add_argument("--cap-per-group", type=int, default=1024,
                   help="Max samples per (layer, expert) group. Higher = better quality but more memory.")
    p.add_argument("--num-prompts", type=int, default=200,
                   help="Number of prompts to generate. Recommended: 200-500 for good coverage.")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Max tokens to generate per prompt. Higher = more samples but slower.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature. Higher = more diverse outputs.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 强制 teacher：禁用 SVD expert manager
    os.environ["NANOVLLM_DISABLE_SVD"] = "1"

    # 构造 runner
    config = Config(
        model=args.model_path,
        tensor_parallel_size=1,
        max_num_batched_tokens=2048,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 创建 collector（从 hf_config 拿层数/专家数/hidden）
    num_layers = config.hf_config.num_hidden_layers
    num_experts = config.hf_config.num_local_experts
    hidden_size = config.hf_config.hidden_size
    collector = MoECalibCollector(
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        cap_per_group=args.cap_per_group,
        seed=args.seed,
    )
    set_global_collector(collector)

    print(f"[Calib] layers={num_layers}, experts={num_experts}, hidden={hidden_size}, cap={args.cap_per_group}")
    runner = ModelRunner(config, rank=0, event=None)

    # prompts：多样化的 prompts 覆盖不同领域和任务类型
    # 增加多样性有助于收集更全面的激活数据，提高蒸馏质量
    base_prompts = [
        # 科技与AI
        "The key to artificial intelligence is",
        "In the future, robots will",
        "Machine learning algorithms can",
        "The development of quantum computing",
        "Neural networks are designed to",
        
        # 科学解释
        "Explain why the sky is blue in one paragraph:",
        "Describe how photosynthesis works:",
        "What causes earthquakes?",
        "How do vaccines work in the human body?",
        "Explain the theory of relativity:",
        
        # 创意写作
        "Write a short story about a lost astronaut:",
        "Once upon a time, in a distant galaxy",
        "The mysterious door opened to reveal",
        "In a world where time travel exists",
        "The last library on Earth contained",
        
        # 教育与学习
        "Give three reasons to learn computer architecture:",
        "Summarize the importance of matrix multiplication:",
        "What are the key principles of economics?",
        "Explain the water cycle in simple terms:",
        "How does the human brain process information?",
        
        # 历史与文化
        "The Renaissance period was significant because",
        "Ancient civilizations developed writing systems to",
        "The Industrial Revolution changed society by",
        "Cultural traditions are important because",
        "Historical events shape our present by",
        
        # 哲学与思考
        "The meaning of life might be",
        "Free will versus determinism:",
        "What is the nature of consciousness?",
        "Ethics in artificial intelligence involves",
        "The relationship between knowledge and wisdom",
        
        # 日常生活
        "How to make a perfect cup of coffee:",
        "The benefits of regular exercise include",
        "Cooking is both an art and a science because",
        "Traveling broadens the mind by",
        "Effective communication requires",
        
        # 技术问题
        "Debugging code requires",
        "The difference between CPU and GPU is",
        "Cloud computing offers advantages such as",
        "Cybersecurity is crucial because",
        "Software engineering best practices include",
        
        # 商业与经济
        "Startup companies face challenges like",
        "Globalization affects economies by",
        "Sustainable business practices involve",
        "The stock market works by",
        "Marketing strategies should focus on",
        
        # 环境与自然
        "Climate change impacts ecosystems through",
        "Renewable energy sources include",
        "Biodiversity is important because",
        "Ocean currents affect weather by",
        "Forest conservation helps by",
    ]
    
    # 如果需要的 prompts 数量少于 base_prompts，随机选择；否则循环使用
    import random
    if args.num_prompts <= len(base_prompts):
        # 随机选择不重复的 prompts
        prompts = random.sample(base_prompts, args.num_prompts)
    else:
        # 先使用所有 base_prompts，然后随机选择补充
        prompts = base_prompts.copy()
        while len(prompts) < args.num_prompts:
            prompts.append(random.choice(base_prompts))
    
    print(f"[Calib] Using {len(prompts)} prompts ({len(set(prompts))} unique templates)")

    t0 = time.time()
    with torch.inference_mode():
        for idx, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
            sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
            seq = Sequence(input_ids, sp)
            # block_table 简化处理（参照 example_mixtral.py）
            seq.block_table = list(range(seq.num_blocks))

            # prefill
            token_ids = runner.run([seq], is_prefill=True)
            if token_ids and token_ids[0] is not None:
                seq.append_token(token_ids[0])

            # decode
            for _ in range(args.max_new_tokens - 1):
                if token_ids and token_ids[0] == tokenizer.eos_token_id:
                    break
                if seq.num_blocks > len(seq.block_table):
                    need = seq.num_blocks - len(seq.block_table)
                    last = seq.block_table[-1] if seq.block_table else -1
                    seq.block_table.extend(range(last + 1, last + 1 + need))
                token_ids = runner.run([seq], is_prefill=False)
                if token_ids and token_ids[0] is not None:
                    seq.append_token(token_ids[0])
                else:
                    break

            if (idx + 1) % 20 == 0:
                print(f"[Calib] {idx+1}/{len(prompts)} prompts, total_added={collector.stats.total_added}")

    calib = collector.export()
    torch.save(calib, args.out)
    dt = time.time() - t0
    
    # 统计信息
    samples = calib["samples"]
    total_samples = sum(s.shape[0] if isinstance(s, torch.Tensor) else 0 for s in samples.values())
    samples_per_expert = total_samples / (num_layers * num_experts) if num_layers * num_experts > 0 else 0
    
    print(f"\n[Calib] Collection complete!")
    print(f"  Total prompts processed: {len(prompts)}")
    print(f"  Total tokens collected: {calib['total_added']}")
    print(f"  Samples per (layer, expert): {samples_per_expert:.1f} (max: {args.cap_per_group})")
    print(f"  Saved to: {args.out}")
    print(f"  Time: {dt:.1f}s")
    
    # 检查是否有专家样本不足
    min_samples = min(
        (s.shape[0] if isinstance(s, torch.Tensor) else 0) 
        for s in samples.values()
    )
    if min_samples < 32:
        print(f"\n  ⚠️  Warning: Some experts have < 32 samples (min: {min_samples})")
        print(f"     Consider increasing --num-prompts or --max-new-tokens")
    elif min_samples < args.cap_per_group * 0.5:
        print(f"\n  ℹ️  Info: Some experts have < 50% capacity (min: {min_samples}/{args.cap_per_group})")
        print(f"     For better quality, consider increasing --num-prompts")
    else:
        print(f"\n  ✓ Good coverage: All experts have sufficient samples (min: {min_samples})")

    # 关闭 collector，避免影响后续
    set_global_collector(None)
    runner.exit()

if __name__ == "__main__":
    main()

