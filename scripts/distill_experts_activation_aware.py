#!/usr/bin/env python3
import os, json, argparse, time
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--calib-path", type=str, required=True)
    p.add_argument("--rank", type=int, default=256)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--chunk", type=int, default=128,
                   help="Batch size for w1/w3 distillation. Larger = faster but more GPU memory. Default: 128")
    p.add_argument("--chunk-w2", type=int, default=32,
                   help="Batch size for w2 distillation. w2 is more complex, so smaller default. Default: 32")
    p.add_argument("--pca-oversample", type=int, default=32)
    p.add_argument("--pca-seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()

def get_expert_info(model_path: str) -> Dict[Tuple[int,int], Dict[str, Tuple[str,str]]]:
    """Map (layer, expert) -> {w1/w2/w3: (filename, tensor_name)}"""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]
    
    expert_info: Dict[Tuple[int, int], Dict[str, Tuple[str, str]]] = {}
    for weight_name, filename in weight_map.items():
        if "block_sparse_moe.experts" not in weight_name:
            continue
        parts = weight_name.split(".")
        layer = int(parts[2])
        expert = int(parts[5])
        wt = parts[6]  # w1/w2/w3
        key = (layer, expert)
        expert_info.setdefault(key, {})[wt] = (filename, weight_name)
    
    # sanity
    for (l,e), d in expert_info.items():
        for wt in ("w1","w2","w3"):
            if wt not in d:
                raise RuntimeError(f"missing {wt} for layer={l}, expert={e}")
    return expert_info

def load_tensor(model_path: str, fname: str, tname: str) -> torch.Tensor:
    path = os.path.join(model_path, fname)
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(tname)

@torch.no_grad()
def pca_exact_u(expert_info, model_path, layer_idx, weight_type, rank, device):
    # C = Σ Wt Wtᵀ, Wt = Wᵀ
    num_experts = max(e for (l,e) in expert_info.keys() if l==layer_idx) + 1
    W0 = load_tensor(model_path, *expert_info[(layer_idx,0)][weight_type])
    Wt0 = W0.T
    d_in, d_out = Wt0.shape
    dev = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    C = torch.zeros((d_in,d_in), dtype=torch.float32, device=dev)
    for e in range(num_experts):
        W = load_tensor(model_path, *expert_info[(layer_idx,e)][weight_type])
        Wt = W.T.contiguous().to(dev, dtype=torch.float32)
        C.add_(Wt @ Wt.T)
        del W, Wt
    evals, evecs = torch.linalg.eigh(C)
    U = evecs[:, -rank:].contiguous()
    U, _ = torch.linalg.qr(U, mode="reduced")
    return U.cpu()

@torch.no_grad()
def pca_sketched_u(expert_info, model_path, layer_idx, weight_type, rank, device, oversample=32, seed=0):
    # randomized 2-pass PCA on C = Σ Wt Wtᵀ
    num_experts = max(e for (l,e) in expert_info.keys() if l==layer_idx) + 1
    W0 = load_tensor(model_path, *expert_info[(layer_idx,0)][weight_type])
    Wt0 = W0.T
    d_in, d_out = Wt0.shape
    k = rank + oversample
    dev = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    g = torch.Generator(device=dev); g.manual_seed(seed + 1337*layer_idx)

    Y = torch.zeros((d_in,k), dtype=torch.float32, device=dev)
    for e in range(num_experts):
        W = load_tensor(model_path, *expert_info[(layer_idx,e)][weight_type])
        Wt = W.T.contiguous().to(dev, dtype=torch.float32)
        Omega = torch.randn((d_out,k), generator=g, device=dev, dtype=torch.float32)
        Y.add_(Wt @ Omega)
        del W, Wt, Omega
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    del Y
    Qt = Q.T.contiguous()
    T = torch.zeros((k,k), dtype=torch.float32, device=dev)
    for e in range(num_experts):
        W = load_tensor(model_path, *expert_info[(layer_idx,e)][weight_type])
        Wt = W.T.contiguous().to(dev, dtype=torch.float32)
        M = Qt @ Wt
        T.add_(M @ M.T)
        del W, Wt, M
    evals, evecs = torch.linalg.eigh(T)
    G = evecs[:, -rank:].contiguous()
    U = (Q @ G).contiguous()
    U, _ = torch.linalg.qr(U, mode="reduced")
    return U.cpu()

@torch.no_grad()
def solve_ridge_from_stream(A: torch.Tensor, B: torch.Tensor, ridge: float) -> torch.Tensor:
    # A: [r,r], B: [r,d]
    r = A.shape[0]
    lam = ridge * (torch.trace(A) / r).clamp_min(1e-12)
    A = A + lam * torch.eye(r, device=A.device, dtype=A.dtype)
    L = torch.linalg.cholesky(A)
    V = torch.cholesky_solve(B, L)
    return V

@torch.no_grad()
def distill_w1w3(X: torch.Tensor, W: torch.Tensor, U_dev: torch.Tensor,
                ridge: float, chunk: int, device: str) -> torch.Tensor:
    # X: [N,H] on any device (preferably GPU if available)
    dev = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    # X might already be on GPU, so only move if needed
    X = X.to(dtype=torch.float32)
    if X.device.type != dev:
        X = X.to(dev, non_blocking=True)
    Wt = W.T.contiguous().to(dev, dtype=torch.float32)       # [H, I]
    r = U_dev.shape[1]
    d_out = W.shape[0]
    A = torch.zeros((r,r), device=dev, dtype=torch.float32)
    B = torch.zeros((r,d_out), device=dev, dtype=torch.float32)
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i+chunk]  # Already on correct device
        Z = Xc @ U_dev                       # [c,r]
        Y = Xc @ Wt                          # [c,d_out]
        A.add_(Z.T @ Z)
        B.add_(Z.T @ Y)
        del Xc, Z, Y
    V = solve_ridge_from_stream(A, B, ridge)
    return V.cpu()

@torch.no_grad()
def distill_w2(X: torch.Tensor, W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor,
               U2_dev: torch.Tensor, ridge: float, chunk: int, device: str) -> torch.Tensor:
    # teacher: H = silu(XW1ᵀ) * (XW3ᵀ), Y2 = H W2ᵀ
    dev = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    # X might already be on GPU, so only move if needed
    X = X.to(dtype=torch.float32)
    if X.device.type != dev:
        X = X.to(dev, non_blocking=True)
    W1t = W1.T.contiguous().to(dev, dtype=torch.float32)   # [H,I]
    W3t = W3.T.contiguous().to(dev, dtype=torch.float32)   # [H,I]
    W2t = W2.T.contiguous().to(dev, dtype=torch.float32)   # [I,H]
    r = U2_dev.shape[1]
    d_out = W2.shape[0]   # H
    A = torch.zeros((r,r), device=dev, dtype=torch.float32)
    B = torch.zeros((r,d_out), device=dev, dtype=torch.float32)
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i+chunk]  # Already on correct device
        gate = Xc @ W1t                                    # [c,I]
        up   = Xc @ W3t                                    # [c,I]
        Hid  = F.silu(gate) * up                           # [c,I]
        Z = Hid @ U2_dev                                   # [c,r]
        Y = Hid @ W2t                                      # [c,H]
        A.add_(Z.T @ Z)
        B.add_(Z.T @ Y)
        del Xc, gate, up, Hid, Z, Y
    V2 = solve_ridge_from_stream(A, B, ridge)
    return V2.cpu()

def main():
    args = parse_args()
    model_path = args.model_path
    out_dir = args.output_dir or os.path.join(model_path, "svd_experts")
    os.makedirs(out_dir, exist_ok=True)
    v_dir = os.path.join(out_dir, "V_experts"); os.makedirs(v_dir, exist_ok=True)

    # Check if calibration file exists
    if not os.path.exists(args.calib_path):
        print(f"❌ Error: Calibration file not found: {args.calib_path}")
        print(f"\n💡 Solution: You need to collect calibration data first!")
        print(f"\n   Run this command to collect calibration data:")
        print(f"   CUDA_VISIBLE_DEVICES=1 uv run python scripts/collect_moe_calib.py \\")
        print(f"     --model-path {model_path} \\")
        print(f"     --out {args.calib_path} \\")
        print(f"     --cap-per-group 1024 \\")
        print(f"     --num-prompts 200 \\")
        print(f"     --max-new-tokens 64")
        print(f"\n   This will generate the calibration file needed for distillation.")
        return 1
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path not found: {model_path}")
        return 1

    # Load calibration data - keep on GPU if device is cuda (faster for computation)
    calib_device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[Distill] Loading calibration data from {args.calib_path}...")
    calib = torch.load(args.calib_path, map_location=calib_device)
    expert_info = get_expert_info(model_path)
    num_layers = max(l for (l,_) in expert_info.keys()) + 1
    num_experts = max(e for (_,e) in expert_info.keys()) + 1

    out_dtype = {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[args.dtype]
    dev = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    
    # Move calibration samples to GPU if using GPU (for faster computation)
    if dev == "cuda" and calib_device == "cpu":
        print("[Distill] Moving calibration samples to GPU for faster computation...")
        for key in calib["samples"]:
            if isinstance(calib["samples"][key], torch.Tensor):
                calib["samples"][key] = calib["samples"][key].to(dev)

    # compute U
    all_U = {}
    U_dev = {}  # per layer per wt
    for l in range(num_layers):
        for wt in ("w1","w2","w3"):
            if wt in ("w1","w3"):
                U = pca_exact_u(expert_info, model_path, l, wt, args.rank, args.device)
            else:
                U = pca_sketched_u(expert_info, model_path, l, wt, args.rank, args.device,
                                   oversample=args.pca_oversample, seed=args.pca_seed)
            # save U
            key = f"layer_{l}_{wt}_U"
            U_save = U.to(out_dtype).contiguous()
            all_U[key] = U_save
            # solve using saved U (cast back to fp32 for correctness)
            U_dev[(l,wt)] = U_save.to(torch.float32).to(dev)

    # distill V
    print(f"\n[Distill] Starting V distillation for {num_layers} layers × {num_experts} experts = {num_layers * num_experts} experts")
    print(f"  Device: {dev}")
    print(f"  Chunk sizes: w1/w3={args.chunk}, w2={args.chunk_w2}")
    
    distill_start = time.time()
    total_experts = num_layers * num_experts
    processed = 0
    
    errs = {"w1":[], "w2":[], "w3":[]}
    for l in range(num_layers):
        layer_start = time.time()
        for e in tqdm(range(num_experts), desc=f"Layer {l}/{num_layers-1}", leave=False):
            X = calib["samples"].get(f"l{l}_e{e}", None)
            if X is None or X.shape[0] < 32:
                # 极端情况：样本太少就跳过/报错
                raise RuntimeError(f"Not enough calib samples for l{l}_e{e}: {None if X is None else X.shape}")

            # load teacher weights (CPU) - 这是主要 I/O 瓶颈
            load_start = time.time()
            W1 = load_tensor(model_path, *expert_info[(l,e)]["w1"])
            W2 = load_tensor(model_path, *expert_info[(l,e)]["w2"])
            W3 = load_tensor(model_path, *expert_info[(l,e)]["w3"])
            load_time = time.time() - load_start

            # 蒸馏计算（GPU 加速）
            compute_start = time.time()
            V1 = distill_w1w3(X, W1, U_dev[(l,"w1")], args.ridge, args.chunk, args.device)
            V3 = distill_w1w3(X, W3, U_dev[(l,"w3")], args.ridge, args.chunk, args.device)
            V2 = distill_w2  (X, W1, W3, W2, U_dev[(l,"w2")], args.ridge, args.chunk_w2, args.device)
            compute_time = time.time() - compute_start

            # save expert V
            save_start = time.time()
            v_tensors = {
                "w1_V": V1.to(out_dtype).contiguous(),
                "w2_V": V2.to(out_dtype).contiguous(),
                "w3_V": V3.to(out_dtype).contiguous(),
            }
            save_file(v_tensors, os.path.join(v_dir, f"layer_{l}_expert_{e}.safetensors"))
            save_time = time.time() - save_start
            
            processed += 1
            # 每处理 32 个 expert 打印一次进度
            if processed % 32 == 0:
                elapsed = time.time() - distill_start
                rate = processed / elapsed
                eta = (total_experts - processed) / rate if rate > 0 else 0
                print(f"  Progress: {processed}/{total_experts} experts ({processed*100//total_experts}%), "
                      f"Rate: {rate:.1f} experts/s, ETA: {eta:.0f}s")
        
        layer_time = time.time() - layer_start
        print(f"  Layer {l} completed in {layer_time:.1f}s ({num_experts/layer_time:.1f} experts/s)")

    # save U
    save_file(all_U, os.path.join(out_dir, "U_matrices.safetensors"))

    # metadata
    meta = {
        "rank": args.rank,
        "dtype": args.dtype,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "ridge": args.ridge,
        "activation_aware": True,
        "calib_path": os.path.abspath(args.calib_path),
        "cap_per_group": int(calib.get("cap_per_group", -1)),
        "total_added": int(calib.get("total_added", -1)),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - distill_start
    print(f"\n[Distill] Distillation complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Average: {total_time/total_experts:.2f}s per expert")
    print(f"  Throughput: {total_experts/total_time:.2f} experts/s")
    print(f"  Output saved to: {out_dir}")
    
    # 性能建议
    if total_time > 600:  # 超过 10 分钟
        print(f"\n  💡 Performance tips:")
        if args.device == "cpu":
            print(f"     - Use --device cuda for 5-10x speedup")
        if args.chunk < 128:
            print(f"     - Increase --chunk (current: {args.chunk}) to reduce overhead")
        if args.chunk_w2 < 32:
            print(f"     - Increase --chunk-w2 (current: {args.chunk_w2}) if GPU memory allows")

if __name__ == "__main__":
    import sys
    sys.exit(main())

