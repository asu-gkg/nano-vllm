#!/usr/bin/env python3
"""
SVD / Shared-U Expert Decomposition Script (Mixtral)

We factor each expert weight (used in forward as W.T) into:
    W.T ≈ U_shared @ V_expert
so that:
    x @ W.T ≈ (x @ U_shared) @ V_expert

Outputs:
    {output_dir}/
    ├── U_matrices.safetensors          # shared U per (layer, w1/w2/w3)
    └── V_experts/
        ├── layer_0_expert_0.safetensors
        ├── layer_0_expert_1.safetensors
        └── ...

Method:
  Uses PCA to compute a real shared subspace U_shared across experts, then computes
  V_e := U_shared^T @ W_e.T for each expert (LS optimal when U is orthonormal).
  - For w1/w3: exact PCA via covariance C=Σ Wt Wt^T and eigh (d_in=hidden).
  - For w2: sketched PCA (randomized) to avoid full eig on large d_in.

Recommended:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/decompose_experts.py --model-path ./Mixtral-8x7B-v0.1 --rank 256
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file


# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Shared-U decomposition for Mixtral experts")
    p.add_argument("--model-path", type=str, required=True, help="Path to Mixtral model directory")
    p.add_argument("--rank", type=int, default=256, help="Rank r (default: 256)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: {model_path}/svd_experts)")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                   help="Output dtype for saved U/V (default: float16)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="Device for heavy computation (default: cuda)")
    # PCA sketch params (used for w2)
    p.add_argument("--pca-oversample", type=int, default=32,
                   help="Oversampling for sketched PCA (default: 32)")
    p.add_argument("--pca-seed", type=int, default=0,
                   help="Random seed for sketched PCA (default: 0)")
    return p.parse_args()


def get_expert_weight_files(model_path: str) -> Dict[Tuple[int, int], Dict[str, Tuple[str, str]]]:
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
        # Example:
        # model.layers.{layer}.block_sparse_moe.experts.{expert}.{w1/w2/w3}.weight
        parts = weight_name.split(".")
        layer_idx = int(parts[2])
        expert_idx = int(parts[5])
        weight_type = parts[6]  # w1/w2/w3

        key = (layer_idx, expert_idx)
        if key not in expert_info:
            expert_info[key] = {}
        expert_info[key][weight_type] = (filename, weight_name)

    # Basic validation: ensure each expert has w1/w2/w3
    for k, m in expert_info.items():
        for wt in ("w1", "w2", "w3"):
            if wt not in m:
                raise RuntimeError(f"Missing {wt} for layer/expert {k}. Found keys={list(m.keys())}")

    return expert_info


def load_expert_weight(model_path: str, filename: str, weight_name: str) -> torch.Tensor:
    """Load a single tensor to CPU"""
    file_path = os.path.join(model_path, filename)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        return f.get_tensor(weight_name)


@torch.no_grad()
def compute_shared_u_pca_exact_cov_eigh(
    expert_info: Dict[Tuple[int, int], Dict[str, Tuple[str, str]]],
    layer_idx: int,
    weight_type: str,
    rank: int,
    model_path: str,
    device: str,
) -> torch.Tensor:
    """
    Exact PCA for w1/w3 (d_in=hidden, moderate).
    C = Σ_e Wt_e @ Wt_e.T  (d_in x d_in), then take top-r eigenvectors.
    Return U_shared: [d_in, rank] CPU float32.
    """
    # Determine num_experts in this layer
    num_experts = max(e for (l, e) in expert_info.keys() if l == layer_idx) + 1

    # Infer shape
    filename0, weight_name0 = expert_info[(layer_idx, 0)][weight_type]
    W0 = load_expert_weight(model_path, filename0, weight_name0)
    Wt0 = W0.T
    d_in, d_out = Wt0.shape

    if rank > min(d_in, d_out):
        raise ValueError(f"rank={rank} > min(d_in,d_out)={min(d_in,d_out)} for {weight_type} shape {Wt0.shape}")

    # Accumulate on GPU if possible (C is d_in x d_in)
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    accum_device = "cuda" if use_cuda else "cpu"

    print(f"    [PCA-exact] {weight_type}: Wt shape={tuple(Wt0.shape)}, C shape=({d_in},{d_in}) on {accum_device}")
    C = torch.zeros((d_in, d_in), dtype=torch.float32, device=accum_device)

    for expert_idx in tqdm(range(num_experts), desc=f"      Accum C ({weight_type})", leave=False):
        filename, weight_name = expert_info[(layer_idx, expert_idx)][weight_type]
        W = load_expert_weight(model_path, filename, weight_name)
        Wt = W.T.contiguous()
        Wt_dev = Wt.to(accum_device, dtype=torch.float32)
        # C += Wt @ Wt.T
        C.add_(Wt_dev @ Wt_dev.T)

        # free
        del W, Wt, Wt_dev

    # Eigendecomposition
    # eigh returns ascending eigenvalues
    evals, evecs = torch.linalg.eigh(C)
    U = evecs[:, -rank:].contiguous()  # top-r (still orthonormal)
    # Optional: re-orthonormalize for safety
    U, _ = torch.linalg.qr(U, mode="reduced")

    U_cpu = U.cpu()

    del C, evals, evecs, U
    if use_cuda:
        torch.cuda.empty_cache()

    return U_cpu


@torch.no_grad()
def compute_shared_u_pca_sketched(
    expert_info: Dict[Tuple[int, int], Dict[str, Tuple[str, str]]],
    layer_idx: int,
    weight_type: str,
    rank: int,
    model_path: str,
    device: str,
    oversample: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """
    Sketched (randomized) PCA to avoid full eig on large d_in (e.g., w2).
    We treat A = [Wt_0, Wt_1, ..., Wt_{E-1}] concatenated by columns.
    Goal: top-r eigenvectors of C = A A^T = Σ Wt_e Wt_e^T.

    Algorithm (2-pass):
      k = rank + oversample
      1) Y = Σ Wt_e @ Ω_e,  Ω_e ~ N(0,1) (d_out x k)
         Q = orth(Y)
      2) T = Σ (Q^T Wt_e)(Q^T Wt_e)^T  (k x k)
         eig(T) -> G (k x rank)
         U = Q @ G

    Return U on CPU float32.
    """
    num_experts = max(e for (l, e) in expert_info.keys() if l == layer_idx) + 1

    filename0, weight_name0 = expert_info[(layer_idx, 0)][weight_type]
    W0 = load_expert_weight(model_path, filename0, weight_name0)
    Wt0 = W0.T
    d_in, d_out = Wt0.shape
    k = rank + max(0, oversample)

    if rank > min(d_in, d_out):
        raise ValueError(f"rank={rank} > min(d_in,d_out)={min(d_in,d_out)} for {weight_type} shape {Wt0.shape}")

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = "cuda" if use_cuda else "cpu"

    print(f"    [PCA-sketched] {weight_type}: Wt shape={tuple(Wt0.shape)}, k={k} on {dev}")

    g = torch.Generator(device=dev)
    g.manual_seed(seed + 1315423911 * (layer_idx + 1) + (7 if weight_type == "w2" else 3))

    # Pass 1: build Y
    Y = torch.zeros((d_in, k), dtype=torch.float32, device=dev)
    for expert_idx in tqdm(range(num_experts), desc=f"      Pass1 Y ({weight_type})", leave=False):
        filename, weight_name = expert_info[(layer_idx, expert_idx)][weight_type]
        W = load_expert_weight(model_path, filename, weight_name)
        Wt = W.T.contiguous().to(dev, dtype=torch.float32)
        Omega = torch.randn((d_out, k), generator=g, device=dev, dtype=torch.float32)
        Y.add_(Wt @ Omega)
        del W, Wt, Omega

    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (d_in x k)
    del Y

    # Pass 2: build T = Σ (Q^T Wt)(Q^T Wt)^T
    T = torch.zeros((k, k), dtype=torch.float32, device=dev)
    Qt = Q.transpose(0, 1).contiguous()  # (k x d_in)

    for expert_idx in tqdm(range(num_experts), desc=f"      Pass2 T ({weight_type})", leave=False):
        filename, weight_name = expert_info[(layer_idx, expert_idx)][weight_type]
        W = load_expert_weight(model_path, filename, weight_name)
        Wt = W.T.contiguous().to(dev, dtype=torch.float32)
        M = Qt @ Wt                # (k x d_out)
        T.add_(M @ M.transpose(0, 1))  # (k x k)
        del W, Wt, M

    evals, evecs = torch.linalg.eigh(T)        # ascending
    G = evecs[:, -rank:].contiguous()          # (k x rank)
    U = (Q @ G).contiguous()                   # (d_in x rank)
    U, _ = torch.linalg.qr(U, mode="reduced")  # re-orthonormalize

    U_cpu = U.cpu()

    del Q, Qt, T, evals, evecs, G, U
    if use_cuda:
        torch.cuda.empty_cache()

    return U_cpu


@torch.no_grad()
def compute_v_and_error(
    W: torch.Tensor,
    U_dev: torch.Tensor,      # (d_in x r) float32 on device
    device: str,
) -> Tuple[torch.Tensor, float]:
    """
    Given W ([out,in] on CPU), compute:
        Wt = W.T on device (float32)
        V = U^T @ Wt  (r x d_out)
        err = ||Wt - U V|| / ||Wt||
    Return (V_cpu_float32, err_float).
    """
    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    Wt = W.T.contiguous().to(dev, dtype=torch.float32)

    V = U_dev.transpose(0, 1) @ Wt  # (r x d_out)

    # reconstruction error on device
    W_hat = U_dev @ V
    num = torch.linalg.norm(Wt - W_hat)
    den = torch.linalg.norm(Wt)
    err = (num / (den + 1e-12)).item()

    V_cpu = V.cpu()

    del Wt, V, W_hat
    if dev == "cuda":
        torch.cuda.empty_cache()

    return V_cpu, err


def main():
    args = parse_args()

    model_path = args.model_path
    rank = args.rank
    output_dir = args.output_dir or os.path.join(model_path, "svd_experts")
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    out_dtype = dtype_map[args.dtype]

    print("=" * 70)
    print("Shared-U Expert Decomposition (Mixtral)")
    print("=" * 70)
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Rank: {rank}")
    print(f"Compute device: {device}")
    print(f"Save dtype: {args.dtype}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory/1024**3:.1f} GB")
    print()

    os.makedirs(output_dir, exist_ok=True)
    v_dir = os.path.join(output_dir, "V_experts")
    os.makedirs(v_dir, exist_ok=True)

    print("Loading expert weight map...")
    expert_info = get_expert_weight_files(model_path)
    num_layers = max(l for (l, _) in expert_info.keys()) + 1
    num_experts = max(e for (_, e) in expert_info.keys()) + 1
    total_experts = num_layers * num_experts
    print(f"Found layers={num_layers}, experts/layer={num_experts}, total experts={total_experts}")
    print()

    # Save U tensors
    all_U_save: Dict[str, torch.Tensor] = {}
    # Error stats
    errors: Dict[str, list] = {"w1": [], "w2": [], "w3": []}

    for layer_idx in range(num_layers):
        print(f"\n{'='*70}")
        print(f"Layer {layer_idx}/{num_layers-1}")
        print(f"{'='*70}")

        # 1) Compute shared U for this layer (fp32 CPU + cached fp32 device)
        U_cpu_fp32: Dict[str, torch.Tensor] = {}
        U_dev_fp32: Dict[str, torch.Tensor] = {}

        for wt in ("w1", "w2", "w3"):
            if wt in ("w1", "w3"):
                print(f"  [U] {wt}: exact PCA (cov + eigh)")
                U = compute_shared_u_pca_exact_cov_eigh(expert_info, layer_idx, wt, rank, model_path, device)
            else:
                # wt == w2
                print(f"  [U] {wt}: sketched PCA (avoid full eig)")
                U = compute_shared_u_pca_sketched(
                    expert_info, layer_idx, wt, rank, model_path, device,
                    oversample=args.pca_oversample, seed=args.pca_seed
                )
                        
            # Keep fp32 CPU for correctness
            U_cpu_fp32[wt] = U.contiguous().to(torch.float32)

            # Save casted version
            u_key = f"layer_{layer_idx}_{wt}_U"
            all_U_save[u_key] = U_cpu_fp32[wt].to(out_dtype).contiguous()

            # Cache device fp32 for V computation
            dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
            U_dev_fp32[wt] = U_cpu_fp32[wt].to(dev, dtype=torch.float32)

        # 2) For each expert, compute V in the shared coordinates (and error)
        pbar = tqdm(range(num_experts), desc="  Computing V & saving experts")
        for expert_idx in pbar:
            key = (layer_idx, expert_idx)
            V_tensors: Dict[str, torch.Tensor] = {}

            for wt in ("w1", "w2", "w3"):
                filename, weight_name = expert_info[key][wt]
                W = load_expert_weight(model_path, filename, weight_name)  # CPU
                V_cpu, err = compute_v_and_error(W, U_dev_fp32[wt], device)
                errors[wt].append(err)

                V_tensors[f"{wt}_V"] = V_cpu.to(out_dtype).contiguous()
                del W, V_cpu

            v_file = os.path.join(v_dir, f"layer_{layer_idx}_expert_{expert_idx}.safetensors")
            save_file(V_tensors, v_file)

        # cleanup per-layer cache
        del U_cpu_fp32, U_dev_fp32
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save all U
    print("\nSaving shared U matrices...")
    u_file = os.path.join(output_dir, "U_matrices.safetensors")
    save_file(all_U_save, u_file)

    # Save metadata
    metadata = {
        "rank": rank,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "dtype": args.dtype,
        "model_path": os.path.basename(model_path),
        "pca_oversample": int(args.pca_oversample),
        "pca_seed": int(args.pca_seed),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Print stats
    print("\n" + "=" * 70)
    print("Decomposition Complete!")
    print("=" * 70)

    print("\nReconstruction Error (relative Frobenius) statistics:")
    for wt in ("w1", "w2", "w3"):
        arr = np.asarray(errors[wt], dtype=np.float64)
        print(f"  {wt}: mean={arr.mean():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

    # File sizes
    u_size = os.path.getsize(u_file) / 1024 / 1024
    v_total_size = 0.0
    for fn in os.listdir(v_dir):
        if fn.endswith(".safetensors"):
            v_total_size += os.path.getsize(os.path.join(v_dir, fn)) / 1024 / 1024

    print("\nFile sizes:")
    print(f"  U_matrices: {u_size:.1f} MB")
    print(f"  V_experts total: {v_total_size:.1f} MB")
    print(f"  V per expert (avg): {v_total_size / (num_layers * num_experts):.1f} MB")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
