"""Batched AR sampling from a trained Ising transformer, with importance weights.

Saves the model samples plus the log-weight log w = -beta_c * E - log p_sym
that ESS / log Z / KL estimators consume.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.ising.energy import BETA_C, energy_torch
from src.model.transformer import IsingTransformer
from src.train import pick_device


@torch.no_grad()
def sample_and_score(
    model: IsingTransformer,
    n_total: int,
    batch_size: int,
    device: torch.device,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw n_total samples from p_sym and return (samples, log_p_sym, log_w).

    log_w[i] = -beta_c * E(s_i) - log p_sym(s_i) is the unnormalized IS log weight
    against the exact Boltzmann distribution at T_c.
    """
    model.eval()
    samples_all = []
    logp_all = []
    logw_all = []
    done = 0
    while done < n_total:
        B = min(batch_size, n_total - done)
        s_pm = model.sample(M=B, device=device, symmetrize=True)  # (B, N, N) int8
        # log p_sym(s) under the model
        s_float = s_pm.to(torch.float32)
        logp_sym = model.log_prob_sym(s_float)  # (B,)
        E = energy_torch(s_float)  # (B,)
        logw = -BETA_C * E - logp_sym
        samples_all.append(s_pm.cpu().numpy().astype(np.int8))
        logp_all.append(logp_sym.cpu().numpy())
        logw_all.append(logw.cpu().numpy())
        done += B
        if progress:
            print(f"  sampled {done}/{n_total}", flush=True)
    return (
        np.concatenate(samples_all, axis=0),
        np.concatenate(logp_all, axis=0),
        np.concatenate(logw_all, axis=0),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--n_samples", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"device = {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = IsingTransformer(
        N=cfg["N"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff_mult=cfg.get("d_ff_mult", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt} (step {ckpt.get('step', '?')}, N={cfg['N']})")

    t0 = time.time()
    samples, logp_sym, logw = sample_and_score(model, args.n_samples, args.batch_size, device)
    dt = time.time() - t0
    print(f"drew {args.n_samples} samples in {dt:.1f}s ({dt/args.n_samples*1000:.2f} ms/sample)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"model_samples_N{cfg['N']}"
    np.save(out_dir / f"{base}.npy", samples)
    np.save(out_dir / f"{base}.logp.npy", logp_sym)
    np.save(out_dir / f"{base}.logw.npy", logw)
    print(f"saved {out_dir / base}.{{npy, logp.npy, logw.npy}}")


if __name__ == "__main__":
    main()
