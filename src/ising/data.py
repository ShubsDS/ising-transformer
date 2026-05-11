"""Generate and cache a ground-truth Wolff sample bank at T_c."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .energy import BETA_C
from .mcmc import wolff_sample
from .observables import derived_observables, primary_moments


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--n_samples", type=int, default=100_000)
    p.add_argument("--sweeps_per_sample", type=int, default=5)
    p.add_argument("--n_therm", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n_chains", type=int, default=4, help="independent chains, concatenated")
    args = p.parse_args()

    out = Path(args.out) if args.out else Path(f"data/ising_N{args.N}_Tc.npy")
    out.parent.mkdir(parents=True, exist_ok=True)

    per_chain = args.n_samples // args.n_chains
    all_samples = []
    all_cluster = []
    t0 = time.time()
    for c in range(args.n_chains):
        s, cs = wolff_sample(
            N=args.N,
            beta=BETA_C,
            n_samples=per_chain,
            sweeps_per_sample=args.sweeps_per_sample,
            n_therm=args.n_therm,
            seed=args.seed + c,
        )
        all_samples.append(s)
        all_cluster.append(cs)
        print(f"  chain {c}: {per_chain} samples in {time.time() - t0:.2f}s")
    samples = np.concatenate(all_samples, axis=0)
    cluster_sizes = np.concatenate(all_cluster, axis=0)
    elapsed = time.time() - t0
    print(f"total: {samples.shape[0]} samples in {elapsed:.2f}s")

    np.save(out, samples)
    np.save(out.with_suffix(".cluster.npy"), cluster_sizes)
    print(f"saved {out} (shape={samples.shape}, dtype={samples.dtype})")

    pm = primary_moments(samples)
    dv = derived_observables(samples)
    print("primary moments:", pm)
    print("derived:", dv)
    print(f"mean cluster size: {cluster_sizes.mean():.2f}")
    print(f"  reference U4* (2D Ising): ~0.6107")
    print(f"  thermodynamic-limit <e>: -sqrt(2) ≈ {-np.sqrt(2):.4f}")


if __name__ == "__main__":
    main()
