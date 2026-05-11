"""Wolff single-cluster algorithm for the 2D ferromagnetic Ising model with periodic BCs.

Wolff is essentially free of critical slowing down at T_c, so it is the natural
ground-truth sampler for this project.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .energy import BETA_C


@njit(cache=True, fastmath=False)
def _wolff_one_flip(s: np.ndarray, p_add: float, rng_state: np.ndarray) -> int:
    """Perform one Wolff cluster flip on the in-place lattice `s` (shape (N, N), values +/-1).

    Returns the size of the flipped cluster. `rng_state` is a length-1 uint64 array
    used as a simple counter seed for numba-friendly randomness (we just call np.random
    inside; numba threads its own RNG).
    """
    N = s.shape[0]
    # Pick a random seed site uniformly.
    seed_i = np.random.randint(0, N)
    seed_j = np.random.randint(0, N)
    old_spin = s[seed_i, seed_j]
    new_spin = -old_spin

    # Stack of sites to process; preallocate to lattice size.
    stack_i = np.empty(N * N, dtype=np.int64)
    stack_j = np.empty(N * N, dtype=np.int64)
    top = 0
    stack_i[top] = seed_i
    stack_j[top] = seed_j
    top += 1
    s[seed_i, seed_j] = new_spin
    cluster_size = 1

    while top > 0:
        top -= 1
        i = stack_i[top]
        j = stack_j[top]
        # Four neighbors with periodic BCs.
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni = (i + di) % N
            nj = (j + dj) % N
            if s[ni, nj] == old_spin and np.random.random() < p_add:
                s[ni, nj] = new_spin
                stack_i[top] = ni
                stack_j[top] = nj
                top += 1
                cluster_size += 1
    rng_state[0] += 1  # unused; kept for ABI stability
    return cluster_size


@njit(cache=True)
def _wolff_run(
    s: np.ndarray,
    beta: float,
    n_therm: int,
    n_samples: int,
    sweeps_per_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a Wolff chain from `s` and return collected samples and cluster sizes."""
    N = s.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta)
    rng_state = np.zeros(1, dtype=np.uint64)

    # Thermalize.
    for _ in range(n_therm):
        _wolff_one_flip(s, p_add, rng_state)

    samples = np.empty((n_samples, N, N), dtype=np.int8)
    cluster_sizes = np.empty(n_samples * sweeps_per_sample, dtype=np.int64)
    csi = 0
    for k in range(n_samples):
        for _ in range(sweeps_per_sample):
            cluster_sizes[csi] = _wolff_one_flip(s, p_add, rng_state)
            csi += 1
        # Copy lattice into samples buffer as int8.
        for i in range(N):
            for j in range(N):
                samples[k, i, j] = s[i, j]
    return samples, cluster_sizes


def wolff_sample(
    N: int,
    beta: float = BETA_C,
    n_samples: int = 100_000,
    sweeps_per_sample: int = 5,
    n_therm: int = 10_000,
    seed: int | None = 0,
    init: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate decorrelated Wolff samples at inverse temperature beta.

    Returns:
        samples: int8 array of shape (n_samples, N, N) with values in {-1, +1}.
        cluster_sizes: int64 array of length n_samples * sweeps_per_sample.
    """
    if seed is not None:
        np.random.seed(seed)
    if init == "random":
        s = np.where(np.random.random((N, N)) < 0.5, np.int8(-1), np.int8(1))
    elif init == "up":
        s = np.ones((N, N), dtype=np.int8)
    else:
        raise ValueError(f"unknown init: {init}")
    # Numba wants a writable array of the right dtype.
    s = s.astype(np.int8, copy=True)
    samples, cluster_sizes = _wolff_run(s, beta, n_therm, n_samples, sweeps_per_sample)
    return samples, cluster_sizes
