"""Energy and basic per-config quantities for the 2D Ising model on a torus."""
from __future__ import annotations

import numpy as np
import torch

# Critical inverse temperature for the 2D ferromagnetic Ising model: beta_c = (1/2) ln(1 + sqrt(2)).
BETA_C: float = 0.5 * np.log(1.0 + np.sqrt(2.0))
T_C: float = 1.0 / BETA_C  # ~2.2691853


def energy_np(s: np.ndarray) -> np.ndarray:
    """Total bond-summed energy E(s) = -J * sum_<ij> s_i s_j for one or many configs.

    s: (..., N, N) array with values in {-1, +1}.
    Returns: scalar per leading dim. With periodic BCs we count each bond once by
    summing s * roll(s, -1, axis=-1) and s * roll(s, -1, axis=-2).
    """
    s = s.astype(np.float64, copy=False)
    nn_right = s * np.roll(s, -1, axis=-1)
    nn_down = s * np.roll(s, -1, axis=-2)
    return -(nn_right.sum(axis=(-1, -2)) + nn_down.sum(axis=(-1, -2)))


def energy_torch(s: torch.Tensor) -> torch.Tensor:
    """Same as energy_np but for a torch tensor of shape (..., N, N) with values in {-1, +1}."""
    s = s.to(torch.float32)
    nn_right = s * torch.roll(s, -1, dims=-1)
    nn_down = s * torch.roll(s, -1, dims=-2)
    return -(nn_right.sum(dim=(-1, -2)) + nn_down.sum(dim=(-1, -2)))


def magnetization_np(s: np.ndarray) -> np.ndarray:
    """Total magnetization M(s) = sum_i s_i, returned per leading dim."""
    return s.astype(np.float64, copy=False).sum(axis=(-1, -2))
