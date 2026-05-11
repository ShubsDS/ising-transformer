"""Pointwise comparison of G(r) and small-k structure factor between model and MC.

No critical-exponent fit at N=16: finite-size + small-r and r ~ N/2 effects make
that exponent biased and useless as a discriminator. We instead report G(r) and
S(k) shell-by-shell with bootstrap CIs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ising.observables import structure_factor, two_point_function


@dataclass
class SpatialReport:
    r: np.ndarray
    G_model: np.ndarray
    G_mc: np.ndarray
    G_model_se: np.ndarray
    G_mc_se: np.ndarray
    z_G: np.ndarray             # per-shell z-score
    S_model: np.ndarray         # full reciprocal lattice power
    S_mc: np.ndarray
    k_small: np.ndarray         # the small-|k| modes selected
    z_S_small: np.ndarray       # z-scores at those modes


def _bootstrap_G(samples: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    M = samples.shape[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, M, size=M)
        boots.append(two_point_function(samples[idx]))
    return np.stack(boots, axis=0)


def _bootstrap_S(samples: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    M = samples.shape[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, M, size=M)
        boots.append(structure_factor(samples[idx]))
    return np.stack(boots, axis=0)


def run_spatial_test(
    s_model: np.ndarray,
    s_mc: np.ndarray,
    n_boot: int = 100,
    seed: int = 0,
    n_small_k: int = 8,
) -> SpatialReport:
    rng = np.random.default_rng(seed)
    G_m = two_point_function(s_model)
    G_c = two_point_function(s_mc)
    boots_m = _bootstrap_G(s_model, n_boot, rng)
    boots_c = _bootstrap_G(s_mc, n_boot, rng)
    se_m = boots_m.std(axis=0, ddof=1)
    se_c = boots_c.std(axis=0, ddof=1)
    z_G = (G_m - G_c) / np.sqrt(se_m ** 2 + se_c ** 2 + 1e-30)

    S_m = structure_factor(s_model)
    S_c = structure_factor(s_mc)
    bS_m = _bootstrap_S(s_model, n_boot, rng)
    bS_c = _bootstrap_S(s_mc, n_boot, rng)
    sS_m = bS_m.std(axis=0, ddof=1)
    sS_c = bS_c.std(axis=0, ddof=1)

    # Pick smallest non-zero |k| modes (closest to the critical, long-wavelength sector).
    N = s_model.shape[-1]
    kx = np.fft.fftfreq(N) * 2 * np.pi
    ky = np.fft.fftfreq(N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Kmag = np.sqrt(KX ** 2 + KY ** 2)
    flat_idx = np.argsort(Kmag.ravel())  # smallest first
    flat_idx = [i for i in flat_idx if Kmag.ravel()[i] > 1e-9][:n_small_k]
    k_small = Kmag.ravel()[flat_idx]
    delta_S = (S_m.ravel()[flat_idx] - S_c.ravel()[flat_idx])
    se_S = np.sqrt(sS_m.ravel()[flat_idx] ** 2 + sS_c.ravel()[flat_idx] ** 2 + 1e-30)
    z_S_small = delta_S / se_S

    r = np.arange(G_m.shape[0])
    return SpatialReport(
        r=r,
        G_model=G_m,
        G_mc=G_c,
        G_model_se=se_m,
        G_mc_se=se_c,
        z_G=z_G,
        S_model=S_m,
        S_mc=S_c,
        k_small=k_small,
        z_S_small=z_S_small,
    )
