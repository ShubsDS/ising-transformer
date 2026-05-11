"""Per-sample and population observables for the 2D Ising model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .energy import T_C, energy_np, magnetization_np


@dataclass
class PrimaryMoments:
    """Independent primary moments used as the headline statistical test."""

    e: float        # <e> = <E> / N^2
    e2: float       # <e^2>
    abs_m: float    # <|m|> = < |M| / N^2 >
    m2: float       # <m^2>
    m4: float       # <m^4>

    def as_vector(self) -> np.ndarray:
        return np.array([self.e, self.e2, self.abs_m, self.m2, self.m4])


def per_sample_quantities(s: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-sample e and m for a stack of configs of shape (M, N, N)."""
    N = s.shape[-1]
    n_sites = N * N
    E = energy_np(s)
    M = magnetization_np(s)
    e = E / n_sites
    m = M / n_sites
    return {"e": e, "m": m, "abs_m": np.abs(m)}


def primary_moments(s: np.ndarray) -> PrimaryMoments:
    q = per_sample_quantities(s)
    e = q["e"]
    abs_m = q["abs_m"]
    return PrimaryMoments(
        e=float(e.mean()),
        e2=float((e ** 2).mean()),
        abs_m=float(abs_m.mean()),
        m2=float((abs_m ** 2).mean()),
        m4=float((abs_m ** 4).mean()),
    )


def derived_observables(s: np.ndarray, T: float = T_C) -> dict[str, float]:
    """Diagnostic (non-primary) observables that are deterministic functions of the moments.

    - U4: Binder cumulant, target ~ 0.6107 at the 2D Ising critical point.
    - chi: magnetic susceptibility (per spin, with N^2 factor in the standard FSS form).
    - C:   specific heat (per spin, FSS form).
    - mean_m: signed magnetization mean; should be ~0 by Z2 symmetry.
    """
    N = s.shape[-1]
    q = per_sample_quantities(s)
    e = q["e"]
    m = q["m"]
    abs_m = q["abs_m"]
    m2 = float((abs_m ** 2).mean())
    m4 = float((abs_m ** 4).mean())
    abs_m_mean = float(abs_m.mean())
    e_mean = float(e.mean())
    e2 = float((e ** 2).mean())
    chi = (N * N) * (m2 - abs_m_mean ** 2) / T
    C = (N * N) * (e2 - e_mean ** 2) / (T ** 2)
    U4 = 1.0 - m4 / (3.0 * m2 ** 2 + 1e-30)
    return {
        "U4": U4,
        "chi": chi,
        "C": C,
        "mean_m": float(m.mean()),
        "mean_abs_m": abs_m_mean,
        "mean_e": e_mean,
    }


def two_point_function(s: np.ndarray) -> np.ndarray:
    """Connected two-point function G(r) for r = 0, 1, ..., N//2 along axis-aligned directions.

    Computed by translation-averaging and averaging over the two lattice directions.
    Returns array of shape (N//2 + 1,).
    """
    N = s.shape[-1]
    s = s.astype(np.float64, copy=False)
    rmax = N // 2
    out = np.zeros(rmax + 1, dtype=np.float64)
    # G(r) along axis -1 then axis -2; average. Use FFT-free roll-and-multiply since N is small.
    for r in range(rmax + 1):
        gh = (s * np.roll(s, -r, axis=-1)).mean(axis=(-1, -2)).mean()
        gv = (s * np.roll(s, -r, axis=-2)).mean(axis=(-1, -2)).mean()
        out[r] = 0.5 * (gh + gv)
    return out


def structure_factor(s: np.ndarray) -> np.ndarray:
    """Structure factor S(k) on the discrete reciprocal lattice, shape (N, N).

    S(k) = (1/N^2) E[ |sum_x s(x) e^{i k.x}|^2 ] averaged over samples.
    Note: k = 0 mode gives N^2 * <m^2>; usually we drop or report it separately.
    """
    N = s.shape[-1]
    s = s.astype(np.float64, copy=False)
    fs = np.fft.fft2(s, axes=(-2, -1))
    power = (np.abs(fs) ** 2) / (N * N)
    return power.mean(axis=0)


def bootstrap_moment_diff(
    s_model: np.ndarray,
    s_mc: np.ndarray,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Bootstrap the joint sampling distribution of (X_model - X_mc) over primary moments.

    Returns dict with 'delta' (mean diff, shape (5,)) and 'cov' (5x5).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Mm = s_model.shape[0]
    Mc = s_mc.shape[0]
    diffs = np.empty((n_boot, 5), dtype=np.float64)
    for b in range(n_boot):
        idx_m = rng.integers(0, Mm, size=Mm)
        idx_c = rng.integers(0, Mc, size=Mc)
        Xm = primary_moments(s_model[idx_m]).as_vector()
        Xc = primary_moments(s_mc[idx_c]).as_vector()
        diffs[b] = Xm - Xc
    delta = diffs.mean(axis=0)
    cov = np.cov(diffs.T)
    return {"delta": delta, "cov": cov, "boot_diffs": diffs}
