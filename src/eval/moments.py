"""Hotelling-T^2 on independent primary moments of model vs. MC.

The five independent primary moments are
    X = (<e>, <e^2>, <|m|>, <m^2>, <m^4>).

We bootstrap the joint sampling distribution of the difference delta = X_model - X_mc
to get its covariance, then form T^2 = delta^T Sigma^-1 delta. Under H_0 this is
chi^2_5; we report T^2 and the p-value, plus per-component z-scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from src.ising.observables import bootstrap_moment_diff, primary_moments


MOMENT_NAMES = ("e", "e2", "abs_m", "m2", "m4")


@dataclass
class MomentReport:
    X_model: np.ndarray
    X_mc: np.ndarray
    delta: np.ndarray
    cov: np.ndarray
    z_per_component: np.ndarray
    T2: float
    p_value: float
    dof: int


def run_moment_test(
    s_model: np.ndarray,
    s_mc: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> MomentReport:
    X_model = primary_moments(s_model).as_vector()
    X_mc = primary_moments(s_mc).as_vector()
    rng = np.random.default_rng(seed)
    boot = bootstrap_moment_diff(s_model, s_mc, n_boot=n_boot, rng=rng)
    delta = X_model - X_mc
    cov = boot["cov"]
    # Mild Tikhonov to keep the inversion well-conditioned for n_boot ~ 500.
    eps = 1e-12 * np.trace(cov) / cov.shape[0]
    cov_reg = cov + eps * np.eye(cov.shape[0])
    diag = np.sqrt(np.diag(cov_reg))
    z = delta / (diag + 1e-30)
    inv = np.linalg.inv(cov_reg)
    T2 = float(delta @ inv @ delta)
    dof = cov.shape[0]
    p_value = float(1.0 - chi2.cdf(T2, df=dof))
    return MomentReport(
        X_model=X_model,
        X_mc=X_mc,
        delta=delta,
        cov=cov,
        z_per_component=z,
        T2=T2,
        p_value=p_value,
        dof=dof,
    )
