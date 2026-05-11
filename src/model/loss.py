"""Symmetrized NLL loss for the Ising AR transformer.

Loss = -E_{s ~ data} [log p_sym(s)], with p_sym(s) = (p_theta(s) + p_theta(-s)) / 2,
computed via logaddexp for numerical stability. This is MLE on the symmetrized
model that we also use at sampling time (with a 50% global Z_2 flip).
"""
from __future__ import annotations

import math

import torch

from .transformer import IsingTransformer


def symmetrized_nll(model: IsingTransformer, s: torch.Tensor) -> torch.Tensor:
    """s: (B, N, N) in {-1, +1}. Returns scalar mean NLL of p_sym."""
    lp_pos = model.log_prob_spins(s)
    lp_neg = model.log_prob_spins(-s)
    lp_sym = torch.logaddexp(lp_pos, lp_neg) - math.log(2.0)
    return -lp_sym.mean()


def per_site_nll(model: IsingTransformer, s: torch.Tensor) -> torch.Tensor:
    """Per-site symmetrized NLL (nats). Useful as a scale-invariant training curve."""
    return symmetrized_nll(model, s) / (model.N * model.N)
