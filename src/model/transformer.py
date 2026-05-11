"""Decoder-only autoregressive transformer over the N x N Ising lattice in raster order.

Position-(i, j) site uses summed row+column learned embeddings, encoding the 2D
adjacency at trivially small parameter cost (2 * N * d_model). The model
processes the BOS-shifted sequence under a causal mask; output logits at sequence
position t predict the spin at lattice site t.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a static causal mask.

    Kept hand-rolled (instead of nn.TransformerEncoderLayer) so we have a clean
    place to add a KV-cache later if sampling becomes a bottleneck.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D_h)
        q = q.transpose(1, 2)  # (B, H, T, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # PyTorch's fused scaled_dot_product_attention with is_causal=True is
        # numerically faithful and fast on MPS.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, self.d_model)
        return self.out(y)


class TransformerBlock(nn.Module):
    """Pre-norm GPT-style block: x = x + attn(norm(x)); x = x + ffn(norm(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class IsingTransformer(nn.Module):
    """Autoregressive transformer over N^2 binary spins in raster-scan order.

    Convention: input tokens are in {0, 1}; the physical spin is s = 2*tok - 1 in {-1, +1}.
    forward(tokens of shape (B, T)) returns logits of shape (B, T+1, 2), where
    logits[:, t, :] predicts the token at lattice site t given tokens[:, :t].
    """

    def __init__(
        self,
        N: int,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.N = N
        self.seq_len = N * N
        self.d_model = d_model

        # Token embedding for spins {0, 1}, plus a single BOS vector.
        self.token_emb = nn.Embedding(2, d_model)
        self.bos_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        # 2D positional embeddings: separate row and column tables, summed.
        # At sequence/site position t, site is (t // N, t % N).
        self.row_emb = nn.Embedding(N, d_model)
        self.col_emb = nn.Embedding(N, d_model)
        rows = torch.arange(N).repeat_interleave(N)  # (N^2,)
        cols = torch.arange(N).repeat(N)
        self.register_buffer("pos_rows", rows, persistent=False)
        self.register_buffer("pos_cols", cols, persistent=False)

        d_ff = d_ff_mult * d_model
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2, bias=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) in {0, 1}, T in [0, N^2]. Returns (B, T+1, 2).

        Note T+1 <= N^2 + 1 in principle; we only ever use up to T = N^2 - 1 in training
        (and up to T = N^2 - 1 in sampling), which produces logits of length N^2.
        """
        B, T = tokens.shape
        bos = self.bos_emb.expand(B, 1, -1)
        if T > 0:
            x = torch.cat([bos, self.token_emb(tokens)], dim=1)  # (B, T+1, d_model)
        else:
            x = bos  # (B, 1, d_model)
        T_in = x.shape[1]
        if T_in > self.seq_len:
            raise ValueError(f"input sequence too long: {T_in} > {self.seq_len}")

        pos = self.row_emb(self.pos_rows[:T_in]) + self.col_emb(self.pos_cols[:T_in])
        x = x + pos.unsqueeze(0)

        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.head(x)  # (B, T+1, 2)

    # ---------- likelihoods ----------
    def log_prob_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N^2) in {0, 1}. Returns (B,) log p_theta(tokens)."""
        inputs = tokens[:, :-1]  # (B, N^2 - 1); forward will prepend BOS to get length N^2.
        logits = self.forward(inputs)  # (B, N^2, 2)
        log_softmax = F.log_softmax(logits, dim=-1)
        target = tokens.unsqueeze(-1)
        gathered = torch.gather(log_softmax, dim=-1, index=target).squeeze(-1)  # (B, N^2)
        return gathered.sum(dim=-1)

    def log_prob_spins(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, N, N) in {-1, +1}. Returns (B,) log p_theta(s)."""
        B = s.shape[0]
        tokens = ((s + 1) // 2).long().view(B, -1)
        return self.log_prob_tokens(tokens)

    def log_prob_sym(self, s: torch.Tensor) -> torch.Tensor:
        """Log of the symmetrized model p_sym(s) = (p(s) + p(-s)) / 2."""
        lp_pos = self.log_prob_spins(s)
        lp_neg = self.log_prob_spins(-s)
        return torch.logaddexp(lp_pos, lp_neg) - math.log(2.0)

    # ---------- sampling ----------
    @torch.no_grad()
    def sample(self, M: int, device: torch.device, symmetrize: bool = True) -> torch.Tensor:
        """Draw M lattice configurations. Returns (M, N, N) in {-1, +1}.

        If symmetrize=True, applies a global Z_2 flip with prob 1/2 per sample so we
        actually sample from p_sym rather than p_theta.
        """
        self.eval()
        N = self.N
        tokens = torch.empty(M, 0, dtype=torch.long, device=device)
        for _ in range(self.seq_len):
            logits = self.forward(tokens)[:, -1, :]  # (M, 2)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (M,)
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
        s = (2 * tokens - 1).view(M, N, N).to(torch.int8)
        if symmetrize:
            flip = torch.where(torch.rand(M, device=device) < 0.5, -1, 1).to(torch.int8)
            s = s * flip.view(M, 1, 1)
        return s
