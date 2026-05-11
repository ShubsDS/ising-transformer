"""Train the Ising autoregressive transformer with symmetrized NLL on Wolff samples."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.model.loss import symmetrized_nll
from src.model.transformer import IsingTransformer


@dataclass
class TrainConfig:
    N: int = 16
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    d_ff_mult: int = 4

    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 500
    n_steps: int = 15_000

    val_frac: float = 0.1
    eval_every: int = 250
    eval_batches: int = 8

    early_stop_patience: int = 2_000  # in optimizer steps

    data_path: str = "data/ising_N16_Tc.npy"
    out_dir: str = "checkpoints"
    seed: int = 0


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_with_warmup(step: int, n_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, n_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=15_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--data_path", type=str, default="data/ising_N16_Tc.npy")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    cfg = TrainConfig(
        N=args.N,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        data_path=args.data_path,
        out_dir=args.out_dir,
        seed=args.seed,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"device = {device}")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    data = np.load(cfg.data_path)
    assert data.shape[1:] == (cfg.N, cfg.N), f"data shape mismatch: {data.shape}"
    print(f"loaded {data.shape[0]} samples of shape {data.shape[1:]}")
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(data.shape[0])
    n_val = int(cfg.val_frac * data.shape[0])
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_data = torch.from_numpy(data[val_idx]).to(torch.int8)
    train_data = torch.from_numpy(data[train_idx]).to(torch.int8)
    print(f"train {train_data.shape[0]}  val {val_data.shape[0]}")

    # Pinned host tensors; move minibatches to device.
    def sample_batch(t: torch.Tensor, B: int) -> torch.Tensor:
        idx = torch.randint(0, t.shape[0], (B,))
        return t[idx]

    # ---- model / opt ----
    model = IsingTransformer(
        N=cfg.N,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff_mult=cfg.d_ff_mult,
    ).to(device)
    print(f"n_params = {model.n_params():,}")
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    sched = LambdaLR(opt, lr_lambda=lambda step: cosine_with_warmup(step, cfg.n_steps, cfg.warmup_steps))

    # ---- training loop ----
    best_val = float("inf")
    best_step = 0
    history: list[dict] = []
    t0 = time.time()
    n_sites = cfg.N * cfg.N
    log_uniform = n_sites * math.log(2.0)

    for step in range(1, cfg.n_steps + 1):
        model.train()
        batch = sample_batch(train_data, cfg.batch_size).to(device).to(torch.long)
        # int8 -> int64 via .long(), values in {-1, +1}.
        loss = symmetrized_nll(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % 50 == 0:
            lr = sched.get_last_lr()[0]
            elapsed = time.time() - t0
            print(
                f"step {step:>6d}/{cfg.n_steps}  train_nll {loss.item():.3f}  "
                f"per_site {loss.item() / n_sites:.4f}  lr {lr:.2e}  t {elapsed:.1f}s",
                flush=True,
            )

        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(cfg.eval_batches):
                    vb = sample_batch(val_data, cfg.batch_size).to(device).to(torch.long)
                    val_losses.append(symmetrized_nll(model, vb).item())
                val_nll = float(np.mean(val_losses))
            elapsed = time.time() - t0
            print(
                f"  >>> step {step}  val_nll {val_nll:.3f}  per_site {val_nll/n_sites:.4f}  "
                f"vs uniform {log_uniform:.3f}  ({elapsed:.1f}s)",
                flush=True,
            )
            history.append({"step": step, "val_nll": val_nll, "train_nll": loss.item(), "time": elapsed})
            if val_nll < best_val - 1e-3:
                best_val = val_nll
                best_step = step
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "step": step}, out_dir / "best.pt")
            elif step - best_step >= cfg.early_stop_patience:
                print(f"early stop at step {step} (best at step {best_step}, val {best_val:.3f})")
                break

    # final save
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "step": step}, out_dir / "last.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump({"config": asdict(cfg), "history": history, "best_val": best_val, "best_step": best_step}, f, indent=2)
    print(f"done. best_val={best_val:.3f} at step {best_step}; checkpoints at {out_dir}")


if __name__ == "__main__":
    main()
