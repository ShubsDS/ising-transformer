"""End-to-end evaluation of a trained Ising transformer.

Loads model + cached MC data, draws fresh model samples (or reuses a cache),
runs the four primary tests (ESS, C2ST, T^2, spatial), and writes
reports/score.json plus a few diagnostic plots.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eval.c2st import run_c2st
from src.eval.ess import compute_ess
from src.eval.moments import MOMENT_NAMES, run_moment_test
from src.eval.spatial import run_spatial_test
from src.ising.observables import derived_observables, per_sample_quantities
from src.model.sample import sample_and_score
from src.model.transformer import IsingTransformer
from src.train import pick_device


def _load_model(ckpt_path: str, device: torch.device) -> tuple[IsingTransformer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = IsingTransformer(
        N=cfg["N"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff_mult=cfg.get("d_ff_mult", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def _compute_logp_sym_for_array(model, s_np: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    out = []
    for i in range(0, s_np.shape[0], batch_size):
        chunk = torch.from_numpy(s_np[i : i + batch_size]).to(device).to(torch.float32)
        out.append(model.log_prob_sym(chunk).cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--mc_data", type=str, default="data/ising_N16_Tc.npy")
    p.add_argument("--model_samples", type=str, default="data/model_samples_N16.npy")
    p.add_argument("--logw_path", type=str, default="data/model_samples_N16.logw.npy")
    p.add_argument("--n_eval", type=int, default=10_000)
    p.add_argument("--sample_batch", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default="reports")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--regenerate_samples", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"device = {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = _load_model(args.ckpt, device)
    N = cfg["N"]
    print(f"loaded model: N={N}, d_model={cfg['d_model']}, layers={cfg['n_layers']}")

    # ---------------- model samples + IS log-weights ----------------
    samples_path = Path(args.model_samples)
    logw_path = Path(args.logw_path)
    if args.regenerate_samples or not samples_path.exists() or not logw_path.exists():
        print(f"generating {args.n_eval} model samples ...")
        t0 = time.time()
        s_model, logp_sym, logw = sample_and_score(
            model, args.n_eval, args.sample_batch, device
        )
        print(f"  done in {time.time() - t0:.1f}s")
        np.save(samples_path, s_model)
        np.save(samples_path.with_suffix(".logp.npy"), logp_sym)
        np.save(logw_path, logw)
    else:
        s_model = np.load(samples_path)[: args.n_eval]
        logw = np.load(logw_path)[: args.n_eval]
        print(f"loaded {s_model.shape[0]} cached model samples")

    # ---------------- MC samples (held-out part) ----------------
    mc_all = np.load(args.mc_data)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(mc_all.shape[0])
    held_out = mc_all[perm[: max(args.n_eval, mc_all.shape[0] // 10)]]
    s_mc = held_out[: args.n_eval]
    train_mc = mc_all[perm[mc_all.shape[0] // 10 :]]  # mimic train split
    print(f"MC: held-out {s_mc.shape[0]}, train-like {train_mc.shape[0]}")

    report: dict = {"config": cfg, "n_eval": s_model.shape[0]}

    # ---------------- ESS via IS to exact pi ----------------
    print("\n== ESS via importance sampling ==")
    ess_rep = compute_ess(logw, n_boot=1000, seed=args.seed)
    print(f"  ESS/M = {ess_rep.ess_over_M:.4f}  (95% CI {ess_rep.ess_over_M_ci})")
    print(f"  Var[log w] = {ess_rep.log_w_var:.3f}   log Z_hat = {ess_rep.log_Z_hat:.3f}")
    print(f"  KL(p_sym || pi) ≈ {ess_rep.kl_psym_pi:.3f}")
    report["ess"] = asdict(ess_rep)

    # ---------------- C2ST ----------------
    print("\n== C2ST (CNN with circular padding) ==")
    c2st_rep = run_c2st(s_model, s_mc, device=device, n_epochs=30, seed=args.seed)
    print(f"  test acc = {c2st_rep.accuracy:.4f}   z = {c2st_rep.z:.2f}   (|z|<2 ⇒ pass)")
    report["c2st"] = asdict(c2st_rep)

    # ---------------- Hotelling T^2 on primary moments ----------------
    print("\n== Primary moments (Hotelling T^2) ==")
    mom_rep = run_moment_test(s_model, s_mc, n_boot=500, seed=args.seed)
    for name, xm, xc, z in zip(MOMENT_NAMES, mom_rep.X_model, mom_rep.X_mc, mom_rep.z_per_component):
        print(f"  {name:>6s}: model {xm:+.5f}  mc {xc:+.5f}  z {z:+.2f}")
    print(f"  T^2 = {mom_rep.T2:.2f} (dof {mom_rep.dof})   p = {mom_rep.p_value:.3f}")
    report["moments"] = {
        "X_model": mom_rep.X_model.tolist(),
        "X_mc": mom_rep.X_mc.tolist(),
        "delta": mom_rep.delta.tolist(),
        "cov": mom_rep.cov.tolist(),
        "z_per_component": mom_rep.z_per_component.tolist(),
        "T2": mom_rep.T2,
        "p_value": mom_rep.p_value,
        "dof": mom_rep.dof,
        "moment_names": list(MOMENT_NAMES),
    }

    # ---------------- Derived diagnostics (U4, chi, C, <m>) ----------------
    print("\n== Derived diagnostics (NOT in composite) ==")
    dv_model = derived_observables(s_model)
    dv_mc = derived_observables(s_mc)
    for k in ("U4", "chi", "C", "mean_m"):
        print(f"  {k:>6s}: model {dv_model[k]:.4f}   mc {dv_mc[k]:.4f}")
    print("  U4 universal target (2D Ising): 0.6107")
    report["derived"] = {"model": dv_model, "mc": dv_mc, "U4_universal": 0.6107}

    # ---------------- Spatial: G(r), small-k S(k) ----------------
    print("\n== G(r) and S(k) (pointwise, with bootstrap CIs) ==")
    sp_rep = run_spatial_test(s_model, s_mc, n_boot=100, seed=args.seed)
    for r, gm, gc, z in zip(sp_rep.r, sp_rep.G_model, sp_rep.G_mc, sp_rep.z_G):
        print(f"  r={r}: G_model {gm:.4f}  G_mc {gc:.4f}  z {z:+.2f}")
    print("  small-|k| S(k) z-scores:", np.round(sp_rep.z_S_small, 2))
    report["spatial"] = {
        "r": sp_rep.r.tolist(),
        "G_model": sp_rep.G_model.tolist(),
        "G_mc": sp_rep.G_mc.tolist(),
        "z_G": sp_rep.z_G.tolist(),
        "k_small": sp_rep.k_small.tolist(),
        "z_S_small": sp_rep.z_S_small.tolist(),
    }

    # ---------------- Memorization / NLL gap ----------------
    print("\n== Memorization check ==")
    n_check = min(2000, train_mc.shape[0], s_mc.shape[0])
    train_chunk = train_mc[:n_check].astype(np.int8)
    held_chunk = s_mc[:n_check].astype(np.int8)
    lp_train = _compute_logp_sym_for_array(model, train_chunk, device)
    lp_held = _compute_logp_sym_for_array(model, held_chunk, device)
    gap = float(lp_train.mean() - lp_held.mean())
    se_gap = float(np.sqrt(lp_train.var() / n_check + lp_held.var() / n_check))
    print(f"  log p_sym (train) = {lp_train.mean():.3f}")
    print(f"  log p_sym (held)  = {lp_held.mean():.3f}")
    print(f"  gap (train - held) = {gap:+.3f}  ± {se_gap:.3f}  (large positive ⇒ memorization)")
    # exact duplicates between model and train
    dup = 0
    train_set = {bytes(s.tobytes()) for s in train_mc[:50_000]}
    for s in s_model:
        if bytes(s.astype(np.int8).tobytes()) in train_set:
            dup += 1
    print(f"  exact duplicates in model samples vs train set: {dup}/{s_model.shape[0]}")
    report["memorization"] = {
        "log_p_sym_train_mean": float(lp_train.mean()),
        "log_p_sym_held_mean": float(lp_held.mean()),
        "nll_gap": gap,
        "nll_gap_se": se_gap,
        "exact_duplicates": dup,
    }

    # ---------------- Pass / fail summary ----------------
    chi2_5_thr = 11.07
    # Pass threshold for ESS/M = 0.1 ("usable as a proposal"); 0.5+ would be "excellent",
    # ~1e-3 just means "not catastrophically collapsed". 0.1 is the right line for "reasonable sampler".
    pass_ess = ess_rep.ess_over_M > 0.1
    pass_c2st = abs(c2st_rep.z) < 2.0
    pass_t2 = mom_rep.T2 < chi2_5_thr
    report["summary"] = {
        "ess_pass": bool(pass_ess),
        "c2st_pass": bool(pass_c2st),
        "T2_pass": bool(pass_t2),
        "overall_pass": bool(pass_ess and pass_c2st and pass_t2),
        "chi2_5_threshold": chi2_5_thr,
    }

    print("\n==== SUMMARY ====")
    print(f"  ESS/M           = {ess_rep.ess_over_M:.4f}   [pass={pass_ess}]")
    print(f"  C2ST |z|        = {abs(c2st_rep.z):.2f}     [pass={pass_c2st}]")
    print(f"  Hotelling T^2   = {mom_rep.T2:.2f}          [pass={pass_t2}, thr={chi2_5_thr}]")
    print(f"  Overall pass    = {report['summary']['overall_pass']}")

    with open(out_dir / "score.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {out_dir/'score.json'}")

    # ---------------- Plots ----------------
    # G(r) plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(sp_rep.r, sp_rep.G_mc, yerr=sp_rep.G_mc_se, marker="o", label="MC (Wolff)", capsize=2)
    ax.errorbar(sp_rep.r, sp_rep.G_model, yerr=sp_rep.G_model_se, marker="s", label="model", capsize=2)
    ax.set_xlabel("r")
    ax.set_ylabel("G(r)")
    ax.set_title(f"Two-point function, N={N}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "G_r.png", dpi=120)
    plt.close(fig)

    # Histograms of e and |m|
    q_m = per_sample_quantities(s_model)
    q_c = per_sample_quantities(s_mc)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(q_c["e"], bins=60, alpha=0.5, label="MC", density=True)
    axes[0].hist(q_m["e"], bins=60, alpha=0.5, label="model", density=True)
    axes[0].set_xlabel("e (energy per spin)"); axes[0].legend()
    axes[1].hist(q_c["abs_m"], bins=60, alpha=0.5, label="MC", density=True)
    axes[1].hist(q_m["abs_m"], bins=60, alpha=0.5, label="model", density=True)
    axes[1].set_xlabel("|m| (magnetization per spin)"); axes[1].legend()
    fig.suptitle(f"Per-sample observables, N={N}")
    fig.tight_layout()
    fig.savefig(out_dir / "observables.png", dpi=120)
    plt.close(fig)

    # log-weight diagnostic
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(logw - logw.max(), bins=60, density=True)
    ax.set_xlabel("log w - max log w")
    ax.set_ylabel("density")
    ax.set_title(f"IS log-weights (ESS/M = {ess_rep.ess_over_M:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "logw_hist.png", dpi=120)
    plt.close(fig)

    print(f"plots in {out_dir}/")


if __name__ == "__main__":
    main()
