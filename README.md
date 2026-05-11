# MARS Ising Lattice — Neural Sampler at $T_c$

A small autoregressive transformer that learns to approximately sample $16\times16$
Ising spin configurations at the 2D critical temperature
$T_c = 2/\ln(1+\sqrt 2) \approx 2.2692$.

The full design — model, training objective, evaluation, and the chi-squared /
classifier-based scoring — is in [`PLAN.md`](PLAN.md). This README is just the
run sequence.

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch numpy numba scipy matplotlib tqdm
```

PyTorch's CUDA wheel is fetched automatically via `uv pip install torch`;
the training script auto-detects CUDA → MPS → CPU.

## End-to-end pipeline

### 1. Generate ground-truth Wolff samples at $T_c$ (~3 s)

```bash
python -m src.ising.data --N 16 --n_samples 100000 --n_chains 4
```

Writes `data/ising_N16_Tc.npy` (100k × 16 × 16 int8). Prints sanity checks against
Onsager $\langle e\rangle \!\to\! -\sqrt 2$ and the universal $U_4^\*\!\approx\!0.6107$.

### 2. Train the AR transformer (5–15 min on a modern CUDA GPU)

```bash
python -m src.train --n_steps 15000 --batch_size 256
```

Defaults: $d_\text{model}{=}128$, 6 layers, 4 heads, 2D row+col positional
embeddings, AdamW + cosine, symmetrized logaddexp NLL on $p_\text{sym}$.
Best checkpoint (lowest val NLL) is saved to `checkpoints/best.pt`.

### 3. Run the full evaluation (~1 min on GPU)

```bash
python -m src.evaluate --ckpt checkpoints/best.pt --n_eval 10000
```

This draws 10k samples from $p_\text{sym}$, computes importance weights
$\log w = -\beta_c E(s) - \log p_\text{sym}(s)$, and runs the four primary tests
described in `PLAN.md §6`:

| Test | What it measures | Pass criterion |
|---|---|---|
| **ESS/M** | importance-sampling effective sample size against the **exact** Boltzmann target | $> 0.1$ ("usable as a proposal") |
| **C2ST** | small CNN distinguishing model vs MC on raw $16\!\times\!16$ spins | $\lvert z \rvert < 2$ |
| **Hotelling $\mathcal{T}^2$** | joint deviation on 5 independent primary moments | $< 11.07$ ($\chi^2_5$ at 5%) |
| Diagnostics | $G(r)$ per shell, $S(k)$ at small $\lvert k\rvert$, $U_4$, $\chi$, $C$, $\langle m\rangle$, NLL gap | reported but not scored |

Outputs:

* `reports/score.json` — full JSON with every metric, including the bootstrap
  covariance of the moments, per-shell $G(r)$ z-scores, and the pass/fail summary.
* `reports/G_r.png` — two-point function model vs MC with error bars.
* `reports/observables.png` — energy and $|m|$ histograms model vs MC.
* `reports/logw_hist.png` — IS log-weight histogram (its tail tells you ESS).

## Files

```
src/
├── ising/
│   ├── mcmc.py          # Wolff single-cluster sampler (Numba JIT)
│   ├── energy.py        # E(s), beta_c, T_c
│   ├── observables.py   # primary moments, U4, chi, C, G(r), S(k), bootstraps
│   └── data.py          # generate / save Wolff samples
├── model/
│   ├── transformer.py   # decoder-only AR transformer, 2D row+col pos emb,
│   │                    # tractable log p(s), batched sampling with Z2 flip
│   ├── loss.py          # logaddexp symmetrized NLL on p_sym
│   └── sample.py        # batched sampling + log-w computation
├── eval/
│   ├── ess.py           # importance-sampling ESS/M, log Z, KL, bootstrap CI
│   ├── c2st.py          # small CNN with circular padding, z-statistic
│   ├── moments.py       # Hotelling T^2 on 5 primary moments
│   └── spatial.py       # G(r) and S(k) with bootstrap CIs
├── train.py             # training loop (early-stop on val)
└── evaluate.py          # orchestrator: writes reports/score.json + plots
```
