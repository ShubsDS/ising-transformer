# Plan: Neural sampler for the 2D Ising lattice at $T_c$ (revised)

## 0. Changelog vs. v1

After review the following faults in v1 were accepted and fixed in this revision:

1. **Composite score conflated dependent observables.** $U_4$, $\chi$, $C$ are deterministic functions of moments already in the score. Now the headline composite is built only from independent primary moments; $U_4,\chi,C$ are reported as diagnostics.
2. **$\hat\eta$ fit at $N=16$ is not a meaningful discriminator.** Replaced with pointwise comparison of $G(r)$ and $S(\mathbf{k})$ versus MC, with bootstrap CIs. Recovering $\eta=1/4$ requires multi-$N$ FSS, kept as a stretch goal.
3. **No joint-distribution test.** Added a small-CNN **classifier two-sample test (C2ST)** on raw $16\times16$ spins as a primary metric.
4. **Flat 1D positional embedding.** Replaced with summed row+column embeddings.
5. **Symmetrized loss was the log-geometric mean.** Replaced with the log-arithmetic-mean form (MLE on the symmetrized model), computed via `logaddexp`.
6. **Hamming nearest-neighbor memorization check.** Replaced by train-vs-held-out NLL gap; NN retained only as a duplicate-detection sanity check.
7. **New:** Added **self-normalized importance-sampling effective sample size (ESS/M)** as a primary metric — it tests the model against the *exact* Boltzmann target, not against a finite MC estimate of it.

## 1. Problem setup

We want a model $p_\theta(s)$ over spin configurations $s\in\{-1,+1\}^{N\times N}$ that approximately samples the Boltzmann distribution of the 2D ferromagnetic Ising model with periodic boundaries,
$$
\pi(s) = \frac{1}{Z}\exp\!\big(\beta J \!\!\sum_{\langle ij\rangle}\!\! s_i s_j\big),\qquad
T_c=\frac{2}{\ln(1+\sqrt 2)}\approx 2.26918531,\quad \beta_c=\tfrac12\ln(1+\sqrt 2).
$$
We take $J=k_B=1$ and $N=16$ (256 spins). $N$ stays a config flag so $\{8,16,32\}$ run without code changes.

## 2. Architecture

**Decoder-only Transformer, autoregressive in raster-scan order.**

- Sequence is the 256 spins in row-major order; each token is binary, mapped to $\pm 1$ for physics.
- $d_{\text{model}}=128$, 6 layers, 4 heads, GELU, pre-norm.
- **2D positional embedding.** A site at $(i,j)$ gets $\mathbf{e}_{ij} = \mathbf{r}_i + \mathbf{c}_j$ where $\{\mathbf{r}_i\}_{i=0}^{N-1}$ and $\{\mathbf{c}_j\}_{j=0}^{N-1}$ are two learned $N\times d_{\text{model}}$ tables. This encodes 2D adjacency at trivially small parameter cost ($2Nd_{\text{model}}$ vs $N^2 d_{\text{model}}$ for the flat version). (Optional ablation: 2D-RoPE relative position.)
- Output head: linear → 2 logits per site → Bernoulli over $\{-1,+1\}$.
- **Tractable exact log-likelihood:** $\log p_\theta(s)=\sum_i \log p_\theta(s_i\mid s_{<i})$.

### 2.1 $\mathbb{Z}_2$ symmetry

The symmetrized model is
$$
p_\text{sym}(s)=\tfrac12[p_\theta(s)+p_\theta(-s)],
$$
which is the model we *use* for sampling (each draw is flipped globally with probability $1/2$) and for likelihood evaluation. For training/eval consistency we must train against the same object: see §4.

## 3. Training-data pipeline (ground truth)

Wolff single-cluster MCMC at $\beta_c$. Wolff has essentially no critical slowing down at $T_c$, which makes it almost mandatory here.

- Pure NumPy with a BFS queue, then Numba-JITed; periodic BCs.
- Thermalize $10^4$ cluster flips, then collect $10^5$ samples, one per $\sim 5$ cluster flips. Save as `int8` in `data/ising_N16_Tc.npy` (~25 MB).
- **Sanity checks on the ground truth itself:**
  - Compare $\langle e\rangle$ at $N=16$ against an Onsager finite-$N$ partition-function reference (well tabulated for small $N$).
  - Check $U_4 \to U_4^\* \approx 0.6107$ to within $\sim 10^{-2}$.
  - Histogram the cluster-size distribution and verify the heavy tail.

## 4. Training objective

The data law $\pi$ is exactly $\mathbb{Z}_2$-symmetric, so the principled MLE is on the symmetrized model:
$$
\mathcal{L}(\theta) = -\mathbb{E}_{s\sim\pi}\!\Big[\log\tfrac12\big(p_\theta(s)+p_\theta(-s)\big)\Big]
= -\mathbb{E}_{s\sim\pi}\!\Big[\mathrm{logaddexp}\!\big(\log p_\theta(s),\,\log p_\theta(-s)\big) - \log 2\Big].
$$
Implemented via `torch.logaddexp` for numerical stability. Each minibatch step does two forward passes (on $s$ and on $-s$); each costs the same as a vanilla AR forward.

Note on the v1 mistake: $-\tfrac12\log p(s)-\tfrac12\log p(-s) = -\log\sqrt{p(s)p(-s)}$ is the *geometric* mean, a strict lower bound (AM-GM) on the symmetrized log-likelihood, and equivalent to vanilla MLE on $p_\theta$ under flip-augmented data. It pushes $p_\theta$ itself to be symmetric. That is also a valid training objective but is *not* MLE on $p_\text{sym}$; since we sample from $p_\text{sym}$ we should train against $p_\text{sym}$.

- Optimizer: AdamW, lr $3\!\times\!10^{-4}$, cosine schedule, weight decay $0.01$, grad clip $1.0$.
- Batch 256; **15k steps** (val NLL plateaus well before 50k at this scale); early-stop on held-out symmetrized NLL with patience 1k steps.
- Hardware: PyTorch with MPS if available, CPU fallback. FP32 throughout (MPS FP16 is hit-or-miss).
- **Wall-clock budget on an M-series MacBook Air:** ~45–90 min total end-to-end (Wolff data ~5 min, training ~30–60 min, sampling + eval ~10–20 min).

## 5. Sampling

Standard temperature-1 AR sampling, batched ($M$ chains advance together). Then global $\mathbb{Z}_2$ flip with probability $1/2$ per sample to draw from $p_\text{sym}$.

## 6. Evaluation: how do we know it's a "reasonable" sampler?

Three independent families of tests. The first is the strongest and most physically grounded; the others are corroborating.

### 6.1 Primary headline metric: self-normalized importance ESS

Because we have both an exact $\log p_\text{sym}(s)$ and the closed-form unnormalized target $\tilde\pi(s) = e^{-\beta_c E(s)}$, we can importance-reweight model samples to the *exact* Boltzmann distribution (not to a finite-sample MC estimate of it). For $M$ samples $s_i\sim p_\text{sym}$:
$$
\log w_i = -\beta_c E(s_i) - \log p_\text{sym}(s_i),\qquad
\mathrm{ESS}/M \;=\; \frac{(\sum_i w_i)^2}{M\sum_i w_i^2}\in(0,1].
$$

- $\mathrm{ESS}/M\to 1$ iff $p_\text{sym}\equiv\pi$. This is a test against the **exact target**, with no MC reference involved.
- We report $\mathrm{ESS}/M$ over $M=10^4$ samples, plus a bootstrap CI.
- Companion diagnostics: $\mathrm{Var}_{p_\text{sym}}[\log w]$ and the reverse KL estimator $\widehat{\mathrm{KL}}(p_\text{sym}\|\pi) = \mathbb{E}_{p_\text{sym}}[\log p_\text{sym} - \log\tilde\pi] - \log\hat Z$, with $\log\hat Z = \mathrm{logsumexp}(\log w_i) - \log M$.
- **Rough thresholds** (these are not theorems, just practical guides): $\mathrm{ESS}/M\!>\!0.5$ ≈ excellent; $>\!0.1$ ≈ usable as a proposal; $<\!10^{-3}$ ≈ poor.

### 6.2 Primary joint-distribution test: classifier two-sample test (C2ST)

- Train a small CNN ($\sim$50k params: two conv layers with circular padding to respect PBC, then GAP + linear) to distinguish 10k model samples from 10k held-out Wolff samples, with a 80/20 train/test split, weight decay, and early stopping on the test split.
- **Test statistic:** held-out accuracy $\hat a$. Under $H_0: p_\theta = \pi$, $\hat a$ should be statistically consistent with $1/2$; specifically $z = \sqrt{4M_\text{test}}(\hat a - 1/2) \sim \mathcal{N}(0,1)$.
- **Decision:** $|z| < 2$ ⇒ indistinguishable at $\sim$5% level.
- A C2ST tests the *full joint* in spin space, which no scalar moment panel can do.

### 6.3 Primary moment test (cleaned up)

We use only **independent primary moments** computed on $10^4$ samples each from the model and from held-out Wolff:
$$
\mathbf{X} = \big(\langle e\rangle,\;\langle e^2\rangle,\;\langle |m|\rangle,\;\langle m^2\rangle,\;\langle m^4\rangle\big)\in\mathbb{R}^5.
$$
Let $\hat{\mathbf{X}}_\theta$, $\hat{\mathbf{X}}_\text{MC}$ be the two sample means and $\hat\Sigma$ a bootstrap-estimated joint covariance of $(\hat{\mathbf{X}}_\theta - \hat{\mathbf{X}}_\text{MC})$. Define
$$
\boxed{\;\mathcal{T}^2 = (\hat{\mathbf{X}}_\theta - \hat{\mathbf{X}}_\text{MC})^\top \hat\Sigma^{-1} (\hat{\mathbf{X}}_\theta - \hat{\mathbf{X}}_\text{MC})\;}
$$
which is Hotelling-$T^2$-like. Under $H_0$ (and with these sample sizes asymptotic normality is fine) $\mathcal{T}^2$ has a $\chi^2_5$ reference distribution; $\mathcal{T}^2\!\lesssim\!11.07$ ⇒ pass at 5%.

Reported as diagnostics, *not* in the composite:
- $U_4 = 1-\langle m^4\rangle/(3\langle m^2\rangle^2)$ (target: $\approx 0.6107$).
- $\chi = N^2(\langle m^2\rangle-\langle |m|\rangle^2)/T_c$.
- $C = N^2(\langle e^2\rangle-\langle e\rangle^2)/T_c^2$.
- $\langle m\rangle$ (symmetry breaking check; should be $\approx 0$ to MC error).

### 6.4 Spatial correlation tests (pointwise, no exponent fit)

Compute on both sample sets:
- The full **two-point function** $G(r)$ averaged over translations and over both lattice directions, for $r=1,\dots,N/2$. Report $G_\theta(r) - G_\text{MC}(r)$ per shell with bootstrap CIs. We do **not** fit a power law at $N=16$ — that would conflate model error with finite-size scaling violation. Recovering $\eta=1/4$ is a multi-$N$ stretch goal (§9).
- The **structure factor** $S(\mathbf{k})$ on the discrete momentum grid, focusing on small-$\mathbf{k}$ modes (which carry the long-wavelength critical content). Report mode-by-mode $z$-scores; the long-wavelength modes are where critical fluctuations live and where a too-short-ranged model will fail visibly.

### 6.5 Memorization / overfitting check

- Compare $\overline{\log p_\theta}(s_\text{train})$ vs $\overline{\log p_\theta}(s_\text{held-out})$ over the actual Wolff splits. A model that has memorized assigns notably higher likelihood to training configs.
- Auxiliary: exact-duplicate check (Hamming distance 0) between generated samples and the training set; expected rate is near zero given $2^{256}$ configuration space, but worth confirming. We drop the Hamming-NN heuristic because pairwise distances concentrate around $128\pm 8$ in 256-D binary and the heuristic can't resolve partial memorization.

### 6.6 Summary scorecard

A run is reported as a single JSON with:
- `ess_over_M` (primary, headline),
- `c2st_accuracy`, `c2st_z` (primary),
- `T2_moments`, `T2_pvalue` (primary, 5 dof),
- `U4_model`, `U4_mc`, `U4_universal=0.6107`,
- `chi_model`, `chi_mc`, `C_model`, `C_mc`,
- `G_r_z[ ]`, `S_k_z[ ]`,
- `train_held_NLL_gap`, `exact_duplicate_rate`,
- per-observable z-scores in the diagnostic block.

A model is declared "indistinguishable" if it passes all three primary tests: $\mathrm{ESS}/M$ not collapsed, C2ST $|z|<2$, $\mathcal{T}^2 < \chi^2_5(0.95)$.

## 7. Layout

```
MARS_ising_lattice/
├── PLAN.md                  # this file
├── pyproject.toml           # uv: torch, numpy, numba, scipy, matplotlib, tqdm
├── src/
│   ├── ising/
│   │   ├── mcmc.py          # Wolff sampler (Numba)
│   │   ├── energy.py        # E(s), torch+numpy
│   │   ├── observables.py   # primary moments, U4, chi, C, G(r), S(k)
│   │   └── data.py
│   ├── model/
│   │   ├── transformer.py   # AR transformer w/ 2D row+col pos embeddings
│   │   ├── loss.py          # logaddexp symmetrized NLL
│   │   └── sample.py        # batched AR sampling, then random Z2 flip
│   ├── eval/
│   │   ├── ess.py           # IS reweighting → ESS/M, KL, Z hat
│   │   ├── c2st.py          # small CNN with circular padding
│   │   ├── moments.py       # Hotelling T^2
│   │   └── spatial.py       # G(r), S(k) with bootstrap CIs
│   ├── train.py
│   └── evaluate.py          # writes reports/score.json + plots
├── data/                    # cached Wolff samples
├── checkpoints/
└── reports/                 # observables.png, G_r.png, S_k.png, score.json
```

## 8. Milestones

1. **Phase 1: ground truth.** Wolff sampler, observables, validation against Onsager + $U_4^\*$. Generate 100k samples.
2. **Phase 2: model.** Transformer with 2D pos embeddings, symmetrized logaddexp loss, AR sampling with global flip. Train; monitor held-out symmetrized NLL.
3. **Phase 3: evaluation.** Implement ESS/IS, C2ST, $\mathcal{T}^2$, $G(r)$ and $S(\mathbf{k})$ with bootstrap. Produce `score.json` and plots.
4. **Phase 4: stretch.** VAN comparison; multi-$N$ FSS check ($\eta\to 1/4$); temperature conditioning.

## 9. Stretch goals

- **VAN (no MC data):** same architecture, loss $\mathbb{E}_{p_\text{sym}}[\beta_c E + \log p_\text{sym}]$ via REINFORCE with moving-average baseline; subject to known issues at criticality (symmetry breaking, gradient variance).
- **Multi-$N$ FSS:** train at $N\in\{8,16,32\}$, check $U_4$ crossings at $T_c$ and effective $\hat\eta(N)$ trending to $1/4$. This is where the universal critical exponents become a meaningful test.
- **Temperature conditioning:** concatenate a learned $\beta$ embedding to every token; train across $\beta\in[\beta_c-0.2,\beta_c+0.2]$; verify the susceptibility peak.
