# Applications of Gaussian Processes

Final project for the Stochastic Processes course, bridging continuous-time stochastic processes and discrete computational inference. The project builds a Gaussian Process (GP) regression framework from scratch, performs Bayesian inference over kernel hyperparameters using Markov Chain Monte Carlo (MCMC), and implements a score-driven diffusion sampler using Langevin dynamics.

**Course:** Stochastic Processes
**Instructor:** Dr. Peyvandi

**Authors**
- Mohammad Parsa Talebizadeh (Student ID: 402100475)
- Seyed Hossein Ahmadi Mousavi (Student ID: 402100334)

---

## Overview

The notebook is organized into four self-contained tasks that progressively move from analytical GP theory to sampling-based Bayesian inference:

1. **GP Priors & Kernels** — Construct covariance matrices from the RBF and Brownian motion kernels and draw sample paths to contrast smooth vs. rough stochastic processes.
2. **GP Regression** — Derive and implement the closed-form posterior predictive distribution for noisy observations, conditioning the prior on training data.
3. **MCMC over Hyperparameters** — Treat the GP length scale as a random variable and sample its posterior with a Metropolis-Hastings sampler operating on the log marginal likelihood.
4. **Langevin Dynamics** — Simulate a continuous-time Itô diffusion via Euler-Maruyama discretization (the Unadjusted Langevin Algorithm) to sample from a bimodal target distribution.

## Background

A stochastic process $\{X(t) \mid t \in T\}$ is a **Gaussian Process** if every finite-dimensional projection is jointly Gaussian, fully specified by a mean function $m(t)$ and covariance kernel $k(s,t)$. This project explores GPs both as priors over function spaces and as a bridge to sampling-based inference methods (MCMC, Langevin dynamics) that are central to stochastic process theory.

## Task 1 — GP Priors & Kernels

Implements two covariance kernels and samples random functions from each:

- **RBF (squared-exponential) kernel** — stationary, infinitely differentiable, producing smooth sample paths:

$$k_{RBF}(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

- **Brownian motion (Wiener process) kernel** — non-stationary with independent increments, producing paths that are continuous everywhere but differentiable nowhere:

$$k_{BM}(s, t) = \min(s, t)$$

**Result:** RBF samples are smooth and locally correlated; Brownian motion samples exhibit rough, cumulative random fluctuations — consistent with the theoretical roughness properties of each kernel.

## Task 2 — GP Regression

Given noisy observations $y_i = f(x_i) + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma_y^2)$, the joint distribution of observed outputs and unobserved test points is partitioned as a multivariate normal, and the **posterior predictive distribution** is derived via Gaussian conditioning:

$$\mu_* = K(X_*, X)\left[K(X,X) + \sigma_y^2 I_n\right]^{-1} y$$

$$\Sigma_* = K(X_*, X_*) - K(X_*, X)\left[K(X,X) + \sigma_y^2 I_n\right]^{-1} K(X, X_*)$$

This is implemented from scratch (matrix inversion, posterior mean/covariance, and sampling) and validated against noisy observations of $f(x) = \sin(x)$.

**Result:** The posterior mean tracks the underlying sinusoid closely, and the 95% confidence band correctly narrows near observed points and widens in unobserved regions.

## Task 3 — MCMC over Kernel Hyperparameters

Rather than fixing the RBF length scale $\ell$, it is treated as a random variable with a **Gamma prior**, $\ell \sim \text{Gamma}(\text{shape}=2.0, \text{scale}=1.0)$. The (unnormalized) posterior is:

$$\log \pi(\ell) = \log p(y \mid X, \ell) + \log p(\ell)$$

where the log marginal likelihood is the analytically tractable GP evidence term. A **Metropolis-Hastings sampler** with a symmetric Gaussian random-walk proposal draws from this posterior in log-space for numerical stability.

**Result (15,000 iterations, 20% burn-in):**

| Metric | Value |
|---|---|
| Posterior mean of $\ell$ | 1.616 |
| Posterior std of $\ell$ | 0.398 |
| Acceptance rate | 63.27% |

The trace plot shows stable mixing and the posterior histogram concentrates around a plausible length scale, confirming the sampler is exploring the target distribution correctly.

## Task 4 — Langevin Dynamics (Unadjusted Langevin Algorithm)

Sampling is reformulated as simulating a continuous-time Itô diffusion whose stationary distribution is exactly the target density $\pi(x)$:

$$dX_t = \frac{1}{2}\nabla \log \pi(X_t)\, dt + dW_t$$

Discretized via Euler-Maruyama (the ULA update):

$$X_{k+1} = X_k + \frac{\epsilon}{2}\nabla \log \pi(X_k) + \sqrt{\epsilon}\, Z_k, \qquad Z_k \sim \mathcal{N}(0, I)$$

The sampler targets a **bimodal Gaussian mixture** $\pi(x) = 0.5\,\mathcal{N}(-2,1) + 0.5\,\mathcal{N}(2,1)$, using a numerically approximated score function.

**Result (100,000 steps, $\epsilon = 0.05$, 20,000 burn-in):**

| Metric | Value |
|---|---|
| Sample mean | 0.049 |
| Sample std | 2.218 |
| Post-burn-in samples | 80,000 |

The empirical histogram closely matches the true bimodal density, confirming the ULA sampler correctly captures both modes of the target distribution.

## Repository Contents

```
.
└── SPFinalProject.ipynb   # Full notebook: theory, implementation, and results for Tasks 1–4
```

## Requirements

- Python 3
- `numpy`
- `matplotlib`
- `scipy`

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

## Usage

Open and run the notebook top to bottom in Jupyter, JupyterLab, or Google Colab:

```bash
jupyter notebook SPFinalProject.ipynb
```

All results are reproducible; a fixed random seed (`np.random.seed(42)`) is set at the start of the notebook.

## References

- Foreman-Mackey, D. et al. (2013). *emcee: The MCMC Hammer.* [arXiv:1202.3665](https://arxiv.org/abs/1202.3665)
- Rasmussen, C. E. & Williams, C. K. I. *Gaussian Processes for Machine Learning.* MIT Press.
