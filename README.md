# Applications of Gaussian Processes

Final project for a Stochastic Processes course, covering Gaussian Process (GP) priors, GP regression, Bayesian hyperparameter inference via Metropolis–Hastings MCMC, and continuous-time sampling via Langevin dynamics.

**Authors**
- Mohammad Parsa Talebizadeh
- Seyed Hossein Ahmadi Mousavi

**Course**: Stochastic Processes — Instructor: Dr. Peyvandi

---

## Overview

This project bridges continuous-time stochastic processes and discrete computational inference. Starting from the definition of a Gaussian Process, it builds a full GP regression framework from scratch, performs Bayesian inference over kernel hyperparameters using MCMC, and simulates a continuous-time Itô diffusion whose stationary distribution matches a target density.

**Objectives**

1. Construct GP prior spaces from foundational covariance kernels.
2. Derive and implement analytical GP regression updates under noisy observations.
3. Build a Metropolis–Hastings sampler to perform Bayesian inference over kernel hyperparameters via the log-marginal likelihood.
4. Simulate a continuous-time Itô diffusion using the Euler–Maruyama scheme (Unadjusted Langevin Algorithm).

All code is implemented from first principles in NumPy — no GP or MCMC libraries are used.

---

## Contents

| Task | Topic | Description |
|------|-------|-------------|
| 1 | GP Priors | RBF and Brownian motion kernels; sampling prior paths |
| 2 | GP Regression | Closed-form posterior predictive mean and covariance under Gaussian noise |
| 3 | MCMC Hyperparameter Inference | Metropolis–Hastings sampling of the GP length-scale posterior |
| 4 | Langevin Dynamics | Euler–Maruyama simulation of Langevin SDE to sample a bimodal target |

---

## Task 1 — Gaussian Process Priors

A stochastic process is a Gaussian Process if every finite-dimensional projection is jointly Gaussian, fully characterized by a mean function and a covariance kernel. Two kernels are implemented and compared:

- **RBF (squared-exponential) kernel** — stationary, infinitely differentiable, producing smooth sample paths:

$$k_{RBF}(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

- **Brownian motion kernel** — non-stationary, with independent increments and paths that are continuous but nowhere differentiable:

$$k_{BM}(s, t) = \min(s, t)$$

Both covariance matrices are constructed explicitly, and sample paths are drawn via `np.random.multivariate_normal`, illustrating the qualitative difference between smooth and rough stochastic process realizations.

## Task 2 — GP Regression

Given noisy observations $y_i = f(x_i) + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma_y^2)$ and a GP prior $f \sim \mathcal{GP}(0, k)$, the joint distribution of training outputs and test-point predictions is a partitioned multivariate Gaussian. Conditioning gives the posterior predictive distribution in closed form:

$$\mu_{\ast} = K(X_{\ast}, X)\left[K(X,X) + \sigma_y^2 I\right]^{-1} y$$

$$\Sigma_{\ast} = K(X_{\ast}, X_{\ast}) - K(X_{\ast}, X)\left[K(X,X) + \sigma_y^2 I\right]^{-1} K(X, X_{\ast})$$

The project fits a GP to noisy samples of $\sin(x)$ and visualizes the posterior mean, 95% credible interval, and posterior sample paths, demonstrating uncertainty shrinkage near observed points.

## Task 3 — MCMC Inference Over Kernel Hyperparameters

Rather than fixing the RBF length scale $\ell$, it is treated as a random variable with a Gamma prior, and its posterior is targeted via a Metropolis–Hastings sampler operating on the log-marginal likelihood:

$$\log p(y \mid X, \ell) = -\frac{1}{2} y^\top K_\ell^{-1} y - \frac{1}{2} \log|K_\ell| - \frac{n}{2}\log(2\pi)$$

A symmetric Gaussian random-walk proposal is used, acceptance is computed in log-space for numerical stability, and the chain is run for 15,000 iterations with a 20% burn-in. Trace plots and posterior histograms are used to diagnose mixing and estimate the posterior mean and standard deviation of $\ell$.

## Task 4 — Langevin Dynamics (Unadjusted Langevin Algorithm)

A continuous-time Itô diffusion is defined so that its stationary distribution is exactly a target density $\pi(x)$:

$$dX_t = \frac{1}{2}\nabla \log \pi(X_t)\, dt + dW_t$$

This is discretized via the Euler–Maruyama scheme:

$$X_{k+1} = X_k + \frac{\epsilon}{2}\nabla \log \pi(X_k) + \sqrt{\epsilon}\, Z_k, \qquad Z_k \sim \mathcal{N}(0, I)$$

The sampler is applied to a bimodal Gaussian mixture target, using a numerically estimated score function. Over 100,000 steps (with a 20,000-step burn-in), the empirical sample distribution is shown to closely match the true bimodal density, validating the ULA as a gradient-based sampling method.

---

## Repository Structure

```
.
├── SPFinalProject.ipynb   # Main notebook (all tasks, derivations, and code)
└── README.md
```

## Requirements

- Python 3
- `numpy`
- `matplotlib`
- `scipy`

Install dependencies with:

```bash
pip install numpy matplotlib scipy
```

## Usage

Open and run the notebook top to bottom in Jupyter, JupyterLab, or Google Colab:

```bash
jupyter notebook SPFinalProject.ipynb
```

Each task is self-contained: it includes the relevant derivation in markdown, an implementation cell, a visualization, and a short results discussion.

## Key Concepts Demonstrated

- Multivariate Gaussian conditioning and the Gaussian Process framework
- Kernel design (stationary vs. non-stationary covariance structures)
- Bayesian nonparametric regression and predictive uncertainty quantification
- Metropolis–Hastings MCMC and log-space numerical stability
- Score-based sampling and Euler–Maruyama discretization of SDEs
- The connection between Langevin dynamics and the Fokker–Planck equation

## References

- Foreman-Mackey, D. et al. (2013). *emcee: The MCMC Hammer*. [arXiv:1202.3665](https://arxiv.org/abs/1202.3665)
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
