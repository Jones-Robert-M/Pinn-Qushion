---
title: Pinn-Qushion
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
---

# Pinn-Qushion

[![CI](https://github.com/Jones-Robert-M/pinn-qushion/actions/workflows/ci.yml/badge.svg)](https://github.com/Jones-Robert-M/pinn-qushion/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Pinn-Qushion trains a neural network to solve the 1D time-dependent Schrödinger equation for a range of quantum potentials, then deploys that network as an interactive web application. Rather than running a numerical solver at query time, the network learns the solution operator: given an initial position x₀, momentum k₀, and time t, it returns the wavefunction Ψ(x,t) directly. The result is sub-millisecond inference on arbitrary initial conditions — at the cost of an offline training problem that turns out to be genuinely difficult to get right.

**[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/JonesRobM/pinn-qushion)**

---

## The physics

The time-dependent Schrödinger equation (TDSE) in natural units (ℏ = m = 1) is:

```text
i ∂Ψ/∂t = [ -½ ∂²/∂x² + V(x) ] Ψ
```

The left side is the time derivative of the quantum state. The right side is the Hamiltonian: kinetic energy plus potential energy. Given an initial wavepacket Ψ(x, 0), this equation determines how the probability distribution |Ψ(x,t)|² evolves.

The initial state is a Gaussian wavepacket centred at x₀ with momentum k₀:

```text
Ψ₀(x) = (1/2πσ²)^(1/4) · exp(-(x-x₀)²/4σ²) · exp(ik₀x)
```

A critical physical constraint is norm conservation: the TDSE is unitary, so ∫|Ψ|²dx = 1 must hold for all t. A wavefunction that loses norm is not a valid quantum state — it is the primary quality metric for the trained models.

Five potential types are supported, covering a range of physically distinct behaviours:

| Potential | Parameters | Physical interest |
| --- | --- | --- |
| Harmonic oscillator | ω = 1.0 | Analytically solvable; wavepacket oscillates without spreading |
| Infinite square well | width = 8.0 | Hard-wall confinement; discrete energy levels |
| Finite square well | width = 6.0, depth = 5.0 | Wavefunction leaks into classically forbidden region |
| Double well | separation = 4.0, depth = 5.0, barrier = 3.0 | Quantum tunnelling between two minima |
| Gaussian well | depth = 5.0, σ = 2.0 | Smooth confinement; quantum dot analogue |

---

## The approach: Physics-Informed Neural Networks

A Physics-Informed Neural Network (PINN) is trained not just to fit data but to satisfy a differential equation. The training loss has four components:

**Physics residual** — the mean squared deviation from the TDSE at randomly sampled collocation points (x, t, x₀, k₀). This is the core constraint that teaches the network the equation of motion.

**Initial condition** — mean squared error against the analytic Gaussian wavepacket at t = 0. This anchors the solution to a specific physical starting state.

**Boundary conditions** — Dirichlet conditions Ψ(±L, t) = 0 enforced at the domain edges.

**Norm conservation** — penalty for |∫|Ψ|²dx − 1| at sampled time points. This is not derivable from the PDE residual alone; without it, the trivial solution Ψ = 0 satisfies the first three losses exactly.

The last point is worth emphasis. The zero solution is a local attractor during training: it satisfies the PDE (trivially), satisfies the boundary conditions (trivially), and costs nothing on the physics residual. The only loss that rules it out is the norm constraint. Getting the gradient balance right — so that norm enforcement wins over the zero-attractor pull — is the central training challenge in this project.

**Curriculum training** addresses this by separating the problem into two phases. Phase 1 (IC-only, ~20k steps) trains the network to reproduce the initial wavepacket before the physics loss is introduced. Phase 2 (full loss, ~180k+ steps) then introduces the PDE residual while the norm constraint is enforced at multiple fixed time points (t = 0, 2, 5, 10, 20) each step. The exponential sampling of the random norm time concentrates the constraint near small t where decay begins.

---

## Architecture

The network is a two-headed MLP built with [Equinox](https://github.com/patrick-kidger/equinox):

```text
Input: (x, t, x₀, k₀)  — 4 scalars per collocation point
         ↓
Shared trunk: 5 × Linear(128) + tanh
         ↓
   ┌─────┴─────┐
head_real    head_imag
   │              │
Ψ_R(x,t)    Ψ_I(x,t)
```

The real and imaginary components of the wavefunction are output separately. The potential V(x) is baked into the model at training time — each potential type has its own trained weights. During training, derivatives ∂Ψ/∂t and ∂²Ψ/∂x² are computed via JAX automatic differentiation through the network.

Input dimension is 4: position x, time t, initial position x₀, initial momentum k₀. The last two make the network a solution operator over a family of initial conditions, not just a single trajectory.

---

## Training

Models are trained with the following configuration:

```bash
.venv/bin/python scripts/train_all.py \
  --potentials harmonic_oscillator \
  --iterations 200000 \
  --lambda-phys 10 --lambda-ic 100 --lambda-bc 10 --lambda-norm 1000 \
  --curriculum-ic-steps 20000 \
  --norm-late-times 0.0 2.0 5.0 10.0 20.0
```

| Hyperparameter | Value | Rationale |
| --- | --- | --- |
| λ_phys | 10 | Physics residual over 10k collocation points |
| λ_ic | 100 | Strong IC enforcement during and after curriculum |
| λ_norm | 1000 | Must dominate gradient competition against zero attractor |
| Curriculum IC steps | 20k | Nail the initial condition before introducing physics |
| Optimizer | Adam + cosine decay | LR: 1e-3 → 1e-5 over full run |
| Gradient clipping | global norm 1.0 | Stability for complex autodiff |

Training is logged to `training_logs/` with per-step loss breakdowns (physics, IC, BC, norm, curriculum phase). Loss curves are saved as PNGs.

---

## Results

*Norm conservation is the current active training challenge — see the Training section for details. The qualitative results below are expected once training converges.*

The app presents four analysis panels alongside the wavefunction animation:

**Survival probability** |C(t)| = |∫Ψ*(x,0)Ψ(x,t)dx| measures how much the evolving state overlaps with its initial configuration. For a harmonic oscillator it should show periodic recurrences at the classical period T = 2π/ω ≈ 6.28.

**Excitation spectrum** |FFT(C(t))| extracts the energy eigenvalues excited by the initial wavepacket. Since C(t) = Σₙ |cₙ|² e^{-iEₙt}, its Fourier transform peaks at the eigenvalues Eₙ weighted by the overlap coefficients |cₙ|². For the harmonic oscillator the analytic eigenvalues are Eₙ = ω(n + ½) = 0.5, 1.5, 2.5, ... — when training is correct, the spectrum peaks align with these values without being told them explicitly during training. Only levels with significant amplitude are annotated.

**Expectation values** ⟨x⟩(t) and ⟨p⟩(t) on a dual-axis plot show the mean position and mean momentum trajectories. These are computed directly from the probability density and wavefunction gradients — no additional model calls. For a harmonic oscillator, both should oscillate sinusoidally at frequency ω, 90° out of phase, tracing an ellipse in phase space (a coherent state). This is one of the most directly interpretable physics checks: if the neural network has correctly learned the harmonic oscillator, the position trajectory should look like a clean sine wave.

**Norm conservation** ∫|Ψ|²dx should remain at 1.0 for all time. Deviation from this is the primary model quality metric and is shown explicitly so the model's current accuracy is visible to the user.

---

## Running locally

```bash
# Clone and install
git clone https://github.com/Jones-Robert-M/pinn-qushion.git
cd pinn-qushion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the app (uses pre-trained weights from weights/)
streamlit run app.py

# Train a model from scratch
.venv/bin/python scripts/train_all.py \
  --potentials harmonic_oscillator \
  --iterations 200000 \
  --lambda-phys 10 --lambda-ic 100 --lambda-norm 1000 \
  --curriculum-ic-steps 20000

# Run tests
pytest
```

Pre-trained weights are not stored in this repository. They are hosted on [Hugging Face](https://huggingface.co/JonesRobM/pinn-qushion-weights) and fetched automatically by the CI deploy pipeline. To run the app locally with trained models, either train from scratch or download weights manually and place them in `weights/`.

---

## Repository structure

```text
pinn_qushion/
    models/         ComplexMLP architecture, PINN wrapper
    potentials/     Five potential energy functions
    training/       Loss functions, collocation sampler, training loop
    analysis/       Autocorrelation and energy spectrum via FFT
    inference.py    ModelManager — weight loading and prediction
scripts/
    train_all.py        Training with curriculum, logging, checkpointing
    evaluate_models.py  Post-training diagnostic plots
    upload_weights.py   Upload production weights to Hugging Face
.github/workflows/
    ci.yml          Test → lint → deploy to Hugging Face Spaces
tests/
    test_potentials.py  Unit tests for all potential functions
    test_models.py      Network architecture and forward pass tests
    test_physics.py     PDE residual and IC physics validation
    test_training.py    Loss function and sampler tests
    test_analysis.py    Autocorrelation and spectrum tests
app.py              Streamlit interface
```

---

## Tech stack

| Component | Technology |
| --- | --- |
| ML framework | JAX + Equinox |
| Optimisation | Optax |
| Web interface | Streamlit |
| Visualisation | Plotly |
| Signal processing | NumPy, SciPy |
| Model hosting | Hugging Face Spaces + Model Hub |
| CI/CD | GitHub Actions |
| Linting | Ruff |

---

## License

MIT
