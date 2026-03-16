---
title: Pinn-Qushion
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Pinn-Qushion

[![CI](https://github.com/rjones/pinn-qushion/actions/workflows/ci.yml/badge.svg)](https://github.com/rjones/pinn-qushion/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Physics-Informed Neural Network (PINN) solver for the 1D time-dependent Schrodinger equation with interactive web visualization.

## Overview

Pinn-Qushion uses deep learning to solve quantum mechanical wavepacket dynamics in real-time. Each potential type has a dedicated pre-trained neural network that satisfies the Schrodinger equation:

```
i * d/dt Psi(x,t) = [-1/2 * d^2/dx^2 + V(x)] Psi(x,t)
```

The models are trained using physics-informed loss functions that enforce the PDE residual, initial conditions, and boundary conditions.

## Features

**Five Pre-trained Potential Types**
- Infinite square well
- Finite square well
- Harmonic oscillator
- Double well (quartic)
- Gaussian well (quantum dot analog)

**Real-time Visualization**
- Probability density evolution
- Wavefunction phase dynamics
- Interactive parameter controls

**Signal Processing**
- Autocorrelation function computation
- FFT-based energy spectrum extraction
- Eigenvalue identification

## Project Structure

```
pinn_qushion/
    potentials/     # Potential energy functions (5 types)
    models/         # Neural network architecture (ComplexMLP, PINN wrapper)
    training/       # Training infrastructure (sampler, loss, trainer)
    analysis/       # Signal processing (autocorrelation, spectrum)
    inference.py    # Model loading and prediction
scripts/
    train_all.py    # Training script with loss curve logging
    evaluate_models.py  # Model diagnostics and visualization
tests/
    test_potentials.py  # Unit tests for potentials
    test_models.py      # Unit tests for neural networks
    test_training.py    # Unit tests for training components
    test_physics.py     # Physics validation tests
    test_analysis.py    # Signal processing tests
app.py              # Streamlit web interface
```

## Installation

```bash
# Clone the repository
git clone https://github.com/rjones/pinn-qushion.git
cd pinn-qushion

# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

**Run the web interface:**
```bash
streamlit run app.py
```

**Train models from scratch:**
```bash
python scripts/train_all.py --iterations 100000
```

**Train a specific potential:**
```bash
python scripts/train_all.py --potentials harmonic_oscillator --iterations 50000
```

**Run tests:**
```bash
pytest tests/
```

## Technical Details

**Neural Network Architecture**
- Two-headed MLP with shared trunk (5 layers, 128 neurons)
- Separate output heads for real and imaginary wavefunction components
- Built with JAX and Equinox for JIT compilation

**Training**
- Physics residual loss (Schrodinger equation)
- Initial condition loss (Gaussian wavepacket)
- Boundary condition loss (Dirichlet)
- Adam optimizer with cosine learning rate decay
- Gradient clipping for numerical stability

**Domain**
- Spatial: x in [-10, 10]
- Temporal: t in [0, 20]
- Initial position: x0 in [-5, 5]
- Initial momentum: k0 in [-3, 3]

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | JAX + Equinox |
| Optimization | Optax |
| Web Interface | Streamlit |
| Visualization | Plotly |
| Signal Processing | NumPy, SciPy |
| Testing | pytest |
| Linting | Ruff |

## Development

```bash
# Run linter
ruff check .

# Run tests with coverage
pytest --cov=pinn_qushion tests/

# Format code
ruff format .
```

## License

MIT
