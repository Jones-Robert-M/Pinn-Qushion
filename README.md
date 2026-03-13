---
title: Pinn-Qushion
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Pinn-Qushion

Interactive PINN solver for the 1D time-dependent Schrödinger equation.

## Features

- **5 Pre-trained Potentials**: Infinite/finite square well, harmonic oscillator, double well, Gaussian well
- **Real-time Visualization**: Watch quantum wavepackets evolve
- **Spectral Analysis**: Extract energy eigenvalues via FFT

## How It Works

This demo uses Physics-Informed Neural Networks (PINNs) trained to satisfy the Schrödinger equation:

$$i\hbar \frac{\partial \Psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2 \Psi}{\partial x^2} + V(x)\Psi$$

Each potential type has a dedicated pre-trained model that can predict wavefunction dynamics for any initial wavepacket configuration.

## Tech Stack

- **ML Framework**: JAX + Equinox
- **Web Interface**: Streamlit + Plotly
- **Signal Processing**: NumPy/SciPy

## Local Development

```bash
pip install -e ".[dev]"
streamlit run app.py
```

## License

MIT
