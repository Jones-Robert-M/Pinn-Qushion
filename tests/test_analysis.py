"""Tests for signal processing and spectral analysis."""

import jax.numpy as jnp
import numpy as np
import pytest

from pinn_qushion.analysis.autocorrelation import compute_autocorrelation
from pinn_qushion.analysis.spectrum import compute_energy_spectrum


class TestAutocorrelation:
    """Tests for autocorrelation function."""

    def test_c0_is_one_for_normalized(self):
        """C(0) = 1 for normalized wavefunction."""
        # Create a simple Gaussian initial state
        x = jnp.linspace(-10, 10, 256)
        dx = x[1] - x[0]

        psi_0 = jnp.exp(-x**2 / 2)
        psi_0 = psi_0 / jnp.sqrt(jnp.sum(jnp.abs(psi_0)**2) * dx)

        # At t=0, psi(t) = psi_0
        psi_t = psi_0

        C = compute_autocorrelation(psi_0, psi_t, dx)

        assert jnp.abs(C - 1.0) < 0.01

    def test_returns_complex(self):
        """Autocorrelation can be complex."""
        x = jnp.linspace(-10, 10, 256)
        dx = x[1] - x[0]

        psi_0 = jnp.exp(-x**2 / 2)
        psi_0 = psi_0 / jnp.sqrt(jnp.sum(jnp.abs(psi_0)**2) * dx)

        # Phase-shifted state
        psi_t = psi_0 * jnp.exp(1j * jnp.pi / 4)

        C = compute_autocorrelation(psi_0, psi_t, dx)

        # Should have non-zero imaginary part
        assert jnp.abs(jnp.imag(C)) > 0.1

    def test_orthogonal_states_zero(self):
        """Orthogonal states have zero autocorrelation."""
        x = jnp.linspace(-10, 10, 256)
        dx = x[1] - x[0]

        # Ground state of harmonic oscillator (even)
        psi_0 = jnp.exp(-x**2 / 2)
        psi_0 = psi_0 / jnp.sqrt(jnp.sum(jnp.abs(psi_0)**2) * dx)

        # First excited state (odd)
        psi_1 = x * jnp.exp(-x**2 / 2)
        psi_1 = psi_1 / jnp.sqrt(jnp.sum(jnp.abs(psi_1)**2) * dx)

        C = compute_autocorrelation(psi_0, psi_1, dx)

        assert jnp.abs(C) < 0.01


class TestEnergySpectrum:
    """Tests for energy spectrum extraction."""

    def test_returns_frequencies_and_amplitudes(self):
        """Spectrum returns energy values and amplitudes."""
        # Create a simple oscillating autocorrelation
        t = jnp.linspace(0, 20, 256)
        dt = t[1] - t[0]

        # Single frequency oscillation
        omega = 2.0
        C_t = jnp.exp(-1j * omega * t)

        energies, amplitudes = compute_energy_spectrum(C_t, dt)

        assert energies.shape[0] > 0
        assert amplitudes.shape == energies.shape

    def test_peak_at_correct_frequency(self):
        """Spectrum has peak at the correct energy."""
        t = jnp.linspace(0, 50, 512)
        dt = t[1] - t[0]

        # Known frequency
        E0 = 1.5
        C_t = jnp.exp(-1j * E0 * t)

        energies, amplitudes = compute_energy_spectrum(C_t, dt)

        # Find peak
        peak_idx = jnp.argmax(amplitudes)
        peak_energy = energies[peak_idx]

        # Should be close to E0
        assert jnp.abs(peak_energy - E0) < 0.2

    def test_multiple_frequencies(self):
        """Spectrum shows multiple peaks for superposition state."""
        t = jnp.linspace(0, 100, 1024)
        dt = t[1] - t[0]

        # Two frequencies
        E1, E2 = 1.0, 3.0
        C_t = 0.5 * jnp.exp(-1j * E1 * t) + 0.5 * jnp.exp(-1j * E2 * t)

        energies, amplitudes = compute_energy_spectrum(C_t, dt)

        # Should have at least 2 significant peaks
        threshold = 0.1 * jnp.max(amplitudes)
        n_peaks = jnp.sum(amplitudes > threshold)

        assert n_peaks >= 2

    def test_amplitudes_non_negative(self):
        """Spectrum amplitudes are non-negative."""
        t = jnp.linspace(0, 20, 256)
        dt = t[1] - t[0]
        C_t = jnp.exp(-1j * 2.0 * t) * jnp.exp(-t / 10)

        energies, amplitudes = compute_energy_spectrum(C_t, dt)

        assert jnp.all(amplitudes >= 0)
