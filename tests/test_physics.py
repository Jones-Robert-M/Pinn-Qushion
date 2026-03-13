"""Physics validation tests for trained PINN models.

These tests verify that the trained models produce physically correct results
by comparing against known analytical solutions.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinn_qushion.analysis import compute_autocorrelation_series, compute_energy_spectrum, find_spectral_peaks
from pinn_qushion.inference import ModelManager, POTENTIAL_CONFIGS


@pytest.fixture
def model_manager():
    """Fixture providing model manager."""
    return ModelManager()


class TestInfiniteSquareWellPhysics:
    """Physics tests for infinite square well."""

    def test_energy_levels_proportional_to_n_squared(self, model_manager):
        """Energy levels should follow E_n ∝ n² pattern."""
        weight_file = model_manager.weights_dir / POTENTIAL_CONFIGS["infinite_square_well"]["weight_file"]
        if not weight_file.exists():
            pytest.skip("Model weights not found")

        model = model_manager.get_model("infinite_square_well")

        # Generate time evolution for a superposition state
        x = jnp.linspace(-10, 10, 256)
        t_values = jnp.linspace(0, 50, 512)
        dx = float(x[1] - x[0])
        dt = float(t_values[1] - t_values[0])

        # Initial state near center with some momentum
        x0, k0 = 0.0, 1.0
        n = len(x)

        # Get initial wavefunction
        x0_arr = jnp.full(n, x0)
        k0_arr = jnp.full(n, k0)
        t0_arr = jnp.zeros(n)
        psi_r_0, psi_i_0 = model.psi(x, t0_arr, x0_arr, k0_arr)
        psi_0 = psi_r_0 + 1j * psi_i_0

        # Time evolution
        psi_series = []
        for t in t_values:
            t_arr = jnp.full(n, t)
            psi_r, psi_i = model.psi(x, t_arr, x0_arr, k0_arr)
            psi_series.append(psi_r + 1j * psi_i)
        psi_series = jnp.stack(psi_series)

        # Compute autocorrelation and spectrum
        C_t = compute_autocorrelation_series(psi_0, psi_series, dx)
        energies, amplitudes = compute_energy_spectrum(C_t, dt)
        peak_energies, _ = find_spectral_peaks(energies, amplitudes, threshold_ratio=0.05)

        # For infinite square well with width L, E_n = n²π²/(2L²)
        # Check that ratios of consecutive peaks follow n² pattern
        if len(peak_energies) >= 3:
            # Ratios should be approximately 4/1, 9/4, 16/9, etc.
            E1 = peak_energies[0]
            for i, E_n in enumerate(peak_energies[1:4], start=2):
                expected_ratio = i**2
                actual_ratio = E_n / E1
                # Allow 20% tolerance for PINN approximation
                assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.20


class TestHarmonicOscillatorPhysics:
    """Physics tests for harmonic oscillator."""

    def test_equally_spaced_energy_levels(self, model_manager):
        """Harmonic oscillator should have equally spaced energy levels."""
        weight_file = model_manager.weights_dir / POTENTIAL_CONFIGS["harmonic_oscillator"]["weight_file"]
        if not weight_file.exists():
            pytest.skip("Model weights not found")

        model = model_manager.get_model("harmonic_oscillator")

        x = jnp.linspace(-10, 10, 256)
        t_values = jnp.linspace(0, 100, 1024)
        dx = float(x[1] - x[0])
        dt = float(t_values[1] - t_values[0])

        x0, k0 = 2.0, 0.5  # Off-center initial state
        n = len(x)

        x0_arr = jnp.full(n, x0)
        k0_arr = jnp.full(n, k0)
        t0_arr = jnp.zeros(n)
        psi_r_0, psi_i_0 = model.psi(x, t0_arr, x0_arr, k0_arr)
        psi_0 = psi_r_0 + 1j * psi_i_0

        psi_series = []
        for t in t_values:
            t_arr = jnp.full(n, t)
            psi_r, psi_i = model.psi(x, t_arr, x0_arr, k0_arr)
            psi_series.append(psi_r + 1j * psi_i)
        psi_series = jnp.stack(psi_series)

        C_t = compute_autocorrelation_series(psi_0, psi_series, dx)
        energies, amplitudes = compute_energy_spectrum(C_t, dt)
        peak_energies, _ = find_spectral_peaks(energies, amplitudes, threshold_ratio=0.05)

        # Check equal spacing: E_{n+1} - E_n should be constant (= ω in natural units)
        if len(peak_energies) >= 3:
            spacings = np.diff(peak_energies[:4])
            mean_spacing = np.mean(spacings)
            # All spacings should be within 20% of mean
            for spacing in spacings:
                assert abs(spacing - mean_spacing) / mean_spacing < 0.20


class TestNormalizationConservation:
    """Tests that probability is conserved during evolution."""

    @pytest.mark.parametrize("potential_name", list(POTENTIAL_CONFIGS.keys()))
    def test_probability_approximately_conserved(self, model_manager, potential_name):
        """Total probability should remain approximately constant."""
        weight_file = model_manager.weights_dir / POTENTIAL_CONFIGS[potential_name]["weight_file"]
        if not weight_file.exists():
            pytest.skip(f"Model weights not found for {potential_name}")

        model = model_manager.get_model(potential_name)

        x = jnp.linspace(-10, 10, 256)
        dx = float(x[1] - x[0])
        n = len(x)

        x0, k0 = 0.0, 1.0
        x0_arr = jnp.full(n, x0)
        k0_arr = jnp.full(n, k0)

        probabilities = []
        for t in [0.0, 5.0, 10.0, 15.0, 20.0]:
            t_arr = jnp.full(n, t)
            psi_r, psi_i = model.psi(x, t_arr, x0_arr, k0_arr)
            prob_density = psi_r**2 + psi_i**2
            total_prob = float(jnp.sum(prob_density) * dx)
            probabilities.append(total_prob)

        # Probabilities should all be similar (within 30% of each other)
        prob_array = np.array(probabilities)
        mean_prob = np.mean(prob_array)
        for p in prob_array:
            assert abs(p - mean_prob) / mean_prob < 0.30
