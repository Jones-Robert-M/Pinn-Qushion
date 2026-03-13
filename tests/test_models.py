"""Tests for neural network models."""

import jax
import jax.numpy as jnp
import pytest

from pinn_qushion.models.complex_mlp import ComplexMLP
from pinn_qushion.models.pinn import PINN
from pinn_qushion.potentials import HarmonicOscillator


class TestComplexMLP:
    """Tests for the two-headed complex MLP."""

    def test_output_shape(self):
        """Output has correct shape (N, 2) for real and imaginary parts."""
        key = jax.random.PRNGKey(0)
        model = ComplexMLP(
            input_dim=4,
            hidden_dim=64,
            num_layers=3,
            key=key,
        )

        x = jnp.ones((100, 4))
        output = model(x)

        assert output.shape == (100, 2)

    def test_output_is_finite(self):
        """Output contains no NaN or Inf values."""
        key = jax.random.PRNGKey(42)
        model = ComplexMLP(
            input_dim=4,
            hidden_dim=128,
            num_layers=5,
            key=key,
        )

        x = jax.random.normal(key, (50, 4))
        output = model(x)

        assert jnp.all(jnp.isfinite(output))

    def test_single_input(self):
        """Model works with single input (batch size 1)."""
        key = jax.random.PRNGKey(1)
        model = ComplexMLP(
            input_dim=4,
            hidden_dim=64,
            num_layers=3,
            key=key,
        )

        x = jnp.ones((1, 4))
        output = model(x)

        assert output.shape == (1, 2)

    def test_jit_compatible(self):
        """Model can be JIT compiled."""
        key = jax.random.PRNGKey(2)
        model = ComplexMLP(
            input_dim=4,
            hidden_dim=64,
            num_layers=3,
            key=key,
        )

        @jax.jit
        def forward(m, x):
            return m(x)

        x = jnp.ones((10, 4))
        output = forward(model, x)

        assert output.shape == (10, 2)

    def test_different_architectures(self):
        """Model works with different layer configurations."""
        key = jax.random.PRNGKey(3)

        for num_layers in [3, 4, 5, 6]:
            for hidden_dim in [32, 64, 128, 256]:
                model = ComplexMLP(
                    input_dim=4,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    key=key,
                )
                x = jnp.ones((5, 4))
                output = model(x)
                assert output.shape == (5, 2)


class TestPINN:
    """Tests for the PINN wrapper."""

    def test_psi_output_shape(self):
        """psi method returns complex wavefunction components."""
        key = jax.random.PRNGKey(0)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(
            potential=potential,
            hidden_dim=64,
            num_layers=3,
            key=key,
        )

        x = jnp.linspace(-5, 5, 50)
        t = jnp.ones(50) * 0.5
        x0 = jnp.zeros(50)
        k0 = jnp.ones(50) * 2.0

        psi_r, psi_i = model.psi(x, t, x0, k0)

        assert psi_r.shape == (50,)
        assert psi_i.shape == (50,)

    def test_psi_derivatives(self):
        """Can compute derivatives of psi with respect to x and t."""
        key = jax.random.PRNGKey(1)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(
            potential=potential,
            hidden_dim=64,
            num_layers=3,
            key=key,
        )

        x = jnp.array([0.0, 1.0, 2.0])
        t = jnp.array([0.1, 0.1, 0.1])
        x0 = jnp.zeros(3)
        k0 = jnp.ones(3)

        # Should not raise
        dpsi_dx = model.psi_x(x, t, x0, k0)
        dpsi_dt = model.psi_t(x, t, x0, k0)
        d2psi_dx2 = model.psi_xx(x, t, x0, k0)

        assert dpsi_dx[0].shape == (3,)  # real part
        assert dpsi_dt[0].shape == (3,)
        assert d2psi_dx2[0].shape == (3,)

    def test_probability_density(self):
        """Probability density is non-negative."""
        key = jax.random.PRNGKey(2)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(
            potential=potential,
            hidden_dim=64,
            num_layers=3,
            key=key,
        )

        x = jnp.linspace(-5, 5, 100)
        t = jnp.ones(100) * 0.5
        x0 = jnp.zeros(100)
        k0 = jnp.ones(100)

        prob = model.probability_density(x, t, x0, k0)

        assert jnp.all(prob >= 0)
        assert prob.shape == (100,)
