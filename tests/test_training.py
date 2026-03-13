"""Tests for training infrastructure."""

import jax
import jax.numpy as jnp
import optax
import pytest

from pinn_qushion.models import PINN
from pinn_qushion.potentials import HarmonicOscillator
from pinn_qushion.training.loss import PINNLoss
from pinn_qushion.training.sampler import CollocationSampler
from pinn_qushion.training.trainer import Trainer


class TestCollocationSampler:
    """Tests for collocation point sampling."""

    def test_sample_shape(self):
        """Sampler returns correct number of points."""
        sampler = CollocationSampler(
            x_range=(-10, 10),
            t_range=(0, 20),
            x0_range=(-5, 5),
            k0_range=(-3, 3),
        )

        key = jax.random.PRNGKey(0)
        x, t, x0, k0 = sampler.sample_interior(key, n_points=1000)

        assert x.shape == (1000,)
        assert t.shape == (1000,)
        assert x0.shape == (1000,)
        assert k0.shape == (1000,)

    def test_sample_ranges(self):
        """Sampled points are within specified ranges."""
        sampler = CollocationSampler(
            x_range=(-10, 10),
            t_range=(0, 20),
            x0_range=(-5, 5),
            k0_range=(-3, 3),
        )

        key = jax.random.PRNGKey(1)
        x, t, x0, k0 = sampler.sample_interior(key, n_points=5000)

        assert jnp.all(x >= -10) and jnp.all(x <= 10)
        assert jnp.all(t >= 0) and jnp.all(t <= 20)
        assert jnp.all(x0 >= -5) and jnp.all(x0 <= 5)
        assert jnp.all(k0 >= -3) and jnp.all(k0 <= 3)

    def test_sample_initial_condition(self):
        """Initial condition sampler returns t=0."""
        sampler = CollocationSampler(
            x_range=(-10, 10),
            t_range=(0, 20),
            x0_range=(-5, 5),
            k0_range=(-3, 3),
        )

        key = jax.random.PRNGKey(2)
        x, t, x0, k0 = sampler.sample_initial(key, n_points=500)

        assert jnp.allclose(t, 0.0)
        assert x.shape == (500,)

    def test_sample_boundary(self):
        """Boundary sampler returns x at boundaries."""
        sampler = CollocationSampler(
            x_range=(-10, 10),
            t_range=(0, 20),
            x0_range=(-5, 5),
            k0_range=(-3, 3),
        )

        key = jax.random.PRNGKey(3)
        x, t, x0, k0 = sampler.sample_boundary(key, n_points=500)

        # All x values should be at ±10
        assert jnp.all((jnp.abs(x + 10) < 1e-6) | (jnp.abs(x - 10) < 1e-6))


class TestPINNLoss:
    """Tests for PINN loss functions."""

    def test_loss_is_scalar(self):
        """Total loss is a scalar value."""
        key = jax.random.PRNGKey(0)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(potential=potential, hidden_dim=32, num_layers=2, key=key)

        loss_fn = PINNLoss(sigma=1.0)

        # Sample points
        x = jnp.linspace(-5, 5, 50)
        t = jnp.ones(50) * 0.5
        x0 = jnp.zeros(50)
        k0 = jnp.ones(50)

        loss = loss_fn.physics_loss(model, x, t, x0, k0)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_ic_loss_is_scalar(self):
        """Initial condition loss is a scalar."""
        key = jax.random.PRNGKey(1)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(potential=potential, hidden_dim=32, num_layers=2, key=key)

        loss_fn = PINNLoss(sigma=1.0)

        x = jnp.linspace(-5, 5, 50)
        t = jnp.zeros(50)
        x0 = jnp.zeros(50)
        k0 = jnp.ones(50)

        loss = loss_fn.initial_condition_loss(model, x, t, x0, k0)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_bc_loss_is_scalar(self):
        """Boundary condition loss is a scalar."""
        key = jax.random.PRNGKey(2)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(potential=potential, hidden_dim=32, num_layers=2, key=key)

        loss_fn = PINNLoss(sigma=1.0)

        x = jnp.array([-10.0] * 25 + [10.0] * 25)
        t = jnp.linspace(0, 20, 50)
        x0 = jnp.zeros(50)
        k0 = jnp.ones(50)

        loss = loss_fn.boundary_condition_loss(model, x, t, x0, k0)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_total_loss_combines_all(self):
        """Total loss combines physics, IC, and BC losses."""
        key = jax.random.PRNGKey(3)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(potential=potential, hidden_dim=32, num_layers=2, key=key)

        loss_fn = PINNLoss(
            sigma=1.0,
            lambda_phys=1.0,
            lambda_ic=10.0,
            lambda_bc=10.0,
        )

        # Interior points
        x_int = jnp.linspace(-5, 5, 30)
        t_int = jnp.ones(30) * 5.0
        x0_int = jnp.zeros(30)
        k0_int = jnp.ones(30)

        # IC points
        x_ic = jnp.linspace(-5, 5, 20)
        t_ic = jnp.zeros(20)
        x0_ic = jnp.zeros(20)
        k0_ic = jnp.ones(20)

        # BC points
        x_bc = jnp.array([-10.0] * 10 + [10.0] * 10)
        t_bc = jnp.linspace(0, 20, 20)
        x0_bc = jnp.zeros(20)
        k0_bc = jnp.ones(20)

        total = loss_fn.total_loss(
            model,
            x_int, t_int, x0_int, k0_int,
            x_ic, t_ic, x0_ic, k0_ic,
            x_bc, t_bc, x0_bc, k0_bc,
        )

        assert total.shape == ()
        assert jnp.isfinite(total)


class TestTrainer:
    """Tests for the training loop."""

    def test_trainer_initializes(self):
        """Trainer initializes with model and optimizer."""
        key = jax.random.PRNGKey(0)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(potential=potential, hidden_dim=32, num_layers=2, key=key)

        optimizer = optax.adam(1e-3)
        trainer = Trainer(model=model, optimizer=optimizer, sigma=1.0)

        assert trainer is not None

    def test_trainer_step_reduces_loss(self):
        """A training step should generally reduce or maintain loss."""
        key = jax.random.PRNGKey(1)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(potential=potential, hidden_dim=32, num_layers=2, key=key)

        optimizer = optax.adam(1e-3)
        trainer = Trainer(model=model, optimizer=optimizer, sigma=1.0)

        sampler = CollocationSampler()

        # Get initial loss
        key, subkey = jax.random.split(key)
        x_int, t_int, x0_int, k0_int = sampler.sample_interior(subkey, 100)
        key, subkey = jax.random.split(key)
        x_ic, t_ic, x0_ic, k0_ic = sampler.sample_initial(subkey, 50)
        key, subkey = jax.random.split(key)
        x_bc, t_bc, x0_bc, k0_bc = sampler.sample_boundary(subkey, 50)

        initial_loss = trainer.compute_loss(
            x_int, t_int, x0_int, k0_int,
            x_ic, t_ic, x0_ic, k0_ic,
            x_bc, t_bc, x0_bc, k0_bc,
        )

        # Run a few training steps
        for _ in range(10):
            key, subkey = jax.random.split(key)
            x_int, t_int, x0_int, k0_int = sampler.sample_interior(subkey, 100)
            key, subkey = jax.random.split(key)
            x_ic, t_ic, x0_ic, k0_ic = sampler.sample_initial(subkey, 50)
            key, subkey = jax.random.split(key)
            x_bc, t_bc, x0_bc, k0_bc = sampler.sample_boundary(subkey, 50)

            trainer.step(
                x_int, t_int, x0_int, k0_int,
                x_ic, t_ic, x0_ic, k0_ic,
                x_bc, t_bc, x0_bc, k0_bc,
            )

        final_loss = trainer.compute_loss(
            x_int, t_int, x0_int, k0_int,
            x_ic, t_ic, x0_ic, k0_ic,
            x_bc, t_bc, x0_bc, k0_bc,
        )

        # Loss should decrease or stay similar (not increase dramatically)
        assert final_loss < initial_loss * 1.5

    def test_get_model_returns_updated_model(self):
        """get_model returns the current trained model."""
        key = jax.random.PRNGKey(2)
        potential = HarmonicOscillator(omega=1.0)
        model = PINN(potential=potential, hidden_dim=32, num_layers=2, key=key)

        optimizer = optax.adam(1e-3)
        trainer = Trainer(model=model, optimizer=optimizer, sigma=1.0)

        returned_model = trainer.get_model()

        # Should be a PINN instance
        assert isinstance(returned_model, PINN)
