"""Physics-Informed Neural Network for Schrödinger equation."""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from pinn_qushion.potentials.base import Potential

from .complex_mlp import ComplexMLP


class PINN(eqx.Module):
    """PINN for solving the 1D time-dependent Schrödinger equation.

    Wraps a ComplexMLP and provides methods for computing the wavefunction
    and its derivatives needed for the physics residual.

    Args:
        potential: The quantum potential V(x)
        hidden_dim: Width of hidden layers
        num_layers: Number of hidden layers
        key: JAX random key
    """

    net: ComplexMLP
    potential: Potential

    def __init__(
        self,
        potential: Potential,
        hidden_dim: int = 128,
        num_layers: int = 5,
        key: jax.Array = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        self.potential = potential
        self.net = ComplexMLP(
            input_dim=4,  # x, t, x0, k0
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            key=key,
        )

    def _forward(
        self, x: jnp.ndarray, t: jnp.ndarray, x0: jnp.ndarray, k0: jnp.ndarray
    ) -> jnp.ndarray:
        """Raw network forward pass."""
        inputs = jnp.stack([x, t, x0, k0], axis=-1)
        return self.net(inputs)

    def psi(
        self, x: jnp.ndarray, t: jnp.ndarray, x0: jnp.ndarray, k0: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute wavefunction Psi(x, t; x0, k0).

        Returns:
            Tuple of (psi_real, psi_imag)
        """
        out = self._forward(x, t, x0, k0)
        return out[:, 0], out[:, 1]

    def _psi_scalar(
        self, x: float, t: float, x0: float, k0: float
    ) -> Tuple[float, float]:
        """Scalar version for differentiation."""
        inputs = jnp.array([[x, t, x0, k0]])
        out = self.net(inputs)
        return out[0, 0], out[0, 1]

    def psi_x(
        self, x: jnp.ndarray, t: jnp.ndarray, x0: jnp.ndarray, k0: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute d(Psi)/dx."""
        grad_fn = jax.grad(lambda x_, t_, x0_, k0_: self._psi_scalar(x_, t_, x0_, k0_)[0])
        grad_fn_i = jax.grad(lambda x_, t_, x0_, k0_: self._psi_scalar(x_, t_, x0_, k0_)[1])

        dpsi_r_dx = jax.vmap(grad_fn)(x, t, x0, k0)
        dpsi_i_dx = jax.vmap(grad_fn_i)(x, t, x0, k0)

        return dpsi_r_dx, dpsi_i_dx

    def psi_t(
        self, x: jnp.ndarray, t: jnp.ndarray, x0: jnp.ndarray, k0: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute d(Psi)/dt."""
        grad_fn = jax.grad(lambda x_, t_, x0_, k0_: self._psi_scalar(x_, t_, x0_, k0_)[0], argnums=1)
        grad_fn_i = jax.grad(lambda x_, t_, x0_, k0_: self._psi_scalar(x_, t_, x0_, k0_)[1], argnums=1)

        dpsi_r_dt = jax.vmap(grad_fn)(x, t, x0, k0)
        dpsi_i_dt = jax.vmap(grad_fn_i)(x, t, x0, k0)

        return dpsi_r_dt, dpsi_i_dt

    def psi_xx(
        self, x: jnp.ndarray, t: jnp.ndarray, x0: jnp.ndarray, k0: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute d²(Psi)/dx²."""
        def second_deriv_r(x_, t_, x0_, k0_):
            first = jax.grad(lambda x__: self._psi_scalar(x__, t_, x0_, k0_)[0])
            return jax.grad(first)(x_)

        def second_deriv_i(x_, t_, x0_, k0_):
            first = jax.grad(lambda x__: self._psi_scalar(x__, t_, x0_, k0_)[1])
            return jax.grad(first)(x_)

        d2psi_r_dx2 = jax.vmap(second_deriv_r)(x, t, x0, k0)
        d2psi_i_dx2 = jax.vmap(second_deriv_i)(x, t, x0, k0)

        return d2psi_r_dx2, d2psi_i_dx2

    def probability_density(
        self, x: jnp.ndarray, t: jnp.ndarray, x0: jnp.ndarray, k0: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute |Psi|² = Psi_R² + Psi_I²."""
        psi_r, psi_i = self.psi(x, t, x0, k0)
        return psi_r**2 + psi_i**2
