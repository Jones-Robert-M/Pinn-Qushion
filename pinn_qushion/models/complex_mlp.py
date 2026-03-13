"""Two-headed MLP for complex wavefunction output."""

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp


class ComplexMLP(eqx.Module):
    """Two-headed MLP that outputs real and imaginary parts of wavefunction.

    Architecture:
        - Shared trunk: N fully-connected layers with tanh activation
        - Two output heads: separate linear layers for Psi_R and Psi_I

    Args:
        input_dim: Dimension of input (typically 4: x, t, x0, k0)
        hidden_dim: Width of hidden layers
        num_layers: Number of hidden layers in the trunk
        key: JAX random key for initialization
    """

    trunk: List[eqx.nn.Linear]
    head_real: eqx.nn.Linear
    head_imag: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        key: jax.Array,
    ):
        keys = jax.random.split(key, num_layers + 2)

        # Build trunk layers
        self.trunk = []
        in_features = input_dim
        for i in range(num_layers):
            self.trunk.append(
                eqx.nn.Linear(in_features, hidden_dim, key=keys[i])
            )
            in_features = hidden_dim

        # Output heads
        self.head_real = eqx.nn.Linear(hidden_dim, 1, key=keys[-2])
        self.head_imag = eqx.nn.Linear(hidden_dim, 1, key=keys[-1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Output tensor of shape (batch, 2) where [:, 0] is real part
            and [:, 1] is imaginary part.
        """
        # Trunk forward pass with tanh activation
        # Use vmap to handle batch dimension
        def single_forward(xi):
            h = xi
            for layer in self.trunk:
                h = jnp.tanh(layer(h))
            psi_real = self.head_real(h)
            psi_imag = self.head_imag(h)
            return jnp.concatenate([psi_real, psi_imag], axis=-1)

        return jax.vmap(single_forward)(x)
