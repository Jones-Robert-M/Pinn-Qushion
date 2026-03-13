"""Collocation point sampling for PINN training."""

from typing import Tuple

import jax
import jax.numpy as jnp


class CollocationSampler:
    """Samples collocation points for PINN training.

    Args:
        x_range: Spatial domain (x_min, x_max)
        t_range: Temporal domain (t_min, t_max)
        x0_range: Initial position range for wavepacket
        k0_range: Initial momentum range for wavepacket
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-10, 10),
        t_range: Tuple[float, float] = (0, 20),
        x0_range: Tuple[float, float] = (-5, 5),
        k0_range: Tuple[float, float] = (-3, 3),
    ):
        self.x_range = x_range
        self.t_range = t_range
        self.x0_range = x0_range
        self.k0_range = k0_range

    def sample_interior(
        self, key: jax.Array, n_points: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample interior collocation points for physics residual.

        Uses uniform random sampling (Latin hypercube could be added later).
        """
        keys = jax.random.split(key, 4)

        x = jax.random.uniform(
            keys[0], (n_points,), minval=self.x_range[0], maxval=self.x_range[1]
        )
        t = jax.random.uniform(
            keys[1], (n_points,), minval=self.t_range[0], maxval=self.t_range[1]
        )
        x0 = jax.random.uniform(
            keys[2], (n_points,), minval=self.x0_range[0], maxval=self.x0_range[1]
        )
        k0 = jax.random.uniform(
            keys[3], (n_points,), minval=self.k0_range[0], maxval=self.k0_range[1]
        )

        return x, t, x0, k0

    def sample_initial(
        self, key: jax.Array, n_points: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample initial condition points (t=0)."""
        keys = jax.random.split(key, 3)

        x = jax.random.uniform(
            keys[0], (n_points,), minval=self.x_range[0], maxval=self.x_range[1]
        )
        t = jnp.zeros(n_points)
        x0 = jax.random.uniform(
            keys[1], (n_points,), minval=self.x0_range[0], maxval=self.x0_range[1]
        )
        k0 = jax.random.uniform(
            keys[2], (n_points,), minval=self.k0_range[0], maxval=self.k0_range[1]
        )

        return x, t, x0, k0

    def sample_boundary(
        self, key: jax.Array, n_points: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample boundary condition points (x = ±L)."""
        keys = jax.random.split(key, 4)

        # Half points at each boundary
        n_half = n_points // 2
        n_other = n_points - n_half

        x_left = jnp.full(n_half, self.x_range[0])
        x_right = jnp.full(n_other, self.x_range[1])
        x = jnp.concatenate([x_left, x_right])

        t = jax.random.uniform(
            keys[0], (n_points,), minval=self.t_range[0], maxval=self.t_range[1]
        )
        x0 = jax.random.uniform(
            keys[1], (n_points,), minval=self.x0_range[0], maxval=self.x0_range[1]
        )
        k0 = jax.random.uniform(
            keys[2], (n_points,), minval=self.k0_range[0], maxval=self.k0_range[1]
        )

        return x, t, x0, k0
