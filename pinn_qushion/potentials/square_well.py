"""Square well potentials."""

import jax.numpy as jnp

from .base import Potential


class InfiniteSquareWell(Potential):
    """Infinite square well potential.

    V(x) = 0 for |x| < width/2
    V(x) = V_max for |x| >= width/2

    Args:
        width: Total width of the well (centered at x=0)
        v_max: Large value representing "infinity" (default: 100.0)
    """

    def __init__(self, width: float = 4.0, v_max: float = 100.0):
        self.width = width
        self.v_max = v_max

    @property
    def name(self) -> str:
        return "infinite_square_well"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half_width = self.width / 2
        inside = jnp.abs(x) < half_width
        return jnp.where(inside, 0.0, self.v_max)


class FiniteSquareWell(Potential):
    """Finite square well potential.

    V(x) = -V0 for |x| < width/2
    V(x) = 0 for |x| >= width/2

    Args:
        width: Total width of the well (centered at x=0)
        depth: Depth of the well (V0 > 0)
    """

    def __init__(self, width: float = 4.0, depth: float = 5.0):
        self.width = width
        self.depth = depth

    @property
    def name(self) -> str:
        return "finite_square_well"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half_width = self.width / 2
        inside = jnp.abs(x) < half_width
        return jnp.where(inside, -self.depth, 0.0)
