"""Double well potential."""

import jax.numpy as jnp

from .base import Potential


class DoubleWell(Potential):
    """Double well (quartic) potential.

    V(x) = a * (x^2 - b^2)^2 - c

    where parameters are derived from separation, depth, and barrier height.

    Args:
        separation: Distance between the two minima
        depth: Depth of each well (from zero reference)
        barrier: Height of the central barrier (from well bottoms)
    """

    def __init__(
        self, separation: float = 4.0, depth: float = 5.0, barrier: float = 3.0
    ):
        self.separation = separation
        self.depth = depth
        self.barrier = barrier

        # Derive quartic parameters: V(x) = a*(x^2 - b^2)^2 + c
        # Minima at x = ±b where b = separation/2
        # V(±b) = c = -depth
        # V(0) = a*b^4 + c = barrier - depth
        # So: a*b^4 = barrier, thus a = barrier / b^4
        self.b = separation / 2
        self.a = barrier / (self.b**4) if self.b > 0 else 1.0
        self.c = -depth

    @property
    def name(self) -> str:
        return "double_well"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.a * (x**2 - self.b**2) ** 2 + self.c
