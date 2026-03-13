"""Gaussian well potential."""

import jax.numpy as jnp

from .base import Potential


class GaussianWell(Potential):
    """Gaussian well potential (quantum dot analog).

    V(x) = -depth * exp(-x^2 / (2 * sigma^2))

    Args:
        depth: Depth of the well (positive value)
        sigma: Width parameter of the Gaussian
    """

    def __init__(self, depth: float = 5.0, sigma: float = 1.0):
        self.depth = depth
        self.sigma = sigma

    @property
    def name(self) -> str:
        return "gaussian_well"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return -self.depth * jnp.exp(-x**2 / (2 * self.sigma**2))
