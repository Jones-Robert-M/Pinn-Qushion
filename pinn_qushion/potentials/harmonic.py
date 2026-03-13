"""Harmonic oscillator potential."""

import jax.numpy as jnp

from .base import Potential


class HarmonicOscillator(Potential):
    """Quantum harmonic oscillator potential.

    V(x) = 0.5 * omega^2 * x^2

    Args:
        omega: Angular frequency of the oscillator
    """

    def __init__(self, omega: float = 1.0):
        self.omega = omega

    @property
    def name(self) -> str:
        return "harmonic_oscillator"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * self.omega**2 * x**2
