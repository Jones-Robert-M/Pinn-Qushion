"""Base class for quantum potentials."""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class Potential(ABC):
    """Abstract base class for quantum potentials V(x)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this potential type."""
        pass

    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the potential at positions x.

        Args:
            x: Array of spatial positions

        Returns:
            Array of potential values V(x)
        """
        pass
