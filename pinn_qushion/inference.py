"""Inference utilities for loading and running pre-trained models."""

from pathlib import Path
from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .models import PINN
from .potentials import (
    DoubleWell,
    FiniteSquareWell,
    GaussianWell,
    HarmonicOscillator,
    InfiniteSquareWell,
)


POTENTIAL_CONFIGS = {
    "infinite_square_well": {
        "class": InfiniteSquareWell,
        "params": {"width": 8.0},
        "weight_file": "infinite_well.eqx",
    },
    "harmonic_oscillator": {
        "class": HarmonicOscillator,
        "params": {"omega": 1.0},
        "weight_file": "harmonic.eqx",
    },
    "finite_square_well": {
        "class": FiniteSquareWell,
        "params": {"width": 6.0, "depth": 5.0},
        "weight_file": "finite_well.eqx",
    },
    "double_well": {
        "class": DoubleWell,
        "params": {"separation": 4.0, "depth": 5.0, "barrier": 3.0},
        "weight_file": "double_well.eqx",
    },
    "gaussian_well": {
        "class": GaussianWell,
        "params": {"depth": 5.0, "sigma": 2.0},
        "weight_file": "gaussian_well.eqx",
    },
}


class ModelManager:
    """Manages loading and caching of pre-trained PINN models."""

    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = Path(weights_dir)
        self._cache: Dict[str, PINN] = {}

    def get_model(self, potential_name: str) -> Optional[PINN]:
        """Load or retrieve cached model for a potential type.

        Args:
            potential_name: One of the keys in POTENTIAL_CONFIGS

        Returns:
            Loaded PINN model, or None if weights don't exist
        """
        if potential_name in self._cache:
            return self._cache[potential_name]

        config = POTENTIAL_CONFIGS.get(potential_name)
        if config is None:
            return None

        weight_path = self.weights_dir / config["weight_file"]

        # Create potential
        potential = config["class"](**config["params"])

        # Create model structure
        key = jax.random.PRNGKey(0)
        model = PINN(
            potential=potential,
            hidden_dim=128,
            num_layers=5,
            key=key,
        )

        # Load weights if they exist
        if weight_path.exists():
            model = eqx.tree_deserialise_leaves(weight_path, model)

        self._cache[potential_name] = model
        return model

    def predict(
        self,
        potential_name: str,
        x: jnp.ndarray,
        t: jnp.ndarray,
        x0: float,
        k0: float,
    ) -> tuple:
        """Run inference for given parameters.

        Args:
            potential_name: Type of potential
            x: Spatial grid points
            t: Time points (can be scalar broadcast)
            x0: Initial wavepacket position
            k0: Initial wavepacket momentum

        Returns:
            Tuple of (psi_real, psi_imag, probability_density)
        """
        model = self.get_model(potential_name)
        if model is None:
            # Return placeholder if no model
            return jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

        # Broadcast parameters
        n = len(x)
        if jnp.ndim(t) == 0:
            t = jnp.full(n, t)
        x0_arr = jnp.full(n, x0)
        k0_arr = jnp.full(n, k0)

        psi_r, psi_i = model.psi(x, t, x0_arr, k0_arr)
        prob = psi_r**2 + psi_i**2

        return psi_r, psi_i, prob
