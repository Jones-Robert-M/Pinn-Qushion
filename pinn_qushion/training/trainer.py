"""Training loop for PINN models."""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pinn_qushion.models.pinn import PINN

from .loss import PINNLoss


class Trainer:
    """Training loop for PINN models.

    Args:
        model: PINN model to train
        optimizer: Optax optimizer
        sigma: Width of initial Gaussian wavepacket
        lambda_phys: Weight for physics residual loss
        lambda_ic: Weight for initial condition loss
        lambda_bc: Weight for boundary condition loss
    """

    def __init__(
        self,
        model: PINN,
        optimizer: optax.GradientTransformation,
        sigma: float = 1.0,
        lambda_phys: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_bc: float = 10.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        self.loss_fn = PINNLoss(
            sigma=sigma,
            lambda_phys=lambda_phys,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc,
        )

    def compute_loss(
        self,
        x_int: jnp.ndarray,
        t_int: jnp.ndarray,
        x0_int: jnp.ndarray,
        k0_int: jnp.ndarray,
        x_ic: jnp.ndarray,
        t_ic: jnp.ndarray,
        x0_ic: jnp.ndarray,
        k0_ic: jnp.ndarray,
        x_bc: jnp.ndarray,
        t_bc: jnp.ndarray,
        x0_bc: jnp.ndarray,
        k0_bc: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute total loss for current model."""
        return self.loss_fn.total_loss(
            self.model,
            x_int, t_int, x0_int, k0_int,
            x_ic, t_ic, x0_ic, k0_ic,
            x_bc, t_bc, x0_bc, k0_bc,
        )

    @eqx.filter_jit
    def _train_step(
        self,
        model: PINN,
        opt_state: optax.OptState,
        x_int: jnp.ndarray,
        t_int: jnp.ndarray,
        x0_int: jnp.ndarray,
        k0_int: jnp.ndarray,
        x_ic: jnp.ndarray,
        t_ic: jnp.ndarray,
        x0_ic: jnp.ndarray,
        k0_ic: jnp.ndarray,
        x_bc: jnp.ndarray,
        t_bc: jnp.ndarray,
        x0_bc: jnp.ndarray,
        k0_bc: jnp.ndarray,
    ) -> Tuple[PINN, optax.OptState, jnp.ndarray]:
        """Single training step with JIT compilation."""

        def loss_wrapper(m):
            return self.loss_fn.total_loss(
                m,
                x_int, t_int, x0_int, k0_int,
                x_ic, t_ic, x0_ic, k0_ic,
                x_bc, t_bc, x0_bc, k0_bc,
            )

        loss, grads = eqx.filter_value_and_grad(loss_wrapper)(model)
        updates, new_opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)

        return new_model, new_opt_state, loss

    def step(
        self,
        x_int: jnp.ndarray,
        t_int: jnp.ndarray,
        x0_int: jnp.ndarray,
        k0_int: jnp.ndarray,
        x_ic: jnp.ndarray,
        t_ic: jnp.ndarray,
        x0_ic: jnp.ndarray,
        k0_ic: jnp.ndarray,
        x_bc: jnp.ndarray,
        t_bc: jnp.ndarray,
        x0_bc: jnp.ndarray,
        k0_bc: jnp.ndarray,
    ) -> jnp.ndarray:
        """Perform one training step and return loss."""
        self.model, self.opt_state, loss = self._train_step(
            self.model,
            self.opt_state,
            x_int, t_int, x0_int, k0_int,
            x_ic, t_ic, x0_ic, k0_ic,
            x_bc, t_bc, x0_bc, k0_bc,
        )
        return loss

    def get_model(self) -> PINN:
        """Return the current trained model."""
        return self.model
