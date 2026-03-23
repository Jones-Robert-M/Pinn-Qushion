"""Loss functions for PINN training."""

import jax.numpy as jnp

from pinn_qushion.models.pinn import PINN


class PINNLoss:
    """Loss functions for training the Schrodinger equation PINN.

    The TDSE in natural units (hbar = m = 1):
        i * dPsi/dt = -0.5 * d^2Psi/dx^2 + V(x) * Psi

    Separating real and imaginary parts:
        dPsi_R/dt = -0.5 * d^2Psi_I/dx^2 + V * Psi_I
        dPsi_I/dt =  0.5 * d^2Psi_R/dx^2 - V * Psi_R

    Args:
        sigma: Width of initial Gaussian wavepacket
        lambda_phys: Weight for physics residual loss
        lambda_ic: Weight for initial condition loss
        lambda_bc: Weight for boundary condition loss
    """

    def __init__(
        self,
        sigma: float = 1.0,
        lambda_phys: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_bc: float = 0.0,
        lambda_norm: float = 1.0,
    ):
        self.sigma = sigma
        self.lambda_phys = lambda_phys
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.lambda_norm = lambda_norm

    def initial_wavepacket(
        self, x: jnp.ndarray, x0: jnp.ndarray, k0: jnp.ndarray
    ) -> tuple:
        """Compute normalized initial Gaussian wavepacket.

        Psi_0(x) = N * exp(-(x-x0)^2/(4*sigma^2)) * exp(i*k0*x)

        where N = (1/(2*pi*sigma^2))^(1/4) is the normalization factor.

        Returns:
            Tuple of (psi_real, psi_imag)
        """
        # Normalization factor for Gaussian wavepacket
        norm = (1.0 / (2 * jnp.pi * self.sigma**2)) ** 0.25
        envelope = norm * jnp.exp(-((x - x0) ** 2) / (4 * self.sigma**2))
        phase = k0 * x
        psi_r = envelope * jnp.cos(phase)
        psi_i = envelope * jnp.sin(phase)
        return psi_r, psi_i

    def physics_loss(
        self,
        model: PINN,
        x: jnp.ndarray,
        t: jnp.ndarray,
        x0: jnp.ndarray,
        k0: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute physics residual loss for the TDSE.

        Residual for real part: dPsi_R/dt + 0.5*d^2Psi_I/dx^2 - V*Psi_I = 0
        Residual for imag part: dPsi_I/dt - 0.5*d^2Psi_R/dx^2 + V*Psi_R = 0
        """
        psi_r, psi_i = model.psi(x, t, x0, k0)
        dpsi_r_dt, dpsi_i_dt = model.psi_t(x, t, x0, k0)
        d2psi_r_dx2, d2psi_i_dx2 = model.psi_xx(x, t, x0, k0)

        V = model.potential(x)

        # TDSE residuals
        res_r = dpsi_r_dt + 0.5 * d2psi_i_dx2 - V * psi_i
        res_i = dpsi_i_dt - 0.5 * d2psi_r_dx2 + V * psi_r

        return jnp.mean(res_r**2 + res_i**2)

    def initial_condition_loss(
        self,
        model: PINN,
        x: jnp.ndarray,
        t: jnp.ndarray,
        x0: jnp.ndarray,
        k0: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute initial condition loss."""
        psi_r_pred, psi_i_pred = model.psi(x, t, x0, k0)
        psi_r_true, psi_i_true = self.initial_wavepacket(x, x0, k0)

        return jnp.mean((psi_r_pred - psi_r_true)**2 + (psi_i_pred - psi_i_true)**2)

    def boundary_condition_loss(
        self,
        model: PINN,
        x: jnp.ndarray,
        t: jnp.ndarray,
        x0: jnp.ndarray,
        k0: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute boundary condition loss (Dirichlet: Psi = 0 at boundaries)."""
        psi_r, psi_i = model.psi(x, t, x0, k0)
        return jnp.mean(psi_r**2 + psi_i**2)

    def normalization_loss(
        self,
        model: PINN,
        x: jnp.ndarray,
        t: jnp.ndarray,
        x0: jnp.ndarray,
        k0: jnp.ndarray,
        dx: float = 0.078125,  # 20/256 grid spacing
    ) -> jnp.ndarray:
        """Compute normalization loss - penalize deviation from unit norm.

        The wavefunction should satisfy integral |Psi|^2 dx = 1 at all times.
        """
        psi_r, psi_i = model.psi(x, t, x0, k0)
        prob = psi_r**2 + psi_i**2
        # Approximate integral using trapezoidal rule
        norm = jnp.sum(prob) * dx
        return (norm - 1.0) ** 2

    def total_loss(
        self,
        model: PINN,
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
        x_norm: jnp.ndarray = None,
        t_norm: jnp.ndarray = None,
        x0_norm: jnp.ndarray = None,
        k0_norm: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Compute total weighted loss."""
        l_phys = self.physics_loss(model, x_int, t_int, x0_int, k0_int)
        l_ic = self.initial_condition_loss(model, x_ic, t_ic, x0_ic, k0_ic)
        l_bc = self.boundary_condition_loss(model, x_bc, t_bc, x0_bc, k0_bc)

        total = self.lambda_phys * l_phys + self.lambda_ic * l_ic + self.lambda_bc * l_bc

        # Add normalization loss if points provided
        if x_norm is not None and self.lambda_norm > 0:
            l_norm = self.normalization_loss(model, x_norm, t_norm, x0_norm, k0_norm)
            total = total + self.lambda_norm * l_norm

        return total
