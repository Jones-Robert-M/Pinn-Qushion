"""Autocorrelation function for quantum wavepacket analysis."""

import jax.numpy as jnp


def compute_autocorrelation(
    psi_0: jnp.ndarray,
    psi_t: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """Compute the autocorrelation (survival probability) C(t).

    C(t) = <psi_0 | psi(t)> = integral of psi_0^*(x) * psi(x,t) dx

    Args:
        psi_0: Initial wavefunction (complex), shape (N,)
        psi_t: Wavefunction at time t (complex), shape (N,)
        dx: Spatial grid spacing

    Returns:
        Complex autocorrelation value
    """
    return jnp.sum(jnp.conj(psi_0) * psi_t) * dx


def compute_autocorrelation_series(
    psi_0: jnp.ndarray,
    psi_series: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """Compute autocorrelation for a time series of wavefunctions.

    Args:
        psi_0: Initial wavefunction (complex), shape (N,)
        psi_series: Wavefunction time series (complex), shape (T, N)
        dx: Spatial grid spacing

    Returns:
        Autocorrelation time series C(t), shape (T,)
    """
    return jnp.array([compute_autocorrelation(psi_0, psi_t, dx) for psi_t in psi_series])
