"""Energy spectrum extraction via FFT."""

from typing import Tuple

import jax.numpy as jnp
import numpy as np


def compute_energy_spectrum(
    C_t: jnp.ndarray,
    dt: float,
    positive_only: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute energy spectrum from autocorrelation function via FFT.

    The energy spectrum is obtained by Fourier transforming C(t):
        S(E) = |FT[C(t)]|

    Since C(t) = sum_n |c_n|^2 * exp(-i * E_n * t), peaks in S(E)
    correspond to energy eigenvalues E_n.

    Args:
        C_t: Autocorrelation time series (complex), shape (T,)
        dt: Time step
        positive_only: If True, return only positive frequencies (energies)

    Returns:
        Tuple of (energies, amplitudes) where both have shape (T//2,) or (T,)
    """
    n = len(C_t)

    # Use numpy FFT (JAX FFT works too, but numpy is simpler for this)
    C_t_np = np.array(C_t)
    spectrum = np.fft.fft(C_t_np)
    frequencies = np.fft.fftfreq(n, dt)

    # Convert frequency to energy (E = hbar * omega, with hbar = 1)
    # For C(t) = exp(-i*E*t), the FFT gives a peak at negative frequency -E/(2*pi)
    # So we use |f| to get positive energies: E = 2*pi*|f|
    energies = 2 * np.pi * np.abs(frequencies)

    # Amplitude is magnitude of FFT
    amplitudes = np.abs(spectrum)

    if positive_only:
        # Remove the zero-energy component (DC)
        # Note: after abs(frequencies), we may have duplicate zero, so keep only unique positive
        positive_mask = energies > 0
        energies = energies[positive_mask]
        amplitudes = amplitudes[positive_mask]

        # Sort by energy
        sort_idx = np.argsort(energies)
        energies = energies[sort_idx]
        amplitudes = amplitudes[sort_idx]

    return jnp.array(energies), jnp.array(amplitudes)


def find_spectral_peaks(
    energies: jnp.ndarray,
    amplitudes: jnp.ndarray,
    threshold_ratio: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find peaks in the energy spectrum.

    Args:
        energies: Energy values
        amplitudes: Spectrum amplitudes
        threshold_ratio: Minimum peak height as fraction of max

    Returns:
        Tuple of (peak_energies, peak_amplitudes)
    """
    threshold = threshold_ratio * jnp.max(amplitudes)

    # Simple peak detection: point higher than both neighbors
    amplitudes_np = np.array(amplitudes)
    is_peak = np.zeros(len(amplitudes), dtype=bool)

    for i in range(1, len(amplitudes) - 1):
        if (amplitudes_np[i] > amplitudes_np[i-1] and
            amplitudes_np[i] > amplitudes_np[i+1] and
            amplitudes_np[i] > threshold):
            is_peak[i] = True

    peak_mask = jnp.array(is_peak)
    return energies[peak_mask], amplitudes[peak_mask]
