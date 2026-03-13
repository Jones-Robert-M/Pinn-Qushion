"""Signal processing and spectral analysis."""

from .autocorrelation import compute_autocorrelation, compute_autocorrelation_series
from .spectrum import compute_energy_spectrum, find_spectral_peaks

__all__ = [
    "compute_autocorrelation",
    "compute_autocorrelation_series",
    "compute_energy_spectrum",
    "find_spectral_peaks",
]
