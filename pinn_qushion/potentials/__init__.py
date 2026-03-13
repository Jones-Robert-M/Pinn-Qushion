"""Quantum potential definitions."""

from .base import Potential
from .double_well import DoubleWell
from .gaussian_well import GaussianWell
from .harmonic import HarmonicOscillator
from .square_well import FiniteSquareWell, InfiniteSquareWell

__all__ = [
    "Potential",
    "InfiniteSquareWell",
    "FiniteSquareWell",
    "HarmonicOscillator",
    "DoubleWell",
    "GaussianWell",
]
