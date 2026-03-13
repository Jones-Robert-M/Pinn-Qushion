"""Tests for quantum potentials."""

import jax.numpy as jnp
import pytest

from pinn_qushion.potentials.base import Potential
from pinn_qushion.potentials.square_well import InfiniteSquareWell, FiniteSquareWell
from pinn_qushion.potentials.harmonic import HarmonicOscillator
from pinn_qushion.potentials.double_well import DoubleWell
from pinn_qushion.potentials.gaussian_well import GaussianWell


def test_potential_is_abstract():
    """Potential base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Potential()


def test_potential_has_call_method():
    """Potential subclasses must implement __call__."""

    class ConcretePotential(Potential):
        @property
        def name(self) -> str:
            return "test"

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return jnp.zeros_like(x)

    pot = ConcretePotential()
    x = jnp.array([0.0, 1.0, 2.0])
    result = pot(x)
    assert result.shape == x.shape


def test_potential_has_name_property():
    """Potential subclasses must have a name property."""

    class ConcretePotential(Potential):
        @property
        def name(self) -> str:
            return "test_potential"

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return jnp.zeros_like(x)

    pot = ConcretePotential()
    assert pot.name == "test_potential"


class TestInfiniteSquareWell:
    """Tests for infinite square well potential."""

    def test_zero_inside_well(self):
        """Potential is zero inside the well."""
        well = InfiniteSquareWell(width=4.0)
        x = jnp.array([-1.5, 0.0, 1.5])
        V = well(x)
        assert jnp.allclose(V, 0.0)

    def test_infinite_outside_well(self):
        """Potential is very large outside the well."""
        well = InfiniteSquareWell(width=4.0)
        x = jnp.array([-3.0, 3.0])
        V = well(x)
        assert jnp.all(V > 1e6)

    def test_name(self):
        """Well has correct name."""
        well = InfiniteSquareWell(width=4.0)
        assert well.name == "infinite_square_well"

    def test_width_parameter(self):
        """Width parameter controls well boundaries."""
        well = InfiniteSquareWell(width=2.0)
        # Inside narrow well
        assert jnp.isclose(well(jnp.array([0.5])), 0.0)
        # Outside narrow well
        assert well(jnp.array([1.5])) > 1e6


class TestFiniteSquareWell:
    """Tests for finite square well potential."""

    def test_negative_inside_well(self):
        """Potential is -V0 inside the well."""
        well = FiniteSquareWell(width=4.0, depth=5.0)
        x = jnp.array([-1.5, 0.0, 1.5])
        V = well(x)
        assert jnp.allclose(V, -5.0)

    def test_zero_outside_well(self):
        """Potential is zero outside the well."""
        well = FiniteSquareWell(width=4.0, depth=5.0)
        x = jnp.array([-3.0, 3.0])
        V = well(x)
        assert jnp.allclose(V, 0.0)

    def test_name(self):
        """Has correct name."""
        well = FiniteSquareWell(width=4.0, depth=5.0)
        assert well.name == "finite_square_well"

    def test_depth_parameter(self):
        """Depth parameter controls potential magnitude."""
        well = FiniteSquareWell(width=4.0, depth=10.0)
        V = well(jnp.array([0.0]))
        assert jnp.isclose(V, -10.0)


class TestHarmonicOscillator:
    """Tests for harmonic oscillator potential."""

    def test_zero_at_origin(self):
        """Potential is zero at x=0."""
        ho = HarmonicOscillator(omega=1.0)
        V = ho(jnp.array([0.0]))
        assert jnp.isclose(V, 0.0)

    def test_parabolic_shape(self):
        """Potential follows V = 0.5 * omega^2 * x^2."""
        omega = 2.0
        ho = HarmonicOscillator(omega=omega)
        x = jnp.array([1.0, 2.0, 3.0])
        V = ho(x)
        expected = 0.5 * omega**2 * x**2
        assert jnp.allclose(V, expected)

    def test_name(self):
        """Has correct name."""
        ho = HarmonicOscillator(omega=1.0)
        assert ho.name == "harmonic_oscillator"

    def test_symmetric(self):
        """Potential is symmetric around origin."""
        ho = HarmonicOscillator(omega=1.5)
        x_pos = jnp.array([1.0, 2.0, 3.0])
        x_neg = -x_pos
        assert jnp.allclose(ho(x_pos), ho(x_neg))


class TestDoubleWell:
    """Tests for double well potential."""

    def test_maximum_at_origin(self):
        """Potential has local maximum at x=0."""
        dw = DoubleWell(separation=2.0, depth=4.0, barrier=2.0)
        V_origin = dw(jnp.array([0.0]))
        V_nearby = dw(jnp.array([0.5]))
        assert V_origin > V_nearby

    def test_minima_at_separation(self):
        """Potential has minima near x = ±separation/2."""
        dw = DoubleWell(separation=4.0, depth=5.0, barrier=2.0)
        # Find approximate minima
        x = jnp.linspace(-3, 3, 100)
        V = dw(x)
        min_idx = jnp.argmin(V)
        # Minimum should be roughly in range [-2.5, -1.5] or [1.5, 2.5]
        assert jnp.abs(x[min_idx]) > 1.0

    def test_name(self):
        """Has correct name."""
        dw = DoubleWell(separation=2.0, depth=4.0, barrier=2.0)
        assert dw.name == "double_well"

    def test_symmetric(self):
        """Potential is symmetric around origin."""
        dw = DoubleWell(separation=2.0, depth=4.0, barrier=2.0)
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(dw(x), dw(-x))


class TestGaussianWell:
    """Tests for Gaussian well potential (quantum dot analog)."""

    def test_minimum_at_origin(self):
        """Potential has minimum at x=0."""
        gw = GaussianWell(depth=5.0, sigma=2.0)
        V_origin = gw(jnp.array([0.0]))
        V_away = gw(jnp.array([3.0]))
        assert V_origin < V_away

    def test_depth_at_origin(self):
        """Potential equals -depth at x=0."""
        gw = GaussianWell(depth=7.0, sigma=2.0)
        V = gw(jnp.array([0.0]))
        assert jnp.isclose(V, -7.0)

    def test_approaches_zero_far_away(self):
        """Potential approaches zero far from origin."""
        gw = GaussianWell(depth=5.0, sigma=1.0)
        V = gw(jnp.array([10.0]))
        assert jnp.abs(V) < 0.01

    def test_name(self):
        """Has correct name."""
        gw = GaussianWell(depth=5.0, sigma=2.0)
        assert gw.name == "gaussian_well"

    def test_width_parameter(self):
        """Sigma controls the width of the well."""
        narrow = GaussianWell(depth=5.0, sigma=1.0)
        wide = GaussianWell(depth=5.0, sigma=3.0)
        x = jnp.array([2.0])
        # Narrow well should have smaller magnitude at x=2
        assert jnp.abs(narrow(x)) < jnp.abs(wide(x))
