"""
Patel et al. 2025 (Eqs. 34-38) use:

* ``linear``     : ``a*x + b``       (rho_el, D)
* ``kappa_peak`` : ``a*r*exp(-r/b)``  (kappa, Eq. 35)
* ``poly2``      : ``a*r^2 + b*r + c`` (rho_plus)
* ``power_law``  : ``a * m**n + c``   (U vs m; m = salt molality)

Additional families for flexibility: ``poly3``, ``exp``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------
# Basic fit families (functions, analytical derivatives, defaults)
# ----------------------------------------------------------------------
def _linear(x, a, b):
    return a * x + b


def _linear_deriv(x, a, b):
    return np.full_like(np.asarray(x, dtype=float), a)


def _kappa_peak(r, a, b):
    """Patel 2025 Eq. 35: kappa = a * r * exp(-r/b)."""
    return a * r * np.exp(-r / b)


def _kappa_peak_deriv(r, a, b):
    e = np.exp(-r / b)
    return a * e * (1.0 - r / b)


def _poly2(x, a, b, c):
    return a * x * x + b * x + c


def _poly2_deriv(x, a, b, c):
    return 2.0 * a * x + b


def _poly3(x, a, b, c, d):
    return a * x**3 + b * x * x + c * x + d


def _poly3_deriv(x, a, b, c, d):
    return 3.0 * a * x * x + 2.0 * b * x + c


def _power_law(x, a, n, c):
    """Patel 2025 Eq. 38: y = a * x**n + c (x must be positive)."""
    return a * np.power(np.asarray(x, dtype=float), n) + c


def _power_law_deriv(x, a, n, c):
    xarr = np.asarray(x, dtype=float)
    return a * n * np.power(xarr, n - 1.0)


def _exp(x, a, b, c):
    return a * np.exp(b * x) + c


def _exp_deriv(x, a, b, c):
    return a * b * np.exp(b * x)


@dataclass(frozen=True)
class FitFamily:
    """Definition of a parametric fit family."""

    name: str
    func: Callable[..., np.ndarray]
    deriv: Callable[..., np.ndarray]
    n_params: int
    default_p0: tuple[float, ...]


_FAMILIES: dict[str, FitFamily] = {
    "linear":     FitFamily("linear",     _linear,     _linear_deriv,     2, (1.0, 0.0)),
    "kappa_peak": FitFamily("kappa_peak", _kappa_peak, _kappa_peak_deriv, 2, (1.0e-3, 0.061)),
    "poly2":      FitFamily("poly2",      _poly2,      _poly2_deriv,      3, (1.0, 0.0, 0.0)),
    "poly3":      FitFamily("poly3",      _poly3,      _poly3_deriv,      4, (1.0, 0.0, 0.0, 0.0)),
    "power_law":  FitFamily("power_law",  _power_law,  _power_law_deriv,  3, (-0.14, 0.56, 0.16)),
    "exp":        FitFamily("exp",        _exp,        _exp_deriv,        3, (1.0, 1.0, 0.0)),
}



# ----------------------------------------------------------------------
# Fit objects wrapping family + fitted parameters
# ----------------------------------------------------------------------
class Fit:
    """A fitted instance: callable and differentiable."""

    def __init__(self, family: FitFamily, params: tuple[float, ...]):
        if len(params) != family.n_params:
            raise ValueError(
                f"{family.name} expects {family.n_params} params, got {len(params)}"
            )
        self.family = family
        self.params = tuple(float(p) for p in params)

    # Call-style evaluation
    def __call__(self, x):
        return self.family.func(np.asarray(x, dtype=float), *self.params)

    def derivative(self, x):
        return self.family.deriv(np.asarray(x, dtype=float), *self.params)

    def __repr__(self) -> str:
        return f"Fit(family={self.family.name!r}, params={self.params!r})"


class CustomFit:
    """custom fit example
    -------
    >>> my_fit = CustomFit(lambda r: np.tanh(r), lambda r: 1 - np.tanh(r)**2)
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        deriv: Callable[[np.ndarray], np.ndarray],
        name: str = "custom",
    ):
        self._func = func
        self._deriv = deriv
        self.name = name
        self.params: tuple = ()

    def __call__(self, x):
        return self._func(np.asarray(x, dtype=float))

    def derivative(self, x):
        return self._deriv(np.asarray(x, dtype=float))

    def __repr__(self) -> str:
        return f"CustomFit(name={self.name!r})"
