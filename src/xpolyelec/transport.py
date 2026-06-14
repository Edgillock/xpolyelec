"""Concentration-dependent transport and thermodynamic properties.

Bundles the four measured functions (kappa, rho_plus, D, U) plus the
electrolyte density fit into a single object that downstream solver code
can query cheaply. Provides:

r -> c(r) via Eq. 5 and the density fit (Eq. 34).
r -> m(r) via r = m * M_EO  → m = r / M_EO.
r -> (dU/d ln m)(r) by chain rule, needed by Eqs. 6, 7, 22a, 22b.
r -> t_minus_0(r)    Eq. 6 (solvent-frame anion transference number).
r -> thermo_factor(r) Eq. 7 (1 + d ln gamma_+- / d ln m).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from xpolyelec.config import Config
from xpolyelec.fits import CustomFit, Fit, FitRegistry


def _load_paper_fig2f_spline() -> PchipInterpolator | None:
    """Load the paper Fig. 2F red-dashed-line data shipped with the demo/sample library."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    path = os.path.join(project_root, "examples", "sample_data", "t0_plus.csv")

    if not os.path.exists(path):
        return None

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    r_vals, t_vals = data[:, 0], data[:, 1]
    return PchipInterpolator(r_vals, t_vals, extrapolate=False)


def _build_fit(spec: dict[str, Any] | Fit | CustomFit) -> Fit | CustomFit:
    """Build a Fit from either a config spec dict or an existing Fit-like object."""
    if isinstance(spec, (Fit, CustomFit)):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"expected dict or Fit-like, got {type(spec).__name__}")
    form = spec["form"]
    params = spec["params"]
    return FitRegistry.from_params(form, params)


@dataclass
class TransportProperties:
    """Container for all concentration-dependent property fits.

    Parameters
    ----------
    kappa, rho_plus, D, U, rho_el : Fit or CustomFit
        Fitted functions of r (or ln m for U).
    M_EO, M_LiTFSI : float
        Molecular weights (g/mol).
    F, R, T : float
        Faraday's constant, gas constant, temperature.
    """

    kappa: Fit | CustomFit
    rho_plus: Fit | CustomFit
    D: Fit | CustomFit
    U: Fit | CustomFit
    rho_el: Fit | CustomFit
    M_EO: float
    M_LiTFSI: float
    F: float
    R: float
    T: float
    # Optional spline of paper Fig. 2F t_+^0(r); built lazily in __post_init__
    _fig2f_spline: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._fig2f_spline is None:
            self._fig2f_spline = _load_paper_fig2f_spline()

    # ------------------------------------------------------------------
    # Construction from a Config object
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Config, overrides: dict[str, Fit | CustomFit] | None = None) -> "TransportProperties":
        """Build a TransportProperties from config.fits + physical constants.

        overrides maps property name → Fit-like, e.g. a CustomFit the user
        constructed programmatically.
        """
        fits_cfg = config.get("fits")
        phys = config.get("physical")
        overrides = overrides or {}
        return cls(
            kappa=overrides.get("kappa", _build_fit(fits_cfg["kappa"])),
            rho_plus=overrides.get("rho_plus", _build_fit(fits_cfg["rho_plus"])),
            D=overrides.get("D", _build_fit(fits_cfg["D"])),
            U=overrides.get("U", _build_fit(fits_cfg["U"])),
            rho_el=overrides.get("rho_el", _build_fit(fits_cfg["rho_el"])),
            M_EO=float(phys["M_EO_g_per_mol"]),
            M_LiTFSI=float(phys["M_LiTFSI_g_per_mol"]),
            F=float(phys["F_C_per_mol"]),
            R=float(phys["R_J_per_mol_K"]),
            T=float(phys["T_K"]),
        )

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------
    def m(self, r):
        
        return np.asarray(r, dtype=float) * 1000.0 / self.M_EO

    def c(self, r):
        
        r_arr = np.asarray(r, dtype=float)
        rho = self.rho_el(r_arr)
        # c = rho * r / (M_EO + r * M_LiTFSI). rho in g/cm^3 -> * 1000 gives g/L;
        # dividing by an effective M in g/mol gives mol/L.
        return 1000.0 * rho * r_arr / (self.M_EO + r_arr * self.M_LiTFSI)

    def c_T(self, r):
        
        return self.c(r)  # kept for API symmetry; ratio is handled in Eq. 22a directly

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def dU_dlnm(self, r):
        
        m = self.m(r)
        m_safe = np.where(m <= 0, 1e-12, m)
        if isinstance(self.U, Fit) and self.U.family.name == "power_law":
            a, n, _c = self.U.params
            return a * n * np.power(m_safe, n)
        # Fallback: finite difference in log-m space
        ln_m = np.log(m_safe)
        eps = 1e-4
        return (self.U(np.exp(ln_m + eps)) - self.U(np.exp(ln_m - eps))) / (2.0 * eps)

    def thermo_factor(self, r):
        
        r_arr = np.asarray(r, dtype=float)
        rp = np.asarray(self.rho_plus(r_arr))
        #against division-by-zero at rho_plus = 1
        rp_safe = np.where(np.isclose(rp, 1.0), 0.999999, rp)
        dUdlnm = self.dU_dlnm(r_arr)
        numerator = self.kappa(r_arr) * dUdlnm**2
        denominator = 2.0 * self.R * self.T * self.D(r_arr) * self.c(r_arr) * (1.0 / rp_safe - 1.0) ** 2
        denominator *= 1.0e-3  
        return numerator / denominator

    def _t_plus_0_analytical(self, r):
        
        r_arr = np.asarray(r, dtype=float)
        rp = np.asarray(self.rho_plus(r_arr))
        rp_safe = np.where(np.isclose(rp, 0.0), 1.0e-9, rp)
        c_mol_cm3 = self.c(r_arr) * 1.0e-3
        kappa = self.kappa(r_arr)
        kappa_safe = np.where(np.isclose(kappa, 0.0), 1.0e-30, kappa)
        dUdlnm = self.dU_dlnm(r_arr)
        dU_safe = np.where(np.isclose(dUdlnm, 0.0), 1.0e-30, dUdlnm)
        return (
            self.F * self.D(r_arr) * c_mol_cm3 * (1.0 / rp_safe - 1.0)
            / (kappa_safe * dU_safe)
        )

    def t_plus_0(self, r):
       
        r_arr = np.asarray(r, dtype=float)
        analytical = self._t_plus_0_analytical(r_arr)
        if self._fig2f_spline is None:
            return analytical
        spline_vals = self._fig2f_spline(r_arr)
        in_range = np.isfinite(spline_vals)
        return np.where(in_range, spline_vals, analytical)

    def t_minus_0(self, r):
        
        return 1.0 - self._t_plus_0_analytical(r)
