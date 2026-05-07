"""Concentration-dependent transport and thermodynamic properties.

Bundles the four measured functions (kappa, rho_plus, D, U) plus the
electrolyte density fit into a single object for downstream solver 
Includes:

r -> c(r) conversion via Eq. 5 and the density fit (Eq. 34).
r -> m(r) via r = m * M_EO  → m = r / M_EO.
r -> (dU/d ln m)(r) by chain rule, needed by Eqs. 6, 7, 22a, 22b.
r -> t_minus_0(r)    Eq. 6 (solvent-frame anion transference number).
r -> thermo_factor(r) Eq. 7 (1 + d ln gamma_+- / d ln m).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from xpolyelec.config import Config
from xpolyelec.fits import CustomFit, Fit, FitRegistry


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

    Parameters:
    kappa, rho_plus, D, U, rho_el : Fit or CustomFit
        Fitted functions of ``r`` (or ``ln m`` for ``U``).
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

    # ------------------------------------------------------------------
    # Construction from a Config object
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Config, overrides: dict[str, Fit | CustomFit] | None = None) -> "TransportProperties":
        """Build a TransportProperties from ``config.fits`` + physical constants.

        ``overrides`` maps property name → Fit-like, e.g. a CustomFit the user
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
        """Salt molality m (kg/mol^{-1} as used by paper Eq. 38).

        From r = m * M_EO where M_EO has units g/mol, so molality in mol/kg is
        m = r * 1000 / M_EO. From paper Eq. 38, m is in
        kg/mol (= mol/kg). use the paper's units.
        """
        return np.asarray(r, dtype=float) * 1000.0 / self.M_EO

    def c(self, r):
        """Molar salt concentration c(r) in mol/L (Eq. 5)."""
        r_arr = np.asarray(r, dtype=float)
        rho = self.rho_el(r_arr)
        # c = rho * r / (M_EO + r * M_LiTFSI). rho in g/cm^3 -> * 1000 gives g/L;
        # dividing by an effective M in g/mol gives mol/L.
        return 1000.0 * rho * r_arr / (self.M_EO + r_arr * self.M_LiTFSI)

    def c_T(self, r):
        """Total solution concentration (Eq. 3 uses c_T/c_0; approximated as 1/v_bar_avg).

        For a binary salt + polymer solvent under the paper's approximations,
        c_T ≈ c + c_0 where c_0 is the monomer concentration. The paper uses
        c_T/c_0 ≈ 1 / (v̄_m * n_m / (n_m*v̄_m + n_s*v̄_s)) for the thermodynamic
        factor. Downstream code only needs the ratio c_T/c_0 which cancels with
        n_m accounting, so we provide a convenience routine.
        """
        return self.c(r)  # kept for API symmetry; ratio is handled in Eq. 22a directly

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def dU_dlnm(self, r):
        """d U / d ln m as a function of r.

        U is fit as U(m) = a*m^n + c (Eq. 38), so dU/dm = a*n*m^(n-1) and
        dU/dlnm = m * dU/dm = a*n*m^n. For custom U fits we fall back to a
        finite-difference on U vs ln m.
        """
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
        """1 + d ln gamma_+- / d ln m (dimensionless) via Eq. 7.

            (1 + d ln g/ d ln m) = kappa(r) * (dU/d ln m)^2 / [ 2 R T D(r) c(r) (1/rho_plus - 1)^2 ]

        Valid only where rho_plus < 1 (physical).
        """
        r_arr = np.asarray(r, dtype=float)
        rp = np.asarray(self.rho_plus(r_arr))
        # Guard against division-by-zero at rho_plus = 1
        rp_safe = np.where(np.isclose(rp, 1.0), 0.999999, rp)
        dUdlnm = self.dU_dlnm(r_arr)
        numerator = self.kappa(r_arr) * dUdlnm**2
        denominator = 2.0 * self.R * self.T * self.D(r_arr) * self.c(r_arr) * (1.0 / rp_safe - 1.0) ** 2
        # Convert c from mol/L to mol/cm^3 for SI consistency with kappa [S/cm]:
        # kappa [S/cm] = A/V/cm ; D [cm^2/s] ; c [mol/cm^3] ; dU [V] → dimensionless.
        denominator *= 1.0e-3  # mol/L -> mol/cm^3
        return numerator / denominator

    def t_minus_0(self, r):
        """Anion transference number w.r.t. solvent velocity (Eq. 6).

            t_-^0 = 1 - F D(r) c(r) (dU/dlnm) (1 - 1/rho_plus) / kappa(r)

        Eq. 6 rearranged. use the algebraic
        rearrangement that is self-consistent with Eq. 7.
        """
        r_arr = np.asarray(r, dtype=float)
        rp = np.asarray(self.rho_plus(r_arr))
        rp_safe = np.where(np.isclose(rp, 0.0), 1.0e-9, rp)
        c_mol_cm3 = self.c(r_arr) * 1.0e-3
        term = self.F * self.D(r_arr) * c_mol_cm3 * self.dU_dlnm(r_arr) * (1.0 - 1.0 / rp_safe)
        return 1.0 - term / self.kappa(r_arr)

    def t_plus_0(self, r):
        """Cation transference number w.r.t. solvent velocity = 1 - t_-^0."""
        return 1.0 - self.t_minus_0(r)
