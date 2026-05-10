"""Core numerical pipeline.

Implements Eqs. 22-27 of Patel 2025:

build_J1_and_J2 → callable J1(r), J2(r) given a StrainModel + StrainContext + TransportProperties.
solve_r_profile → three-step iterative procedure that finds the
r(x/L) profile for a target (ravg, iL).

All code is pure numpy/scipy and has no dependency on the Simulation API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from xpolyelec.strain.base import StrainContext, StrainModel
from xpolyelec.transport import TransportProperties


# ----------------------------------------------------------------------
# J1 and J2 builders
# ----------------------------------------------------------------------
@dataclass
class JFunctions:
    """Container for the two integrands J1(r) and J2(r).

    J1 (Eq. 24) is integrated over r to produce the implicit r(x/L) profile.
    J2 (Eq. 27) is integrated over r to produce the concentration + strain
    overpotential.
    """

    J1: Callable[[np.ndarray], np.ndarray]
    J2: Callable[[np.ndarray], np.ndarray]
    Gamma_conc: Callable[[np.ndarray], np.ndarray]
    Gamma_strain: Callable[[np.ndarray], np.ndarray]


def build_J1_and_J2(
    transport: TransportProperties,
    strain_model: StrainModel,
    ctx: StrainContext,
) -> JFunctions:
    """Build callable J1(r), J2(r) from the transport properties + strain model.

    Derivation from Patel 2025 Eqs. 4, 20-27:

    For the Crosslink Model (Eqs. 20-22, 25), the strain contribution to the
    electrolyte chemical potential mu_es adds the dimensionless piece
    (1/(2RT)) * r * d mu_strain/dr * (1/(1+Theta))`` inside the bracket in Eq. 22a; 
    absorb it as an additive correction to Gamma_conc:
    Gamma_strain(r) = Gamma_conc^Baseline(r) * ( r * d mu_strain / dr ) / ( 2 R T (1 + Theta) )

    so J1 = Gamma_conc * (1 + r * d mu_strain/dr / (2 R T (1+Theta))).

    For J2 (Eq. 27), the concentration piece is (2RT/F) * (1 - t_-^0) *
    (1 + Theta) / r and the strain piece is t_-^0 * d mu_strain / dr / F.
    """
    RT = transport.R * transport.T
    F = transport.F


    def _gamma_conc_baseline(r):
        r = np.asarray(r, dtype=float)
        c_mol_cm3 = transport.c(r) * 1.0e-3  # mol/L -> mol/cm^3
        D = transport.D(r)
        t_minus = transport.t_minus_0(r)
        tf = transport.thermo_factor(r)
        # Guard against t_minus or r approaching 0
        t_safe = np.where(np.abs(t_minus) < 1e-12, 1e-12, t_minus)
        r_safe = np.where(np.abs(r) < 1e-12, 1e-12, r)
        # Gamma_conc^Baseline = D * c * (1 + Theta) / (r * t_-^0)
        return D * c_mol_cm3 * tf / (r_safe * t_safe)

    def Gamma_conc(r):
        return _gamma_conc_baseline(r)

    def Gamma_strain(r):
        r = np.asarray(r, dtype=float)
        if strain_model.name == "none":
            return np.zeros_like(r)
        gc = _gamma_conc_baseline(r)
        tf = transport.thermo_factor(r)
        # Protect against (1+Theta) -> 0 numerical noise
        denom = 2.0 * RT * np.where(np.abs(tf) < 1e-12, 1e-12, tf)
        dmu_dr = strain_model.d_mu_strain_d_r(r, ctx)
        return gc * (r * dmu_dr / denom)

    def J1(r):
        return Gamma_conc(r) + Gamma_strain(r)

    def J2(r):
        # Paper Eq. 27
        r = np.asarray(r, dtype=float)
        t_minus = transport.t_minus_0(r)
        tf = transport.thermo_factor(r)
        # Concentration piece: (2RT/F) * (1 - t_-^0)(1 + Theta) / r
        conc_term = (2.0 * RT / F) * (1.0 - t_minus) * tf / r
        if strain_model.name == "none":
            return conc_term
        # Strain piece: t_-^0 * d mu_strain / dr / F
        strain_term = t_minus * strain_model.d_mu_strain_d_r(r, ctx) / F
        return conc_term + strain_term

    return JFunctions(J1=J1, J2=J2, Gamma_conc=Gamma_conc, Gamma_strain=Gamma_strain)
