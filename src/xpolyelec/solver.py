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
