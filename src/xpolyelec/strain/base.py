"""Strain-model abstract base + shared kinematics helpers.

Every strain model expresses the chemical-potential contribution
mu_strain(r; context) (J/mol) and its derivative d mu_strain / dr.
The solver uses the derivative to build Gamma_strain in Eq. 22b and the
integrand J_strain in Eq. 27.

Three structural assumptions of the paper are exposed here as switches:

1.kinematics: "affine_isotropic" (paper) vs. "uniaxial".
2.phi_p_model: "paper" (phi_p = 1 - vs*c) vs. a user callable.
3.lambda_crit: symmetric delta vs. independent extension/contraction.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from xpolyelec.transport import TransportProperties


# ----------------------------------------------------------------------
# Kinematics helpers
# ----------------------------------------------------------------------
def phi_p_paper(c_mol_L: np.ndarray, vbar_s_nm3: float) -> np.ndarray:
    """Paper's polymer-volume-fraction relation: phi_p = 1 - vbar_s * c.

    c is converted from mol/L to molecules/nm^3 before multiplication so the
    product is dimensionless.
    """
    # c [mol/L] * N_A / 1e24 -> molecules/nm^3
    N_A = 6.022_140_76e23
    c_per_nm3 = np.asarray(c_mol_L, dtype=float) * N_A / 1.0e24 * 1.0e-3  
    c_per_nm3 = np.asarray(c_mol_L, dtype=float) * N_A / 1.0e24
    return 1.0 - vbar_s_nm3 * c_per_nm3


def phi_p0_from_ravg(tp: TransportProperties, ravg: float, vbar_s_nm3: float) -> float:
    """Initial (relaxed) polymer volume fraction evaluated at r = ravg."""
    c_avg = float(tp.c(ravg))
    return float(phi_p_paper(np.asarray([c_avg]), vbar_s_nm3)[0])


def lambda_from_phi(phi_p: np.ndarray, phi_p0: float, kinematics: str) -> np.ndarray:
    """Convert polymer volume fractions to local extension lambda."""
    phi_p = np.asarray(phi_p, dtype=float)
    ratio = phi_p0 / np.clip(phi_p, 1.0e-12, None)
    if kinematics == "affine_isotropic":
        return np.sqrt(ratio)
    if kinematics == "uniaxial":
        return ratio
    raise ValueError(f"unknown kinematics {kinematics!r}")


# ----------------------------------------------------------------------
# Strain context
# ----------------------------------------------------------------------
@dataclass
class StrainContext:

    transport: TransportProperties
    ravg: float
    N: float
    lambda_crit_ext: float
    lambda_crit_con: float
    kinematics: str
    phi_p_model: str | Callable[[np.ndarray], np.ndarray]
    vbar_m_nm3: float
    vbar_s_nm3: float
    extra: dict[str, Any] = field(default_factory=dict)

    # Cached composition at ravg
    @property
    def phi_p0(self) -> float:
        """phi_p at the relaxed (initial) reference state r = ravg."""
        if callable(self.phi_p_model):
            c_avg = float(self.transport.c(self.ravg))
            return float(self.phi_p_model(np.asarray([c_avg]))[0])
        return phi_p0_from_ravg(self.transport, self.ravg, self.vbar_s_nm3)

    def phi_p(self, r) -> np.ndarray:
        """phi_p at general concentration r."""
        c = self.transport.c(r)
        if callable(self.phi_p_model):
            return np.asarray(self.phi_p_model(np.asarray(c)), dtype=float)
        return phi_p_paper(np.asarray(c), self.vbar_s_nm3)

    def lambda_of_r(self, r) -> np.ndarray:
        """Local extension lambda as a function of r."""
        return lambda_from_phi(self.phi_p(r), self.phi_p0, self.kinematics)

    def lambda_crit_for(self, lam: np.ndarray) -> np.ndarray:
        """Return the relevant lambda_crit for each point (ext vs con)."""
        lam = np.asarray(lam, dtype=float)
        return np.where(lam >= 1.0, self.lambda_crit_ext, self.lambda_crit_con)


# ----------------------------------------------------------------------
# Abstract base
# ----------------------------------------------------------------------
class StrainModel(ABC):
    """Abstract strain-model interface."""

    name: str = "abstract"

    @abstractmethod
    def mu_strain(self, r, ctx: StrainContext) -> np.ndarray:
        """Strain chemical-potential contribution mu_strain(r) [J/mol]."""

    @abstractmethod
    def d_mu_strain_d_r(self, r, ctx: StrainContext) -> np.ndarray:
        """Derivative d mu_strain / d r [J/mol]."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
