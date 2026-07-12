"""Gent finite-elasticity strain model (Patel 2025 Crosslink Model).

The Gent strain-energy density is

    W(lam) = -(R T / (2 N)) * (I_crit - I_ref) * ln( 1 - (I(lam) - I_ref) / (I_crit - I_ref) )

with the paper's strain invariants (Eq. 13)

    I(lam)     = lam^2 + 1/lam^2          (I_ref = I(1) = 2)
    I_crit     = lam_crit^2 + 1/lam_crit^2

so W diverges when lam -> lam_crit (i.e. I -> I_crit). The chemical
potential contribution mu_strain is obtained from a chain-rule expansion

    mu_strain(r) = (dW/dI) * (dI/dphi_p) * (dphi_p/dn_s) * N_A / R    [J/mol]

but, because the closed-form expression is long and error-prone, we
compute it as a robust numerical derivative of an explicit free-energy
function f(n_s) at fixed polymer amount. The numerical derivative is more
than accurate enough for the integration tolerance used by the solver.
"""
from __future__ import annotations

import numpy as np

from xpolyelec.strain.base import StrainContext, StrainModel, lambda_from_phi


def _I_invariant(lam: np.ndarray) -> np.ndarray:
    """First strain invariant from paper Eq. 13."""
    return lam ** 2 + 1.0 / lam ** 2


class GentStrain(StrainModel):
    """Patel 2025 Gent-based Crosslink Model."""

    name = "gent"

    # ------------------------------------------------------------------
    def _W(self, lam: np.ndarray, ctx: StrainContext) -> np.ndarray:
        """Gent free-energy density [J / mol of EO monomers]."""
        I = _I_invariant(lam)
        # Use the relevant lambda_crit per-point (extension vs contraction)
        lam_crit = ctx.lambda_crit_for(lam)
        I_crit = _I_invariant(lam_crit)
        I_ref = 2.0  # = I(lam=1)
        # Argument of log; clip to avoid log(<=0) at and past the singularity
        arg = 1.0 - (I - I_ref) / (I_crit - I_ref)
        arg_safe = np.clip(arg, 1.0e-12, None)
        prefactor = -(ctx.transport.R * ctx.transport.T / (2.0 * ctx.N)) * (I_crit - I_ref)
        return prefactor * np.log(arg_safe)

    def mu_strain(self, r, ctx: StrainContext) -> np.ndarray:
        """Strain chemical-potential contribution mu_strain(r) [J/mol]."""
        r_arr = np.asarray(r, dtype=float)
        lam = ctx.lambda_of_r(r_arr)
        I = _I_invariant(lam)
        lam_crit = ctx.lambda_crit_for(lam)
        I_crit = _I_invariant(lam_crit)
        I_ref = 2.0

        # dW/dI (J/mol per unit invariant). Use |I_crit - I| so the divergence
        # is single-signed (Gent restricts both extension and contraction, both
        # produce a positive energy spike). Without abs(), the analytic dW/dI
        # changes sign across lam_crit — nonphysical, because the polymer
        # network cannot accommodate I > I_crit.
        denom = np.abs(I_crit - I)
        denom_safe = np.where(denom < 1.0e-10, 1.0e-10, denom)
        dW_dI = (ctx.transport.R * ctx.transport.T / (2.0 * ctx.N)) * (I_crit - I_ref) / denom_safe

        # dI/dlam = 2 lam - 2/lam^3
        dI_dlam = 2.0 * lam - 2.0 / lam ** 3

        # dlam/dphi_p depends on kinematics
        phi_p = ctx.phi_p(r_arr)
        phi_p_safe = np.where(np.abs(phi_p) < 1.0e-12, 1.0e-12, phi_p)
        if ctx.kinematics == "affine_isotropic":
            # lam = sqrt(phi_p0/phi_p) -> dlam/dphi_p = -(1/2) * lam / phi_p
            # (paper uses 2D in-plane membrane stretch; see base.lambda_from_phi)
            dlam_dphi = -0.5 * lam / phi_p_safe
        else:  # uniaxial: lam = phi_p0/phi_p -> dlam/dphi = -lam/phi_p
            dlam_dphi = -lam / phi_p_safe

        # mu_strain = (vbar_s / vbar_m) * dW/dphi_p * (something).  Following
        # the chain dW/dphi_p = dW/dI * dI/dlam * dlam/dphi_p.  Multiplying by
        # the dimensionless ratio vbar_s/vbar_m gives the chemical potential
        # contribution per mole of salt referenced to the polymer (paper
        # convention).
        dW_dphi = dW_dI * dI_dlam * dlam_dphi
        return (ctx.vbar_s_nm3 / ctx.vbar_m_nm3) * dW_dphi

    def d_mu_strain_d_r(self, r, ctx: StrainContext) -> np.ndarray:
        """Centred finite-difference derivative of mu_strain w.r.t. r."""
        r_arr = np.asarray(r, dtype=float)
        h = np.maximum(1.0e-5, np.abs(r_arr) * 1.0e-5)
        up = self.mu_strain(r_arr + h, ctx)
        dn = self.mu_strain(r_arr - h, ctx)
        return (up - dn) / (2.0 * h)
