"""Gent finite-elasticity strain model (Patel 2025 Crosslink Model).

Implementation of Patel et al. 2025, Eq. 19:

    mu_strain(r) = - (R T I_crit vbar_s) / (2 N vbar_m) * (phi_p0 * phi_p)^0.5 * (lambda^4 - 1) / ( -(I_crit + 2) * lambda^3 + lambda^5 + lambda )

where
    I = lambda^2 + 1/lambda^2 (Eq. 13)
    I_crit = lambda_crit^2 + 1/lambda_crit^2 (Eq. 14)
    lambda = (phi_p0 / phi_p)^0.5 (Eq. 16)
    phi_p = 1 - vbar_s * c (paper)


The derivative d mu_strain / d r is evaluated by centered finite
differences. 
"""
from __future__ import annotations

import numpy as np

from xpolyelec.strain.base import StrainContext, StrainModel


class GentStrain(StrainModel):
    """Patel 2025 Gent-based Crosslink Model."""

    name = "gent"

    def mu_strain(self, r, ctx: StrainContext) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        phi_p = ctx.phi_p(r_arr)
        phi_p0 = ctx.phi_p0
        lam = ctx.lambda_of_r(r_arr)
        # Strain invariants (paper Eq. 13/14); we allow asymmetric lambda_crit
        lam_crit = ctx.lambda_crit_for(lam)
        I = lam**2 + 1.0 / lam**2
        I_crit = lam_crit**2 + 1.0 / lam_crit**2

        # Denominator: -(I_crit + 2)*lam^3 + lam^5 + lam
        denom = -(I_crit + 2.0) * lam**3 + lam**5 + lam

        # Prefactor: -(R T I_crit vbar_s) / (2 N vbar_m)
        #   vbar_s and vbar_m have the same units (nm^3) so the ratio is
        #   dimensionless; R T has units J/mol; result has units J/mol.
        tp = ctx.transport
        prefactor = -(tp.R * tp.T * I_crit * ctx.vbar_s_nm3) / (2.0 * ctx.N * ctx.vbar_m_nm3)

        numer = (phi_p0 * phi_p) ** 0.5 * (lam**4 - 1.0)

        # prevents lam == lam_crit singularity: clip denom near zero
        eps = 1.0e-14
        denom_safe = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)

        return prefactor * numer / denom_safe

    def d_mu_strain_d_r(self, r, ctx: StrainContext) -> np.ndarray:
        """Centered finite-difference derivative w.r.t. r."""
        r_arr = np.asarray(r, dtype=float)
        # Adaptive step: 1e-5 in r, but no smaller than 1e-8 absolute
        h = np.maximum(1.0e-5, np.abs(r_arr) * 1.0e-5)
        up = self.mu_strain(r_arr + h, ctx)
        dn = self.mu_strain(r_arr - h, ctx)
        return (up - dn) / (2.0 * h)
