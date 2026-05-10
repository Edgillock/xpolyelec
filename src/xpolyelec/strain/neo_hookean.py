"""Neo-Hookean strain model (Gaussian-chain limit, no finite extensibility).

For an incompressible isotropic Neo-Hookean solid,

G_strain = (n_x k_B T / 2) * (I - 3)

where I = lambda^2 + 2/lambda for uniaxial or I = lambda^2 + 1/lambda^2 in the
paper's 2D-isotropic convention. Taking d/d n_s with n_x = n_m / N fixed,
and using phi_p = n_m vbar_m / (n_m vbar_m + n_s vbar_s),

mu_strain = N_av * d G_strain / d n_s = (R T vbar_s / (N vbar_m)) * (phi_p0 * phi_p)^0.5 * (lambda^4 - 1) / lambda^3

This removes the small-strain limit of the Gent model (I_crit -> infinity),
where the denominator -(I_crit + 2)*lam^3 + lam^5 + lam / I_crit ~ -lam^3,
yielding the above closed form up to sign.

No finite-extensibility singularity.

Parameters
----------
Only N is used (plus phi_p0 via the kinematics).
"""
from __future__ import annotations

import numpy as np

from xpolyelec.strain.base import StrainContext, StrainModel


class NeoHookeanStrain(StrainModel):
    """Gaussian-chain (no finite extensibility) strain model."""

    name = "neo_hookean"

    def mu_strain(self, r, ctx: StrainContext) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        phi_p = ctx.phi_p(r_arr)
        phi_p0 = ctx.phi_p0
        lam = ctx.lambda_of_r(r_arr)
        tp = ctx.transport
        prefactor = (tp.R * tp.T * ctx.vbar_s_nm3) / (ctx.N * ctx.vbar_m_nm3)
        # (phi_p0 phi_p)^0.5 * (lam^4 - 1) / lam^3
        return prefactor * (phi_p0 * phi_p) ** 0.5 * (lam**4 - 1.0) / lam**3

    def d_mu_strain_d_r(self, r, ctx: StrainContext) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        h = np.maximum(1.0e-5, np.abs(r_arr) * 1.0e-5)
        return (self.mu_strain(r_arr + h, ctx) - self.mu_strain(r_arr - h, ctx)) / (2.0 * h)
