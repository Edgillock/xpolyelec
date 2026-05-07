"""Mooney-Rivlin strain model.

Mooney-Rivlin generalises Neo-Hookean by adding a second invariant I_2: W = C1 * (I_1 - 3) + C2 * (I_2 - 3)

For an incompressible isotropic solid with the paper's stretch kinematics,
I_1 = lam^2 + 1/lam^2   (paper's 2D convention)
I_2 = lam^-2 + lam^2   (equal to I_1 under this 2D simplification)

To make Mooney-Rivlin genuinely distinct from Neo-Hookean in the 2D convention
we use the 3D form for I_2:
    I_1_3D = lam^2 + 2/lam      (uniaxial) or  lam^2 + 1/lam^2  (2D)
    I_2_3D = 2 lam + lam^-2      (uniaxial)

Here we implement the simplest chemically sensible version:

W_MR = (n_x k_B T / 2) * [ (1 - alpha) * (lam^2 + 1/lam^2 - 2) + alpha * (lam + 1/lam - 2) ]

alpha = C2/(C1+C2) is a mixing fraction (set in config as
strain_model.mooney_rivlin_C_ratio). alpha = 0 -> Neo-Hookean;
alpha -> 1 weights the second invariant more strongly.

The derivative d mu_strain / d r is obtained by finite difference as for
the Gent model.
"""
from __future__ import annotations

import numpy as np

from xpolyelec.strain.base import StrainContext, StrainModel


class MooneyRivlinStrain(StrainModel):
    """Mooney-Rivlin two-parameter strain model (no finite extensibility)."""

    name = "mooney_rivlin"

    def mu_strain(self, r, ctx: StrainContext) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        phi_p = ctx.phi_p(r_arr)
        phi_p0 = ctx.phi_p0
        lam = ctx.lambda_of_r(r_arr)
        alpha = float(ctx.extra.get("mooney_rivlin_C_ratio", 0.0))
        alpha = max(0.0, min(1.0, alpha))

        tp = ctx.transport

        # d/d ns of W_MR(lam(ns)).  Through the chain rule
        #   d lam / d ns = lam' = -0.5 * lam * (1/phi_p) * (d phi_p / d ns)
        # and  d phi_p / d ns = -phi_p^2 * vbar_s / (n_m * vbar_m) (Eq. 17b)
        # the algebra collapses (see Neo-Hookean case) to a prefactor times
        # a lambda-only function. For Mooney-Rivlin, the lambda-only function is
        # a weighted sum of the two invariants' derivatives.
        prefactor = (tp.R * tp.T * ctx.vbar_s_nm3) / (ctx.N * ctx.vbar_m_nm3) * (phi_p0 * phi_p) ** 0.5

        # d/d lam of (lam^2 + 1/lam^2 - 2) = 2 lam - 2/lam^3
        #   -> under kinematic chain, contributes factor lam^2 * (1 - 1/lam^4)
        #      proportional to (lam^4 - 1)/lam^3 (matches Neo-Hookean in shape).
        nh_term = (lam**4 - 1.0) / lam**3

        # d/d lam of (lam + 1/lam - 2) = 1 - 1/lam^2
        #   -> under kinematic chain, contributes factor lam * (1 - 1/lam^2)
        #      proportional to (lam^2 - 1)/lam^2
        mr_term = (lam**2 - 1.0) / lam**2

        return prefactor * ((1.0 - alpha) * nh_term + alpha * mr_term)

    def d_mu_strain_d_r(self, r, ctx: StrainContext) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        h = np.maximum(1.0e-5, np.abs(r_arr) * 1.0e-5)
        return (self.mu_strain(r_arr + h, ctx) - self.mu_strain(r_arr - h, ctx)) / (2.0 * h)
