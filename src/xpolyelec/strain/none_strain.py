"""Baseline: Newman's model: no strain contribution."""
from __future__ import annotations

import numpy as np

from xpolyelec.strain.base import StrainContext, StrainModel


class NoStrain(StrainModel):
    """mu_strain = 0. Recovers Newman's concentrated solution theory.

    This is the Baseline Model in the paper Patel 2025 (equivalent to N -> infinity).
    """

    name = "none"

    def mu_strain(self, r, ctx: StrainContext) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=float))

    def d_mu_strain_d_r(self, r, ctx: StrainContext) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=float))
