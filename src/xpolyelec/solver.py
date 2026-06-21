"""Core numerical pipeline.

func: build_J1_and_J2 → callable J1(r), J2(r) given a StrainModel + StrainContext + TransportProperties.
func: solve_r_profile → three-step iterative procedure that finds the r(x/L) profile for a target (ravg, iL).
func: compute_potential_drop → phi_ohmic, phi_conc, phi_strain, phi_total.
func: sweep_iL → current-voltage relationship over a range of iL.

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


def _dc_dr_centred(transport: TransportProperties, r, h: float = 1.0e-5):
    """Centred finite difference of c(r) (mol/L per unit r)."""
    r_arr = np.asarray(r, dtype=float)
    h_arr = np.maximum(h, np.abs(r_arr) * h)
    return (transport.c(r_arr + h_arr) - transport.c(r_arr - h_arr)) / (2.0 * h_arr)


def build_J1_and_J2(
    transport: TransportProperties,
    strain_model: StrainModel,
    ctx: StrainContext,
) -> JFunctions:
    """Build callable J1(r), J2(r) from the transport properties + strain model.

    For the Crosslink Model the strain piece (Eq. 22b) is added; it is
    handled by Gamma_strain below.

    J2 (Eq. 27): (2RT/F) (1 - t_-^0)(1 + Theta) / r  +  strain term.
    """
    RT = transport.R * transport.T
    F = transport.F

    def _gamma_conc_baseline(r):
        r = np.asarray(r, dtype=float)
        # c(r) returns mol/L; convert to mol/cm^3 to be consistent with
        # D [cm^2/s] and kappa [S/cm].
        dc_dr_mol_cm3 = _dc_dr_centred(transport, r) * 1.0e-3
        D = transport.D(r)
        tf = transport.thermo_factor(r)                # (1 + Theta)
        t_minus = transport.t_minus_0(r)
        t_safe = np.where(np.abs(t_minus) < 1e-12, 1e-12, t_minus)
        # Paper Eq. 4 → Gamma_conc^Baseline = D (1+Theta) dc/dr / t_-^0
        return D * tf * dc_dr_mol_cm3 / t_safe

    def Gamma_conc(r):
        return _gamma_conc_baseline(r)

    def Gamma_strain(r):
        r = np.asarray(r, dtype=float)
        if strain_model.name == "none":
            return np.zeros_like(r)
        gc = _gamma_conc_baseline(r)
        tf = transport.thermo_factor(r)
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


# ----------------------------------------------------------------------
# r(x/L) profile solver
# ----------------------------------------------------------------------
@dataclass
class Profile:
    """Result of solve_r_profile."""

    x_over_L: np.ndarray
    r: np.ndarray
    lam: np.ndarray
    ravg_target: float
    ravg_achieved: float
    iL: float
    iterations: int
    converged: bool


def _integrate_J1(J1: Callable, r_lo: float, r_hi: float, abs_tol: float, rel_tol: float) -> float:
    if r_hi <= r_lo:
        return 0.0
    val, _err = quad(J1, r_lo, r_hi, epsabs=abs_tol, epsrel=rel_tol, limit=200)
    return float(val)


def solve_r_profile(
    J: JFunctions,
    ctx: StrainContext,
    *,
    iL: float,
    ravg: float,
    n_points: int = 1001,
    r0_bracket: tuple[float, float] = (1.0e-4, 0.30),
    ravg_tol: float = 1.0e-6,
    max_iter: int = 80,
    F: float = 96485.33212,
    quad_abs_tol: float = 1.0e-10,   # kept for API compatibility; unused
    quad_rel_tol: float = 1.0e-8,    # kept for API compatibility; unused
    n_r_grid: int = 2001,
) -> Profile:
    """Three-step iterative procedure from Patel 2025 (fast inversion form).
    precompute J1 on a dense r-grid once, then use cumulative trapezoid
    integration to build F(r) = integral of J1 from r_min to r. For any trial
    r0 (= r at x/L = 0), the profile satisfies F(r(x/L)) - F(r0) = -(iL/F) * (x/L)
    """
    xL = np.linspace(0.0, 1.0, n_points)
    r_min = max(1.0e-5, r0_bracket[0])
    r_max = min(0.5, r0_bracket[1])
    r_grid = np.linspace(r_min, r_max, n_r_grid)

    # Evaluate J1 across the grid (vectorised).
    J1_vals = np.asarray(J.J1(r_grid), dtype=float)
    # Guard against NaN / inf from singularities
    J1_vals = np.where(np.isfinite(J1_vals), J1_vals, 0.0)

    J1_abs = np.abs(J1_vals)
    j1_scale = float(np.median(J1_abs[J1_abs > 0])) if (J1_abs > 0).any() else 0.0
    if j1_scale > 0.0:
        clip_hi = 1.0e3 * j1_scale
        J1_abs = np.minimum(J1_abs, clip_hi)

    dr = np.diff(r_grid)
    trap = 0.5 * (J1_abs[:-1] + J1_abs[1:]) * dr
    F_vals = np.concatenate(([0.0], np.cumsum(trap)))

    F_mono = np.maximum.accumulate(F_vals)

    def _profile_for_r0(r0: float) -> np.ndarray:
        F_at_r0 = float(np.interp(r0, r_grid, F_mono))
        target = F_at_r0 - (iL / F) * xL
        return np.interp(target, F_mono, r_grid,
                         left=r_grid[0], right=r_grid[-1])

    def _ravg_residual(r0: float) -> float:
        r_vals = _profile_for_r0(r0)
        return float(np.trapezoid(r_vals, xL) - ravg)

    # Outer brentq on r0
    lo, hi = r_min, r_max
    f_lo = _ravg_residual(lo)
    f_hi = _ravg_residual(hi)
    converged = False
    iterations = 0
    r0_final = float(ravg)
    if f_lo * f_hi < 0.0:
        try:
            r0_final, info = brentq(
                _ravg_residual, lo, hi, xtol=ravg_tol, maxiter=max_iter, full_output=True
            )
            converged = bool(info.converged)
            iterations = int(info.iterations)
        except Exception:
            pass
    else:
        # As a fallback, choose r0 = ravg (often near the true solution).
        r0_final = float(ravg)

    r_profile = _profile_for_r0(r0_final)
    lam = ctx.lambda_of_r(r_profile)
    ravg_achieved = float(np.trapezoid(r_profile, xL))
    return Profile(
        x_over_L=xL,
        r=r_profile,
        lam=lam,
        ravg_target=float(ravg),
        ravg_achieved=ravg_achieved,
        iL=float(iL),
        iterations=iterations,
        converged=converged,
    )


# ----------------------------------------------------------------------
# Potential drop
# ----------------------------------------------------------------------
@dataclass
class PotentialDrop:
    iL: float
    ravg: float
    delta_phi_ohmic: float
    delta_phi_conc: float
    delta_phi_strain: float
    delta_phi_total: float
    converged: bool


def compute_potential_drop(
    J: JFunctions,
    transport: TransportProperties,
    strain_model: StrainModel,
    ctx: StrainContext,
    profile: Profile,
    *,
    quad_abs_tol: float = 1.0e-10,
    quad_rel_tol: float = 1.0e-8,
) -> PotentialDrop:
    """Evaluate phi_ohmic, phi_conc, phi_strain from a solved r profile."""
    xL = profile.x_over_L
    r_prof = profile.r
    iL = profile.iL
    RT_over_F = (transport.R * transport.T) / transport.F

    # --- Ohmic (integral over x/L) ---
    kappa_vals = np.asarray(transport.kappa(r_prof))
    integrand_ohmic = 1.0 / np.clip(kappa_vals, 1.0e-30, None)
    delta_phi_ohmic = float(iL * np.trapezoid(integrand_ohmic, xL))

    # --- Concentration & strain (integrals over r from r(x/L=1) to r(x/L=0)) ---
    r_lo = float(r_prof[-1])  # x/L = 1
    r_hi = float(r_prof[0])   # x/L = 0

    def conc_integrand(r):
        t_minus = transport.t_minus_0(r)
        tf = transport.thermo_factor(r)
        return 2.0 * RT_over_F * (1.0 - t_minus) * tf / r

    def strain_integrand(r):
        if strain_model.name == "none":
            return 0.0
        t_minus = transport.t_minus_0(r)
        dmu_dr = strain_model.d_mu_strain_d_r(np.asarray([r]), ctx)[0]
        return t_minus * dmu_dr / transport.F

    try:
        dphi_conc, _ = quad(conc_integrand, r_lo, r_hi, epsabs=quad_abs_tol, epsrel=quad_rel_tol, limit=200)
    except Exception:
        dphi_conc = 0.0
    try:
        dphi_strain, _ = quad(strain_integrand, r_lo, r_hi, epsabs=quad_abs_tol, epsrel=quad_rel_tol, limit=200)
    except Exception:
        dphi_strain = 0.0

    total = delta_phi_ohmic + float(dphi_conc) + float(dphi_strain)
    return PotentialDrop(
        iL=iL,
        ravg=profile.ravg_target,
        delta_phi_ohmic=delta_phi_ohmic,
        delta_phi_conc=float(dphi_conc),
        delta_phi_strain=float(dphi_strain),
        delta_phi_total=total,
        converged=profile.converged,
    )


# ----------------------------------------------------------------------
# iL sweep (current-voltage relationship)
# ----------------------------------------------------------------------
@dataclass
class IVCurve:
    iL: np.ndarray
    delta_phi_total: np.ndarray
    delta_phi_ohmic: np.ndarray
    delta_phi_conc: np.ndarray
    delta_phi_strain: np.ndarray
    ravg: float
    converged: np.ndarray
    i_lim: float | None


def sweep_iL(
    J: JFunctions,
    transport: TransportProperties,
    strain_model: StrainModel,
    ctx: StrainContext,
    *,
    ravg: float,
    iL_values: np.ndarray,
    n_points: int = 201,
    r0_bracket: tuple[float, float] = (1.0e-4, 0.30),
    ravg_tol: float = 1.0e-6,
    quad_abs_tol: float = 1.0e-10,
    quad_rel_tol: float = 1.0e-8,
) -> IVCurve:
    """Sweep iL → solve profile + compute pih for each value. Used for Fig. 7."""
    iL_values = np.asarray(iL_values, dtype=float)
    dphi_tot = np.empty_like(iL_values)
    dphi_oh = np.empty_like(iL_values)
    dphi_co = np.empty_like(iL_values)
    dphi_st = np.empty_like(iL_values)
    conv = np.zeros_like(iL_values, dtype=bool)

    for k, iL in enumerate(iL_values):
        prof = solve_r_profile(
            J, ctx,
            iL=iL, ravg=ravg, n_points=n_points, r0_bracket=r0_bracket,
            ravg_tol=ravg_tol, F=transport.F,
            quad_abs_tol=quad_abs_tol, quad_rel_tol=quad_rel_tol,
        )
        pd = compute_potential_drop(
            J, transport, strain_model, ctx, prof,
            quad_abs_tol=quad_abs_tol, quad_rel_tol=quad_rel_tol,
        )
        dphi_tot[k] = pd.delta_phi_total
        dphi_oh[k] = pd.delta_phi_ohmic
        dphi_co[k] = pd.delta_phi_conc
        dphi_st[k] = pd.delta_phi_strain
        conv[k] = prof.converged

    # Limiting current: largest iL where the profile still converges and
    # total potential hasn't exploded.
    try:
        finite = np.isfinite(dphi_tot) & conv
        reasonable = dphi_tot < 1.0e3  # V/cm; anything larger is runaway
        mask = finite & reasonable
        i_lim = float(iL_values[mask].max()) if mask.any() else None
    except Exception:
        i_lim = None

    return IVCurve(
        iL=iL_values,
        delta_phi_total=dphi_tot,
        delta_phi_ohmic=dphi_oh,
        delta_phi_conc=dphi_co,
        delta_phi_strain=dphi_st,
        ravg=float(ravg),
        converged=conv,
        i_lim=i_lim,
    )
