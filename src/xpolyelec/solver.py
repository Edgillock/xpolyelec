"""Core numerical pipeline.

1. Gamma_strain now follows Eq. 22b's literal structure — c(r)*D(r)/(4*t_minus*(1+Theta))
   scaling the strain chemical-potential derivative — instead of rescaling
   Gamma_conc by r/(2RT). Old behaviour kept available via
   `strain_form="rescaled"` for A/B comparison; new literal form is
   `strain_form="literal"` (default).
2. solve_r_profile's clipping is now adaptive: instead of a single global
   1000x-median clip, we clip locally based on a rolling window so that
   sharp-but-legitimate singularity shapes near lambda_crit aren't
   flattened by whatever the global median happens to be elsewhere in
   the r-range. Controlled by `local_clip_window` and `local_clip_factor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from xpolyelec.strain.base import StrainContext, StrainModel
from xpolyelec.transport import TransportProperties


@dataclass
class JFunctions:
    J1: Callable[[np.ndarray], np.ndarray]
    J2: Callable[[np.ndarray], np.ndarray]
    Gamma_conc: Callable[[np.ndarray], np.ndarray]
    Gamma_strain: Callable[[np.ndarray], np.ndarray]


def _dc_dr_centred(transport: TransportProperties, r, h: float = 1.0e-5):
    r_arr = np.asarray(r, dtype=float)
    h_arr = np.maximum(h, np.abs(r_arr) * h)
    return (transport.c(r_arr + h_arr) - transport.c(r_arr - h_arr)) / (2.0 * h_arr)


def build_J1_and_J2(
    transport: TransportProperties,
    strain_model: StrainModel,
    ctx: StrainContext,
    strain_form: str = "literal",
) -> JFunctions:
    RT = transport.R * transport.T
    F = transport.F

    def _gamma_conc_baseline(r):
        r = np.asarray(r, dtype=float)
        dc_dr_mol_cm3 = _dc_dr_centred(transport, r) * 1.0e-3
        D = transport.D(r)
        tf = transport.thermo_factor(r)
        t_minus = transport.t_minus_0(r)
        t_safe = np.where(np.abs(t_minus) < 1e-12, 1e-12, t_minus)
        return D * tf * dc_dr_mol_cm3 / t_safe

    def Gamma_conc(r):
        return _gamma_conc_baseline(r)

    def Gamma_strain(r):
        r = np.asarray(r, dtype=float)
        if strain_model.name == "none":
            return np.zeros_like(r)

        t_minus = transport.t_minus_0(r)
        t_safe = np.where(np.abs(t_minus) < 1e-12, 1e-12, t_minus)
        tf = transport.thermo_factor(r)
        dmu_dr = strain_model.d_mu_strain_d_r(r, ctx)

        if strain_form == "literal":
            c_mol_cm3 = transport.c(r) * 1.0e-3
            D = transport.D(r)
            denom = 4.0 * t_safe * np.where(np.abs(tf) < 1e-12, 1e-12, tf)
            return -(c_mol_cm3 * D) / denom * (dmu_dr / RT)
        elif strain_form == "rescaled":
            gc = _gamma_conc_baseline(r)
            denom = 2.0 * RT * np.where(np.abs(tf) < 1e-12, 1e-12, tf)
            return gc * (r * dmu_dr / denom)
        else:
            raise ValueError(f"unknown strain_form: {strain_form!r}")

    def J1(r):
        return Gamma_conc(r) + Gamma_strain(r)

    def J2(r):
        r = np.asarray(r, dtype=float)
        t_minus = transport.t_minus_0(r)
        tf = transport.thermo_factor(r)
        conc_term = (2.0 * RT / F) * (1.0 - t_minus) * tf / r
        if strain_model.name == "none":
            return conc_term
        strain_term = t_minus * strain_model.d_mu_strain_d_r(r, ctx) / F
        return conc_term + strain_term

    return JFunctions(J1=J1, J2=J2, Gamma_conc=Gamma_conc, Gamma_strain=Gamma_strain)


@dataclass
class Profile:
    x_over_L: np.ndarray
    r: np.ndarray
    lam: np.ndarray
    ravg_target: float
    ravg_achieved: float
    iL: float
    iterations: int
    converged: bool


def _rolling_clip(values: np.ndarray, window: int, factor: float) -> np.ndarray:
    """Clip each point based on a local rolling median rather than a
    single global median, so legitimate sharp peaks near singularities
    keep more of their shape while still bounding runaway values.
    """
    n = len(values)
    if n == 0:
        return values
    half = max(1, window // 2)
    out = np.empty_like(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = values[lo:hi]
        nz = window_vals[window_vals > 0]
        local_med = float(np.median(nz)) if nz.size > 0 else 0.0
        cap = factor * local_med if local_med > 0 else values[i]
        out[i] = min(values[i], cap) if cap > 0 else values[i]
    return out


def solve_r_profile(
    J: JFunctions,
    ctx: StrainContext,
    *,
    iL: float,
    ravg: float,
    n_points: int = 201,
    r0_bracket: tuple[float, float] = (1.0e-4, 0.30),
    ravg_tol: float = 1.0e-6,
    max_iter: int = 80,
    F: float = 96485.33212,
    quad_abs_tol: float = 1.0e-10,
    quad_rel_tol: float = 1.0e-8,
    n_r_grid: int = 2001,
    use_local_clip: bool = True,
    local_clip_window: int = 41,
    local_clip_factor: float = 50.0,
    global_clip_factor: float = 1.0e3,
) -> Profile:
    xL = np.linspace(0.0, 1.0, n_points)
    r_min = max(1.0e-5, r0_bracket[0])
    r_max = min(0.5, r0_bracket[1])
    r_grid = np.linspace(r_min, r_max, n_r_grid)

    J1_vals = np.asarray(J.J1(r_grid), dtype=float)
    J1_vals = np.where(np.isfinite(J1_vals), J1_vals, 0.0)

    J1_abs = np.abs(J1_vals)
    if use_local_clip:
        J1_abs = _rolling_clip(J1_abs, local_clip_window, local_clip_factor)
    else:
        j1_scale = float(np.median(J1_abs[J1_abs > 0])) if (J1_abs > 0).any() else 0.0
        if j1_scale > 0.0:
            clip_hi = global_clip_factor * j1_scale
            J1_abs = np.minimum(J1_abs, clip_hi)

    dr = np.diff(r_grid)
    trap = 0.5 * (J1_abs[:-1] + J1_abs[1:]) * dr
    F_vals = np.concatenate(([0.0], np.cumsum(trap)))
    F_mono = np.maximum.accumulate(F_vals)

    def _profile_for_r0(r0: float) -> np.ndarray:
        F_at_r0 = float(np.interp(r0, r_grid, F_mono))
        target = F_at_r0 - (iL / F) * xL
        return np.interp(target, F_mono, r_grid, left=r_grid[0], right=r_grid[-1])

    def _ravg_residual(r0: float) -> float:
        r_vals = _profile_for_r0(r0)
        return float(np.trapezoid(r_vals, xL) - ravg)

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
    xL = profile.x_over_L
    r_prof = profile.r
    iL = profile.iL
    RT_over_F = (transport.R * transport.T) / transport.F

    kappa_vals = np.asarray(transport.kappa(r_prof))
    integrand_ohmic = 1.0 / np.clip(kappa_vals, 1.0e-30, None)
    delta_phi_ohmic = float(iL * np.trapezoid(integrand_ohmic, xL))

    r_lo = float(r_prof[-1])
    r_hi = float(r_prof[0])

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

    try:
        finite = np.isfinite(dphi_tot) & conv
        reasonable = dphi_tot < 1.0e3
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
