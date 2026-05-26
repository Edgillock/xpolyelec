"""matplotlib reproductions of Figs. 2-7 from Patel 2025.

plot_fig2  transport property fits vs r   (6 sub-panels)
plot_fig3  J1(r) for chosen models        (overlay)
plot_fig4  r(x/L) profile + lambda(x/L)
plot_fig5  J2(r) for chosen models        (overlay)
plot_fig6  phi/L decomposition vs x/L
plot_fig7  phi_ss/L vs iL  (current-voltage relationship)
"""
from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from xpolyelec.solver import IVCurve, JFunctions, Profile
from xpolyelec.transport import TransportProperties


# ----------------------------------------------------------------------
# Fig. 2: transport property fits
# ----------------------------------------------------------------------
def plot_fig2(
    transport: TransportProperties,
    raw_data: dict[str, np.ndarray] | None = None,
    r_grid: np.ndarray | None = None,
):
    """Reproduce Fig. 2: rho_el, kappa, rho_plus, D, U, t_-^0 vs r.

    raw_data (optional) maps property name → (2, n) array of experimental
    (x, y) points to overlay as scatter.
    """
    if r_grid is None:
        r_grid = np.linspace(1.0e-3, 0.30, 300)
    raw_data = raw_data or {}

    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle("Fig. 2 — Transport and thermodynamic properties of xPEO/LiTFSI")

    # A: rho_el
    ax = axs[0, 0]
    ax.plot(r_grid, transport.rho_el(r_grid), "k--", label="Fit")
    if "rho_el" in raw_data:
        ax.plot(raw_data["rho_el"][0], raw_data["rho_el"][1], "o", mfc="C0", mec="k", label="Experiment")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\rho_{\mathrm{el}}$ [g/cm$^3$]")
    ax.set_title("A")
    ax.legend(fontsize=8)

    # B: kappa (log-y)
    ax = axs[0, 1]
    ax.semilogy(r_grid, transport.kappa(r_grid), "k--", label="Fit")
    if "kappa" in raw_data:
        ax.semilogy(raw_data["kappa"][0], raw_data["kappa"][1], "o", mfc="C0", mec="k", label="Experiment")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\kappa$ [S/cm]")
    ax.set_title("B")
    ax.legend(fontsize=8)

    # C: rho_plus
    ax = axs[0, 2]
    ax.plot(r_grid, transport.rho_plus(r_grid), "k--", label="Fit")
    if "rho_plus" in raw_data:
        ax.plot(raw_data["rho_plus"][0], raw_data["rho_plus"][1], "o", mfc="C0", mec="k", label="Experiment")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\rho_+$")
    ax.set_title("C")
    ax.legend(fontsize=8)

    # D: D (log-y)
    ax = axs[1, 0]
    ax.semilogy(r_grid, np.maximum(transport.D(r_grid), 1.0e-30), "k--", label="Fit")
    if "D" in raw_data:
        ax.semilogy(raw_data["D"][0], raw_data["D"][1], "o", mfc="C0", mec="k", label="Experiment")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$D$ [cm$^2$/s]")
    ax.set_title("D")
    ax.legend(fontsize=8)

    # E: U vs ln m
    ax = axs[1, 1]
    m_grid = r_grid * 1000.0 / transport.M_EO
    ln_m_grid = np.log(m_grid)
    ax.plot(ln_m_grid, transport.U(m_grid), "k--", label="Fit")
    if "U" in raw_data:
        ax.plot(raw_data["U"][0], raw_data["U"][1], "o", mfc="C0", mec="k", label="Experiment")
    ax.set_xlabel(r"$\ln\,m$ [mol/kg]")
    ax.set_ylabel(r"$U$ [V]")
    ax.set_title("E")
    ax.legend(fontsize=8)

    # F: t_-^0
    ax = axs[1, 2]
    ax.plot(r_grid, transport.t_minus_0(r_grid), "k--", label=r"$t_-^0(r)$")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$t_-^0$")
    ax.set_title("F")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, axs


# ----------------------------------------------------------------------
# Fig. 3: J1(r)
# ----------------------------------------------------------------------
def plot_fig3(
    models: dict[str, JFunctions],
    r_grid: np.ndarray | None = None,
):
    """Overlay J1(r) for one or more model labels."""
    if r_grid is None:
        r_grid = np.linspace(0.01, 0.29, 400)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, J in models.items():
        ax.plot(r_grid, J.J1(r_grid), label=label)
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$J_1(r)$ [mol/(cm·s)]")
    ax.set_title("Fig. 3 — $J_1(r) = \\Gamma_{\\mathrm{conc}} + \\Gamma_{\\mathrm{strain}}$")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
# Fig. 4: r(x/L) and lambda(x/L)
# ----------------------------------------------------------------------
def plot_fig4(profiles: dict[str, Profile]):
    """Overlay r(x/L) and lambda(x/L) for one or more profiles."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for label, prof in profiles.items():
        axs[0].plot(prof.x_over_L, prof.r, label=label)
        axs[1].plot(prof.x_over_L, prof.lam, label=label)
    axs[0].set_xlabel(r"$x/L$")
    axs[0].set_ylabel(r"$r(x/L)$")
    axs[0].set_title("Fig. 4B — concentration profile")
    axs[0].legend(fontsize=8)

    axs[1].set_xlabel(r"$x/L$")
    axs[1].set_ylabel(r"$\lambda(x/L)$")
    axs[1].set_title("Fig. 4C — local extension")
    axs[1].axhline(1.0, color="grey", lw=0.5)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    return fig, axs


# ----------------------------------------------------------------------
# Fig. 5: J2(r)
# ----------------------------------------------------------------------
def plot_fig5(models: dict[str, JFunctions], r_grid: np.ndarray | None = None):
    if r_grid is None:
        r_grid = np.linspace(0.01, 0.29, 400)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, J in models.items():
        ax.plot(r_grid, J.J2(r_grid), label=label)
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$J_2(r)$ [V]")
    ax.set_title(r"Fig. 5 — $J_2(r)$ (concentration + strain overpotential integrand)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
# Fig. 6: Δφ decomposition
# ----------------------------------------------------------------------
def plot_fig6(potentials: dict[str, dict[str, float]]):
    """decomposition of phi_ohmic, phi_conc, phi_strain, phi_total.

    potentials maps model label → {"ohmic":..., "conc":..., "strain":...,
    "total":...} in V/cm.
    """
    labels = list(potentials.keys())
    components = ["ohmic", "conc", "strain", "total"]

    def _get(p, comp):
        # Accept either a dict {"ohmic":..,} or a PotentialDrop dataclass
        # with attributes delta_phi_ohmic / _conc / _strain / _total.
        if isinstance(p, dict):
            return p.get(comp, 0.0)
        return float(getattr(p, f"delta_phi_{comp}", 0.0) or 0.0)

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, comp in enumerate(components):
        vals = [_get(potentials[m], comp) for m in labels]
        ax.bar(x + (i - 1.5) * width, vals, width, label=comp)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta\phi / L$ [V/cm]")
    ax.set_title("Fig. 6 — Potential-drop decomposition")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
# Fig. 7: current-voltage relationship
# ----------------------------------------------------------------------
def plot_fig7(
    curves: dict[str, IVCurve],
    experimental: np.ndarray | None = None,
):
    """Overlay phi_ss/L vs iL for one or more IVCurves.

    experimental (optional): (2, n) array of (iL, phi/L) measured points.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, c in curves.items():
        ax.plot(c.iL, c.delta_phi_total, "--", label=label)
    if experimental is not None:
        ax.plot(experimental[0], experimental[1], "ko", mfc="k", label="Experiment")
    ax.set_xlabel(r"$iL$ [A/cm]")
    ax.set_ylabel(r"$\Delta\phi_{ss}$ [V]")
    ax.set_title("Fig. 7 — Current-voltage relationship")
    ax.legend()
    fig.tight_layout()
    return fig, ax
