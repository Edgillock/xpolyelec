"""Reproduce Figs. 2-7 of Patel 2025 using the bundled paper fits.

Run from the repository root::

    python examples/reproduce_figures.py
"""
from __future__ import annotations

import numpy as np

from xpolyelec import Simulation


def main() -> None:
    # --- Baseline simulation with paper fits as defaults ---
    baseline = Simulation(verbose=False)
    baseline.set_strain_model("none")

    # --- Crosslink simulation (Gent, N=16, Δ=0.06) ---
    crosslink = Simulation(verbose=False)
    crosslink.set_strain_model("gent")

    # Fig. 2: property fits
    baseline.plot("fig2", save="fig2_properties.png")

    # Fig. 3: J1(r) overlay
    baseline.plot(
        "fig3",
        {"Baseline": baseline.J, "Crosslink (Gent)": crosslink.J},
        save="fig3_J1.png",
    )

    # Fig. 4: r(x/L), λ(x/L) at iL = 3.2e-4 mA/cm = 3.2e-7 A/cm (paper's stability limit)
    iL = 3.2e-7  # A/cm (SI-consistent)
    prof_b = baseline.compute_profile(iL=iL, ravg=0.08)
    prof_c = crosslink.compute_profile(iL=iL, ravg=0.08)
    baseline.plot(
        "fig4",
        {"Baseline": prof_b, "Crosslink (Gent)": prof_c},
        save="fig4_profiles.png",
    )

    # Fig. 5: J2(r) overlay
    baseline.plot(
        "fig5",
        {"Baseline": baseline.J, "Crosslink (Gent)": crosslink.J},
        save="fig5_J2.png",
    )

    # Fig. 6: Δφ decomposition at iL_lim
    pd_b = baseline.compute_potential(iL=iL, ravg=0.08)
    pd_c = crosslink.compute_potential(iL=iL, ravg=0.08)
    baseline.plot(
        "fig6",
        {
            "Baseline": {
                "ohmic": pd_b.delta_phi_ohmic,
                "conc": pd_b.delta_phi_conc,
                "strain": pd_b.delta_phi_strain,
                "total": pd_b.delta_phi_total,
            },
            "Crosslink": {
                "ohmic": pd_c.delta_phi_ohmic,
                "conc": pd_c.delta_phi_conc,
                "strain": pd_c.delta_phi_strain,
                "total": pd_c.delta_phi_total,
            },
        },
        save="fig6_decomposition.png",
    )

    # Fig. 7: current-voltage sweep (A/cm; paper's 1e-6 to 5e-4 mA/cm = 1e-9 to 5e-7 A/cm)
    iL_range = (1.0e-9, 5.0e-7)
    iv_b = baseline.compute_iv_curve(ravg=0.08, iL_range=iL_range, n=20, log_spaced=True)
    iv_c = crosslink.compute_iv_curve(ravg=0.08, iL_range=iL_range, n=20, log_spaced=True)
    baseline.plot(
        "fig7",
        {"Baseline": iv_b, "Crosslink (Gent)": iv_c},
        save="fig7_iv_curve.png",
    )

    print("Done. Figures written: fig2_properties.png ... fig7_iv_curve.png")


if __name__ == "__main__":
    main()
