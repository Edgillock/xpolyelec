"""Compare all four strain models side-by-side.

Demonstrates the strategy-pattern architecture: the same Simulation pipeline
is driven with four different strain models and their results overlaid.

Run::

    python examples/compare_strain_models.py
"""
from __future__ import annotations

from xpolyelec import Simulation


def main() -> None:
    models = ["none", "gent", "neo_hookean", "mooney_rivlin"]
    sims = {}
    for m in models:
        s = Simulation(verbose=False)
        s.set_strain_model(m)
        sims[m] = s

    sim_ref = sims["none"]

    # J1 overlay
    sim_ref.plot(
        "fig3",
        {m: s.J for m, s in sims.items()},
        save="compare_J1.png",
    )

    # Profiles at a modest iL
    iL = 2.0e-4
    profiles = {m: s.compute_profile(iL=iL, ravg=0.08) for m, s in sims.items()}
    sim_ref.plot("fig4", profiles, save="compare_profiles.png")

    # Current-voltage sweep
    iv = {}
    for m, s in sims.items():
        iv[m] = s.compute_iv_curve(ravg=0.08, iL_range=(1.0e-6, 5.0e-4), n=15, log_spaced=True)
    sim_ref.plot("fig7", iv, save="compare_iv_curve.png")

    print("Strain-model comparison figures saved.")


if __name__ == "__main__":
    main()
