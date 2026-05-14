"""example of the intended user workflow.

1. Load user-provided CSVs (one per property) from a directory.
2. Switch to the Gent Crosslink Model and tweak N.
3. Compute Fig. 7 data and plot.
4. Save the modified config as a new JSON.
"""
from __future__ import annotations

from pathlib import Path

from xpolyelec import Simulation


HERE = Path(__file__).parent


def main() -> None:
    sim = Simulation()  # verbose=True by default -> prints changes

    #1. Load the sample CSVs 
    sim.load_csv_data(HERE / "sample_data")

    #2. Switch model + tweak parameters
    sim.set_strain_model("gent")
    sim.set_param("strain_model.N_monomers_per_strand", 20)
    sim.set_asymmetric_lambda_crit(ext=1.08, con=0.93)

    #3. Compute + plot
    # iL in A/cm; paper range is 1e-6 to 5e-4 mA/cm = 1e-9 to 5e-7 A/cm.
    iv = sim.compute_iv_curve(ravg=0.08, iL_range=(1.0e-9, 5.0e-7), n=15, log_spaced=True)
    sim.plot("fig7", {"My run": iv}, save="my_fig7.png")

    #4. Interactive save (reply 'y' at the prompt) 
    # sim.save_config()  # uncomment for interactive prompt
    sim.save_config("my_config.json", interactive=False)


if __name__ == "__main__":
    main()
