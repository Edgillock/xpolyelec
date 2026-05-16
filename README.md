# xpolyelec

A reproducibility package for **Patel et al.**, "Effect of Network Strain on Ion Transport in Crosslinked Solid Polymer Electrolytes," *Journal of the Electrochemical Society* **172**, 120517 (2025).

The package implements the Baseline and Crosslink transport models of the paper, produces Figs 2–7 from raw CSV data, and lets you compare the **Gent**, **Neo-Hookean**, and **Mooney-Rivlin** strain models by switching the μ\_strain contribution to the electrolyte electrochemical potential.

## Features

- Pure **NumPy / SciPy / Matplotlib / Pandas** — no heavy optimization or ML stack.
- **API-first design**: call it as a Python module, not a CLI.
- **JSON config** with defaults matching Patel 2025 (60 °C); any change is displayed to the terminal and can be saved to a new JSON on exit.
- **Four strain models**: `none` (Baseline), `gent`, `neo_hookean`, `mooney_rivlin`.
- **Three assumption toggles**: kinematics (affine_isotropic vs uniaxial), φ\_p model (paper vs callable), symmetric vs asymmetric λ\_crit.
- **Bring-your-own-data**: load separate CSVs, a combined CSV, or a directory, and fit the four transport properties in one call.
- **Fast solver** (~10 ms per (iL, ravg) point) via precomputed cumulative integral + linear inversion.

## Installation

Python 3.12 or newer is required.

```bash
git clone <this repo>
cd xpolyelec
pip install -e .
```

## Quick start

```python
from xpolyelec import Simulation

# Defaults: Patel 2025 60 °C fits, Baseline model.
sim = Simulation()

# Switch to Gent strain model:
sim.set_strain_model("gent")

# Change any config value — the change is echoed to the terminal.
sim.set_param("strain_model.delta_lambda", 0.08)

# Solve the steady-state profile and potential drop.
prof = sim.compute_profile(iL=3.2e-7, ravg=0.08)   # iL in A/cm
pd   = sim.compute_potential(iL=3.2e-7, ravg=0.08)

# Reproduce paper figures.
sim.plot("fig4", {"Gent": prof}, save="fig4.png")

# Save the (possibly-tweaked) config to a new JSON.
sim.save_config()   # prompts for a filename interactively
```

Run the bundled example to regenerate all six figures:

```bash
python examples/reproduce_figures.py
```

Or load your own CSVs:

```python
sim = Simulation()
sim.load_csv_data("examples/sample_data/")   # directory with kappa.csv, D.csv, ...
sim.plot("fig2", save="fig2_myfits.png")
```

## Layout

```
xpolyelec/
├── pyproject.toml
├── src/xpolyelec/
│   ├── config.py          # Config (dotted-path access, change echo, save prompt)
│   ├── fits.py            # FitRegistry with linear/kappa_peak/poly2/poly3/power_law/exp
│   ├── io.py              # load_csv / load_directory / load_combined
│   ├── transport.py       # TransportProperties (c, m, t_-^0, thermo_factor)
│   ├── strain/
│   │   ├── base.py        # StrainModel ABC + StrainContext + kinematics
│   │   ├── none_strain.py
│   │   ├── gent.py
│   │   ├── neo_hookean.py
│   │   └── mooney_rivlin.py
│   ├── solver.py          # J1, J2, solve_r_profile, compute_potential_drop, sweep_iL
│   ├── plotting.py        # plot_fig2 … plot_fig7
│   ├── api.py             # Simulation (top-level class)
│   └── defaults/
│       ├── default_config.json
│       └── paper_fits.json
├── examples/
│   ├── reproduce_figures.py
│   ├── compare_strain_models.py
│   ├── user_csv_workflow.py
│   └── sample_data/       # Synthesized CSVs from paper fits + noise
└── tests/
    ├── test_config.py
    ├── test_fits.py
    ├── test_io.py
    ├── test_strain_models.py
    └── test_solver.py
```

## Units and conventions

- `iL` is **length-normalized current density** in **A / cm** (SI). Paper Fig. 7 uses mA / cm; multiply the paper value by 10⁻³ to pass to `compute_profile`. At 60 °C and L = 250 µm, the paper reports iL\_lim = 3.2 × 10⁻⁴ mA / cm = **3.2 × 10⁻⁷ A / cm** for stable operation.
- `r` is the Li⁺ / EO molar ratio (Patel Eq. 4), dimensionless.
- `Δφ_total` is returned in **V** (over the full cell thickness L, not per unit length).
- All transport-property fits follow the paper's Eqs. 34–38:
  - ρ\_el = 2.29 r + 1.21 (linear)
  - κ = a · r · exp(−r/b) with (a, b) = (0.001 or 0.01, 0.061) — peak near r = b
  - ρ\_+ = 9.42 r² − 2.18 r + 0.17 (poly2)
  - D = −8.93 × 10⁻⁹ r + 3.49 × 10⁻⁹ (linear)
  - U(m) = −0.14 m^0.56 + 0.16 with m in kg / mol (power_law)
- The kappa prefactor in `default_config.json` is set to `0.01` so the absolute magnitude matches typical xPEO / LiTFSI at 60 °C (≈ 2 × 10⁻⁴ S / cm at peak). The paper-reported value of `0.001` is preserved in `paper_fits.json` for reference.

## Running the tests

```bash
cd xpolyelec
pytest -q
```

Fifteen tests cover config change-tracking, CSV I/O round-trip, fit-family derivatives, strain-model sanity checks, and end-to-end solver behaviour for all four strain models.

---

## How the physics maps to the code

| Paper equation(s) | Code location |
|---|---|
| Eq. 4 (steady-state anion flux = 0) | `solver.build_J1_and_J2::_gamma_conc_baseline` |
| Eq. 5 (c(r) from ρ\_el) | `transport.TransportProperties.c` |
| Eq. 6 (t\_−⁰) | `transport.TransportProperties.t_minus_0` |
| Eq. 7 (1 + d ln γ / d ln m) | `transport.TransportProperties.thermo_factor` |
| Eqs. 10–19 (μ\_strain derivation, Gent) | `strain/gent.py` |
| Eq. 17a, 17b (φ\_p, φ\_p0) | `strain/base.py::phi_p_paper`, `StrainContext.phi_p0` |
| Eqs. 22a, 22b, 23 (J1) | `solver.build_J1_and_J2`, `solver.solve_r_profile` |
| Eqs. 26a–c (Δφ decomposition) | `solver.compute_potential_drop` |
| Eq. 27 (J2) | `solver.build_J1_and_J2` (J2 lambda) |
| Eqs. 34–38 (measured-property fits) | `defaults/paper_fits.json`, `fits.py` |

## Citation

If this package helps with your research, please cite the original paper:

Patel, A. N. et al. *Journal of the Electrochemical Society* **172**, 120517 (2025). DOI: [10.1149/1945-7111/adaed0](https://doi.org/10.1149/1945-7111/adaed0) (or as published).

This package is released under the MIT license.
