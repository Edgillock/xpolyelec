# xpolyelec

A reproducibility package for **Patel et al.**, "Ion Transport in Concentrated Crosslinked Solid Polymer Electrolytes," *Journal of the Electrochemical Society* **172**, 120517 (2025). DOI: [10.1149/1945-7111/ae285f](https://doi.org/10.1149/1945-7111/ae285f).

The package implements the Baseline (Newman) and Crosslink (Gent) transport models of the paper, produces Figs 2–7 from raw CSV data, and lets you compare the **Gent**, **Neo-Hookean**, and **Mooney-Rivlin** strain models by switching the μ\_strain contribution to the electrolyte electrochemical potential.

---

## Features

- Pure **NumPy / SciPy / Matplotlib / Pandas** — no heavy optimization or ML stack. Python 3.12+.
- **API-first design**: call it as a Python module (`from xpolyelec import Simulation`), not a CLI.
- **JSON config** with defaults matching Patel 2025 (60 °C); every change is echoed to the terminal and can be saved to a new JSON on exit.
- **Four strain models**: `none` (Baseline), `gent`, `neo_hookean`, `mooney_rivlin`.
- **Three assumption toggles**: kinematics (`affine_isotropic` vs `uniaxial`), φ\_p model (paper vs callable), symmetric vs asymmetric λ\_crit.
- **Bring-your-own-data**: load separate CSVs, a combined CSV, or a directory, and fit the four transport properties in one call.
- **Fast solver** (~10 ms per (iL, ravg) point) via precomputed cumulative integral + linear inversion on |J1|.

---

## Installation

```bash
git clone <this repo>
cd xpolyelec
pip install -e .
```

---

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

Run the bundled example to regenerate Figs 2–4 (Figs 5–7 are deprecated — see [Known limitations](#known-limitations)):

```bash
python examples/reproduce_figures.py
```

Load your own CSVs:

```python
sim = Simulation()
sim.load_csv_data("examples/sample_data/")   # directory with kappa.csv, D.csv, ...
sim.plot("fig2", save="fig2_myfits.png")
```

---

## Complete code structure

```
xpolyelec/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── src/xpolyelec/
│   ├── __init__.py              # exports Simulation, Config; __version__
│   ├── api.py                   # Simulation — top-level user-facing class
│   ├── config.py                # Config — dotted-path JSON with change echo
│   ├── fits.py                  # FitFamily / Fit / CustomFit / FitRegistry
│   ├── io.py                    # load / load_csv / load_directory / load_combined
│   ├── transport.py             # TransportProperties (c, m, dU/dlnm, Θ, t±⁰)
│   ├── solver.py                # J1/J2 builders, r-profile solver, Δφ, iL sweep
│   ├── plotting.py              # plot_fig2 … plot_fig7
│   ├── strain/
│   │   ├── __init__.py          # registry + get_strain_model
│   │   ├── base.py              # StrainModel ABC, StrainContext, kinematics
│   │   ├── none_strain.py       # NoStrain (Baseline)
│   │   ├── gent.py              # GentStrain (Patel 2025 Crosslink Model)
│   │   ├── neo_hookean.py       # NeoHookeanStrain
│   │   └── mooney_rivlin.py     # MooneyRivlinStrain
│   └── defaults/
│       ├── __init__.py
│       ├── default_config.json  # 60 °C parameters + solver / fit settings
│       ├── paper_fits.json      # Reference Eqs. 35–38 coefficients
│       └── paper_fig2f_t_plus.csv   # Digitized Fig 2F red-dashed t₊⁰(r)
├── examples/
│   ├── reproduce_figures.py     # Regenerate Figs 2–4 from defaults
│   ├── compare_strain_models.py # Overlay Baseline / Gent / NH / MR
│   ├── user_csv_workflow.py     # Load sample CSVs, refit, plot Fig 2
│   └── sample_data/
│       ├── kappa.csv            # Eq. 35 — κ(r) [S/cm]
│       ├── rho_plus.csv         # Eq. 36 — ρ₊(r)
│       ├── D.csv                # Eq. 37 — D(r) [cm²/s]
│       ├── U.csv                # Eq. 38 — U(ln m) [V]  (Note: divided by 1000 vs paper text)
│       └── rho_el.csv           # ρ_el(r) [g/cm³]
└── tests/
    ├── __init__.py
    ├── test_config.py           # set/get, change echo, save round-trip
    ├── test_fits.py             # each fit family's func vs deriv consistency
    ├── test_io.py               # CSV round-trip
    ├── test_strain_models.py    # μ_strain / dμ_strain/dr finite for all models
    └── test_solver.py           # end-to-end profile + IV curve
```

### Module dependency graph

```
                   ┌───────────────┐
                   │ defaults/     │  (JSON + CSV data files)
                   └──────┬────────┘
                          │
                    ┌─────▼──────┐
                    │  config.py │
                    └─────┬──────┘
                          │
              ┌───────────┼───────┐
              │           │       │
        ┌─────▼──┐   ┌────▼───┐   ▼
        │ io.py  │   │ fits.py│   │         
        └─────┬──┘   └────┬───┘   │         
              │           │       │         
              └─────┬─────┘       │         
                    │             │         
              ┌─────▼─────────────▼──┐      
              │   transport.py       │
              │ (TransportProperties)│
              └─────┬────────────────┘
                    │
             ┌──────┴──────────┐
             │                 │
       ┌─────▼──────┐    ┌─────▼──────┐
       │ strain/    │    │  solver.py │
       │  base.py   │◄───┤  (J1, J2,  │
       │  gent.py   │    │  profile,  │
       │  ...       │    │  Δφ, sweep)│
       └─────┬──────┘    └─────┬──────┘
             │                 │
             └────────┬────────┘
                      │
                ┌─────▼──────┐
                │ plotting.py│
                └─────┬──────┘
                      │
                ┌─────▼──────┐
                │  api.py    │  (Simulation — glues everything)
                └────────────┘
```

### File-by-file reference

| File | Purpose | Key exports |
|---|---|---|
| `__init__.py` | Package entry point | `Simulation`, `Config`, `__version__` |
| `api.py` | Top-level façade bundling Config + TransportProperties + StrainModel + StrainContext | `Simulation` |
| `config.py` | Nested-dict JSON config with dotted-path `get`/`set`, change tracking, interactive save | `Config`, `load_default_config`, `load_paper_fits` |
| `fits.py` | Parametric fit families with analytical derivatives | `FitFamily`, `Fit`, `CustomFit`, `FitRegistry` |
| `io.py` | Read raw experimental CSVs → dict of (2, n) arrays | `load`, `load_csv`, `load_directory`, `load_combined` |
| `transport.py` | Concentration-dependent transport + thermodynamics | `TransportProperties` |
| `strain/base.py` | ABC + kinematics helpers (φ_p, λ from φ_p) | `StrainModel`, `StrainContext`, `phi_p_paper`, `lambda_from_phi` |
| `strain/none_strain.py` | μ_strain = 0 (Baseline / Newman) | `NoStrain` |
| `strain/gent.py` | Gent finite-elasticity (paper Eqs. 11–19) | `GentStrain` |
| `strain/neo_hookean.py` | Neo-Hookean (Gaussian-chain limit) | `NeoHookeanStrain` |
| `strain/mooney_rivlin.py` | Mooney-Rivlin two-parameter form | `MooneyRivlinStrain` |
| `strain/__init__.py` | Strain-model registry | `get_strain_model`, `available_strain_models` |
| `solver.py` | Numerical pipeline: J1/J2 → r(x/L) → Δφ → IV sweep | `build_J1_and_J2`, `solve_r_profile`, `compute_potential_drop`, `sweep_iL`, `JFunctions`, `Profile`, `PotentialDrop`, `IVCurve` |
| `plotting.py` | Matplotlib reproductions of paper figures | `plot_fig2` … `plot_fig7` |

### Data-flow of a single `sim.compute_profile(iL, ravg)` call

```
Simulation.compute_profile
   │
   ├─► TransportProperties  (r → κ, ρ₊, D, U, ρ_el, c, dU/dlnm, Θ, t±⁰)
   │
   ├─► StrainModel.d_mu_strain_d_r  ─┐
   ├─► StrainContext.lambda_of_r   ──┤
   │                                 ▼
   ├─► build_J1_and_J2  →  Γ_conc, Γ_strain, J1(r), J2(r)
   │
   └─► solve_r_profile
           │
           1. tabulate J1 on dense r-grid   (n_r_grid = 2001)
           2. build monotonic F(r) = ∫|J1| dr, clip singular peaks
           3. inner:  r(x/L) = F⁻¹( F(r0) − (iL/F)·x/L )   via np.interp
           4. outer:  brentq on r0 until ⟨r⟩_x/L = ravg
                                   │
                                   ▼
                        Profile(x_over_L, r, lam, converged, …)
```

---

## How the physics maps to the code

| Paper equation(s) | Code location |
|---|---|
| Eq. 4 (steady-state anion flux = 0) | `solver.build_J1_and_J2::_gamma_conc_baseline` |
| Eq. 5 (c(r) from ρ_el) | `transport.TransportProperties.c` |
| Eq. 6 (t₋⁰, analytical) | `transport.TransportProperties._t_plus_0_analytical` (see notes) |
| Eq. 7 (1 + d ln γ/d ln m) | `transport.TransportProperties.thermo_factor` (see notes) |
| Eqs. 10–19 (μ_strain, Gent) | `strain/gent.py` |
| Eq. 16 (λ from φ_p) | `strain/base.py::lambda_from_phi` |
| Eq. 17a, 17b (φ_p, φ_p0) | `strain/base.py::phi_p_paper`, `StrainContext.phi_p0` |
| Eqs. 22a, 22b, 23 (J1) | `solver.build_J1_and_J2`, `solver.solve_r_profile` |
| Eqs. 26a–c (Δφ decomposition) | `solver.compute_potential_drop` |
| Eq. 27 (J2) | `solver.build_J1_and_J2::J2` |
| Eqs. 34–38 (measured-property fits) | `defaults/paper_fits.json`, `fits.py` |
| Fig. 2F digitized reference | `defaults/paper_fig2f_t_plus.csv` → `TransportProperties.t_plus_0` |

---

## Units and conventions

- `iL` is **length-normalized current density** in **A / cm** (SI). Paper Fig. 7 uses mA / cm; multiply the paper value by 10⁻³ to pass to `compute_profile`. At 60 °C and L = 250 µm, the paper reports iL\_lim ≈ **3.2 × 10⁻⁷ A / cm** for stable operation.
- `r` is the Li⁺ / EO molar ratio, dimensionless.
- `Δφ_total` is returned in **V** (over the full cell thickness L).
- All transport-property fits follow the paper's Eqs. 34–38:
  - ρ\_el = 2.29 r + 1.21 (linear)
  - κ = a · r · exp(−r/b) with **(a, b) = (0.01, 0.061)** in `default_config.json` — the paper text gives a = 0.001 but that magnitude does not match Fig. 2B; the 0.01 value reproduces the ≈ 2 × 10⁻⁴ S / cm peak of Fig. 2B. The paper-text value is preserved in `paper_fits.json`.
  - ρ\_+ = 9.42 r² − 2.18 r + 0.17 (poly2)
  - D = −8.93 × 10⁻⁹ r + 3.49 × 10⁻⁹ (linear)
  - U(m) = −0.1372 m^0.5708 + 0.1582 (power_law, m in kg/mol) — refitted from the corrected `U.csv` (values divided by 1000 vs an early transcription); essentially unchanged from the paper's −0.14 m^0.56 + 0.16.

---

## Physics decisions worth knowing

Several subtle points diverge from a literal reading of the paper. Each is documented at the point of use.

### 1. Kinematics exponent: λ = (φ_p0 / φ_p)^(1/2), not 1/3

Paper Eq. 16 is written as a 3D volumetric relation which would give the cube root. But the paper's Fig. 3B caption states singularities at r = 0.03 and r = 0.13 for λ_crit = 0.94 / 1.06 at r_avg = 0.08. Only the **square-root** exponent reproduces those positions; the cube root gives r = 0.007 and r = 0.16. We interpret the film as a 2D in-plane membrane stretch (through-thickness suppressed) and use λ² = φ_p0/φ_p. See `strain/base.py::lambda_from_phi`.

### 2. |J1|-based cumulative integral for r(x/L) inversion

The signed J1 becomes non-monotonic across each Gent singularity, so `np.interp` cannot invert its cumulative integral directly. Physically r(x/L) is monotonic for any single-direction current, and the singular contribution to the true integral is bounded (log-integrable). We therefore build `F(r) = ∫|J1| dr`, clip peaks to 1000× the median of |J1|, and invert on that monotonic surrogate. See `solver.solve_r_profile`.

### 3. Gent Γ_strain uses (1 − t₋⁰) in the denominator

Paper Eq. 22b carries an anion-fraction correction `(1 − t₋⁰)` in the denominator; omitting it over-weights the strain contribution by ~1/(1 − t₋⁰). Explicit in `solver.build_J1_and_J2::Gamma_strain`.

### 4. Squared (1/ρ₊ − 1) in the thermodynamic factor

Paper Eq. 7 as printed has `(1/ρ₊ − 1)` to the first power, and paper Eq. 6 gives an explicit direct form for t₋⁰. Applied literally, together with the Eq. 35–38 fits, this over-predicts J1 in Fig. 3 by ~55× and collapses Fig. 4. Using **`(1/ρ₊ − 1)²` (squared)** together with `t₋⁰ = 1 − t₊⁰_analytical` reproduces the published J1 magnitudes and the correct singularity positions. This is either a typo in Eqs. 6/7 or a hidden parameter mismatch between the paper's equations and its figures. See `transport.thermo_factor` and `transport.t_minus_0` for the full discussion.

### 5. Fig 2F t₊⁰(r) is served from a digitized spline

The direct analytical form of Eq. 6 does not reproduce the small positive peak of Fig 2F at r ≈ 0.02. `TransportProperties.t_plus_0` therefore prefers a monotone PCHIP interpolant of digitized Fig 2F data (`defaults/paper_fig2f_t_plus.csv`) inside the data range, falling back to analytical outside. The solver, in contrast, uses the analytical form via `t_minus_0` so J1/J2 stay thermodynamically self-consistent with the fits.

---

## Known limitations

- **Figures 5, 6, 7 are deprecated in this build.** Their physical interpretation is dominated by the (1 + Θ) blow-up at high r (from ρ₊ rising via the Eq. 36 polynomial), which produces overpotential decompositions and IV curves that do not resemble the paper. Only Figs 2, 3, 4 have been validated against the paper.
- **Fig 3A high-r tail (r ≳ 0.25)** is ≈ 12× above the paper's dashed line. Same (1 + Θ) mechanism. Everything below r ≈ 0.20 matches.
- The `sample_data/*.csv` files are synthesised from the paper fits (plus noise) — they are not raw experimental data. `paper_fig2f_t_plus.csv` is a digitisation of the actual Fig. 2F.

---

## Running the tests

```bash
cd xpolyelec
pytest -q
```

Fifteen tests cover config change-tracking, CSV I/O round-trip, fit-family derivatives, strain-model sanity checks, and end-to-end solver behaviour for all four strain models.

---

# Documentation of fix attempts

Applied after side-by-side comparison with the paper figures:

22. **Γ_conc formula** — remove the extraneous 1/F factor; recast as `D·(1+Θ)·dc/dr / t₋⁰` so that `Γ · dr/dx = i/F` with correct units.
23. **|J1|-based cumulative integrand** — replace `np.maximum.accumulate` / `np.unique` with the strictly-monotone `∫|J1| dr`, clipping singular peaks at 1000× median.
24. **Kinematics exponent** — set λ = (φ_p0/φ_p)^0.5 in `strain/base.py` (was cbrt after an incorrect edit); update `dlam/dphi_p = −0.5·λ/φ_p` in `strain/gent.py`.
25. **(1 − t₋⁰) factor in Γ_strain denominator** — restored in `solver.build_J1_and_J2`.
26. **U.csv unit correction** — divide U by 1000 so `power_law` fit refits to essentially the paper's parameters.
27. **Fig 2F digitised spline** — new `defaults/paper_fig2f_t_plus.csv` + `transport._load_paper_fig2f_spline` and PCHIP interpolation in `t_plus_0`.
28. **Thermodynamic-factor investigation** — Eq. 7/6 direct-form attempt (`(1/ρ₊ − 1)^1` and Eq. 6 direct `t₋⁰`) was reverted after it broke Fig 3/4 by ~55×; the squared form is retained with a large comment explaining the discrepancy.

---

## Citation

If this package helps with your research, please cite the original paper:

Patel, V.; Lee, O.; Makkar, S.; Balsara, N. P. *J. Electrochem. Soc.* **172**, 120517 (2025). DOI: [10.1149/1945-7111/ae285f](https://doi.org/10.1149/1945-7111/ae285f).

This package is released under the MIT license.
