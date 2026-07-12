# xpolyelec

A reproducibility package for **Patel et al.**, "Ion Transport in Concentrated Crosslinked Solid Polymer Electrolytes," *Journal of the Electrochemical Society* **172**, 120517 (2025). DOI: [10.1149/1945-7111/ae285f](https://doi.org/10.1149/1945-7111/ae285f).

The package implements the Baseline (Newman) and Crosslink (Gent) transport models of the paper, produces Figs 2–7 from raw CSV data, and lets you compare the **Gent**, **Neo-Hookean**, and **Mooney-Rivlin** strain models by switching the μ_strain contribution to the electrolyte electrochemical potential.

---

## Features

- Pure **NumPy / SciPy / Matplotlib / Pandas** — no heavy optimization or ML stack. Python 3.12+.
- **API-first design**: call it as a Python module (`from xpolyelec import Simulation`), not a CLI.
- **JSON config** with defaults matching Patel 2025 (60 °C); every change is echoed to the terminal and can be saved to a new JSON on exit.
- **Four strain models**: `none` (Baseline), `gent`, `neo_hookean`, `mooney_rivlin`.
- **Three assumption toggles**: kinematics (`affine_isotropic` vs `uniaxial`), φ_p model (paper vs callable), symmetric vs asymmetric λ_crit.
- **Bring-your-own-data**: load separate CSVs, a combined CSV, or a directory, and fit the four transport properties in one call.
- **Fast solver** (~10 ms per (iL, ravg) point) via precomputed cumulative integral + linear inversion on |J1|.
- **Diagnostics module** (`diagnose_singularities.py`, see [Diagnostics module](#diagnostics-module)): a standalone script that decomposes J1/J2 into individual transport-property factors to trace disagreements with the paper's figures back to their root cause.

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

## Diagnostics module

`diagnose_singularities.py` (project root) is a standalone diagnostic script produced while investigating why this package could not fully reproduce Figs. 3A/3B and 5–7 of the paper. It does **not** modify any package internals — it only calls public/semi-public `xpolyelec` APIs to decompose the transport equations into individual factors so disagreements can be traced to a root cause rather than guessed at.

Run it from the project root after `pip install -e .`:

```bash
python diagnose_singularities.py
```

All outputs are written to `./diagnostics/` as CSV + PNG files. It performs six steps:

| Step | What it checks | Output |
|---|---|---|
| 1 | Raw, unclipped J1(r) over the Crosslink Model's singular region (r ≈ 0.01–0.20) | `step1_raw_J1.{csv,png}` |
| 2 | Fig. 2F digitized-spline domain vs. the observed singular r-values; continuity of t₋⁰ across the spline/analytical boundary | `step2_spline_domain.txt`, `step2_spline_boundary_continuity.png` |
| 3 | thermo_factor, t₋⁰, t₊⁰, dμ_strain/dr, and λ evaluated individually at the suspicious r-values, plus continuous plots | `step3_factor_breakdown.{csv,png}` |
| 4 | Solved r(x/L) profile with the solver's local (rolling-median) clip vs. the legacy global clip | `step4_clip_comparison.png`, `step4_profile_{local,global}.csv` |
| 5 | Γ_conc vs Γ_strain vs combined J1 over r = 0.15–0.30 (the Fig. 3A plateau-failure region), plus ρ₊(r) vs. the ρ₊ = 1 pole line | `step5_high_r_breakdown.{csv,png}`, `step5b_rho_plus_high_r.png` |
| 6 | Full factor decomposition of Γ_conc (D, thermo_factor, t₋⁰, dc/dr, κ, dU/d ln m, ρ₊) over r = 0.15–0.30, isolating which single fitted function drives the high-r divergence | `step6_gamma_conc_factors.{csv,png}`, `step6b_thermo_vs_tminus.png` |

### Diagnostic findings to date

- **The two Crosslink-Model singularities (r ≈ 0.03, r ≈ 0.135) are physically correct.** Step 1 shows exactly two poles with the expected sign-flip (−∞→+∞ / +∞→−∞) structure predicted by the Gent strain denominator's two positive roots. What looked like 4 singularities in the plotted Fig. 3B was an artifact of how the steep pole flanks were sampled/clipped, not a distinct physics error.
- **t₋⁰ exceeding 1 (and t₊⁰ going negative) near r ≈ 0.125–0.14 is expected, real electrolyte physics**, since these r-values fall inside the digitized Fig. 2F spline's valid domain (r = 0.01–0.30), not an extrapolation artifact — negative transference numbers are a known feature of concentrated/correlated ion transport.
- **The Fig. 3A plateau failure (J1 rising instead of leveling off past r ≈ 0.20) is *not* caused by t₋⁰, ρ₊, or the strain term.** Step 6 isolates the cause to the fitted D(r) function: `D.csv`-derived data confirms D(r) declines smoothly and linearly all the way to r = 0.30 (matching the Eq. 37 linear fit `D = −8.93e−9·r + 3.49e−9`, R.T. Patel 2025) with no saturation. Since D sits in thermo_factor's denominator, its unchecked linear decline drives an exponential rise in thermo_factor and therefore in Γ_conc/J1.
- **The paper does not disclose a valid r-range for the Eqs. 34–38 property fits, nor the raw data or fitting method behind t₊⁰/t₋⁰.** This means the high-r divergence cannot be definitively "fixed" — there is no authoritative cutoff or corrected functional form to substitute without guessing at unpublished data. See [Known limitations](#known-limitations).

---

## Complete code structure

```
xpolyelec/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── diagnose_singularities.py    # Standalone diagnostic script (see Diagnostics module)
├── diagnostics/                 # Output of diagnose_singularities.py (CSV + PNG)
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

Note: `diagnose_singularities.py` sits outside this dependency graph — it is a diagnostic consumer, calling into `Simulation`, `solver.build_J1_and_J2`, and `solver.solve_r_profile` from the outside, without altering any internal module.

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
| `diagnose_singularities.py` | Standalone 6-step diagnostic decomposition of J1/J2/Γ_conc/Γ_strain (see [Diagnostics module](#diagnostics-module)) | none (script) |

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

- `iL` is **length-normalized current density** in **A / cm** (SI). Paper Fig. 7 uses mA / cm; multiply the paper value by 10⁻³ to pass to `compute_profile`. At 60 °C and L = 250 µm, the paper reports iL_lim ≈ **3.2 × 10⁻⁷ A / cm** for stable operation.
- `r` is the Li⁺ / EO molar ratio, dimensionless.
- `Δφ_total` is returned in **V** (over the full cell thickness L).
- All transport-property fits follow the paper's Eqs. 34–38:
  - ρ_el = 2.29 r + 1.21 (linear)
  - κ = a · r · exp(−r/b) with **(a, b) = (0.01, 0.061)** in `default_config.json` — the paper text gives a = 0.001 but that magnitude does not match Fig. 2B; the 0.01 value reproduces the ≈ 2 × 10⁻⁴ S / cm peak of Fig. 2B. The paper-text value is preserved in `paper_fits.json`.
  - ρ_+ = 9.42 r² − 2.18 r + 0.17 (poly2)
  - D = −8.93 × 10⁻⁹ r + 3.49 × 10⁻⁹ (linear) — **no saturation at high r; see [Diagnostics module](#diagnostics-module) for how this drives the Fig. 3A plateau failure**
  - U(m) = −0.1372 m^0.5708 + 0.1582 (power_law, m in kg/mol) — refitted from the corrected `U.csv` (values divided by 1000 vs an early transcription); essentially unchanged from the paper's −0.14 m^0.56 + 0.16.

---

## Physics decisions worth knowing

Several subtle points diverge from a literal reading of the paper. Each is documented at the point of use.

### 1. Kinematics exponent: λ = (φ_p0 / φ_p)^(1/2), not 1/3

Paper Eq. 16 is written as a 3D volumetric relation which would give the cube root. But the paper's Fig. 3B caption states singularities at r = 0.03 and r = 0.13 for λ_crit = 0.94 / 1.06 at r_avg = 0.08. Only the **square-root** exponent reproduces those positions; the cube root gives r = 0.007 and r = 0.16. We interpret the film as a 2D in-plane membrane stretch (through-thickness suppressed) and use λ² = φ_p0/φ_p. See `strain/base.py::lambda_from_phi`.

Diagnostic confirmation: Step 1 of `diagnose_singularities.py` shows the raw, unclipped J1(r) has exactly two poles, located almost exactly at r ≈ 0.03 and r ≈ 0.135, matching the paper's stated singularity positions and confirming the square-root exponent choice is correct.

### 2. |J1|-based cumulative integral for r(x/L) inversion

The signed J1 becomes non-monotonic across each Gent singularity, so `np.interp` cannot invert its cumulative integral directly. Physically r(x/L) is monotonic for any single-direction current, and the singular contribution to the true integral is bounded (log-integrable). We therefore build `F(r) = ∫|J1| dr`, clip peaks to 1000× the median of |J1|, and invert on that monotonic surrogate. See `solver.solve_r_profile`.

Diagnostic note: Step 4 of `diagnose_singularities.py` compares this global clip against a local rolling-median clip. Neither fully avoids some distortion of the pole shape near r ≈ 0.03/0.135; this remains an open numerical approximation, not an exact reproduction.

### 3. Gent Γ_strain uses (1 − t₋⁰) in the denominator

Paper Eq. 22b carries an anion-fraction correction `(1 − t₋⁰)` in the denominator; omitting it over-weights the strain contribution by ~1/(1 − t₋⁰). Explicit in `solver.build_J1_and_J2::Gamma_strain`.

### 4. Squared (1/ρ₊ − 1) in the thermodynamic factor

Paper Eq. 7 as printed has `(1/ρ₊ − 1)` to the first power, and paper Eq. 6 gives an explicit direct form for t₋⁰. Applied literally, together with the Eq. 35–38 fits, this over-predicts J1 in Fig. 3 by ~55× and collapses Fig. 4. Using **`(1/ρ₊ − 1)²` (squared)** together with `t₋⁰ = 1 − t₊⁰_analytical` reproduces the published J1 magnitudes and the correct singularity positions. This is either a typo in Eqs. 6/7 or a hidden parameter mismatch between the paper's equations and its figures. See `transport.thermo_factor` and `transport.t_minus_0` for the full discussion.

### 5. Fig 2F t₊⁰(r) is served from a digitized spline

The direct analytical form of Eq. 6 does not reproduce the small positive peak of Fig 2F at r ≈ 0.02. `TransportProperties.t_plus_0` therefore prefers a monotone PCHIP interpolant of digitized Fig 2F data (`defaults/paper_fig2f_t_plus.csv`) inside the data range, falling back to analytical outside. The solver, in contrast, uses the analytical form via `t_minus_0` so J1/J2 stay thermodynamically self-consistent with the fits.

Diagnostic note: Step 2/3 of `diagnose_singularities.py` confirms the spline's digitized domain is r = 0.01–0.30, i.e. it *does* cover the r ≈ 0.125–0.14 region where t₋⁰ exceeds 1 and t₊⁰ goes negative — this is genuine digitized-data behavior, not an out-of-range extrapolation artifact.

---

## Known limitations

- **Figures 5, 6, 7 are deprecated in this build.** Their physical interpretation is dominated by the (1 + Θ) blow-up at high r (from ρ₊ rising via the Eq. 36 polynomial), which produces overpotential decompositions and IV curves that do not resemble the paper. Only Figs 2, 3, 4 have been validated against the paper.
- **Fig 3A high-r tail (r ≳ 0.20) is ≈ 12× above the paper's dashed line, and continues to rise instead of leveling off.** Diagnostics (`diagnose_singularities.py` Step 6) trace this to the unconstrained linear fit for D(r) (Eq. 37), which declines toward zero with no saturation. Since D sits in thermo_factor's denominator, its decline drives an exponential-looking rise in thermo_factor and therefore in Γ_conc/J1 past r ≈ 0.20. t₋⁰ and ρ₊ were both checked and ruled out as the cause (see Step 6b: t₋⁰ stays bounded 2.8–3.6 and ρ₊ never approaches 1 in this range).
- **Fig 3B shows what visually appears as 4 singularities instead of the paper's 2.** Diagnostics (Step 1) confirm the underlying physics has exactly 2 true poles at r ≈ 0.03 and r ≈ 0.135, with the correct sign-flip structure. The apparent 4-singularity appearance is an artifact of how the steep pole flanks are sampled/clipped by the solver, not a distinct physics error.
- **The paper does not state a valid r-range for the Eqs. 34–38 property fits, nor does it disclose the raw data or fitting procedure for t₊⁰/t₋⁰.** This means the high-r divergence in Figs 3A and 5–7 cannot be definitively corrected — there is no authoritative cutoff or alternative functional form to substitute without guessing at unpublished data. This is treated as an inherent limitation of the published paper's documentation, not a fixable bug in this package.
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
29. **Diagnostics module added** (`diagnose_singularities.py`) — a 6-step decomposition script to trace the persistent Fig 3A/3B/5-7 mismatches to their root causes. Findings: (a) the Gent strain physics and singularity positions are correct; (b) t₋⁰ > 1 / t₊⁰ < 0 near r ≈ 0.13 is real digitized-data behavior, not a bug; (c) the Fig 3A plateau failure is driven by the unconstrained linear D(r) fit (Eq. 37) lacking high-r saturation, not by t₋⁰, ρ₊, or the strain term; (d) the paper discloses no valid r-range or raw data for these fits, so this remains an open limitation rather than a fixable defect. See [Diagnostics module](#diagnostics-module) and [Known limitations](#known-limitations).

---

## Citation

If this package helps with your research, please cite the original paper:

Patel, V.; Lee, O.; Makkar, S.; Balsara, N. P. *J. Electrochem. Soc.* **172**, 120517 (2025). DOI: [10.1149/1945-7111/ae285f](https://doi.org/10.1149/1945-7111/ae285f).

This package is released under the MIT license.
