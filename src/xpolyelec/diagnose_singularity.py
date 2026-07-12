"""
performs the 5 diagnostic steps:
  1. Raw unclipped J1(r) over the singularity region.
  2. Fig 2F spline domain vs the four suspicious r-values.
  3. thermo_factor / t_minus_0 / d_mu_strain_d_r individually near each point.
  4. Global-clip vs local-clip comparison of the solved profile.
  5. Gamma_conc vs Gamma_strain breakdown at high r (Fig 3A plateau check).

Outputs are written to ./diagnostics/ as CSV + PNG so they can be shared.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from xpolyelec import Simulation
from xpolyelec.solver import build_J1_and_J2, solve_r_profile, _dc_dr_centred

OUTDIR = "diagnostics"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------
# Setup: Gent / Crosslink model at default (60 C) config, via Simulation
# facade so we don't have to guess low-level constructor signatures.
# ---------------------------------------------------------------------
sim = Simulation()
sim.set_strain_model("gent")

attrs = [a for a in dir(sim) if not a.startswith("_")]
print("Simulation public attributes:", attrs)


def _get_first(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n), n
    return None, None


transport, t_name = _get_first(sim, ["transport", "_transport", "props", "transport_properties"])
strain_model, sm_name = _get_first(sim, ["strain_model", "_strain_model", "model"])
ctx, ctx_name = _get_first(sim, ["ctx", "_ctx", "strain_context", "context"])

print(f"Resolved: transport -> sim.{t_name}, strain_model -> sim.{sm_name}, ctx -> sim.{ctx_name}")

if transport is None or strain_model is None or ctx is None:
    raise RuntimeError(
        f"Could not auto-resolve transport/strain_model/ctx from Simulation. "
        f"Public attributes were: {attrs}. Please check src/xpolyelec/api.py "
        "and edit the _get_first(...) name lists above."
    )

J = build_J1_and_J2(transport, strain_model, ctx, strain_form="literal")

POINTS_OF_INTEREST = [0.03, 0.04, 0.125, 0.14]

# =====================================================================
# STEP 1: raw unclipped J1(r) on a dense grid
# =====================================================================
r_dense = np.linspace(0.01, 0.20, 5000)
J1_raw = np.asarray(J.J1(r_dense), dtype=float)

np.savetxt(
    os.path.join(OUTDIR, "step1_raw_J1.csv"),
    np.column_stack([r_dense, J1_raw]),
    delimiter=",", header="r,J1_raw", comments="",
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r_dense, J1_raw, lw=1)
for rp in POINTS_OF_INTEREST:
    ax.axvline(rp, color="red", ls="--", lw=0.8, alpha=0.6)
finite = J1_raw[np.isfinite(J1_raw)]
if finite.size:
    cap = 5 * np.nanpercentile(np.abs(finite), 90)
    ax.set_ylim(-cap, cap)
ax.set_xlabel("r")
ax.set_ylabel("J1(r) [raw, unclipped]")
ax.set_title("Step 1: Raw J1(r), no clipping")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "step1_raw_J1.png"), dpi=150)
plt.close(fig)

# =====================================================================
# STEP 2: Fig 2F spline domain vs suspicious r-values
# =====================================================================
spline = getattr(transport, "_fig2f_spline", None)
spline_domain = (float(spline.x.min()), float(spline.x.max())) if spline is not None else None

with open(os.path.join(OUTDIR, "step2_spline_domain.txt"), "w") as f:
    f.write(f"Fig2F spline domain: {spline_domain}\n")
    f.write(f"Points of interest: {POINTS_OF_INTEREST}\n")
    if spline_domain is not None:
        for rp in POINTS_OF_INTEREST:
            near_edge = (
                abs(rp - spline_domain[0]) < 0.01
                or abs(rp - spline_domain[1]) < 0.01
            )
            f.write(f"  r={rp}: near spline boundary = {near_edge}\n")

if spline_domain is not None:
    lo, hi = spline_domain
    r_edge_lo = np.linspace(max(0.001, lo - 0.02), lo + 0.02, 400)
    r_edge_hi = np.linspace(hi - 0.02, hi + 0.02, 400)
    tm_lo = transport.t_minus_0(r_edge_lo)
    tm_hi = transport.t_minus_0(r_edge_hi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(r_edge_lo, tm_lo)
    axes[0].axvline(lo, color="red", ls="--")
    axes[0].set_title(f"t_minus_0 near spline LOW edge (r={lo:.4f})")
    axes[0].set_xlabel("r")
    axes[1].plot(r_edge_hi, tm_hi)
    axes[1].axvline(hi, color="red", ls="--")
    axes[1].set_title(f"t_minus_0 near spline HIGH edge (r={hi:.4f})")
    axes[1].set_xlabel("r")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "step2_spline_boundary_continuity.png"), dpi=150)
    plt.close(fig)

# =====================================================================
# STEP 3: individual factors at the four suspicious points
# =====================================================================
rows = []
for rp in POINTS_OF_INTEREST:
    r_arr = np.array([rp])
    tf = float(transport.thermo_factor(r_arr)[0])
    tm = float(transport.t_minus_0(r_arr)[0])
    tp = float(transport.t_plus_0(r_arr)[0])
    dmu = float(np.asarray(strain_model.d_mu_strain_d_r(r_arr, ctx))[0])
    lam = float(np.asarray(ctx.lambda_of_r(r_arr))[0])
    rows.append((rp, tf, tm, tp, dmu, lam))

with open(os.path.join(OUTDIR, "step3_factor_breakdown.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["r", "thermo_factor", "t_minus_0", "t_plus_0", "d_mu_strain_d_r", "lambda"])
    w.writerows(rows)

r_fine = np.linspace(0.015, 0.16, 3000)
tf_fine = transport.thermo_factor(r_fine)
tm_fine = transport.t_minus_0(r_fine)
dmu_fine = np.asarray(strain_model.d_mu_strain_d_r(r_fine, ctx))
lam_fine = np.asarray(ctx.lambda_of_r(r_fine))

fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
for ax, y, name in zip(
    axes, [tf_fine, tm_fine, dmu_fine, lam_fine],
    ["thermo_factor (1+Theta)", "t_minus_0", "d_mu_strain_d_r", "lambda"],
):
    ax.plot(r_fine, y, lw=1)
    for rp in POINTS_OF_INTEREST:
        ax.axvline(rp, color="red", ls="--", lw=0.7, alpha=0.6)
    ax.set_ylabel(name)
axes[-1].set_xlabel("r")
fig.suptitle("Step 3: individual factor behaviour near suspicious r-values")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "step3_factor_breakdown.png"), dpi=150)
plt.close(fig)

# =====================================================================
# STEP 4: global-clip vs local-clip solved profile comparison
# =====================================================================
prof_local = solve_r_profile(J, ctx, iL=3.2e-7, ravg=0.08, use_local_clip=True)
prof_global = solve_r_profile(J, ctx, iL=3.2e-7, ravg=0.08, use_local_clip=False)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(prof_local.x_over_L, prof_local.r, label="local (rolling) clip")
ax.plot(prof_global.x_over_L, prof_global.r, label="global (flat) clip", ls="--")
ax.set_xlabel("x/L")
ax.set_ylabel("r(x/L)")
ax.legend()
ax.set_title("Step 4: local vs global clip solved profile (ravg=0.08)")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "step4_clip_comparison.png"), dpi=150)
plt.close(fig)

np.savetxt(
    os.path.join(OUTDIR, "step4_profile_local.csv"),
    np.column_stack([prof_local.x_over_L, prof_local.r]),
    delimiter=",", header="x_over_L,r", comments="",
)
np.savetxt(
    os.path.join(OUTDIR, "step4_profile_global.csv"),
    np.column_stack([prof_global.x_over_L, prof_global.r]),
    delimiter=",", header="x_over_L,r", comments="",
)

# =====================================================================
# STEP 5: Gamma_conc vs Gamma_strain breakdown at high r (Fig 3A plateau)
# =====================================================================
r_high = np.linspace(0.15, 0.30, 2000)
gc_high = J.Gamma_conc(r_high)
gs_high = J.Gamma_strain(r_high)
j1_high = gc_high + gs_high

np.savetxt(
    os.path.join(OUTDIR, "step5_high_r_breakdown.csv"),
    np.column_stack([r_high, gc_high, gs_high, j1_high]),
    delimiter=",", header="r,Gamma_conc,Gamma_strain,J1", comments="",
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r_high, gc_high, label="Gamma_conc")
ax.plot(r_high, gs_high, label="Gamma_strain")
ax.plot(r_high, j1_high, label="J1 = sum", lw=2, color="black")
ax.axvline(0.20, color="gray", ls=":", label="r=0.20 (paper plateau onset)")
ax.set_xlabel("r")
ax.set_ylabel("magnitude")
ax.legend()
ax.set_title("Step 5: high-r breakdown (Fig 3A plateau check)")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "step5_high_r_breakdown.png"), dpi=150)
plt.close(fig)

rho_plus_high = transport.rho_plus(r_high)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(r_high, rho_plus_high)
ax.axhline(1.0, color="red", ls="--", label="rho_plus = 1 (thermo_factor pole)")
ax.set_xlabel("r")
ax.set_ylabel("rho_plus(r)")
ax.legend()
ax.set_title("Step 5b: rho_plus(r) at high r (Eq. 36 fit)")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "step5b_rho_plus_high_r.png"), dpi=150)
plt.close(fig)

# =====================================================================
# STEP 6: Gamma_conc factor decomposition over r=0.15-0.30
# (isolates which single factor drives the Fig 3A plateau failure)
# =====================================================================
D_vals = transport.D(r_high)
tf_vals = transport.thermo_factor(r_high)
tm_vals = transport.t_minus_0(r_high)
dcdr_vals = _dc_dr_centred(transport, r_high) * 1.0e-3  # mol/cm^3 per unit r

kappa_vals = transport.kappa(r_high)
dUdlnm_vals = transport.dU_dlnm(r_high)
c_vals = transport.c(r_high) * 1.0e-3  # mol/cm^3

gamma_conc_recomputed = D_vals * tf_vals * dcdr_vals / np.where(np.abs(tm_vals) < 1e-12, 1e-12, tm_vals)

np.savetxt(
    os.path.join(OUTDIR, "step6_gamma_conc_factors.csv"),
    np.column_stack([
        r_high, D_vals, tf_vals, tm_vals, dcdr_vals,
        kappa_vals, dUdlnm_vals, rho_plus_high, c_vals, gamma_conc_recomputed,
    ]),
    delimiter=",",
    header="r,D,thermo_factor,t_minus_0,dc_dr,kappa,dU_dlnm,rho_plus,c,Gamma_conc",
    comments="",
)

fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
panels = [
    (D_vals, "D(r)", False),
    (tf_vals, "thermo_factor (1+Theta)", False),
    (tm_vals, "t_minus_0", False),
    (dcdr_vals, "dc/dr [mol/cm^3 per unit r]", False),
    (kappa_vals, "kappa(r)", True),
    (dUdlnm_vals, "dU/d ln m", False),
    (rho_plus_high, "rho_plus(r)", False),
    (gamma_conc_recomputed, "Gamma_conc (combined)", True),
]
for ax, (vals, name, logscale) in zip(axes.flat, panels):
    ax.plot(r_high, vals, lw=1.2)
    ax.set_ylabel(name)
    if logscale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
axes[-1, 0].set_xlabel("r")
axes[-1, 1].set_xlabel("r")
fig.suptitle("Step 6: Gamma_conc factor breakdown over r=0.15-0.30 (Fig 3A plateau region)")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "step6_gamma_conc_factors.png"), dpi=150)
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(r_high, tf_vals, color="tab:blue", label="thermo_factor (1+Theta)")
ax1.set_xlabel("r")
ax1.set_ylabel("thermo_factor", color="tab:blue")
ax1.set_yscale("log")
ax2 = ax1.twinx()
ax2.plot(r_high, tm_vals, color="tab:red", label="t_minus_0")
ax2.set_ylabel("t_minus_0", color="tab:red")
ax2.axhline(0, color="tab:red", ls=":", lw=0.8)
fig.suptitle("Step 6b: thermo_factor vs t_minus_0, r=0.15-0.30")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "step6b_thermo_vs_tminus.png"), dpi=150)
plt.close(fig)


def _rel_growth(vals):
    v0, v1 = vals[0], vals[-1]
    if v0 == 0:
        return np.inf
    return (v1 - v0) / abs(v0)


print("\n--- Step 6 relative growth from r=0.15 to r=0.30 ---")
for vals, name, _ in panels:
    print(f"  {name}: {vals[0]:.4e} -> {vals[-1]:.4e}  (rel growth = {_rel_growth(vals):.3f})")

print("\nMinimum |t_minus_0| in range:", np.min(np.abs(tm_vals)), "at r =", r_high[np.argmin(np.abs(tm_vals))])
print("Maximum thermo_factor in range:", np.max(tf_vals), "at r =", r_high[np.argmax(tf_vals)])

# =====================================================================
# Final summary printout
# =====================================================================
print("\n--- Diagnostics complete ---")
print("All CSVs and PNGs written to ./diagnostics/")
print("Spline domain:", spline_domain)
print("Step 3 factor breakdown rows (r, thermo_factor, t_minus_0, t_plus_0, d_mu_strain_d_r, lambda):")
for row in rows:
    print(" ", row)

