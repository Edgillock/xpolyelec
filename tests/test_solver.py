import numpy as np

from xpolyelec import Simulation


def test_profile_converges_baseline():
    s = Simulation(verbose=False)
    s.set_strain_model("none")
    # iL = 1e-7 A/cm ~ 1e-4 mA/cm (well within paper Fig. 7 range)
    prof = s.compute_profile(iL=1.0e-7, ravg=0.08)
    assert prof.converged
    assert prof.r.min() > 0
    assert prof.r.max() < 0.5
    ravg_recomputed = np.trapezoid(prof.r, prof.x_over_L)
    assert abs(ravg_recomputed - 0.08) < 1e-3


def test_iv_curve_monotonic_total_potential():
    s = Simulation(verbose=False)
    s.set_strain_model("none")
    iv = s.compute_iv_curve(ravg=0.08, iL_range=(1e-9, 2e-7), n=6, log_spaced=True)
    # phi_total should grow with iL (ignore runaway points)
    finite = np.isfinite(iv.delta_phi_total) & (iv.delta_phi_total < 500)
    vals = iv.delta_phi_total[finite]
    diffs = np.diff(vals)
    # Allow small non-monotonicities from solver tolerance
    assert (diffs >= -1e-3).all()


def test_end_to_end_all_models_produce_finite_output():
    for name in ("none", "gent", "neo_hookean", "mooney_rivlin"):
        s = Simulation(verbose=False)
        s.set_strain_model(name)
        pd = s.compute_potential(iL=5e-8, ravg=0.08)
        assert np.isfinite(pd.delta_phi_total)
        assert pd.delta_phi_ohmic > 0
