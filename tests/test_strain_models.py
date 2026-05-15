import numpy as np

from xpolyelec import Simulation
from xpolyelec.strain import available_strain_models


def test_all_strain_models_instantiate():
    for name in available_strain_models():
        s = Simulation(verbose=False)
        s.set_strain_model(name)
        # mu_strain should be finite at a typical r
        r = np.array([0.05, 0.08, 0.12])
        val = s.strain_model.mu_strain(r, s.context)
        assert np.all(np.isfinite(val))


def test_none_strain_is_zero():
    s = Simulation(verbose=False)
    s.set_strain_model("none")
    r = np.linspace(0.01, 0.2, 10)
    assert np.allclose(s.strain_model.mu_strain(r, s.context), 0.0)
    assert np.allclose(s.strain_model.d_mu_strain_d_r(r, s.context), 0.0)


def test_gent_symmetric_lambda_crit_has_plateau_near_edges():
    s = Simulation(verbose=False)
    s.set_strain_model("gent")
    # Far from ravg, lambda approaches lambda_crit -> |mu_strain| grows
    r = np.array([0.08, 0.04, 0.03])  # moving away from ravg=0.08
    vals = np.abs(s.strain_model.mu_strain(r, s.context))
    assert vals[-1] > vals[0]
