import numpy as np

from xpolyelec.fits import FitRegistry


def test_linear_fit_recovers_params():
    x = np.linspace(0, 1, 50)
    y = 3.0 * x + 0.5
    fit = FitRegistry.fit("linear", x, y)
    assert np.allclose(fit.params, (3.0, 0.5), atol=1e-8)


def test_kappa_peak_paper_values():
    # kappa = 0.001 r exp(-r/0.061) at r=0.061 peaks at 0.001*0.061/e
    kp = FitRegistry.from_params("kappa_peak", (0.001, 0.061))
    assert kp(0.061) > 0.0
    # Derivative is zero at r = b (peak location)
    assert abs(kp.derivative(0.061)) < 1e-10


def test_power_law_derivative():
    pl = FitRegistry.from_params("power_law", (-0.14, 0.56, 0.16))
    # analytical d/dm = a*n*m^(n-1)
    m = 2.0
    expected = -0.14 * 0.56 * m ** (0.56 - 1)
    assert np.isclose(pl.derivative(m), expected, atol=1e-10)


def test_poly2_round_trip():
    x = np.linspace(0, 0.3, 20)
    y = 9.42 * x ** 2 - 2.18 * x + 0.17
    fit = FitRegistry.fit("poly2", x, y)
    assert np.allclose(fit.params, (9.42, -2.18, 0.17), atol=1e-8)
