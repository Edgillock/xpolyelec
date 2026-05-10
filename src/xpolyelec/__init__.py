"""xpolyelec — Ion transport in concentrated crosslinked solid polymer electrolytes.

Reproducibility package for:

    Vivaan Patel, Ondrea Lee, Shreya Makkar, and Nitash P. Balsara,
    "Ion Transport in Concentrated Crosslinked Solid Polymer Electrolytes",
    J. Electrochem. Soc. 172, 120517 (2025).
    DOI: 10.1149/1945-7111/ae285f

The top-level entry point is the :class:`Simulation` class:

    >>> from xpolyelec import Simulation
    >>> sim = Simulation()
    >>> sim.set_strain_model("gent")
    >>> profile = sim.compute_profile(ravg=0.08, iL=3.2e-4)
    >>> curve = sim.compute_iv_curve(ravg=0.08, iL_range=(1e-5, 1e-3), n=40)
"""

from xpolyelec.api import Simulation
from xpolyelec.config import Config

__all__ = ["Simulation", "Config"]
__version__ = "0.1.0"
