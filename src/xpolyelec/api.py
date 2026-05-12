"""Top-level user-facing API: the `Simulation` class.

Includes classes:

class: Config (with change-tracking + save prompt)
class: TransportProperties (built from config fits or user CSVs)
class: StrainModel (selected by name)
class: StrainContext (rebuilt whenever parameters change)

available methods:

load_csv_data: fit transport properties from raw CSVs
set_strain_model: switch strain model
set_param: tweak any dotted-path parameter
set_kinematics: set_asymmetric_lambda_crit
compute_profile: r(x/L) at one (iL, ravg)
compute_potential: Delta Phi decomposition at one (iL, ravg)
compute_iv_curve: Fig. 7 sweep
plot: dispatcher to :mod:`xpolyelec.plotting`
save_config: interactive save prompt
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from xpolyelec import io as _io
from xpolyelec import plotting as _plotting
from xpolyelec.config import Config
from xpolyelec.fits import CustomFit, Fit, FitRegistry
from xpolyelec.solver import (
    IVCurve,
    JFunctions,
    PotentialDrop,
    Profile,
    build_J1_and_J2,
    compute_potential_drop,
    solve_r_profile,
    sweep_iL,
)
from xpolyelec.strain import (
    StrainContext,
    StrainModel,
    available_strain_models,
    get_strain_model,
)
from xpolyelec.transport import TransportProperties


class Simulation:
    """The main entry point. Defaults match Patel et al. 2025 paper exactly.

    Parameters
    
    config : Config or dict or path, optional
        Starting configuration. If None, the bundled default is loaded.
    verbose : bool, optional
        If True (default), parameter changes are echoed to stdout.
    """

    def __init__(
        self,
        config: Config | dict[str, Any] | str | Path | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        if isinstance(config, Config):
            self.config = config
        elif isinstance(config, dict):
            self.config = Config(config, verbose=verbose)
        elif isinstance(config, (str, Path)):
            self.config = Config.from_file(config, verbose=verbose)
        else:
            self.config = Config.default(verbose=verbose)

        self._fit_overrides: dict[str, Fit | CustomFit] = {}
        self._rebuild_all()

    # ------------------------------------------------------------------
    # Internal: rebuild the derived objects after config changes
    # ------------------------------------------------------------------
    def _rebuild_all(self) -> None:
        self.transport = TransportProperties.from_config(self.config, overrides=self._fit_overrides)
        self._rebuild_strain()

    def _rebuild_strain(self) -> None:
        sm_cfg = self.config.get("strain_model")
        phys = self.config.get("physical")
        ModelCls = get_strain_model(sm_cfg["name"])
        self.strain_model: StrainModel = ModelCls()
        delta = float(sm_cfg.get("delta_lambda", 0.06))
        lam_ext = sm_cfg.get("lambda_crit_ext")
        lam_con = sm_cfg.get("lambda_crit_con")
        if lam_ext is None:
            lam_ext = 1.0 + delta
        if lam_con is None:
            lam_con = 1.0 - delta
        self.context = StrainContext(
            transport=self.transport,
            ravg=float(phys["ravg"]),
            N=float(sm_cfg.get("N_monomers_per_strand", 16)),
            lambda_crit_ext=float(lam_ext),
            lambda_crit_con=float(lam_con),
            kinematics=str(sm_cfg.get("kinematics", "affine_isotropic")),
            phi_p_model=str(sm_cfg.get("phi_p_model", "paper")),
            vbar_m_nm3=float(phys["vbar_m_nm3"]),
            vbar_s_nm3=float(phys["vbar_s_nm3"]),
            extra={"mooney_rivlin_C_ratio": sm_cfg.get("mooney_rivlin_C_ratio", 0.0)},
        )
        self.J = build_J1_and_J2(self.transport, self.strain_model, self.context)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_csv_data(self, source: str | Path | dict[str, str | Path]) -> None:
        """Load raw CSVs and refit all transport properties.

        'source' can be a directory, a dict of {property: path}, or a
        combined CSV.
        """
        raw = _io.load(source, config=self.config)
        self._raw_data = raw
        fits_cfg = self.config.get("fits")
        for prop, arr in raw.items():
            x, y = arr[0], arr[1]
            # For U, CSVs store ln_m but U is fit against m (power_law).
            if prop == "U":
                x = np.exp(x)
            form = fits_cfg[prop]["form"]
            fit = FitRegistry.fit(form, x, y)
            # Update config with fitted params so everything stays consistent
            self.config.set(f"fits.{prop}.params", list(fit.params))
        self._fit_overrides.clear()
        self._rebuild_all()

    def set_custom_fit(self, prop: str, fit: Fit | CustomFit) -> None:
        """Inject a user-provided fit for one transport property."""
        if prop not in {"kappa", "rho_plus", "D", "U", "rho_el"}:
            raise ValueError(f"unknown property {prop!r}")
        self._fit_overrides[prop] = fit
        if self.config.verbose:
            print(f"[xpolyelec] fits.{prop}: -> {fit!r}")
        self._rebuild_all()

    # ------------------------------------------------------------------
    # Configuration setters (all change-tracked via Config)
    # ------------------------------------------------------------------
    def set_strain_model(self, name: str) -> None:
        """Switch active strain model ('none', 'gent', 'neo_hookean', 'mooney_rivlin')."""
        if name not in available_strain_models():
            raise KeyError(f"unknown strain model {name!r}. Available: {available_strain_models()}")
        self.config.set("strain_model.name", name)
        self._rebuild_strain()

    def set_param(self, path: str, value: Any) -> None:
        """Tweak any dotted-path parameter. Rebuilds derived objects as needed."""
        self.config.set(path, value)
        # Conservatively rebuild everything; cost is negligible.
        self._rebuild_all()

    def set_kinematics(self, kinematics: str) -> None:
        if kinematics not in {"affine_isotropic", "uniaxial"}:
            raise ValueError(f"kinematics must be 'affine_isotropic' or 'uniaxial'; got {kinematics!r}")
        self.config.set("strain_model.kinematics", kinematics)
        self._rebuild_strain()

    def set_asymmetric_lambda_crit(self, ext: float, con: float) -> None:
        """Set independent extension and contraction limits (overrides delta_lambda)."""
        self.config.set("strain_model.lambda_crit_ext", float(ext))
        self.config.set("strain_model.lambda_crit_con", float(con))
        self._rebuild_strain()

    def set_phi_p_model(self, model: str) -> None:
        """For now only 'paper' is supported as a named string; pass a callable via context for custom."""
        self.config.set("strain_model.phi_p_model", model)
        self._rebuild_strain()

    # ------------------------------------------------------------------
    # Computational entry points
    # ------------------------------------------------------------------
    def compute_profile(self, *, iL: float, ravg: float | None = None) -> Profile:
        """Solve r(x/L) at a given (iL, ravg)."""
        if ravg is not None:
            self.config.set("physical.ravg", float(ravg))
            self._rebuild_strain()
        solver_cfg = self.config.get("solver")
        return solve_r_profile(
            self.J, self.context,
            iL=iL,
            ravg=self.context.ravg,
            n_points=int(solver_cfg["n_xL_points"]),
            r0_bracket=tuple(solver_cfg["r0_bracket"]),
            ravg_tol=float(solver_cfg["ravg_tol"]),
            max_iter=int(solver_cfg["ravg_max_iter"]),
            F=self.transport.F,
            quad_abs_tol=float(solver_cfg["quad_abs_tol"]),
            quad_rel_tol=float(solver_cfg["quad_rel_tol"]),
        )

    def compute_potential(self, *, iL: float, ravg: float | None = None) -> PotentialDrop:
        prof = self.compute_profile(iL=iL, ravg=ravg)
        solver_cfg = self.config.get("solver")
        return compute_potential_drop(
            self.J, self.transport, self.strain_model, self.context, prof,
            quad_abs_tol=float(solver_cfg["quad_abs_tol"]),
            quad_rel_tol=float(solver_cfg["quad_rel_tol"]),
        )

    def compute_iv_curve(
        self,
        *,
        ravg: float | None = None,
        iL_range: tuple[float, float] | None = None,
        n: int | None = None,
        log_spaced: bool | None = None,
    ) -> IVCurve:
        """Sweep iL → produce Fig. 7 data."""
        if ravg is not None:
            self.config.set("physical.ravg", float(ravg))
            self._rebuild_strain()
        scfg = self.config.get("solver")
        sweep = scfg["iL_sweep"]
        iL_min = iL_range[0] if iL_range else float(sweep["iL_min"])
        iL_max = iL_range[1] if iL_range else float(sweep["iL_max"])
        n = int(n) if n is not None else int(sweep["n_points"])
        log_spaced = bool(log_spaced) if log_spaced is not None else bool(sweep["log_spaced"])
        iL_vals = (
            np.logspace(np.log10(iL_min), np.log10(iL_max), n)
            if log_spaced
            else np.linspace(iL_min, iL_max, n)
        )
        return sweep_iL(
            self.J, self.transport, self.strain_model, self.context,
            ravg=self.context.ravg, iL_values=iL_vals,
            n_points=int(scfg["n_xL_points"]),
            r0_bracket=tuple(scfg["r0_bracket"]),
            ravg_tol=float(scfg["ravg_tol"]),
            quad_abs_tol=float(scfg["quad_abs_tol"]),
            quad_rel_tol=float(scfg["quad_rel_tol"]),
        )

    # ------------------------------------------------------------------
    # Plotting dispatcher
    # ------------------------------------------------------------------
    def plot(
        self,
        which: str,
        *args,
        save: str | Path | None = None,
        show: bool = False,
        **kwargs,
    ):
        """Generate a named figure. ``which`` ∈ {'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7'}."""
        fn_map = {
            "fig2": lambda: _plotting.plot_fig2(self.transport, *args, **kwargs),
            "fig3": lambda: _plotting.plot_fig3(*args, **kwargs),
            "fig4": lambda: _plotting.plot_fig4(*args, **kwargs),
            "fig5": lambda: _plotting.plot_fig5(*args, **kwargs),
            "fig6": lambda: _plotting.plot_fig6(*args, **kwargs),
            "fig7": lambda: _plotting.plot_fig7(*args, **kwargs),
        }
        if which not in fn_map:
            raise KeyError(f"unknown figure {which!r}. Available: {list(fn_map)}")
        fig, axes = fn_map[which]()
        if save is not None:
            fig.savefig(save, dpi=150, bbox_inches="tight")
            if self.config.verbose:
                print(f"[xpolyelec] saved figure to {save}", file=sys.stdout)
        if show:
            plt.show()
        return fig, axes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_config(self, path: str | Path | None = None, *, interactive: bool = True) -> Path | None:
        """Save the current config to JSON.

        If path is given, writes unconditionally. Otherwise prompts the
        user to confirm and choose a filename (iff interactive=True).
        """
        return self.config.save(path, interactive=interactive)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @verbose.setter
    def verbose(self, v: bool) -> None:
        self.config.verbose = bool(v)

    def __repr__(self) -> str:
        return (
            f"Simulation(strain_model={self.strain_model.name!r}, "
            f"ravg={self.context.ravg!r}, verbose={self.verbose})"
        )
