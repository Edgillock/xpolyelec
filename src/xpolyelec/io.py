from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xpolyelec.config import Config

_PROPERTY_KEYS = ("kappa", "rho_plus", "D", "U", "rho_el")


def _xy(df: pd.DataFrame, xcol: str, ycol: str) -> np.ndarray:
    """Extract a clean (2, n) array of (x, y) rows, dropping NaNs."""
    sub = df[[xcol, ycol]].dropna()
    return np.vstack([sub[xcol].to_numpy(float), sub[ycol].to_numpy(float)])


def load_csv(path: str | Path, prop: str, config: Config | None = None) -> np.ndarray:
    """Load a single-property CSV.

    Parameters
    ----------
    path : str or Path
        CSV file on disk.
    prop : str
        One of ``{'kappa', 'rho_plus', 'D', 'U', 'rho_el'}``.
    config : Config, optional
        Used to look up the expected column names. If omitted, defaults are
        used from the bundled config.
    """
    if prop not in _PROPERTY_KEYS:
        raise ValueError(f"unknown property {prop!r}; expected one of {_PROPERTY_KEYS}")
    cfg = config or Config.default(verbose=False)
    cols = cfg.get(f"io.default_columns.{prop}")
    df = pd.read_csv(path)
    xcol, ycol = cols[0], cols[1]
    if xcol not in df.columns or ycol not in df.columns:
        # Fall back to the first two numeric columns
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < 2:
            raise ValueError(
                f"CSV {path} must contain columns {cols!r} or at least two numeric columns"
            )
        xcol, ycol = numeric[0], numeric[1]
    return _xy(df, xcol, ycol)


def load_directory(directory: str | Path, config: Config | None = None) -> dict[str, np.ndarray]:
    """Load all per-property CSVs from a directory.

    Expected filenames (any missing file is skipped):
        kappa.csv, rho_plus.csv, D.csv, U.csv, rho_el.csv
    """
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a directory")
    out: dict[str, np.ndarray] = {}
    for prop in _PROPERTY_KEYS:
        p = d / f"{prop}.csv"
        if p.is_file():
            out[prop] = load_csv(p, prop, config=config)
    if not out:
        raise FileNotFoundError(f"no recognised CSVs found in {d}")
    return out


def load_combined(path: str | Path) -> dict[str, np.ndarray]:
    """Load a combined CSV whose columns are a subset of the property names.

    Recognised columns:
        r, kappa, rho_plus, D, rho_el, ln_m or m, U
    """
    df = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    if "r" in df.columns:
        for ycol in ("kappa", "rho_plus", "D", "rho_el"):
            if ycol in df.columns:
                out[ycol] = _xy(df, "r", ycol)
    if "U" in df.columns:
        if "ln_m" in df.columns:
            out["U"] = _xy(df, "ln_m", "U")
        elif "m" in df.columns:
            sub = df[["m", "U"]].dropna()
            out["U"] = np.vstack([np.log(sub["m"].to_numpy(float)), sub["U"].to_numpy(float)])
    if not out:
        raise ValueError(f"no recognised property columns found in {path}")
    return out


def load(source: str | Path | dict[str, str | Path], config: Config | None = None) -> dict[str, np.ndarray]:
    """High-level entry point: auto-detect format.

    ``source`` may be:
        * A directory → :func:`load_directory`
        * A dict of ``{prop: path}`` → loaded piecewise via :func:`load_csv`
        * A file → :func:`load_combined`
    """
    if isinstance(source, dict):
        return {prop: load_csv(path, prop, config=config) for prop, path in source.items()}
    p = Path(source)
    if p.is_dir():
        return load_directory(p, config=config)
    if p.is_file():
        return load_combined(p)
    raise FileNotFoundError(f"{source!r} is neither a file, directory, nor dict")
