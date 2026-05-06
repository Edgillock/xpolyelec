"""Configuration management with change tracking and interactive save.

The :class:`Config` class is a thin wrapper around a nested dict loaded
from a JSON file. It provides:

Nested access via dotted paths (cfg.get("strain_model.N_monomers_per_strand")).
Change tracking: every modification is recorded in cfg._changes and, if
 verbose=True, printed to stdout as "<dotted.path>: <old> -> <new>".
:meth: save which, when called, prompts the user (input()) whether to
  persist the current configuration to a new JSON file.

The schema is intentionally permissive so that the strain-model subclasses
and custom fit callables can attach extra keys without needing migrations.
"""
from __future__ import annotations

import copy
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Any


def load_default_config() -> dict[str, Any]:
    """Load the bundled default configuration."""
    with resources.files("xpolyelec.defaults").joinpath("default_config.json").open(
        "r", encoding="utf-8"
    ) as fh:
        return json.load(fh)


def load_paper_fits() -> dict[str, Any]:
    """Load the bundled Patel-2025 fit coefficients."""
    with resources.files("xpolyelec.defaults").joinpath("paper_fits.json").open(
        "r", encoding="utf-8"
    ) as fh:
        return json.load(fh)


class Config:
    """Configuration with change tracking.

    Parameters
    ----------
    data : dict, optional
        Initial configuration. If omitted, the bundled default is loaded.
    verbose : bool, optional
        True (default), every modification is printed.
        False for silent operation.
    """

    def __init__(self, data: dict[str, Any] | None = None, verbose: bool = True) -> None:
        self._data: dict[str, Any] = copy.deepcopy(data) if data is not None else load_default_config()
        self.verbose: bool = bool(self._data.pop("verbose", verbose)) if data is None else verbose
        self._changes: list[tuple[str, Any, Any]] = []

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, verbose: bool = True) -> "Config":
        """Load a configuration from a JSON file on disk."""
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls(json.load(fh), verbose=verbose)

    @classmethod
    def default(cls, verbose: bool = True) -> "Config":
        """Load the bundled default configuration."""
        return cls(load_default_config(), verbose=verbose)

    # ------------------------------------------------------------------
    # Dotted-path accessors
    # ------------------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        """Return value at a dotted path, or ``default`` if missing."""
        node: Any = self._data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, path: str, value: Any) -> None:
        """Set value at a dotted path. Records the change and optionally prints it.

        Creates intermediate dicts as needed.
        """
        keys = path.split(".")
        node: dict[str, Any] = self._data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        old = node.get(keys[-1], None)
        if old == value:
            return
        node[keys[-1]] = value
        self._changes.append((path, old, value))
        if self.verbose:
            print(f"[xpolyelec] {path}: {old!r} -> {value!r}", file=sys.stdout)

    def update(self, mapping: dict[str, Any]) -> None:
        """Apply a dict of ``{dotted.path: value}`` updates."""
        for path, value in mapping.items():
            self.set(path, value)