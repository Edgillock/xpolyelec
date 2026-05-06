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
    
    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def data(self) -> dict[str, Any]:
        """Return a deep copy of the underlying dict (read-only contract)."""
        return copy.deepcopy(self._data)

    @property
    def changes(self) -> list[tuple[str, Any, Any]]:
        """List of (dotted_path, old, new) tuples accumulated since load."""
        return list(self._changes)

    def has_unsaved_changes(self) -> bool:
        return len(self._changes) > 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path | None = None, *, interactive: bool = True) -> Path | None:
        """Persist the config to JSON.

        If ``path`` is given, writes unconditionally.
        If ``path`` is None and ``interactive`` is True, prompts the user via
        ``input()`` whether to save and for a filename. Returns the path
        written to, or None if the user declined / there are no changes.
        """
        if path is not None:
            out = Path(path)
            with out.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=4, sort_keys=False)
            self._changes.clear()
            if self.verbose:
                print(f"[xpolyelec] config saved to {out}", file=sys.stdout)
            return out

        if not interactive:
            return None

        if not self.has_unsaved_changes():
            if self.verbose:
                print("[xpolyelec] no changes to save", file=sys.stdout)
            return None

        print("[xpolyelec] The following changes have been made:")
        for p, old, new in self._changes:
            print(f"    {p}: {old!r} -> {new!r}")
        reply = input("[xpolyelec] Save current config as a new JSON file? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            if self.verbose:
                print("[xpolyelec] save cancelled", file=sys.stdout)
            return None
        fname = input("[xpolyelec] Filename (e.g. my_config.json): ").strip()
        if not fname:
            print("[xpolyelec] empty filename, save cancelled", file=sys.stdout)
            return None
        return self.save(fname, interactive=False)

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Config(verbose={self.verbose}, unsaved_changes={len(self._changes)})"
