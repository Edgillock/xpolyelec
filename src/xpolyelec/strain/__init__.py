"""Polymer-network strain models.

Different strain models for the same solver pipeline. 
The active model is selected via
config.strain_model.name; 
the built-in names are:

    "none"           -> NoStrain          (Baseline = Newman concentrated solution theory)
    "gent"           -> GentStrain        (Patel 2025 Crosslink Model, Eqs. 11-19)
    "neo_hookean"    -> NeoHookeanStrain  (Gaussian-chain limit, no finite-extensibility)
    "mooney_rivlin"  -> MooneyRivlinStrain (two-parameter generalisation)

Use :func: get_strain_model as a default.
"""
from xpolyelec.strain.base import StrainContext, StrainModel
from xpolyelec.strain.gent import GentStrain
from xpolyelec.strain.mooney_rivlin import MooneyRivlinStrain
from xpolyelec.strain.neo_hookean import NeoHookeanStrain
from xpolyelec.strain.none_strain import NoStrain

_REGISTRY: dict[str, type[StrainModel]] = {
    "none": NoStrain,
    "gent": GentStrain,
    "neo_hookean": NeoHookeanStrain,
    "mooney_rivlin": MooneyRivlinStrain,
}


def get_strain_model(name: str) -> type[StrainModel]:
    """Return the StrainModel class for a registered name."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown strain model {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def available_strain_models() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "StrainModel",
    "StrainContext",
    "NoStrain",
    "GentStrain",
    "NeoHookeanStrain",
    "MooneyRivlinStrain",
    "get_strain_model",
    "available_strain_models",
]
