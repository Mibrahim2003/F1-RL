"""Grip pipeline — pure tire/weather/surface factors (TECHNICAL_DESIGN.md §4, plan §A).

The whole realism stack collapses to one scalar::

    effective_grip = mu_base * tire_factor(compound, wear) * weather_factor(weather)
                     * surface_factor(surface_zone)

That scalar gates the friction circle in the dynamic model. Adding a realism feature means
writing one factor here, not touching the physics core. This module is **pure**: no
``Track``, no torch, no gym — so both the env and the live ``SimLoop`` import the same
tables and agree on grip. SI-unitless multipliers; ``mu_base`` is the **effective base-grip
coefficient** — it lumps mechanical tire grip with a baseline aero contribution and is
calibrated per car so a clean optimal lap lands near the real pole. It is **not** a literal
tire-road friction value and routinely calibrates above one (see TECHNICAL_DESIGN §4).
``compound`` indices match :class:`~f1rl.physics.base.CarState`:
``0 soft, 1 medium, 2 hard, 3 intermediate, 4 wet``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Surface-zone labels (returned by the env's surface classifier, consumed here).
ASPHALT = "asphalt"
KERB = "kerb"
GRASS = "grass"
GRAVEL = "gravel"

_N_COMPOUNDS = 5  # soft, medium, hard, intermediate, wet
_WEATHER_KEYS = ("dry", "damp", "wet")

# Defaults (all overridable from the ``tires`` / ``weather`` / ``surface`` config blocks).
_DEFAULT_MU_BASE = 1.05
_DEFAULT_COMPOUND_GRIP = (1.00, 0.97, 0.94, 0.88, 0.82)
_DEFAULT_COMPOUND_WEAR_FALLOFF = (0.30, 0.22, 0.16, 0.20, 0.20)
_DEFAULT_WEATHER = {"dry": 1.0, "damp": 0.8, "wet": 0.6}
_DEFAULT_SURFACE = {"asphalt": 1.0, "kerb": 0.9, "grass": 0.4, "gravel": 0.3}


@dataclass(frozen=True)
class TireParams:
    """Per-compound grip and wear falloff tables, plus the effective base-grip coefficient.

    ``mu_base`` is a lumped, calibrated grip coefficient, not a literal road friction value —
    see the module docstring and TECHNICAL_DESIGN §4.
    """

    mu_base: float = _DEFAULT_MU_BASE
    compound_grip: tuple[float, ...] = _DEFAULT_COMPOUND_GRIP
    compound_wear_falloff: tuple[float, ...] = _DEFAULT_COMPOUND_WEAR_FALLOFF
    start_compound: int = 0

    @classmethod
    def from_config(cls, cfg: Any) -> TireParams:
        """Build from a ``tires`` config node (mapping/OmegaConf) or fall back to defaults."""
        node = _tires_node(cfg)
        get = _getter(node)
        grip = _as_float_tuple(get("compound_grip", cls.compound_grip))
        falloff = _as_float_tuple(get("compound_wear_falloff", cls.compound_wear_falloff))
        if len(grip) != _N_COMPOUNDS:
            raise ValueError(
                f"tires.compound_grip must have {_N_COMPOUNDS} entries, got {len(grip)}"
            )
        if len(falloff) != _N_COMPOUNDS:
            raise ValueError(
                f"tires.compound_wear_falloff must have {_N_COMPOUNDS} entries, got {len(falloff)}"
            )
        return cls(
            mu_base=float(get("mu_base", cls.mu_base)),
            compound_grip=grip,
            compound_wear_falloff=falloff,
            start_compound=int(get("start_compound", cls.start_compound)),
        )


@dataclass(frozen=True)
class WeatherParams:
    """Weather grip multipliers and the wet-sampling probability."""

    dry_factor: float = 1.0
    damp_factor: float = 0.8
    wet_factor: float = 0.6
    p_wet: float = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> WeatherParams:
        node = _weather_node(cfg)
        get = _getter(node)
        return cls(
            dry_factor=float(get("dry_factor", cls.dry_factor)),
            damp_factor=float(get("damp_factor", cls.damp_factor)),
            wet_factor=float(get("wet_factor", cls.wet_factor)),
            p_wet=float(get("p_wet", cls.p_wet)),
        )

    def factor(self, weather: str) -> float:
        """Grip multiplier for ``"dry"`` / ``"damp"`` / ``"wet"`` (unknown -> dry)."""
        if weather == "wet":
            return self.wet_factor
        if weather == "damp":
            return self.damp_factor
        return self.dry_factor


@dataclass(frozen=True)
class SurfaceParams:
    """Grip multiplier per Phase-2 surface zone."""

    asphalt: float = 1.0
    kerb: float = 0.9
    grass: float = 0.4
    gravel: float = 0.3

    @classmethod
    def from_config(cls, cfg: Any) -> SurfaceParams:
        node = _surface_node(cfg)
        get = _getter(node)
        return cls(
            asphalt=float(get("asphalt", cls.asphalt)),
            kerb=float(get("kerb", cls.kerb)),
            grass=float(get("grass", cls.grass)),
            gravel=float(get("gravel", cls.gravel)),
        )

    def factor(self, surface_zone: str) -> float:
        """Grip multiplier for a surface-zone label (unknown -> asphalt = 1.0)."""
        return {
            ASPHALT: self.asphalt,
            KERB: self.kerb,
            GRASS: self.grass,
            GRAVEL: self.gravel,
        }.get(surface_zone, self.asphalt)


def tire_factor(compound: int, wear: float, params: TireParams) -> float:
    """Per-compound base grip times a linear wear falloff ``(1 - falloff * wear)``.

    Monotone: wear up -> grip down; at equal wear soft > medium > hard. Clamped at 0 so a
    fully worn tire never produces negative grip.
    """
    idx = int(compound)
    if idx < 0 or idx >= _N_COMPOUNDS:
        idx = 0
    w = 0.0 if wear < 0.0 else 1.0 if wear > 1.0 else float(wear)
    base = params.compound_grip[idx]
    falloff = params.compound_wear_falloff[idx]
    return max(0.0, base * (1.0 - falloff * w))


def weather_factor(weather: str, params: WeatherParams) -> float:
    """Weather grip multiplier (dry ~1.0 > damp > wet)."""
    return params.factor(weather)


def surface_factor(surface_zone: str, params: SurfaceParams) -> float:
    """Surface-zone grip multiplier (asphalt 1.0 > kerb > grass > gravel)."""
    return params.factor(surface_zone)


def effective_grip(
    compound: int,
    wear: float,
    weather: str,
    surface_zone: str,
    tires: TireParams,
    weather_params: WeatherParams,
    surface_params: SurfaceParams,
) -> float:
    """The one grip scalar = ``mu_base * tire * weather * surface`` (gates the friction circle)."""
    return (
        tires.mu_base
        * tire_factor(compound, wear, tires)
        * weather_factor(weather, weather_params)
        * surface_factor(surface_zone, surface_params)
    )


# --- internals --------------------------------------------------------------------------


def _getter(node: Any):
    return node.get if hasattr(node, "get") else (lambda k, d: getattr(node, k, d))


def _tires_node(cfg: Any) -> Any:
    return cfg.tires if hasattr(cfg, "tires") and cfg.tires is not None else cfg


def _weather_node(cfg: Any) -> Any:
    return cfg.weather if hasattr(cfg, "weather") and cfg.weather is not None else cfg


def _surface_node(cfg: Any) -> Any:
    return cfg.surface if hasattr(cfg, "surface") and cfg.surface is not None else cfg


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    """Coerce a config list/sequence (incl. OmegaConf ListConfig) to a tuple of floats."""
    return tuple(float(v) for v in value)
