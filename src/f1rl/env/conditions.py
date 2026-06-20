"""Track/weather conditions + the grip provider (TECHNICAL_DESIGN.md §4, §10, plan §B).

Part 1 was dry-only: a single constant grip scalar fed the kinematic step. Part 2 grows this
into the **grip pipeline** owner: it holds the tire/weather/surface tables and the current
weather, classifies the surface zone under the car, and returns the one grip scalar that
gates the friction circle::

    grip_at(...) = mu_base * tire(compound, wear) * weather * surface

It stays **pure NumPy** (no torch, no gym) so the env *and* the live :class:`SimLoop` import
the same provider and agree on grip — train and serve never skew. The env passes the
``track_query`` outputs (``nearest_idx``, ``signed_lateral``, ``half_width``) straight in, so
there is no second projection. Curriculum overrides (``mu_base``, ``weather``) mutate this
holder via the env, so it is a plain (mutable) dataclass.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from f1rl.physics.tires import (
    ASPHALT,
    GRASS,
    GRAVEL,
    KERB,
    SurfaceParams,
    TireParams,
    WeatherParams,
    effective_grip,
)

# Clip bound for the normalized grip indicator in ObservationV2 (generous Box).
_DEFAULT_GRIP_INDICATOR_HI = 2.0


@dataclass
class Conditions:
    """Environmental conditions gating tire grip and the grip-pipeline provider."""

    grip: float = 1.0  # constant fallback (kinematic / pipeline disabled)
    weather: str = "dry"  # "dry" | "damp" | "wet"; current episode weather
    tires: TireParams = field(default_factory=TireParams)
    weather_params: WeatherParams = field(default_factory=WeatherParams)
    surface: SurfaceParams = field(default_factory=SurfaceParams)
    grip_indicator_hi: float = _DEFAULT_GRIP_INDICATOR_HI

    @classmethod
    def from_config(cls, cfg: Any) -> Conditions:
        """Build from a root config (reads ``sim.grip``, ``tires``, ``weather``, ``surface``)."""
        node = cfg.sim if hasattr(cfg, "sim") and cfg.sim is not None else cfg
        get = node.get if hasattr(node, "get") else (lambda k, d: getattr(node, k, d))
        weather_params = WeatherParams.from_config(cfg)
        weather = "dry"
        wnode = cfg.weather if hasattr(cfg, "weather") and cfg.weather is not None else None
        if wnode is not None:
            wget = wnode.get if hasattr(wnode, "get") else (lambda k, d: getattr(wnode, k, d))
            weather = str(wget("condition", "dry"))
        return cls(
            grip=float(get("grip", cls.grip)),
            weather=weather,
            tires=TireParams.from_config(cfg),
            weather_params=weather_params,
            surface=SurfaceParams.from_config(cfg),
        )

    @property
    def mu_base(self) -> float:
        """Effective base-grip coefficient (grip-indicator normalizer; curriculum target).

        Lumped/calibrated, not a literal road friction value — see TECHNICAL_DESIGN §4.
        """
        return self.tires.mu_base

    def set_mu_base(self, mu_base: float) -> None:
        """Curriculum override: swap the base friction coefficient (rebuilds the tire table)."""
        self.tires = dataclasses.replace(self.tires, mu_base=float(mu_base))

    def set_weather(self, weather: str) -> None:
        """Curriculum override: set the current weather (``dry`` | ``damp`` | ``wet``)."""
        self.weather = str(weather)

    def classify_surface(self, track: Any, idx: int, signed_lateral: float) -> str:
        """Label the surface zone under the car from the lateral offset and the Phase-2 bands.

        Asphalt within the half-width, kerb in the kerb band, then grass/gravel — by
        ``surface_zones`` (0 grass, 1 gravel) when present, else by the grass/gravel widths.
        """
        d = abs(float(signed_lateral))
        hw = float(
            track.half_width_left[idx] if signed_lateral >= 0.0 else track.half_width_right[idx]
        )
        if d <= hw:
            return ASPHALT
        kerb = float(track.kerb_width[idx])
        if d <= hw + kerb:
            return KERB
        if track.surface_zones is not None:
            return GRAVEL if int(track.surface_zones[idx]) == 1 else GRASS
        grass = float(track.grass_width[idx])
        if d <= hw + kerb + grass:
            return GRASS
        return GRAVEL

    def grip_at(
        self, track: Any, idx: int, signed_lateral: float, wear: float, compound: int
    ) -> float:
        """The grip scalar at the car: ``mu_base * tire * weather * surface`` (gates the circle)."""
        zone = self.classify_surface(track, idx, signed_lateral)
        return effective_grip(
            compound,
            wear,
            self.weather,
            zone,
            self.tires,
            self.weather_params,
            self.surface,
        )

    def grip_indicator(
        self, track: Any, idx: int, signed_lateral: float, wear: float, compound: int
    ) -> float:
        """Normalized, clipped grip at the car for ObservationV2 (``grip_at / mu_base``)."""
        g = self.grip_at(track, idx, signed_lateral, wear, compound)
        mu = self.mu_base if self.mu_base > 0.0 else 1.0
        v = g / mu
        hi = self.grip_indicator_hi
        return 0.0 if v < 0.0 else hi if v > hi else v
