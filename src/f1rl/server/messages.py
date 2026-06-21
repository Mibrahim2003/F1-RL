"""Inbound client -> server WebSocket messages (Pydantic v2).

The browser sends six message kinds — ``input``, ``mode``, ``control``, ``record``,
``track``, ``policy`` — all discriminated on a ``type`` field. :func:`parse_client_message`
dispatches on that field and returns ``None`` for anything unknown or malformed, so the socket
loop never raises on bad input. Outbound ``state`` frames are produced by
:class:`f1rl.sim.loop.SimLoop` and are not modeled here. ``SurfaceEdit`` models the
``POST /track/{id}/surfaces`` HTTP body.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class InputMessage(BaseModel):
    """Manual-drive control input. Axes are clamped to their valid ranges."""

    type: Literal["input"] = "input"
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    reset: bool = False

    @field_validator("steer")
    @classmethod
    def _clamp_steer(cls, v: float) -> float:
        return _clamp(v, -1.0, 1.0)

    @field_validator("throttle", "brake")
    @classmethod
    def _clamp_pedal(cls, v: float) -> float:
        return _clamp(v, 0.0, 1.0)

    @property
    def longitudinal(self) -> float:
        """Combined throttle/brake command in ``[-1, 1]``."""
        return _clamp(self.throttle - self.brake, -1.0, 1.0)


class ModeMessage(BaseModel):
    """Switch the session between manual drive, autopilot watch, and replay."""

    type: Literal["mode"] = "mode"
    mode: Literal["manual", "watch", "replay"]


class ControlMessage(BaseModel):
    """Transport control for the live sim (and replay UI)."""

    type: Literal["control"] = "control"
    action: Literal["play", "pause", "restart"]
    speed: int | None = None

    @field_validator("speed")
    @classmethod
    def _check_speed(cls, v: int | None) -> int | None:
        if v is not None and v not in (1, 2, 4):
            raise ValueError("speed must be one of 1, 2, 4")
        return v


class RecordMessage(BaseModel):
    """Start or stop recording the live run to a trajectory file."""

    type: Literal["record"] = "record"
    action: Literal["start", "stop"]


class TrackMessage(BaseModel):
    """Switch the live session to a different circuit (Phase 2 track selector)."""

    type: Literal["track"] = "track"
    id: str


class PolicyMessage(BaseModel):
    """Select what drives the car in watch mode (Phase 3a live-policy view).

    ``source == "autopilot"`` restores the centerline pure-pursuit follower; ``source ==
    "checkpoint"`` loads the trained checkpoint named by ``id`` (a directory under the
    server's checkpoints root). A bad/missing checkpoint never crashes the socket — the
    server replies with a ``policy_error`` event and stays on the autopilot.
    """

    type: Literal["policy"] = "policy"
    source: Literal["autopilot", "checkpoint"]
    id: str | None = None


class WeatherMessage(BaseModel):
    """Set the live weather for the watch session (Phase 3b grip pipeline).

    Changes the grip immediately via the shared :class:`~f1rl.env.conditions.Conditions`
    provider on the session's :class:`~f1rl.sim.loop.SimLoop`, so the car loses grip in the
    wet — the same factor the training env uses.
    """

    type: Literal["weather"] = "weather"
    condition: Literal["dry", "damp", "wet"]


class FieldMessage(BaseModel):
    """Set the live field size (Phase 5 many-cars view).

    ``n_agents == 1`` is the single-car path (a :class:`~f1rl.sim.loop.SimLoop`); ``> 1``
    switches the watch session to a :class:`~f1rl.sim.loop.FieldSimLoop` that drives N cars
    with one shared pilot on a starting grid. Field mode is watch-only (one policy drives the
    whole grid); the active checkpoint/autopilot is re-applied across the switch.
    """

    type: Literal["field"] = "field"
    n_agents: int = 1

    @field_validator("n_agents")
    @classmethod
    def _check_n(cls, v: int) -> int:
        if not (1 <= v <= 22):
            raise ValueError("n_agents must be in [1, 22]")
        return v


class SurfaceEdit(BaseModel):
    """Edited surface band widths for ``POST /track/{id}/surfaces`` (uniform, meters).

    All bands are optional; only the provided ones are applied. Values are bound-checked so a
    bad slider value can never write a degenerate track. ``condition`` (dry/wet) selects the
    grip-pipeline weather for the live session (Phase 3b); the saved band widths feed the
    surface-zone classifier.
    """

    half_width_left: float | None = None
    half_width_right: float | None = None
    kerb_width: float | None = None
    grass_width: float | None = None
    gravel_width: float | None = None
    condition: Literal["dry", "wet"] | None = None

    @field_validator("half_width_left", "half_width_right")
    @classmethod
    def _check_half(cls, v: float | None) -> float | None:
        if v is not None and not (0.5 <= v <= 25.0):
            raise ValueError("half width must be in [0.5, 25] m")
        return v

    @field_validator("kerb_width", "grass_width", "gravel_width")
    @classmethod
    def _check_band(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 50.0):
            raise ValueError("band width must be in [0, 50] m")
        return v


ClientMessage = (
    InputMessage
    | ModeMessage
    | ControlMessage
    | RecordMessage
    | TrackMessage
    | PolicyMessage
    | WeatherMessage
    | FieldMessage
)

_PARSERS: dict[str, type[BaseModel]] = {
    "input": InputMessage,
    "mode": ModeMessage,
    "control": ControlMessage,
    "record": RecordMessage,
    "track": TrackMessage,
    "policy": PolicyMessage,
    "weather": WeatherMessage,
    "field": FieldMessage,
}


def parse_client_message(data: dict) -> ClientMessage | None:
    """Parse a raw client message dict into a typed model.

    Returns ``None`` for unknown types or validation failures so the socket loop can
    quietly ignore bad input rather than crashing.
    """
    if not isinstance(data, dict):
        return None
    model = _PARSERS.get(data.get("type"))
    if model is None:
        return None
    try:
        return model.model_validate(data)
    except Exception:
        return None
