"""Inbound client -> server WebSocket messages (Pydantic v2).

The browser sends four message kinds — ``input``, ``mode``, ``control``, ``record`` —
all discriminated on a ``type`` field. :func:`parse_client_message` dispatches on that
field and returns ``None`` for anything unknown or malformed, so the socket loop never
raises on bad input. Outbound ``state`` frames are produced by :class:`f1rl.sim.loop.SimLoop`
and are not modeled here.
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


ClientMessage = InputMessage | ModeMessage | ControlMessage | RecordMessage

_PARSERS: dict[str, type[BaseModel]] = {
    "input": InputMessage,
    "mode": ModeMessage,
    "control": ControlMessage,
    "record": RecordMessage,
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
