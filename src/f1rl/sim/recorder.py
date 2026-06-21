"""Recorded-trajectory format: the shared interchange between live sim, replay, and
(later) the cloud clip renderer.

Single-car file shape (Phase 1)::

    {
      "meta":   {"track_id": "oval", "dt": 0.05, "seed": 42, "created": "<ISO-8601>"},
      "frames": [{"t": 0.0, "car": {"x","y","yaw","speed"}, "telemetry": {...}}, ...]
    }

Multi-car file shape (Phase 5 — one recorder for the whole field)::

    {
      "meta":   {"track_id": "monza", "dt": 0.05, "seed": 42, "n_agents": 4, "created": ...},
      "frames": [{"t": 0.0, "cars": [{"id","x","y","yaw","speed","team","telemetry"}, ...]}, ...]
    }

A single car is just a one-element ``cars`` array, so the multi-car format is a superset; the
loader/validator accepts **either** a single-car ``car`` frame or a multi-car ``cars`` frame so
the Phase 1/4 one-car replay keeps working unchanged. ``load_trajectory`` schema-validates on
load and raises :class:`TrajectoryError` on anything malformed, so the UI can surface a clear
error instead of crashing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TrajectoryError(ValueError):
    """Raised when a trajectory file is missing required structure."""


class TrajectoryRecorder:
    """Accumulates frames for one run (single car or the whole field) and writes them to disk."""

    def __init__(self, track_id: str, dt: float, seed: int, *, n_agents: int | None = None) -> None:
        self.meta: dict[str, Any] = {
            "track_id": track_id,
            "dt": dt,
            "seed": seed,
            "created": datetime.now(UTC).isoformat(),
        }
        if n_agents is not None:
            self.meta["n_agents"] = int(n_agents)
        self.frames: list[dict[str, Any]] = []

    def append(self, t: float, car: dict[str, float], telemetry: dict[str, Any]) -> None:
        """Append one single-car frame. ``car`` is the kinematic subset ``{x, y, yaw, speed}``."""
        self.frames.append({"t": round(t, 4), "car": car, "telemetry": telemetry})

    def append_cars(self, t: float, cars: list[dict[str, Any]]) -> None:
        """Append one multi-car frame. Each entry is ``{id, x, y, yaw, speed, [team], telemetry}``.

        A one-element list records a single car under the multi-car schema (backward compatible
        with the live ``cars[]`` frame); the field eval driver and the server share this path.
        """
        self.frames.append({"t": round(t, 4), "cars": cars})

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "frames": self.frames}

    def save(self, path: str | Path) -> Path:
        """Write the trajectory as JSON, creating parent directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return p

    def __len__(self) -> int:
        return len(self.frames)


_REQUIRED_META = {"track_id", "dt"}
_REQUIRED_CAR = {"x", "y", "yaw", "speed"}


def _validate_car(car: Any, where: str) -> None:
    if not isinstance(car, dict) or not _REQUIRED_CAR.issubset(car):
        raise TrajectoryError(f"{where} car must contain {sorted(_REQUIRED_CAR)}")


def validate_trajectory(data: Any) -> dict[str, Any]:
    """Validate a parsed trajectory dict; return it unchanged or raise TrajectoryError.

    Accepts both the single-car (``car``) and the multi-car (``cars``) frame schemas.
    """
    if not isinstance(data, dict):
        raise TrajectoryError("trajectory must be a JSON object")
    meta = data.get("meta")
    frames = data.get("frames")
    if not isinstance(meta, dict) or not _REQUIRED_META.issubset(meta):
        raise TrajectoryError(f"meta must contain {sorted(_REQUIRED_META)}")
    if not isinstance(frames, list) or not frames:
        raise TrajectoryError("frames must be a non-empty list")
    for i, f in enumerate(frames):
        if not isinstance(f, dict) or "t" not in f:
            raise TrajectoryError(f"frame {i} missing 't'")
        if "cars" in f:
            cars = f["cars"]
            if not isinstance(cars, list) or not cars:
                raise TrajectoryError(f"frame {i} 'cars' must be a non-empty list")
            for car in cars:
                _validate_car(car, f"frame {i}")
        elif "car" in f:
            _validate_car(f["car"], f"frame {i}")
        else:
            raise TrajectoryError(f"frame {i} missing 'car' or 'cars'")
    return data


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Read and validate a trajectory file from disk."""
    p = Path(path)
    if not p.exists():
        raise TrajectoryError(f"trajectory not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TrajectoryError(f"invalid JSON: {e}") from e
    return validate_trajectory(data)
