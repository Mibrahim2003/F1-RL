"""Recorded-trajectory format: the shared interchange between live sim, replay, and
(later) the cloud clip renderer.

File shape (per the Phase 1 spec)::

    {
      "meta":   {"track_id": "oval", "dt": 0.05, "seed": 42, "created": "<ISO-8601>"},
      "frames": [{"t": 0.0, "car": {"x","y","yaw","speed"}, "telemetry": {...}}, ...]
    }

``load_trajectory`` schema-validates on load and raises :class:`TrajectoryError` on
anything malformed, so the UI can surface a clear error instead of crashing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TrajectoryError(ValueError):
    """Raised when a trajectory file is missing required structure."""


class TrajectoryRecorder:
    """Accumulates frames for one run and writes them to disk."""

    def __init__(self, track_id: str, dt: float, seed: int) -> None:
        self.meta = {
            "track_id": track_id,
            "dt": dt,
            "seed": seed,
            "created": datetime.now(UTC).isoformat(),
        }
        self.frames: list[dict[str, Any]] = []

    def append(self, t: float, car: dict[str, float], telemetry: dict[str, Any]) -> None:
        """Append one frame. ``car`` is the kinematic subset ``{x, y, yaw, speed}``."""
        self.frames.append({"t": round(t, 4), "car": car, "telemetry": telemetry})

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


def validate_trajectory(data: Any) -> dict[str, Any]:
    """Validate a parsed trajectory dict; return it unchanged or raise TrajectoryError."""
    if not isinstance(data, dict):
        raise TrajectoryError("trajectory must be a JSON object")
    meta = data.get("meta")
    frames = data.get("frames")
    if not isinstance(meta, dict) or not _REQUIRED_META.issubset(meta):
        raise TrajectoryError(f"meta must contain {sorted(_REQUIRED_META)}")
    if not isinstance(frames, list) or not frames:
        raise TrajectoryError("frames must be a non-empty list")
    for i, f in enumerate(frames):
        if not isinstance(f, dict) or "t" not in f or "car" not in f:
            raise TrajectoryError(f"frame {i} missing 't' or 'car'")
        car = f["car"]
        if not isinstance(car, dict) or not _REQUIRED_CAR.issubset(car):
            raise TrajectoryError(f"frame {i} car must contain {sorted(_REQUIRED_CAR)}")
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
