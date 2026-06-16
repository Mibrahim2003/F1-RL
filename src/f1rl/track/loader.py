"""Track loading and the cached-circuit catalog.

The procedural oval is built on demand (Phase 1). Every real circuit (Phase 2) is loaded
from its cached ``data/tracks/<id>.npz`` produced offline by :mod:`f1rl.track.build`. This
module is runtime-safe: it never imports FastF1/Shapely/requests and never touches the
network — it only reads cached ``.npz`` files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from f1rl.track.oval import OvalParams, build_oval
from f1rl.track.schema import Track

DEFAULT_TRACKS_DIR = Path("data/tracks")
# |curvature| (1/m) above which a sample is "in a corner" — used to count turns for the catalog.
_TURN_KAPPA = 0.004


def load_track(
    track_id: str = "oval", cfg: Any | None = None, tracks_dir: Path = DEFAULT_TRACKS_DIR
) -> Track:
    """Load a track by id.

    Args:
        track_id: ``"oval"`` builds the procedural oval. Any other id loads
            ``data/tracks/<id>.npz``.
        cfg: Optional config node with oval geometry overrides (oval only).
        tracks_dir: Directory holding cached ``.npz`` circuits.

    Raises:
        FileNotFoundError: For a real circuit whose ``.npz`` has not been built yet.
    """
    if track_id == "oval":
        params = OvalParams.from_config(cfg) if cfg is not None else None
        return build_oval(params, name="oval")

    cache = tracks_dir / f"{track_id}.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"Cached track '{track_id}' not found at {cache}. Build it first:\n"
            f"  .venv/Scripts/python.exe scripts/build_all_tracks.py {track_id}"
        )
    return Track.from_npz(cache)


def list_tracks(tracks_dir: Path = DEFAULT_TRACKS_DIR) -> list[dict[str, Any]]:
    """Lightweight catalog of every built circuit, for the selector.

    Always includes the procedural oval. Each entry: ``id``, ``name``, ``country``,
    ``length``, ``official_length_m``, ``turns``, ``source``, ``low_confidence``.
    """
    catalog: list[dict[str, Any]] = [_summary(build_oval(name="oval"))]
    if tracks_dir.is_dir():
        for p in sorted(tracks_dir.glob("*.npz")):
            try:
                catalog.append(_summary(Track.from_npz(p)))
            except Exception:
                # A corrupt/old cache file should not break the whole catalog.
                continue
    return catalog


def _summary(track: Track) -> dict[str, Any]:
    return {
        "id": track.name,
        "name": track.name,
        "country": track.country,
        "length": round(track.length, 1),
        "official_length_m": track.official_length_m,
        "turns": _count_turns(track.curvature),
        "source": track.source,
        "low_confidence": track.low_confidence,
    }


def _count_turns(curvature: np.ndarray) -> int:
    """Count corners as contiguous runs of above-threshold curvature around the loop."""
    in_corner = np.abs(curvature) >= _TURN_KAPPA
    if not in_corner.any():
        return 0
    # Count rising edges on the circular array (closed loop), so a corner spanning the
    # seam is not double-counted.
    prev = np.roll(in_corner, 1)
    return int(np.sum(in_corner & ~prev))
