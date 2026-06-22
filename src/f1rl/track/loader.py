"""Track loading and the cached-circuit catalog.

The procedural oval is built on demand (Phase 1). Every real circuit (Phase 2) is loaded
from its cached ``data/tracks/<id>.npz`` produced offline by :mod:`f1rl.track.build`. This
module is runtime-safe: it never imports FastF1/Shapely/requests and never touches the
network — it only reads cached ``.npz`` files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.track.oval import OvalParams, build_oval
from f1rl.track.schema import Track

DEFAULT_TRACKS_DIR = Path("data/tracks")
# |curvature| (1/m) above which a sample is "in a corner" — used to count turns for the catalog.
_TURN_KAPPA = 0.004

# Pre-baked static payloads served verbatim by the web backend (no per-request numpy load or
# JSON re-serialize). Produced offline next to each ``.npz`` by the build pipeline / bake script
# and committed to the repo, so a circuit is "preloaded" on the next app open. The loaders below
# fall back to recomputing from the ``.npz`` when a payload is missing/corrupt, so a stale or
# absent bake never breaks an endpoint — it just costs the old per-request work.
API_SUFFIX = ".api.json"
CATALOG_FILE = "_catalog.json"


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


# ----- pre-baked static payloads (the web backend serves these verbatim) ---------------------


def track_api_path(track_id: str, tracks_dir: Path = DEFAULT_TRACKS_DIR) -> Path:
    """Path of the pre-baked ``GET /track/<id>`` JSON payload for ``track_id``."""
    return tracks_dir / f"{track_id}{API_SUFFIX}"


def write_track_api(track: Track, tracks_dir: Path = DEFAULT_TRACKS_DIR) -> Path:
    """Bake ``track``'s API payload to ``<id>.api.json`` (the exact ``GET /track`` body).

    Called by the build pipeline and the surface editor so the committed payload always
    mirrors the ``.npz``. Runtime-safe — no FastF1, no network.
    """
    tracks_dir.mkdir(parents=True, exist_ok=True)
    path = track_api_path(track.name, tracks_dir)
    path.write_text(json.dumps(track.to_api_dict()), encoding="utf-8")
    return path


def write_catalog(tracks_dir: Path = DEFAULT_TRACKS_DIR) -> Path:
    """Bake the full selector catalog (oval + every built circuit) to ``_catalog.json``.

    This is the verbatim ``GET /api/tracks`` body (minus the ``{"tracks": ...}`` wrapper the
    route adds), so the server avoids reloading every ``.npz`` just to summarize it.
    """
    tracks_dir.mkdir(parents=True, exist_ok=True)
    path = tracks_dir / CATALOG_FILE
    path.write_text(json.dumps(list_tracks(tracks_dir)), encoding="utf-8")
    return path


def bake_all(tracks_dir: Path = DEFAULT_TRACKS_DIR) -> list[str]:
    """Bake every cached ``.npz`` to its ``<id>.api.json`` and refresh ``_catalog.json``.

    Returns the baked circuit ids. Runtime-safe: reads only existing ``.npz`` caches, so it
    regenerates the served payloads without rebuilding circuits from the network.
    """
    baked: list[str] = []
    if tracks_dir.is_dir():
        for p in sorted(tracks_dir.glob("*.npz")):
            try:
                write_track_api(Track.from_npz(p), tracks_dir)
            except Exception:
                continue  # a corrupt cache should not abort the rest of the bake
            baked.append(p.stem)
    write_catalog(tracks_dir)
    return baked
