"""Procedural oval circuit (Phase 1).

A "stadium" centerline: two parallel straights joined by two semicircular ends, built
in SI meters at real proportions. Start/finish sits at sample 0 (start of the bottom
straight). Geometry is analytic; tangents, normals, arc length, and signed curvature are
derived on the closed loop so the result matches the :class:`~f1rl.track.schema.Track`
contract used by every later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from f1rl.track.geometry import arc_length, frames, signed_curvature
from f1rl.track.schema import Track


@dataclass(frozen=True)
class OvalParams:
    """Tunable oval geometry, SI meters."""

    straight_length: float = 900.0  # length of each straight, m
    corner_radius: float = 180.0  # radius of each semicircular end, m
    half_width: float = 7.0  # half the asphalt width (total ~14 m), m
    kerb_width: float = 1.0  # red/white band past the asphalt edge, m
    runoff_width: float = 12.0  # grass beyond the kerb, m (the oval has no gravel)
    spacing: float = 5.0  # target centerline sample spacing, m

    @classmethod
    def from_config(cls, cfg: Any) -> OvalParams:
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d: getattr(cfg, k, d))
        return cls(
            straight_length=float(get("straight_length", cls.straight_length)),
            corner_radius=float(get("corner_radius", cls.corner_radius)),
            half_width=float(get("half_width", cls.half_width)),
            kerb_width=float(get("kerb_width", cls.kerb_width)),
            runoff_width=float(get("runoff_width", cls.runoff_width)),
            spacing=float(get("spacing", cls.spacing)),
        )


def build_oval(params: OvalParams | None = None, name: str = "oval") -> Track:
    """Build the procedural oval as a closed :class:`Track`.

    The loop is traversed counter-clockwise; both corners are left-handers, so curvature
    is positive on the arcs and zero on the straights, and the normal points left
    (toward the inside of the corners).
    """
    p = params or OvalParams()
    pts = _stadium_points(p)
    return _track_from_centerline(pts, p, name)


def _stadium_points(p: OvalParams) -> np.ndarray:
    """Sample the stadium centerline CCW at ~uniform arc-length spacing (no duplicates)."""
    r = p.corner_radius
    hs = p.straight_length / 2.0
    n_straight = max(2, round(p.straight_length / p.spacing))
    n_arc = max(2, round(np.pi * r / p.spacing))

    segments: list[np.ndarray] = []

    # 1. Bottom straight: (-hs, -r) -> (+hs, -r), heading +x.
    t = np.linspace(0.0, 1.0, n_straight, endpoint=False)
    segments.append(np.column_stack([-hs + 2 * hs * t, np.full_like(t, -r)]))

    # 2. Right semicircle: center (hs, 0), angle -90deg -> +90deg (CCW).
    a = np.linspace(-np.pi / 2, np.pi / 2, n_arc, endpoint=False)
    segments.append(np.column_stack([hs + r * np.cos(a), r * np.sin(a)]))

    # 3. Top straight: (+hs, +r) -> (-hs, +r), heading -x.
    segments.append(np.column_stack([hs - 2 * hs * t, np.full_like(t, r)]))

    # 4. Left semicircle: center (-hs, 0), angle +90deg -> +270deg (CCW).
    a = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc, endpoint=False)
    segments.append(np.column_stack([-hs + r * np.cos(a), r * np.sin(a)]))

    return np.vstack(segments)


def _track_from_centerline(centerline: np.ndarray, p: OvalParams, name: str) -> Track:
    n = len(centerline)

    s, seg_len = arc_length(centerline, closed=True)
    tangent, normal = frames(centerline, closed=True)
    curvature = signed_curvature(tangent, seg_len, closed=True)

    half = np.full(n, p.half_width)
    gradient = np.zeros(n)
    # Oval surface bands: a thin constant kerb, grass runoff, no gravel.
    official = 2 * p.straight_length + 2 * np.pi * p.corner_radius

    return Track(
        name=name,
        centerline=centerline.astype(float),
        tangent=tangent.astype(float),
        normal=normal.astype(float),
        s=s.astype(float),
        curvature=curvature.astype(float),
        half_width_left=half,
        half_width_right=half,
        kerb_width=np.full(n, p.kerb_width),
        grass_width=np.full(n, p.runoff_width),
        gravel_width=np.zeros(n),
        gradient=gradient,
        closed=True,
        country="Proving Ground",
        official_length_m=float(official),
        source="procedural",
    )
