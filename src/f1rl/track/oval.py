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

from f1rl.track.schema import Track


@dataclass(frozen=True)
class OvalParams:
    """Tunable oval geometry, SI meters."""

    straight_length: float = 900.0  # length of each straight, m
    corner_radius: float = 180.0  # radius of each semicircular end, m
    half_width: float = 7.0  # half the asphalt width (total ~14 m), m
    runoff_width: float = 12.0  # grass/gravel beyond each edge, m
    spacing: float = 5.0  # target centerline sample spacing, m

    @classmethod
    def from_config(cls, cfg: Any) -> OvalParams:
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d: getattr(cfg, k, d))
        return cls(
            straight_length=float(get("straight_length", cls.straight_length)),
            corner_radius=float(get("corner_radius", cls.corner_radius)),
            half_width=float(get("half_width", cls.half_width)),
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

    # Cumulative arc length (chord-based) along the closed loop.
    deltas = np.diff(centerline, axis=0, append=centerline[:1])
    seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])

    # Unit tangent via central differences with wraparound.
    fwd = np.roll(centerline, -1, axis=0) - centerline
    bwd = centerline - np.roll(centerline, 1, axis=0)
    tangent = fwd + bwd
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    # Left normal (rotate tangent +90deg).
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    # Signed curvature = d(heading)/ds, central difference with wraparound. The
    # denominator is the two-segment span around each sample (seam-safe via seg_len).
    heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    dtheta = np.angle(np.exp(1j * (np.roll(heading, -1) - np.roll(heading, 1))))
    ds_span = seg_len + np.roll(seg_len, 1)
    curvature = dtheta / ds_span

    half = np.full(n, p.half_width)
    runoff = np.full(n, p.runoff_width)
    gradient = np.zeros(n)

    return Track(
        name=name,
        centerline=centerline.astype(float),
        tangent=tangent.astype(float),
        normal=normal.astype(float),
        s=s.astype(float),
        curvature=curvature.astype(float),
        half_width_left=half,
        half_width_right=half,
        runoff_width=runoff,
        gradient=gradient,
        closed=True,
    )
