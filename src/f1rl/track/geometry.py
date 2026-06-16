"""Shared centerline geometry (TECHNICAL_DESIGN.md §6).

Both the procedural oval (Phase 1) and the FastF1/OSM build pipeline (Phase 2) need the
same tangent / left-normal / arc-length / signed-curvature derivation, so it lives here
once. The closed-loop math is seam-safe: central differences wrap around index 0, and the
curvature denominator is the two-segment span around each sample.
"""

from __future__ import annotations

import numpy as np


def arc_length(centerline: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative arc length ``s`` (N,) and per-segment chord length ``seg_len`` (N,).

    For a closed loop ``seg_len[i]`` is the chord from sample ``i`` to ``i+1`` (wrapping at
    the seam); ``s`` starts at 0 and excludes the closing chord. For an open path the final
    segment length is duplicated so ``seg_len`` stays length ``N``.
    """
    if closed:
        deltas = np.diff(centerline, axis=0, append=centerline[:1])
        seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
        s = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        return s, seg_len
    deltas = np.diff(centerline, axis=0)
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    seg_len = np.concatenate([seg, seg[-1:]])
    return s, seg_len


def frames(centerline: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangent (N, 2) and left normal (N, 2) via central differences.

    The left normal is the tangent rotated +90° (CCW), pointing left of travel.
    """
    if closed:
        fwd = np.roll(centerline, -1, axis=0) - centerline
        bwd = centerline - np.roll(centerline, 1, axis=0)
    else:
        fwd = np.empty_like(centerline)
        bwd = np.empty_like(centerline)
        fwd[:-1] = centerline[1:] - centerline[:-1]
        fwd[-1] = fwd[-2]
        bwd[1:] = centerline[1:] - centerline[:-1]
        bwd[0] = bwd[1]
    tangent = fwd + bwd
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal


def signed_curvature(tangent: np.ndarray, seg_len: np.ndarray, closed: bool) -> np.ndarray:
    """Signed curvature (N,) = d(heading)/ds, central difference (wrap-aware for closed).

    Positive curvature turns left (CCW); negative turns right.
    """
    heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    if closed:
        dtheta = np.angle(np.exp(1j * (np.roll(heading, -1) - np.roll(heading, 1))))
        ds_span = seg_len + np.roll(seg_len, 1)
    else:
        dtheta = np.empty_like(heading)
        dtheta[1:-1] = np.angle(np.exp(1j * (heading[2:] - heading[:-2])))
        dtheta[0] = np.angle(np.exp(1j * (heading[1] - heading[0])))
        dtheta[-1] = np.angle(np.exp(1j * (heading[-1] - heading[-2])))
        ds_span = seg_len + np.roll(seg_len, 1)
        ds_span[0] = seg_len[0]
        ds_span[-1] = seg_len[-1]
    return dtheta / ds_span


def derive_geometry(
    centerline: np.ndarray, closed: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: return ``(tangent, normal, s, curvature)`` for a centerline."""
    s, seg_len = arc_length(centerline, closed)
    tangent, normal = frames(centerline, closed)
    curvature = signed_curvature(tangent, seg_len, closed)
    return tangent, normal, s, curvature
