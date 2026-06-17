"""Offscreen eval-clip renderer (TECHNICAL_DESIGN.md §11, plan §C — TRAINING-ONLY).

Turns a recorded :class:`~f1rl.sim.recorder.TrajectoryRecorder` trajectory + a
:class:`~f1rl.track.schema.Track` into an mp4: the asphalt ribbon (``centerline ±
half_width``) drawn once into a background, then an oriented car glyph composited per frame,
encoded with imageio.

Headless: ``SDL_VIDEODRIVER=dummy`` is set **before** importing pygame so this runs on a
cloud box with no display. This module is never imported by ``env/`` or the training hot
path — only by the eval callback and the evaluate CLI. SI meters in; a meters→pixels camera
transform fits the whole track into the frame.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# Force the dummy SDL driver before pygame is imported anywhere in this process.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from f1rl.sim.recorder import load_trajectory  # noqa: E402
from f1rl.track.schema import Track  # noqa: E402

# Broadcast-ish palette (RGB).
_COLOR_BG = (18, 20, 24)
_COLOR_ASPHALT = (60, 63, 70)
_COLOR_EDGE = (210, 210, 215)
_COLOR_CENTERLINE = (90, 95, 105)
_COLOR_CAR = (235, 70, 60)
_COLOR_START = (240, 220, 80)

_DEFAULT_SIZE = (960, 720)  # px (width, height)
_DEFAULT_MARGIN = 0.08  # fraction of frame reserved as a border
_DEFAULT_FPS = 20  # matches control_hz
_CAR_LENGTH_M = 5.0  # glyph length (rectangle); width is half this
_CAR_WIDTH_M = 2.0


class _Camera:
    """Meters→pixels transform that fits the whole track into the frame (y flipped up)."""

    def __init__(self, track: Track, size: tuple[int, int], margin: float) -> None:
        w, h = size
        mn_x, mn_y, mx_x, mx_y = track.bounds()
        # Pad the bounds by the widest half-width so the asphalt edge is never clipped.
        pad = float(np.max(track.half_width_left) + np.max(track.half_width_right))
        mn_x -= pad
        mn_y -= pad
        mx_x += pad
        mx_y += pad
        span_x = max(mx_x - mn_x, 1e-6)
        span_y = max(mx_y - mn_y, 1e-6)
        usable_w = w * (1.0 - 2.0 * margin)
        usable_h = h * (1.0 - 2.0 * margin)
        self._scale = min(usable_w / span_x, usable_h / span_y)
        # Center the track in the frame.
        self._off_x = (w - self._scale * span_x) * 0.5 - self._scale * mn_x
        self._off_y = (h - self._scale * span_y) * 0.5 - self._scale * mn_y
        self._h = h
        self._mn_y = mn_y
        self._mx_y = mx_y

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        px = self._scale * x + self._off_x
        # Flip y so world-up maps to screen-up.
        py = self._h - (self._scale * y + self._off_y)
        return px, py

    def to_px_array(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=np.float64)
        out[:, 0] = self._scale * pts[:, 0] + self._off_x
        out[:, 1] = self._h - (self._scale * pts[:, 1] + self._off_y)
        return out

    @property
    def scale(self) -> float:
        return self._scale


def render_trajectory(
    track: Track,
    trajectory: dict[str, Any] | str | Path,
    out_path: str | Path,
    *,
    size: tuple[int, int] = _DEFAULT_SIZE,
    fps: int = _DEFAULT_FPS,
    margin: float = _DEFAULT_MARGIN,
    max_frames: int | None = None,
) -> Path:
    """Render a recorded trajectory over ``track`` to an mp4 at ``out_path``.

    Args:
        track: The circuit (asphalt ribbon drawn from ``centerline ± half_width``).
        trajectory: A trajectory dict (``{"meta", "frames"}``) or a path to a JSON file.
        out_path: Destination mp4 path (parent dirs created).
        size: Frame size in pixels ``(width, height)``.
        fps: Output frame rate (match the control rate for real-time playback).
        margin: Border fraction reserved around the track.
        max_frames: Optional cap on rendered frames (keeps eval clips short).

    Returns:
        The written mp4 path.
    """
    import imageio.v2 as imageio
    import pygame

    if isinstance(trajectory, (str, Path)):
        trajectory = load_trajectory(trajectory)
    frames = trajectory.get("frames", [])
    if not frames:
        raise ValueError("trajectory has no frames to render")
    if max_frames is not None:
        frames = frames[:max_frames]

    pygame.init()
    try:
        surface = pygame.Surface(size)
        cam = _Camera(track, size, margin)
        background = _draw_track_background(pygame, size, track, cam)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out), fps=fps, macro_block_size=None)
        try:
            for frame in frames:
                surface.blit(background, (0, 0))
                car = frame["car"]
                _draw_car(pygame, surface, cam, float(car["x"]), float(car["y"]), float(car["yaw"]))
                # pygame surface -> (H, W, 3) RGB array for imageio.
                arr = pygame.surfarray.array3d(surface)
                arr = np.transpose(arr, (1, 0, 2))  # (W,H,3) -> (H,W,3)
                writer.append_data(arr)
        finally:
            writer.close()
    finally:
        pygame.quit()
    return out


def _draw_track_background(pygame: Any, size: tuple[int, int], track: Track, cam: _Camera) -> Any:
    """Draw the static asphalt ribbon + edges + start line once into a background surface."""
    bg = pygame.Surface(size)
    bg.fill(_COLOR_BG)

    c = np.asarray(track.centerline, dtype=np.float64)
    n = np.asarray(track.normal, dtype=np.float64)
    hl = np.asarray(track.half_width_left, dtype=np.float64).reshape(-1, 1)
    hr = np.asarray(track.half_width_right, dtype=np.float64).reshape(-1, 1)
    left = c + n * hl
    right = c - n * hr

    left_px = cam.to_px_array(left)
    right_px = cam.to_px_array(right)

    # Asphalt ribbon: a filled polygon left-edge forward then right-edge backward.
    ribbon = np.concatenate([left_px, right_px[::-1]], axis=0)
    pygame.draw.polygon(bg, _COLOR_ASPHALT, [(float(p[0]), float(p[1])) for p in ribbon])

    closed = bool(track.closed)
    _draw_polyline(pygame, bg, left_px, _COLOR_EDGE, width=2, closed=closed)
    _draw_polyline(pygame, bg, right_px, _COLOR_EDGE, width=2, closed=closed)
    _draw_polyline(pygame, bg, cam.to_px_array(c), _COLOR_CENTERLINE, width=1, closed=closed)

    # Start/finish line: left edge to right edge at sample 0.
    p0_l = cam.to_px(float(left[0, 0]), float(left[0, 1]))
    p0_r = cam.to_px(float(right[0, 0]), float(right[0, 1]))
    pygame.draw.line(bg, _COLOR_START, p0_l, p0_r, 3)
    return bg


def _draw_polyline(
    pygame: Any,
    surface: Any,
    pts_px: np.ndarray,
    color: tuple[int, int, int],
    width: int,
    closed: bool,
) -> None:
    points = [(float(p[0]), float(p[1])) for p in pts_px]
    if len(points) >= 2:
        pygame.draw.lines(surface, color, closed, points, width)


def _draw_car(pygame: Any, surface: Any, cam: _Camera, x: float, y: float, yaw: float) -> None:
    """Draw an oriented rectangle car glyph (SI meters → pixels), nose along ``yaw``."""
    half_l = _CAR_LENGTH_M * 0.5
    half_w = _CAR_WIDTH_M * 0.5
    # Body-frame corners (forward = +x): nose-left, nose-right, tail-right, tail-left.
    corners = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=np.float64,
    )
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    world = corners @ rot.T + np.array([x, y], dtype=np.float64)
    px = [cam.to_px(float(p[0]), float(p[1])) for p in world]
    pygame.draw.polygon(surface, _COLOR_CAR, px)
