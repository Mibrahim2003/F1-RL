"""A simple centerline-following autopilot for the watch-live mode.

This is **not** a learned policy — it is a pure-pursuit follower that proves the live
path (engine -> socket -> viewport) before any PPO policy exists. It steers toward a
lookahead point on the centerline and sets throttle/brake from the curvature ahead.
"""

from __future__ import annotations

import math

from f1rl.physics.base import CarState
from f1rl.track.schema import Track


class CenterlineAutopilot:
    """Pure-pursuit centerline follower producing ``(steer, longitudinal)`` commands."""

    def __init__(
        self,
        track: Track,
        max_steer_rad: float,
        lookahead_m: float = 35.0,
        target_lat_accel: float = 16.0,  # m/s^2, sets cornering speed
        v_max: float = 90.0,  # m/s speed cap
    ) -> None:
        self.track = track
        self.max_steer = max_steer_rad
        self.target_lat_accel = target_lat_accel
        self.v_max = v_max
        n = len(track.centerline)
        mean_spacing = track.length / n
        self.lookahead_steps = max(1, round(lookahead_m / mean_spacing))

    def control(self, state: CarState) -> tuple[float, float]:
        track = self.track
        n = len(track.centerline)
        idx = track.nearest_index(state.x, state.y)
        target = track.centerline[(idx + self.lookahead_steps) % n]

        desired_yaw = math.atan2(target[1] - state.y, target[0] - state.x)
        err = _wrap_pi(desired_yaw - state.yaw)
        steer = _clamp(err / self.max_steer, -1.0, 1.0)

        # Cornering speed target from the sharpest curvature in the lookahead window.
        window = range(idx, idx + self.lookahead_steps + 1)
        kappa = max(abs(float(track.curvature[i % n])) for i in window)
        if kappa > 1e-4:
            v_target = min(self.v_max, math.sqrt(self.target_lat_accel / kappa))
        else:
            v_target = self.v_max

        longitudinal = 0.7 if state.vx < v_target else -0.5
        return steer, longitudinal


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
