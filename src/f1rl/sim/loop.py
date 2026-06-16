"""Fixed-step simulation loop (TECHNICAL_DESIGN.md §3).

One control step at ``control_hz`` runs ``substeps`` physics substeps of ``dt_physics``
each (5 × 0.01 s = 0.05 s = 20 Hz). The loop owns the car state, the physics stepper, and
the lap timer, and emits a JSON-ready state frame per step. It never renders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from f1rl.physics.base import CarState, PhysicsModel
from f1rl.sim.timing import LapTimer
from f1rl.track.schema import Track


@dataclass(frozen=True)
class SimConfig:
    """Loop timing and the constant grip scalar."""

    control_hz: int = 20
    substeps: int = 5
    dt_physics: float = 0.01
    grip: float = 1.0

    @classmethod
    def from_config(cls, cfg: Any) -> SimConfig:
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d: getattr(cfg, k, d))
        return cls(
            control_hz=int(get("control_hz", cls.control_hz)),
            substeps=int(get("substeps", cls.substeps)),
            dt_physics=float(get("dt_physics", cls.dt_physics)),
            grip=float(get("grip", cls.grip)),
        )

    @property
    def dt_control(self) -> float:
        return 1.0 / self.control_hz


class SimLoop:
    """Drives one car around one track at a fixed control rate."""

    def __init__(
        self,
        physics: PhysicsModel,
        track: Track,
        sim_cfg: SimConfig,
        pole_time_s: float,
        total_laps: int,
    ) -> None:
        self.physics = physics
        self.track = track
        self.cfg = sim_cfg
        self.total_laps = int(total_laps)
        self.timer = LapTimer(track, pole_time_s)
        self.reset()

    def _start_state(self) -> CarState:
        c = self.track.centerline[0]
        tan = self.track.tangent[0]
        return CarState(x=float(c[0]), y=float(c[1]), yaw=math.atan2(tan[1], tan[0]))

    def reset(self) -> None:
        self.t = 0.0
        self.timer.reset()
        self.state = self._start_state()

    def step(self, steer: float, longitudinal: float) -> dict[str, Any]:
        """Advance one control step and return the state frame."""
        for _ in range(self.cfg.substeps):
            self.state = self.physics.step(
                self.state, steer, longitudinal, self.cfg.grip, self.cfg.dt_physics
            )
        self.t += self.cfg.dt_control
        timing = self.timer.update(self.state.x, self.state.y, self.t)
        return self._frame(timing)

    def _frame(self, timing: Any) -> dict[str, Any]:
        s = self.state
        return {
            "type": "state",
            "t": round(self.t, 4),
            "car": {
                "x": round(s.x, 3),
                "y": round(s.y, 3),
                "yaw": round(s.yaw, 5),
                "speed": round(s.speed, 3),
            },
            "telemetry": {
                "speed_kmh": round(s.speed * 3.6),
                "lap_time": round(timing.lap_time, 3),
                "delta_to_pole": round(timing.delta_to_pole, 3),
                "lap": min(timing.lap, self.total_laps),
                "lap_total": self.total_laps,
                "best_lap": round(timing.best_lap, 3) if timing.best_lap is not None else None,
                "last_lap": round(timing.last_lap, 3) if timing.last_lap is not None else None,
                "progress": round(timing.progress, 4),
            },
        }
