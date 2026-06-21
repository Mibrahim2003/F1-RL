"""Fixed-step simulation loop (TECHNICAL_DESIGN.md §3).

One control step at ``control_hz`` runs ``substeps`` physics substeps of ``dt_physics``
each (5 × 0.01 s = 0.05 s = 20 Hz). The loop owns the car state, the physics stepper, and
the lap timer, and emits a JSON-ready state frame per step. It never renders.

Phase 5: the state frame carries a ``cars: [...]`` array (a single car is a one-element
array), and a :class:`FieldSimLoop` drives **N cars** on one shared track with one shared
pilot for the live field view. The single-car ``car``/``telemetry`` top-level keys are kept so
the Phase 1/4 one-car frontend path keeps working unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from f1rl.env.conditions import Conditions
from f1rl.env.observations import track_query
from f1rl.physics.base import CarState, PhysicsModel
from f1rl.sim.timing import LapTimer
from f1rl.track.schema import Track

_COMPOUND_NAMES = ("soft", "medium", "hard", "intermediate", "wet")
# Default team colors for the live field (render only; mirrors configs/default.yaml grid block).
_DEFAULT_TEAM_COLORS = ("#e10600", "#00d2be", "#0600ef", "#ff8700", "#006f62")


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
        conditions: Conditions | None = None,
    ) -> None:
        self.physics = physics
        self.track = track
        self.cfg = sim_cfg
        self.total_laps = int(total_laps)
        self.timer = LapTimer(track, pole_time_s)
        # The grip provider (shared with the env via the same Conditions.grip_at). When None,
        # the loop uses the constant sim grip — the Phase-1 kinematic behavior.
        self.conditions = conditions
        self._grip = float(sim_cfg.grip)
        self.reset()

    def _start_state(self) -> CarState:
        c = self.track.centerline[0]
        tan = self.track.tangent[0]
        compound = int(self.conditions.tires.start_compound) if self.conditions else 0
        return CarState(
            x=float(c[0]), y=float(c[1]), yaw=math.atan2(tan[1], tan[0]), compound=compound
        )

    def reset(self) -> None:
        self.t = 0.0
        self.timer.reset()
        self.state = self._start_state()
        self._grip = float(self.cfg.grip)

    def place(self, idx: int, lateral_m: float = 0.0) -> None:
        """Place the car at centerline sample ``idx`` (optionally ``lateral_m`` sideways).

        Used by :class:`FieldSimLoop` to spread the field over a starting grid; heading is the
        centerline tangent at ``idx``. Resets the timer so the new start is the lap origin.
        """
        n = len(self.track.centerline)
        i = int(idx) % n
        c = self.track.centerline[i]
        tan = self.track.tangent[i]
        x = float(c[0])
        y = float(c[1])
        if lateral_m != 0.0:
            nrm = self.track.normal[i]
            x += float(nrm[0]) * lateral_m
            y += float(nrm[1]) * lateral_m
        compound = int(self.conditions.tires.start_compound) if self.conditions else 0
        self.t = 0.0
        self.timer.reset()
        yaw = math.atan2(float(tan[1]), float(tan[0]))
        self.state = CarState(x=x, y=y, yaw=yaw, compound=compound)
        self._grip = float(self.cfg.grip)

    def set_weather(self, weather: str) -> None:
        """Set the live weather (``dry`` | ``damp`` | ``wet``); changes grip immediately."""
        if self.conditions is not None:
            self.conditions.set_weather(weather)

    def _step_grip(self) -> float:
        """Grip for this step: the shared pipeline (surface/weather/wear) or constant fallback."""
        if self.conditions is None:
            return self.cfg.grip
        idx, _s, signed_lateral, _hw, _heading = track_query(
            self.track, self.state.x, self.state.y, self.state.yaw
        )
        return self.conditions.grip_at(
            self.track, idx, signed_lateral, self.state.tire_wear, self.state.compound
        )

    def step(self, steer: float, longitudinal: float) -> dict[str, Any]:
        """Advance one control step and return the state frame."""
        grip = self._step_grip()
        self._grip = grip
        for _ in range(self.cfg.substeps):
            self.state = self.physics.step(
                self.state, steer, longitudinal, grip, self.cfg.dt_physics
            )
        self.t += self.cfg.dt_control
        timing = self.timer.update(self.state.x, self.state.y, self.t)
        return self._frame(timing)

    def _pose(self) -> dict[str, float]:
        s = self.state
        return {
            "x": round(s.x, 3),
            "y": round(s.y, 3),
            "yaw": round(s.yaw, 5),
            "speed": round(s.speed, 3),
        }

    def _telemetry(self, timing: Any) -> dict[str, Any]:
        s = self.state
        return {
            "speed_kmh": round(s.speed * 3.6),
            "lap_time": round(timing.lap_time, 3),
            "delta_to_pole": round(timing.delta_to_pole, 3),
            "lap": min(timing.lap, self.total_laps),
            "lap_total": self.total_laps,
            "best_lap": round(timing.best_lap, 3) if timing.best_lap is not None else None,
            "last_lap": round(timing.last_lap, 3) if timing.last_lap is not None else None,
            "progress": round(timing.progress, 4),
            # Phase 3b grip-pipeline readouts.
            "compound": _COMPOUND_NAMES[s.compound] if 0 <= s.compound < 5 else "soft",
            "tire_wear": round(s.tire_wear, 4),
            "grip": round(self._grip, 4),
            "weather": self.conditions.weather if self.conditions is not None else "dry",
        }

    def _frame(self, timing: Any) -> dict[str, Any]:
        """One-car state frame. Carries ``cars: [entry]`` (Phase 5) plus the legacy
        top-level ``car``/``telemetry`` so the one-car frontend path is unchanged."""
        pose = self._pose()
        telemetry = self._telemetry(timing)
        car_entry = {
            "id": "car_0",
            "team": 0,
            **pose,
            "telemetry": telemetry,
        }
        return {
            "type": "state",
            "t": round(self.t, 4),
            "car": pose,
            "telemetry": telemetry,
            "cars": [car_entry],
        }


class FieldSimLoop:
    """Drive a FIELD of N cars on one shared track with one shared pilot (Phase 5 live view).

    Each car is its own :class:`SimLoop` (own ``CarState`` + ``LapTimer`` + grip provider) on
    the shared track/physics; one pilot (the centerline autopilot or a checkpoint
    :class:`~f1rl.sim.policy_pilot.PolicyPilot`) drives every car via the unchanged
    ``control(state)`` interface — so one policy laps the whole grid. No car observes another
    and no collision is computed (Phase 6). The emitted frame is the same ``cars: [...]`` shape
    as :meth:`SimLoop._frame`, with a track-position ``gap_m`` per car (leader = furthest along
    by ``completed_laps * length + s``).
    """

    def __init__(
        self,
        physics: PhysicsModel,
        track: Track,
        sim_cfg: SimConfig,
        pole_time_s: float,
        total_laps: int,
        n_agents: int,
        *,
        conditions_factory: Any | None = None,
        reset_mode: str = "grid",
        grid_spacing_m: float = 12.0,
        grid_lateral_m: float = 3.0,
        team_colors: tuple[str, ...] = _DEFAULT_TEAM_COLORS,
    ) -> None:
        self.track = track
        self.cfg = sim_cfg
        self.total_laps = int(total_laps)
        self.n_agents = max(1, int(n_agents))
        self.reset_mode = reset_mode
        self.grid_spacing_m = float(grid_spacing_m)
        self.grid_lateral_m = float(grid_lateral_m)
        self.team_colors = team_colors or _DEFAULT_TEAM_COLORS
        self.cars = [
            SimLoop(
                physics,
                track,
                sim_cfg,
                pole_time_s,
                total_laps,
                conditions_factory() if conditions_factory is not None else None,
            )
            for _ in range(self.n_agents)
        ]
        self.t = 0.0
        self.reset()

    def reset(self) -> None:
        """Place every car on the starting grid (distinct, non-overlapping slots at S/F)."""
        self.t = 0.0
        for i, (idx, lateral) in enumerate(self._slots()):
            self.cars[i].place(idx, lateral)

    def set_weather(self, weather: str) -> None:
        for car in self.cars:
            car.set_weather(weather)

    def _slots(self) -> list[tuple[int, float]]:
        # Two-column grid laid out forward from the S/F line (front row furthest along), so the
        # track-position gap reads monotonically from the front of the grid back.
        s = self.track.s
        n_points = len(self.track.centerline)
        half = self.grid_lateral_m * 0.5
        n_rows = (self.n_agents + 1) // 2
        slots: list[tuple[int, float]] = []
        for i in range(self.n_agents):
            row = i // 2
            col = i % 2
            target_s = (n_rows - row) * self.grid_spacing_m
            idx = int(min(range(n_points), key=lambda k: abs(float(s[k]) - target_s)))
            lateral = half if col == 0 else -half
            hw = float(
                self.track.half_width_left[idx]
                if lateral >= 0
                else self.track.half_width_right[idx]
            )
            lateral = max(-(hw - 0.5), min(hw - 0.5, lateral))
            slots.append((idx, lateral))
        return slots

    def step(self, pilot: Any) -> dict[str, Any]:
        """Advance every car one control step with the shared ``pilot``; return a field frame."""
        entries: list[dict[str, Any]] = []
        progresses: list[float] = []
        length = float(self.track.length)
        for i, car in enumerate(self.cars):
            steer, longitudinal = pilot.control(car.state)
            frame = car.step(steer, longitudinal)
            telem = frame["telemetry"]
            total_progress = (car.timer.completed_laps + float(telem["progress"])) * length
            progresses.append(total_progress)
            entries.append(
                {
                    "id": f"car_{i}",
                    "team": i % len(self.team_colors),
                    **frame["car"],
                    "telemetry": telem,
                    "_total_progress": total_progress,
                }
            )
        self.t = self.cars[0].t if self.cars else self.t

        leader = max(progresses) if progresses else 0.0
        for e in entries:
            e["gap_m"] = round(leader - e.pop("_total_progress"), 2)

        # The leader's telemetry mirrors the single-car keys so the legacy HUD still reads it.
        leader_i = max(range(len(entries)), key=lambda k: progresses[k]) if entries else 0
        lead = entries[leader_i] if entries else None
        return {
            "type": "state",
            "t": round(self.t, 4),
            "car": {k: lead[k] for k in ("x", "y", "yaw", "speed")} if lead else None,
            "telemetry": lead["telemetry"] if lead else None,
            "cars": entries,
        }
