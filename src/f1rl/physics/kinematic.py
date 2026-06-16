"""Kinematic bicycle model (TECHNICAL_DESIGN.md §5, Phase 1).

No tire slip: the turn radius follows from wheelbase, steering, and speed. Longitudinal
motion is force-based (engine/brake minus aerodynamic drag and rolling resistance) so the
car has believable acceleration and a sensible top speed. Integrated with semi-implicit
Euler. Every constant comes from config — nothing load-bearing is hardcoded in logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from f1rl.physics.base import CarState

G = 9.81  # m/s^2


@dataclass(frozen=True)
class KinematicParams:
    """Tunable kinematic-model constants, all SI."""

    wheelbase: float = 3.6  # m
    mass: float = 798.0  # kg
    max_steer: float = math.radians(18.0)  # rad at the wheels
    max_engine_force: float = 6000.0  # N (sets acceleration and, with drag, top speed)
    max_brake_force: float = 16000.0  # N
    drag_coeff: float = 0.70  # N / (m/s)^2  (lumped 0.5*rho*Cd*A)
    rolling_coeff: float = 0.015  # dimensionless; rolling resistance = c * m * g

    @classmethod
    def from_config(cls, cfg: Any) -> KinematicParams:
        """Build params from a mapping/OmegaConf node.

        ``max_steer`` is given in **degrees** in config (key ``max_steer_deg``) and
        converted to radians here.
        """
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d: getattr(cfg, k, d))
        return cls(
            wheelbase=float(get("wheelbase", cls.wheelbase)),
            mass=float(get("mass", cls.mass)),
            max_steer=math.radians(float(get("max_steer_deg", math.degrees(cls.max_steer)))),
            max_engine_force=float(get("max_engine_force", cls.max_engine_force)),
            max_brake_force=float(get("max_brake_force", cls.max_brake_force)),
            drag_coeff=float(get("drag_coeff", cls.drag_coeff)),
            rolling_coeff=float(get("rolling_coeff", cls.rolling_coeff)),
        )


class KinematicBicycle:
    """A no-slip bicycle model. Implements :class:`~f1rl.physics.base.PhysicsModel`."""

    def __init__(self, params: KinematicParams | None = None) -> None:
        self.params = params or KinematicParams()

    def step(
        self,
        state: CarState,
        steer: float,
        longitudinal: float,
        grip: float,  # noqa: ARG002 - part of the interface; unused by the kinematic model
        dt: float,
    ) -> CarState:
        p = self.params
        steer = _clamp(steer, -1.0, 1.0)
        longitudinal = _clamp(longitudinal, -1.0, 1.0)
        delta = steer * p.max_steer

        v = state.vx

        # Longitudinal force: throttle drives, brake/resistance oppose motion.
        if longitudinal >= 0.0:
            f_drive = longitudinal * p.max_engine_force
        else:
            f_drive = longitudinal * p.max_brake_force  # negative -> braking

        f_drag = p.drag_coeff * v * abs(v)
        f_roll = p.rolling_coeff * p.mass * G * _sign(v)
        accel = (f_drive - f_drag - f_roll) / p.mass

        v_new = v + accel * dt
        # The car never rolls backwards under braking/coasting: clamp through zero.
        if v > 0.0 and v_new < 0.0 and longitudinal <= 0.0:
            v_new = 0.0
        elif v_new < 0.0:
            # No reverse gear in Phase 1.
            v_new = 0.0

        # Heading from the no-slip turn radius, then advance the pose.
        yaw_rate = (v_new * math.tan(delta) / p.wheelbase) if p.wheelbase > 0 else 0.0
        yaw_new = state.yaw + yaw_rate * dt
        x_new = state.x + v_new * math.cos(yaw_new) * dt
        y_new = state.y + v_new * math.sin(yaw_new) * dt

        return CarState(
            x=x_new,
            y=y_new,
            yaw=_wrap_pi(yaw_new),
            vx=v_new,
            vy=0.0,
            yaw_rate=yaw_rate,
            tire_wear=state.tire_wear,
            compound=state.compound,
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _sign(v: float) -> float:
    return 1.0 if v > 0.0 else -1.0 if v < 0.0 else 0.0


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))
