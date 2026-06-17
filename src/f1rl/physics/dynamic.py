"""Dynamic bicycle model with a friction circle (TECHNICAL_DESIGN.md §5, plan §A).

Body-frame state ``(vx, vy, r=yaw_rate)``, linear lateral tire forces, and the **friction
circle** — the central realism abstraction: per-axle combined (longitudinal + lateral) tire
force is capped at ``grip * normal_load``, so the car loses grip when overdriven. The grip
scalar comes from the pipeline (:mod:`f1rl.physics.tires`) and is passed in by the env; the
step itself is a **pure function** of ``(state, steer, longitudinal, grip, dt)`` — no
globals, no rendering, no track lookups.

Two numerical guards make the model stable at low speed (the #1 dynamic-model bug):

- ``vx_safe = max(vx, v_eps)`` in the slip-angle ``atan2`` denominators, and
- a low-speed blend (Kong et al. 2015): below ``v_blend`` the yaw rate / lateral velocity
  blend toward the no-slip kinematic prediction, so standstill and pit-crawl never blow up.

Integrated with semi-implicit Euler over the env's physics substeps. SI units throughout;
every constant comes from config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from f1rl.physics.base import CarState

G = 9.81  # m/s^2


@dataclass(frozen=True)
class DynamicParams:
    """Tunable dynamic-model constants, all SI."""

    wheelbase: float = 3.6  # m (= lf + lr)
    mass: float = 798.0  # kg
    yaw_inertia: float = 1200.0  # Iz, kg m^2
    cg_front_ratio: float = 0.54  # lf / wheelbase
    cornering_stiffness_front: float = 180000.0  # Cf, N/rad
    cornering_stiffness_rear: float = 200000.0  # Cr, N/rad
    max_steer: float = math.radians(18.0)  # rad at the wheels
    max_engine_force: float = 6000.0  # N
    max_brake_force: float = 16000.0  # N
    drag_coeff: float = 0.70  # N / (m/s)^2
    rolling_coeff: float = 0.015  # dimensionless
    downforce_coeff: float = 4.0  # N / (m/s)^2; adds to the friction-circle max force
    wear_rate: float = 0.0  # tire wear per (slip * speed * dt); 0 disables wear
    wear_slip_ref: float = 1.0  # divides the slip*speed wear load
    v_blend: float = 3.0  # m/s; below this blend toward the kinematic prediction
    v_eps: float = 0.5  # m/s; slip-angle singularity guard

    @property
    def lf(self) -> float:
        """CG-to-front-axle distance (m)."""
        return self.cg_front_ratio * self.wheelbase

    @property
    def lr(self) -> float:
        """CG-to-rear-axle distance (m)."""
        return (1.0 - self.cg_front_ratio) * self.wheelbase

    @classmethod
    def from_config(cls, cfg: Any) -> DynamicParams:
        """Build params from a ``physics`` config node. ``max_steer`` is given in degrees."""
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d: getattr(cfg, k, d))
        return cls(
            wheelbase=float(get("wheelbase", cls.wheelbase)),
            mass=float(get("mass", cls.mass)),
            yaw_inertia=float(get("yaw_inertia", cls.yaw_inertia)),
            cg_front_ratio=float(get("cg_front_ratio", cls.cg_front_ratio)),
            cornering_stiffness_front=float(
                get("cornering_stiffness_front", cls.cornering_stiffness_front)
            ),
            cornering_stiffness_rear=float(
                get("cornering_stiffness_rear", cls.cornering_stiffness_rear)
            ),
            max_steer=math.radians(float(get("max_steer_deg", math.degrees(cls.max_steer)))),
            max_engine_force=float(get("max_engine_force", cls.max_engine_force)),
            max_brake_force=float(get("max_brake_force", cls.max_brake_force)),
            drag_coeff=float(get("drag_coeff", cls.drag_coeff)),
            rolling_coeff=float(get("rolling_coeff", cls.rolling_coeff)),
            downforce_coeff=float(get("downforce_coeff", cls.downforce_coeff)),
            wear_rate=float(get("wear_rate", cls.wear_rate)),
            wear_slip_ref=float(get("wear_slip_ref", cls.wear_slip_ref)),
            v_blend=float(get("v_blend", cls.v_blend)),
            v_eps=float(get("v_eps", cls.v_eps)),
        )


def friction_circle_limit(grip: float, mass: float, vx: float, downforce_coeff: float) -> float:
    """Total combined-force limit ``grip * mass * g + downforce_coeff * vx**2`` (N).

    The aero term is an additive grip floor from downforce (plan §A). Pure and exact so the
    unit test can assert the realized combined tire force never exceeds this (per axle the
    limit is split by static normal-load fraction).
    """
    return grip * mass * G + downforce_coeff * vx * vx


def clamp_to_circle(fx: float, fy: float, f_max: float) -> tuple[float, float]:
    """Scale ``(fx, fy)`` down onto the circle of radius ``f_max`` if it lies outside it.

    Direction is preserved. ``hypot(out) <= f_max`` always holds (the load-bearing realism:
    a tire over its grip circle cannot produce more force, it slides).
    """
    if f_max <= 0.0:
        return 0.0, 0.0
    mag = math.hypot(fx, fy)
    if mag <= f_max or mag == 0.0:
        return fx, fy
    s = f_max / mag
    return fx * s, fy * s


class DynamicBicycle:
    """Dynamic bicycle model. Implements :class:`~f1rl.physics.base.PhysicsModel`."""

    def __init__(self, params: DynamicParams | None = None) -> None:
        self.params = params or DynamicParams()

    def step(
        self,
        state: CarState,
        steer: float,
        longitudinal: float,
        grip: float,
        dt: float,
    ) -> CarState:
        p = self.params
        steer = _clamp(steer, -1.0, 1.0)
        longitudinal = _clamp(longitudinal, -1.0, 1.0)
        delta = steer * p.max_steer

        vx, vy, r = state.vx, state.vy, state.yaw_rate

        f = self._forces(vx, vy, r, delta, longitudinal, grip)
        fx_total = f["fx_f"] + f["fx_r"]
        fyf, fyr = f["fyf"], f["fyr"]

        # Equations of motion in the body frame (centripetal coupling via vy*r, vx*r).
        ax = (fx_total - fyf * math.sin(delta)) / p.mass + vy * r
        ay = (fyf * math.cos(delta) + fyr) / p.mass - vx * r
        r_dot = (p.lf * fyf * math.cos(delta) - p.lr * fyr) / p.yaw_inertia

        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        r_new = r + r_dot * dt

        # No reverse gear this phase: clamp the car out of backward motion.
        if vx_new < 0.0:
            vx_new = 0.0

        # Low-speed blend toward the no-slip kinematic prediction (Kong et al. 2015).
        if vx_new < p.v_blend:
            w = vx_new / p.v_blend if p.v_blend > 0.0 else 1.0
            r_kin = vx_new * math.tan(delta) / p.wheelbase if p.wheelbase > 0.0 else 0.0
            r_new = w * r_new + (1.0 - w) * r_kin
            vy_new = w * vy_new  # kinematic lateral velocity is 0

        yaw_new = _wrap_pi(state.yaw + r_new * dt)
        x_new = state.x + (vx_new * math.cos(yaw_new) - vy_new * math.sin(yaw_new)) * dt
        y_new = state.y + (vx_new * math.sin(yaw_new) + vy_new * math.cos(yaw_new)) * dt

        # Wear advances with combined slip and speed; wear_rate = 0 disables it.
        slip_load = (abs(f["alpha_f"]) + abs(f["alpha_r"])) * max(vx, 0.0)
        ref = p.wear_slip_ref if p.wear_slip_ref > 0.0 else 1.0
        wear_new = _clamp(state.tire_wear + p.wear_rate * (slip_load / ref) * dt, 0.0, 1.0)

        return CarState(
            x=x_new,
            y=y_new,
            yaw=yaw_new,
            vx=vx_new,
            vy=vy_new,
            yaw_rate=r_new,
            tire_wear=wear_new,
            compound=state.compound,
        )

    def tire_forces(
        self, state: CarState, steer: float, longitudinal: float, grip: float
    ) -> dict[str, float]:
        """Expose the post-friction-circle tire forces and limits (for tests/debug, pure).

        Returns the per-axle longitudinal/lateral forces after the friction-circle reduction,
        the realized combined force, and ``f_max`` = :func:`friction_circle_limit`. The unit
        test asserts ``combined <= f_max`` (the friction circle holds).
        """
        steer = _clamp(steer, -1.0, 1.0)
        longitudinal = _clamp(longitudinal, -1.0, 1.0)
        delta = steer * self.params.max_steer
        f = self._forces(state.vx, state.vy, state.yaw_rate, delta, longitudinal, grip)
        fx_total = f["fx_f"] + f["fx_r"]
        fy_total = f["fyf"] + f["fyr"]
        f["fx_total"] = fx_total
        f["fy_total"] = fy_total
        f["combined"] = math.hypot(fx_total, fy_total)
        f["f_max"] = friction_circle_limit(
            grip, self.params.mass, state.vx, self.params.downforce_coeff
        )
        return f

    # ----- internals --------------------------------------------------------------------

    def _forces(
        self, vx: float, vy: float, r: float, delta: float, longitudinal: float, grip: float
    ) -> dict[str, float]:
        """Per-axle longitudinal/lateral tire forces after the friction-circle reduction."""
        p = self.params

        # Longitudinal demand (same force model as the kinematic step).
        if longitudinal >= 0.0:
            f_drive = longitudinal * p.max_engine_force
        else:
            f_drive = longitudinal * p.max_brake_force
        f_drag = p.drag_coeff * vx * abs(vx)
        f_roll = p.rolling_coeff * p.mass * G * _sign(vx)
        fx_cmd = f_drive - f_drag - f_roll

        # Slip angles, guarded against the vx -> 0 singularity.
        vx_safe = max(vx, p.v_eps)
        alpha_f = math.atan2(vy + p.lf * r, vx_safe) - delta
        alpha_r = math.atan2(vy - p.lr * r, vx_safe)

        # Linear lateral tire forces.
        fyf = -p.cornering_stiffness_front * alpha_f
        fyr = -p.cornering_stiffness_rear * alpha_r

        # Friction circle: split the total limit and the longitudinal demand by static
        # normal-load fraction, then cap each axle's combined force.
        f_max = friction_circle_limit(grip, p.mass, vx, p.downforce_coeff)
        frac_f = p.lr / p.wheelbase if p.wheelbase > 0.0 else 0.5
        frac_r = 1.0 - frac_f
        fx_f, fyf = clamp_to_circle(fx_cmd * frac_f, fyf, f_max * frac_f)
        fx_r, fyr = clamp_to_circle(fx_cmd * frac_r, fyr, f_max * frac_r)

        return {
            "fx_f": fx_f,
            "fx_r": fx_r,
            "fyf": fyf,
            "fyr": fyr,
            "alpha_f": alpha_f,
            "alpha_r": alpha_r,
        }


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _sign(v: float) -> float:
    return 1.0 if v > 0.0 else -1.0 if v < 0.0 else 0.0


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))
