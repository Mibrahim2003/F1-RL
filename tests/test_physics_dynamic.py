"""Dynamic bicycle model contract (spec §5/§b, plan "Dynamic bicycle model").

Built from the public contract: ``DynamicBicycle.step(state, steer, longitudinal, grip, dt)``
returns a new ``CarState``; the **friction circle** caps combined tire force at
``grip*m*g (+ aero)``; turning produces yaw rate / lateral velocity; a straight line stays a
straight line; the model is **stable at vx≈0** (no NaN); higher grip allows tighter/faster
cornering. No implementation internals are mirrored — only the documented behavior.
"""

from __future__ import annotations

import math

import pytest

from f1rl.physics import CarState, DynamicBicycle, make_physics
from f1rl.physics.dynamic import (
    DynamicParams,
    clamp_to_circle,
    friction_circle_limit,
)
from f1rl.utils.config import load_config


def _run(model, state, steer, longitudinal, n, grip=1.5, dt=0.01):
    for _ in range(n):
        state = model.step(state, steer, longitudinal, grip, dt)
    return state


# --- friction circle (the load-bearing realism) -----------------------------------------


def test_clamp_to_circle_never_exceeds_radius():
    for fx, fy, fmax in [(1000, 0, 500), (300, 400, 1000), (5000, 5000, 2000), (0, 0, 100)]:
        ox, oy = clamp_to_circle(fx, fy, fmax)
        assert math.hypot(ox, oy) <= fmax + 1e-6


def test_clamp_to_circle_preserves_direction_and_inside_unchanged():
    # Inside the circle: unchanged. Outside: same direction, scaled onto the boundary.
    assert clamp_to_circle(100, 0, 500) == (100, 0)
    ox, oy = clamp_to_circle(1000, 0, 500)
    assert ox == pytest.approx(500) and oy == pytest.approx(0)


def test_friction_limit_grows_with_grip_and_aero():
    base = friction_circle_limit(1.0, 798.0, 0.0, 0.0)
    more_grip = friction_circle_limit(2.0, 798.0, 0.0, 0.0)
    with_aero = friction_circle_limit(1.0, 798.0, 50.0, 5.0)
    assert more_grip > base
    assert with_aero > base
    assert base == pytest.approx(1.0 * 798.0 * 9.81)


def test_realized_combined_force_within_friction_circle():
    # Hard combined cornering + throttle must never exceed the friction-circle limit.
    model = DynamicBicycle()
    for grip in (0.8, 1.2, 2.0):
        for vx in (20.0, 45.0, 70.0):
            st = CarState(vx=vx, vy=2.0, yaw_rate=0.3)
            f = model.tire_forces(st, steer=1.0, longitudinal=1.0, grip=grip)
            assert f["combined"] <= f["f_max"] + 1e-3


# --- turning vs straight-line -----------------------------------------------------------


def test_turning_produces_yaw_rate_and_lateral_velocity():
    model = DynamicBicycle()
    end = _run(model, CarState(vx=40.0), steer=0.5, longitudinal=0.2, n=20)
    assert abs(end.yaw_rate) > 1e-3
    assert abs(end.vy) > 1e-3
    assert end.yaw_rate > 0.0  # left steer -> positive yaw


def test_straight_line_stays_straight_and_accelerates():
    model = DynamicBicycle()
    end = _run(model, CarState(vx=10.0), steer=0.0, longitudinal=1.0, n=100)
    assert end.vx > 10.0
    assert end.vy == pytest.approx(0.0, abs=1e-9)
    assert end.yaw_rate == pytest.approx(0.0, abs=1e-9)
    assert end.y == pytest.approx(0.0, abs=1e-6)


def test_straight_line_matches_kinematic_longitudinal():
    # With no steer, the dynamic model's longitudinal motion ~ the kinematic model's (same
    # force model). Compare end speed after identical commands.
    from f1rl.physics import KinematicBicycle, KinematicParams

    shared = dict(mass=798.0, max_engine_force=9000.0, drag_coeff=0.7, rolling_coeff=0.015)
    dyn = DynamicBicycle(DynamicParams(**shared))
    kin = KinematicBicycle(KinematicParams(**shared))
    s_dyn = _run(dyn, CarState(vx=5.0), 0.0, 1.0, n=300)
    s_kin = CarState(vx=5.0)
    for _ in range(300):
        s_kin = kin.step(s_kin, 0.0, 1.0, 1.0, 0.01)
    assert s_dyn.vx == pytest.approx(s_kin.vx, rel=1e-3)


# --- low-speed stability (the #1 dynamic-model bug) -------------------------------------


def test_stable_at_standstill_no_nan():
    model = DynamicBicycle()
    s = CarState()  # vx = 0
    for _ in range(200):
        s = model.step(s, steer=1.0, longitudinal=0.0, grip=1.5, dt=0.01)
        for v in (s.x, s.y, s.yaw, s.vx, s.vy, s.yaw_rate):
            assert math.isfinite(v)
    assert s.vx >= 0.0  # never rolls backward


def test_stable_at_pit_crawl_speed():
    model = DynamicBicycle()
    s = CarState(vx=1.0)
    for _ in range(500):
        s = model.step(s, steer=1.0, longitudinal=0.1, grip=1.5, dt=0.01)
        assert math.isfinite(s.vy) and math.isfinite(s.yaw_rate)
    assert abs(s.vy) < 50.0  # no blow-up


# --- grip affects cornering -------------------------------------------------------------


def test_higher_grip_allows_tighter_cornering():
    # At a fixed speed and full steer, low grip saturates the front tire's friction circle and
    # the car washes out (low yaw rate); higher grip sustains more lateral force, so it corners
    # tighter (higher yaw rate) and reaches a higher sustained lateral acceleration ~ grip*g.
    def peak_yaw(grip):
        model = DynamicBicycle()
        s = CarState(vx=55.0)
        peak = 0.0
        for _ in range(15):
            s = model.step(s, steer=1.0, longitudinal=0.5, grip=grip, dt=0.01)
            peak = max(peak, abs(s.yaw_rate))
        return peak

    assert peak_yaw(0.5) < peak_yaw(1.2) < peak_yaw(2.5)


# --- factory + wear ---------------------------------------------------------------------


def test_make_physics_dynamic_returns_dynamic_model():
    cfg = load_config("experiment/rbr_dynamic")
    model = make_physics(cfg)
    assert isinstance(model, DynamicBicycle)


def test_wear_advances_only_when_rate_positive_and_slipping():
    no_wear = DynamicBicycle(DynamicParams(wear_rate=0.0))
    s = _run(no_wear, CarState(vx=40.0), steer=0.6, longitudinal=0.2, n=50)
    assert s.tire_wear == pytest.approx(0.0)

    worn = DynamicBicycle(DynamicParams(wear_rate=0.01))
    s2 = _run(worn, CarState(vx=40.0), steer=0.6, longitudinal=0.2, n=50)
    assert s2.tire_wear > 0.0
    assert s2.tire_wear <= 1.0
