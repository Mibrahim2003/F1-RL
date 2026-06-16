"""Kinematic bicycle model: straight line, constant-radius turn, determinism."""

from __future__ import annotations

import math

import pytest

from f1rl.physics import CarState, KinematicBicycle
from f1rl.physics.kinematic import KinematicParams


def _run(model: KinematicBicycle, state: CarState, steer, longitudinal, n, dt=0.01):
    for _ in range(n):
        state = model.step(state, steer, longitudinal, grip=1.0, dt=dt)
    return state


def test_straight_line_accelerates_along_x():
    model = KinematicBicycle()
    end = _run(model, CarState(), steer=0.0, longitudinal=1.0, n=200)
    assert end.x > 0.0
    assert end.vx > 0.0
    assert end.y == pytest.approx(0.0, abs=1e-9)
    assert end.yaw == pytest.approx(0.0, abs=1e-9)


def test_top_speed_is_bounded_by_drag():
    # At full throttle the car approaches a finite top speed (engine == drag + rolling).
    model = KinematicBicycle()
    end = _run(model, CarState(), steer=0.0, longitudinal=1.0, n=20000)
    p = model.params
    expected_top = math.sqrt(p.max_engine_force / p.drag_coeff)
    assert end.vx == pytest.approx(expected_top, rel=0.05)


def test_braking_stops_without_reversing():
    model = KinematicBicycle()
    moving = CarState(vx=40.0)
    end = _run(model, moving, steer=0.0, longitudinal=-1.0, n=1000)
    assert end.vx == pytest.approx(0.0, abs=1e-6)
    assert end.x >= moving.x  # moved forward while stopping, never backward


def test_constant_radius_turn_matches_bicycle_geometry():
    # Single step from a known speed: yaw_rate should equal v * tan(delta) / L.
    params = KinematicParams()
    model = KinematicBicycle(params)
    v0 = 30.0
    steer = 0.5
    state = model.step(CarState(vx=v0), steer=steer, longitudinal=0.0, grip=1.0, dt=0.01)
    delta = steer * params.max_steer
    expected_yaw_rate = state.vx * math.tan(delta) / params.wheelbase
    assert state.yaw_rate == pytest.approx(expected_yaw_rate, rel=1e-6)
    assert state.yaw > 0.0  # left turn => positive yaw


def test_circle_radius_is_consistent():
    # Driving with fixed steer at ~steady speed traces a circle of radius L / tan(delta).
    params = KinematicParams()
    model = KinematicBicycle(params)
    state = CarState(vx=25.0)
    steer = 0.6
    xs, ys = [], []
    for _ in range(2000):
        # Light throttle to hold speed against drag.
        state = model.step(state, steer=steer, longitudinal=0.15, grip=1.0, dt=0.01)
        xs.append(state.x)
        ys.append(state.y)
    delta = steer * params.max_steer
    expected_r = params.wheelbase / math.tan(delta)
    # Fit radius from the spread of the traced path (diameter ~ max extent in x or y).
    diameter = max(max(xs) - min(xs), max(ys) - min(ys))
    assert diameter / 2.0 == pytest.approx(expected_r, rel=0.1)


def test_determinism():
    model = KinematicBicycle()
    a = _run(model, CarState(), steer=0.3, longitudinal=0.8, n=500)
    b = _run(model, CarState(), steer=0.3, longitudinal=0.8, n=500)
    assert (a.x, a.y, a.yaw, a.vx) == (b.x, b.y, b.yaw, b.vx)
