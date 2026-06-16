"""Lap timing: synthetic crossing counts, and an end-to-end loop+autopilot lap."""

from __future__ import annotations

import pytest

from f1rl.physics import KinematicBicycle
from f1rl.physics.kinematic import KinematicParams
from f1rl.sim.autopilot import CenterlineAutopilot
from f1rl.sim.loop import SimConfig, SimLoop
from f1rl.sim.timing import LapTimer
from f1rl.track.oval import build_oval


def test_lap_detection_fires_once_per_crossing():
    track = build_oval()
    timer = LapTimer(track, pole_time_s=52.0)
    n = len(track.centerline)
    dt = 0.05
    t = 0.0
    timing = None
    # Drive exactly along the centerline for three full loops.
    for _ in range(3):
        for i in range(n):
            p = track.centerline[i]
            t += dt
            timing = timer.update(float(p[0]), float(p[1]), t)
    # Three passes through the start line => two completed laps after the first.
    assert timing.completed_laps == 2
    # Each lap took exactly n steps of dt.
    assert timing.last_lap == pytest.approx(n * dt, rel=0.02)


def test_loop_with_autopilot_completes_laps():
    track = build_oval()
    params = KinematicParams()
    model = KinematicBicycle(params)
    loop = SimLoop(model, track, SimConfig(), pole_time_s=52.0, total_laps=30)
    ap = CenterlineAutopilot(track, max_steer_rad=params.max_steer)

    frame = None
    for _ in range(20 * 200):  # 200 s of sim at 20 Hz
        steer, lon = ap.control(loop.state)
        frame = loop.step(steer, lon)

    assert loop.timer.completed_laps >= 1
    assert loop.timer.best_lap is not None
    assert 20.0 < loop.timer.best_lap < 150.0
    # Delta-to-pole is a finite number and the frame carries the expected fields.
    assert isinstance(frame["telemetry"]["delta_to_pole"], float)
    assert frame["telemetry"]["lap"] >= 1
