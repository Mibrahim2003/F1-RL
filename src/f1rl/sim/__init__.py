"""Simulation orchestration for the live app: fixed-step loop, lap timing, recording,
and a centerline autopilot for the watch-live mode. None of this touches rendering."""

from f1rl.sim.autopilot import CenterlineAutopilot
from f1rl.sim.loop import SimConfig, SimLoop
from f1rl.sim.recorder import TrajectoryError, TrajectoryRecorder, load_trajectory
from f1rl.sim.timing import LapTimer

__all__ = [
    "SimConfig",
    "SimLoop",
    "LapTimer",
    "TrajectoryRecorder",
    "TrajectoryError",
    "load_trajectory",
    "CenterlineAutopilot",
]
