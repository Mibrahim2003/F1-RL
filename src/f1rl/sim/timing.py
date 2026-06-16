"""Lap timing and delta-to-pole on a closed circuit.

Progress is tracked as arc length along the centerline. A lap completes when progress
wraps forward across the start/finish line (sample 0). Delta-to-pole compares the current
lap's elapsed time against the pole reference scaled by how far around the lap the car is,
so a car exactly on the pole pace reads ~0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from f1rl.track.schema import Track


@dataclass
class Timing:
    """A timing snapshot for one control step."""

    lap: int  # current lap number shown to the user (1-based, capped elsewhere)
    completed_laps: int  # laps fully finished
    lap_time: float  # elapsed time in the current lap, seconds
    last_lap: float | None  # last completed lap time, seconds
    best_lap: float | None  # best completed lap time, seconds
    delta_to_pole: float  # current pace vs pole reference, seconds (+slower / -faster)
    progress: float  # fraction of the lap completed, 0..1

    def as_dict(self) -> dict:
        return asdict(self)


class LapTimer:
    """Tracks laps, best/last lap, and delta-to-pole from car position over time."""

    def __init__(self, track: Track, pole_time_s: float) -> None:
        self.track = track
        self.length = track.length
        self.pole = float(pole_time_s)
        self.reset()

    def reset(self) -> None:
        self.completed_laps = 0
        self.lap_start_t = 0.0
        self.best_lap: float | None = None
        self.last_lap: float | None = None
        self._prev_s: float | None = None

    def _progress_s(self, x: float, y: float) -> float:
        return float(self.track.s[self.track.nearest_index(x, y)])

    def update(self, x: float, y: float, t: float) -> Timing:
        """Advance timing to time ``t`` given the car at ``(x, y)``."""
        s_car = self._progress_s(x, y)
        L = self.length

        if self._prev_s is not None and self._prev_s > 0.7 * L and s_car < 0.3 * L:
            lap_time = t - self.lap_start_t
            self.last_lap = lap_time
            self.best_lap = lap_time if self.best_lap is None else min(self.best_lap, lap_time)
            self.completed_laps += 1
            self.lap_start_t = t
        self._prev_s = s_car

        elapsed = t - self.lap_start_t
        progress = s_car / L if L > 0 else 0.0
        delta = elapsed - self.pole * progress
        return Timing(
            lap=self.completed_laps + 1,
            completed_laps=self.completed_laps,
            lap_time=elapsed,
            last_lap=self.last_lap,
            best_lap=self.best_lap,
            delta_to_pole=delta,
            progress=progress,
        )
