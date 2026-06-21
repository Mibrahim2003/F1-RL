"""Multi-agent racing environment — a FIELD of N cars on one shared circuit (Phase 5).

``RacingParallelEnv`` is the PettingZoo :class:`~pettingzoo.ParallelEnv` analogue of
:class:`~f1rl.env.single_agent.RacingEnv`: it holds **N homogeneous cars on one circuit**,
steps them all at once, and returns per-agent ``(obs, reward, terminated, truncated, info)``
dicts. It reuses the single-agent core verbatim — :func:`~f1rl.env.single_agent.step_one_car`
and :func:`~f1rl.env.single_agent.reset_car` — once per car, so the physics, ObservationV2
(length 22, ``OBS_VERSION = 2``), reward, grip pipeline, lap timing, and termination are all
**unchanged**. No car observes another and no collision is computed — that is Phase 6.

What is per-car vs per-circuit is load-bearing (the Phase 5 trap):

- **Per car** (a :class:`~f1rl.env.single_agent.CarRuntime` each): ``CarState``, its **own**
  ``LapTimer`` instance (``LapTimer(track, pole)``, *never* the pooled
  ``CircuitEntry.lap_timer``), and the projection state ``prev_s``/``grip_idx``/``grip_lat``/
  ``wrong_way_count``.
- **Per circuit** (the pool entry, read-only, shared by the whole field): ``Track``,
  ``EdgeCache``, and the resolved ``pole_time_s``.

**Constant SuperSuit-visible width.** The raw env follows standard PettingZoo: a terminated or
truncated car is removed from ``self.agents`` on the next step (so it passes
``parallel_api_test``). The *constant* agent width the SuperSuit vectorizer requires comes from
the ``black_death_v3`` wrapper applied in the training stack (it re-pads removed agents to zero
obs/reward), **not** from the raw env. One car finishing or failing never ends the others; the
episode ends when ``self.agents`` is empty (all cars done) or the per-car step limit fires.

Field size (``n_agents``) is a **per-run constant** set at construction — it is the vector-env
width and is grown by warm-starting successive runs, never by the in-place curriculum (which
stays a pool-widening / conditions knob). See ``.claude/specs/phase-5-many-cars.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import gymnasium.utils.seeding as gym_seeding
import numpy as np
from pettingzoo import ParallelEnv

from f1rl.env.observations import observation_space
from f1rl.env.pool import CircuitPool, pool_ids_from_config
from f1rl.env.single_agent import (
    CarStepConfig,
    apply_conditions,
    reset_car,
    resolve_weather,
    step_one_car,
)
from f1rl.sim.timing import LapTimer

_DEFAULT_TEAM_COLORS = ("#e10600", "#00d2be", "#0600ef", "#ff8700", "#006f62")


@dataclass(frozen=True)
class GridParams:
    """Field layout config (the ``grid:`` block): size, reset mode, spacing, team colors."""

    n_agents: int = 1
    reset_mode: str = "scattered"  # "scattered" (train) | "grid" (eval/demo)
    grid_spacing_m: float = 12.0
    grid_lateral_m: float = 3.0
    team_colors: tuple[str, ...] = _DEFAULT_TEAM_COLORS

    @classmethod
    def from_config(cls, cfg: Any, n_agents: int | None = None) -> GridParams:
        node = getattr(cfg, "grid", None)
        get = (
            node.get
            if node is not None and hasattr(node, "get")
            else (lambda k, d: getattr(node, k, d) if node is not None else d)
        )
        n = int(n_agents if n_agents is not None else get("n_agents", cls.n_agents))
        colors = get("team_colors", None)
        team_colors = tuple(str(c) for c in colors) if colors else _DEFAULT_TEAM_COLORS
        return cls(
            n_agents=n,
            reset_mode=str(get("reset_mode", cls.reset_mode)),
            grid_spacing_m=float(get("grid_spacing_m", cls.grid_spacing_m)),
            grid_lateral_m=float(get("grid_lateral_m", cls.grid_lateral_m)),
            team_colors=team_colors,
        )


class RacingParallelEnv(ParallelEnv):
    """N homogeneous cars on one shared circuit; one shared policy drives them all."""

    metadata = {"name": "racing_parallel_v0", "is_parallelizable": True}

    def __init__(self, cfg: Any, n_agents: int | None = None, seed: int | None = None) -> None:
        """Build the field env from a root config node.

        Args:
            cfg: Root config carrying ``sim``/``physics``/``obs``/``reward``/``env``/``grid``
                blocks, ``track_id``/``track``, and ``circuits`` (the pool).
            n_agents: Field size override; falls back to ``cfg.grid.n_agents``. Must be >= 1.
            seed: Base seed for the env RNG (the circuit draw + start placement).
        """
        self.cfg = cfg
        self.grid = GridParams.from_config(cfg, n_agents)
        if self.grid.n_agents < 1:
            raise ValueError(f"grid.n_agents must be >= 1, got {self.grid.n_agents}")
        self.render_mode = None

        track_id = _cfg_get(cfg, "track_id", "oval")
        self.np_random, _ = gym_seeding.np_random(seed)

        # Phase 4 pool, built once (read-only Track/EdgeCache/pole per entry). The field shares
        # one drawn entry per episode; lap timers are per car (built fresh each reset).
        pool_ids = pool_ids_from_config(cfg, track_id)
        self.pool = CircuitPool(pool_ids, cfg)
        initial = track_id if track_id in self.pool else self.pool.ids[0]
        self._entry = self.pool.entries[initial]

        # The shared, read-only per-car step config (physics/sim/limits/obs/reward/conditions).
        self._car_cfg = CarStepConfig.from_config(cfg)
        self.conditions = self._car_cfg.conditions
        self._weather_mode = str(self.conditions.weather)

        self.possible_agents = [f"car_{i}" for i in range(self.grid.n_agents)]
        self.agents = list(self.possible_agents)
        self._cars: dict[str, Any] = {}

        # Homogeneous spaces, identical for every agent (the unchanged ObservationV2 + action).
        self._obs_space = observation_space()
        self._act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Provided as dicts too, for tooling that reads the attributes directly.
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}

    # ----- PettingZoo ParallelEnv API ---------------------------------------------------

    def observation_space(self, agent: str) -> gym.spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> gym.spaces.Space:
        return self._act_space

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None:
            self.np_random, _ = gym_seeding.np_random(seed)
        self.agents = list(self.possible_agents)

        # One circuit for the whole field, drawn from the env RNG (reproducible from the seed).
        cid = self.pool.sample(self.np_random)
        self._entry = self.pool.entries[cid]
        resolve_weather(self._car_cfg, self._weather_mode, self.np_random)

        slots = self._place_slots(options)
        self._cars = {}
        obs: dict[str, np.ndarray] = {}
        infos: dict[str, dict[str, Any]] = {}
        for i, agent in enumerate(self.possible_agents):
            idx, lateral = slots[i]
            # Per-car OWN lap timer (cheap), bound to the shared circuit + its pole.
            lap_timer = LapTimer(self._entry.track, self._entry.pole_time_s)
            car, ob, timing = reset_car(
                self._entry, self._car_cfg, idx, lap_timer, lateral_m=lateral
            )
            self._cars[agent] = car
            obs[agent] = ob
            infos[agent] = {
                "lap_time": timing.lap_time,
                "off_track": 0.0,
                "progress": timing.progress,
                "completed_laps": timing.completed_laps,
                "start_index": int(idx),
                "circuit_id": self._entry.track_id,
                "pole_time_s": self._entry.pole_time_s,
                "team": i % len(self.grid.team_colors),
            }
        return obs, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        obs: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        for agent in self.agents:
            car = self._cars[agent]
            ob, rew, terminated, truncated, info = step_one_car(
                self._entry, self._car_cfg, car, actions[agent]
            )
            info["team"] = self.possible_agents.index(agent) % len(self.grid.team_colors)
            obs[agent] = ob
            rewards[agent] = rew
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = info
            if terminated or truncated:
                car.done = True

        # Standard PettingZoo: drop done agents from the live set for the NEXT step. The
        # SuperSuit `black_death_v3` wrapper re-pads them so the vectorizer width stays constant.
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        return obs, rewards, terminations, truncations, infos

    def render(self) -> None:  # rendering is offline/web-only (never inside the env)
        return None

    def close(self) -> None:
        return None

    def reseed(self, seed: int | None) -> None:
        """Reseed the env RNG (circuit draw + start placement); used to decorrelate copies."""
        self.np_random, _ = gym_seeding.np_random(seed)

    # ----- curriculum hooks (broadcast onto the raw env; see train/selfplay.py) ---------

    def set_track_pool(self, circuits: list[str] | str | None = None) -> None:
        """Curriculum hook: narrow/widen the active circuit draw set (sampling-side, no retrain)."""
        self.pool.set_active(circuits)

    def apply_conditions(
        self,
        *,
        mu_base: float | None = None,
        wear_rate: float | None = None,
        weather: str | None = None,
    ) -> None:
        """Curriculum hook: push grip/wear/weather overrides into the shared conditions."""
        apply_conditions(self._car_cfg, mu_base=mu_base, wear_rate=wear_rate, weather=weather)
        if weather is not None:
            self._weather_mode = str(weather)

    # ----- placement --------------------------------------------------------------------

    def _place_slots(self, options: dict[str, Any] | None) -> list[tuple[int, float]]:
        """Return ``(centerline_idx, lateral_m)`` per agent for the configured reset mode.

        ``scattered`` (train): distinct seeded centerline indices, no lateral offset — the field
        smeared around the whole lap for state-space coverage. ``grid`` (eval/demo): distinct,
        non-overlapping two-column slots queued back from the start/finish line.
        """
        track = self._entry.track
        n_points = len(track.centerline)
        n = self.grid.n_agents

        if options is not None and "start_indices" in options:
            idxs = [int(j) % n_points for j in options["start_indices"]][:n]
            return [(j, 0.0) for j in idxs]

        if self.grid.reset_mode == "grid":
            return self._grid_slots(n_points, n)

        # scattered: distinct indices without replacement (no two cars share a start sample).
        size = min(n, n_points)
        drawn = self.np_random.choice(n_points, size=size, replace=False)
        idxs = [int(j) for j in drawn]
        while len(idxs) < n:  # only if n_agents > n_points (degenerate tiny track)
            idxs.append(int(self.np_random.integers(0, n_points)))
        return [(j, 0.0) for j in idxs]

    def _grid_slots(self, n_points: int, n: int) -> list[tuple[int, float]]:
        """Two-column starting grid just past the S/F line; front row = furthest along.

        Rows are laid out forward from the line (``target_s`` decreasing down the grid) so no
        car straddles the seam at ``s ~= length`` — which keeps the track-position ordering
        (and any gap derived from it) monotonic from the front of the grid back.
        """
        track = self._entry.track
        s = np.asarray(track.s, dtype=np.float64)
        half = self.grid.grid_lateral_m * 0.5
        n_rows = (n + 1) // 2
        slots: list[tuple[int, float]] = []
        for i in range(n):
            row = i // 2
            col = i % 2
            target_s = (n_rows - row) * self.grid.grid_spacing_m
            idx = int(np.argmin(np.abs(s - target_s)))
            lateral = half if col == 0 else -half
            # Keep the slot on the asphalt: clamp to just inside the half-width on that side.
            hw = float(track.half_width_left[idx] if lateral >= 0 else track.half_width_right[idx])
            lateral = float(np.clip(lateral, -(hw - 0.5), hw - 0.5))
            slots.append((idx, lateral))
        return slots


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
