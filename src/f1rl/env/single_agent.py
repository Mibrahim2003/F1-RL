"""Single-agent racing environment (TECHNICAL_DESIGN.md §10, plan §A).

``RacingEnv`` owns the car state, the physics stepper (built through
:func:`f1rl.physics.make_physics`, never a concrete model), the track, the conditions,
the reward, the termination logic, the lap tracker, and an optional trajectory recorder.
It runs its own fixed-step loop (``substeps`` physics substeps per control step) — it does
**not** import :class:`~f1rl.sim.loop.SimLoop`, so it stays rendering-free and
recorder-optional. It passes ``gymnasium.utils.env_checker.check_env``.

Config-driven throughout: every tunable (sim timing, physics, obs params, reward weights,
target laps, step limit, start randomization, termination thresholds) comes from config.
``step`` never renders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import gymnasium.utils.seeding as gym_seeding
import numpy as np

from f1rl.env.conditions import Conditions
from f1rl.env.observations import (
    ObsParams,
    build_observation,
    observation_space,
    track_query,
)
from f1rl.env.pool import CircuitPool, pool_ids_from_config
from f1rl.env.rewards import RewardWeights, reward_v1, reward_v2
from f1rl.physics import make_physics
from f1rl.physics.base import CarState
from f1rl.track.schema import Track

# Default env limits / termination thresholds (overridable from the ``env:`` config block).
_DEFAULT_TARGET_LAPS = 1
_DEFAULT_MAX_STEPS = 4000  # ~200 s at 20 Hz, generous for one lap of red_bull_ring
_DEFAULT_OFFTRACK_LIMIT_M = 12.0  # meters past the asphalt edge -> failure termination
_DEFAULT_WRONG_WAY_STEPS = 40  # consecutive negative-progress steps -> wrong-way failure
_DEFAULT_WRONG_WAY_DS_M = -0.05  # ds below this (m) counts as going backward
_DEFAULT_START_SPEED = 0.0  # m/s at reset
_DEFAULT_START_RANDOMIZE = True  # random centerline index at reset (start-state randomization)
_DEFAULT_FAILURE_REWARD = -10.0  # one-off penalty added on a failure termination


@dataclass(frozen=True)
class SimParams:
    """Loop timing for the env's own fixed-step stepping (mirrors ``sim/loop.SimConfig``)."""

    control_hz: int = 20
    substeps: int = 5
    dt_physics: float = 0.01

    @classmethod
    def from_config(cls, cfg: Any) -> SimParams:
        node = cfg.sim if hasattr(cfg, "sim") and cfg.sim is not None else cfg
        get = node.get if hasattr(node, "get") else (lambda k, d: getattr(node, k, d))
        return cls(
            control_hz=int(get("control_hz", cls.control_hz)),
            substeps=int(get("substeps", cls.substeps)),
            dt_physics=float(get("dt_physics", cls.dt_physics)),
        )

    @property
    def dt_control(self) -> float:
        return 1.0 / self.control_hz


@dataclass(frozen=True)
class EnvLimits:
    """Termination / truncation limits, all from config."""

    target_laps: int = _DEFAULT_TARGET_LAPS
    max_steps: int = _DEFAULT_MAX_STEPS
    offtrack_limit_m: float = _DEFAULT_OFFTRACK_LIMIT_M
    wrong_way_steps: int = _DEFAULT_WRONG_WAY_STEPS
    wrong_way_ds_m: float = _DEFAULT_WRONG_WAY_DS_M
    start_speed: float = _DEFAULT_START_SPEED
    start_randomize: bool = _DEFAULT_START_RANDOMIZE
    failure_reward: float = _DEFAULT_FAILURE_REWARD

    @classmethod
    def from_config(cls, cfg: Any) -> EnvLimits:
        node = cfg.env if hasattr(cfg, "env") and cfg.env is not None else cfg
        get = node.get if hasattr(node, "get") else (lambda k, d: getattr(node, k, d))
        # target_laps falls back to the per-track total_laps when not set on env.
        target_laps = get("target_laps", None)
        if target_laps is None and hasattr(cfg, "track") and cfg.track is not None:
            tnode = cfg.track
            tget = tnode.get if hasattr(tnode, "get") else (lambda k, d: getattr(tnode, k, d))
            target_laps = tget("target_laps", cls.target_laps)
        if target_laps is None:
            target_laps = cls.target_laps
        return cls(
            target_laps=int(target_laps),
            max_steps=int(get("max_steps", cls.max_steps)),
            offtrack_limit_m=float(get("offtrack_limit_m", cls.offtrack_limit_m)),
            wrong_way_steps=int(get("wrong_way_steps", cls.wrong_way_steps)),
            wrong_way_ds_m=float(get("wrong_way_ds_m", cls.wrong_way_ds_m)),
            start_speed=float(get("start_speed", cls.start_speed)),
            start_randomize=bool(get("start_randomize", cls.start_randomize)),
            failure_reward=float(get("failure_reward", cls.failure_reward)),
        )


class RacingEnv(gym.Env):
    """One car learning one circuit on the configured physics. Passes the env checker."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        *,
        track: Track | None = None,
        seed: int | None = None,
        recorder: Any | None = None,
    ) -> None:
        """Build the env from a root config node.

        Args:
            cfg: Root config (OmegaConf/mapping) carrying ``sim``, ``physics``, ``obs``,
                ``reward``, ``env`` blocks and ``track_id``/``track``.
            track: Optional pre-loaded :class:`Track`; loaded from config when omitted.
            seed: Optional base seed for the env RNG (also accepted via ``reset(seed=...)``).
            recorder: Optional trajectory recorder (eval only); appended per step when set.
        """
        super().__init__()
        self.cfg = cfg

        track_id = _cfg_get(cfg, "track_id", "oval")

        # RNG seeded by reset(seed=...) (Gymnasium contract); seed it now for the pool draw.
        self.np_random, _ = gym_seeding.np_random(seed)

        # Phase 4: build the circuit pool once (per-id Track + EdgeCache + LapTimer + pole). An
        # empty/absent circuits.pool falls back to [track_id] -> the Phase 3b single-circuit
        # behavior. A preloaded `track` (legacy single-track path) seeds that id's entry.
        pool_ids = pool_ids_from_config(cfg, track_id)
        preloaded = {track_id: track} if track is not None else None
        self.pool = CircuitPool(pool_ids, cfg, preloaded=preloaded)

        # Pin one circuit per worker (fallback) draws once now; else draw fresh each reset.
        self._pinned_id: str | None = (
            self.pool.sample(self.np_random) if self.pool.pin_per_worker else None
        )
        # Bind an initial active circuit (the configured track_id when in the pool, else the
        # first pool id) so attributes exist before the first reset.
        initial = track_id if track_id in self.pool else self.pool.ids[0]
        self._bind_circuit(initial)

        self.physics = make_physics(cfg)
        self.sim = SimParams.from_config(cfg)
        self.limits = EnvLimits.from_config(cfg)
        self.obs_params = ObsParams.from_config(cfg)
        self.reward_weights = RewardWeights.from_config(cfg)
        self.conditions = Conditions.from_config(cfg)

        # The grip pipeline gates the dynamic model; the kinematic model ignores grip, so the
        # constant fallback is fine there (no surface lookup overhead).
        self.use_pipeline = str(_cfg_get(getattr(cfg, "physics", None) or cfg, "model", "")) == (
            "dynamic"
        )
        self.start_compound = int(self.conditions.tires.start_compound)
        # Weather mode: a concrete condition ("dry"/"damp"/"wet") or "sampled" (the curriculum
        # sets this; reset() then draws wet with probability weather.p_wet).
        self._weather_mode = str(self.conditions.weather)

        self.recorder = recorder

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = observation_space()

        # Episode state, set in reset().
        self.state: CarState = CarState()
        self._t = 0.0
        self._step_count = 0
        self._prev_s = 0.0
        self._wrong_way_count = 0
        # Last centerline projection (idx, signed_lateral) — reused next step for the grip
        # pipeline (pre-step surface/wear), so each track_query is used once (no re-projection).
        self._grip_idx = 0
        self._grip_lat = 0.0

    # ----- gymnasium API ----------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)  # reseeds self.np_random when seed is not None

        # Phase 4: draw a circuit from the active pool and rebind the per-circuit state before
        # anything reads self.track. Pinned worker uses its fixed id; else a fresh draw each
        # episode (reproducible from the seed). One-circuit pool always returns the same id.
        cid = self._pinned_id if self._pinned_id is not None else self.pool.sample(self.np_random)
        self._bind_circuit(cid)

        self._resolve_weather()
        idx = self._sample_start_index(options)
        c = self.track.centerline[idx]
        tan = self.track.tangent[idx]
        yaw = math.atan2(float(tan[1]), float(tan[0]))
        self.state = CarState(
            x=float(c[0]),
            y=float(c[1]),
            yaw=yaw,
            vx=float(self.limits.start_speed),
            compound=self.start_compound,
        )

        self._t = 0.0
        self._step_count = 0
        self._wrong_way_count = 0
        self.lap_timer.reset()
        # Seed the lap timer's internal previous-s so it does not spuriously detect a lap.
        timing = self.lap_timer.update(self.state.x, self.state.y, self._t)
        self._prev_s = float(self.track.s[idx])
        self._grip_idx = int(idx)
        self._grip_lat = 0.0  # starts on the centerline

        obs = self._build_obs()
        info = {
            "lap_time": timing.lap_time,
            "off_track": 0.0,
            "progress": timing.progress,
            "completed_laps": timing.completed_laps,
            "start_index": int(idx),
            "circuit_id": self.track_id,
            "pole_time_s": self._pole,
        }
        if self.recorder is not None:
            self._record(timing)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        steer, longitudinal = self._map_action(action)

        # Grip for this control step from the pre-step surface/wear/weather (plan §6). Reuses
        # the projection stored from the previous step / reset — no extra track_query.
        grip = self._step_grip()
        for _ in range(self.sim.substeps):
            self.state = self.physics.step(
                self.state, steer, longitudinal, grip, self.sim.dt_physics
            )
        self._t += self.sim.dt_control
        self._step_count += 1

        # Recompute the projection once, shared by reward + off-track + obs build + next grip.
        idx, s_along, signed_lateral, half_width, _heading = track_query(
            self.track, self.state.x, self.state.y, self.state.yaw
        )
        off_track_m = max(0.0, abs(signed_lateral) - half_width)
        self._grip_idx = int(idx)
        self._grip_lat = float(signed_lateral)

        timing = self.lap_timer.update(self.state.x, self.state.y, self._t)

        if self.reward_weights.version >= 2:
            slip = abs(self.state.vy) / max(abs(self.state.vx), 1.0)
            reward, terms = reward_v2(
                self._prev_s, s_along, off_track_m, self.track.length, self.reward_weights, slip
            )
        else:
            reward, terms = reward_v1(
                self._prev_s, s_along, off_track_m, self.track.length, self.reward_weights
            )
        self._prev_s = s_along

        # Wrong-way tracking: sustained backward progress.
        if terms["ds"] < self.limits.wrong_way_ds_m:
            self._wrong_way_count += 1
        else:
            self._wrong_way_count = 0

        terminated, truncated, term_reason = self._check_termination(timing, off_track_m)
        if terminated and term_reason in ("offtrack", "wrong_way"):
            reward += self.limits.failure_reward
            terms["failure"] = self.limits.failure_reward
            terms["total"] = reward

        obs = self._build_obs()
        info = {
            "lap_time": timing.lap_time,
            "off_track": off_track_m,
            "progress": timing.progress,
            "completed_laps": timing.completed_laps,
            "last_lap": timing.last_lap,
            "best_lap": timing.best_lap,
            "reward_terms": terms,
            "termination": term_reason,
            "steps": self._step_count,
            "grip": grip,
            "tire_wear": self.state.tire_wear,
            "compound": self.state.compound,
            "weather": self.conditions.weather,
            "circuit_id": self.track_id,
            "pole_time_s": self._pole,
        }
        if self.recorder is not None:
            self._record(timing)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ----- circuit pool (Phase 4) -------------------------------------------------------

    def _bind_circuit(self, cid: str) -> None:
        """Rebind every per-circuit binding to pool entry ``cid`` (the swap point).

        These four bindings (plus the active pole) are the *entire* per-circuit state ``step``
        reads through ``self.*``; the per-step projection state (``_grip_idx``/``_grip_lat``/
        ``_prev_s``) is re-seeded for the drawn circuit later in ``reset`` before the first
        ``step``, so no stale per-track state carries across a swap. The lap timer is per
        circuit and is ``reset()`` each episode, so it never fires spuriously across a swap.
        """
        entry = self.pool.entries[cid]
        self.track = entry.track
        self.track_id = cid
        self.edge_cache = entry.edge_cache
        self.lap_timer = entry.lap_timer
        self._pole = entry.pole_time_s

    def set_track_pool(self, circuits: list[str] | str | None = None) -> None:
        """Curriculum hook: set the active circuit set (pool widening), no obs-layout change.

        Mirrors :meth:`apply_conditions` — called on the workers via ``VecEnv.env_method``,
        takes effect from the next ``reset``. Empty/None/"all" means the full configured pool.
        Pure sampling-side change, so it is safe mid-run (no retrain).
        """
        self.pool.set_active(circuits)

    # ----- curriculum hooks (called on the workers via VecEnv.env_method) ---------------

    def apply_conditions(
        self,
        *,
        mu_base: float | None = None,
        wear_rate: float | None = None,
        weather: str | None = None,
    ) -> None:
        """Push curriculum condition overrides into this worker (no obs-layout change).

        Conditions only — ``mu_base`` and ``weather`` go to the grip pipeline, ``wear_rate``
        to the dynamic physics. Takes effect from the next ``reset``/``step``; never touches
        the observation layout, so it is safe mid-run (no retrain).
        """
        if mu_base is not None:
            self.conditions.set_mu_base(float(mu_base))
        if weather is not None:
            self._weather_mode = str(weather)
        if wear_rate is not None and hasattr(self.physics, "params"):
            params = self.physics.params
            if hasattr(params, "wear_rate"):
                import dataclasses

                self.physics.params = dataclasses.replace(params, wear_rate=float(wear_rate))

    def _resolve_weather(self) -> None:
        """Set the episode weather: a fixed condition, or sample wet with ``p_wet`` if sampled."""
        if self._weather_mode == "sampled":
            p_wet = float(self.conditions.weather_params.p_wet)
            weather = "wet" if float(self.np_random.random()) < p_wet else "dry"
        else:
            weather = self._weather_mode
        self.conditions.set_weather(weather)

    # ----- helpers ----------------------------------------------------------------------

    def _step_grip(self) -> float:
        """Grip scalar for the next physics step (pipeline when dynamic, else constant)."""
        if not self.use_pipeline:
            return self.conditions.grip
        return self.conditions.grip_at(
            self.track, self._grip_idx, self._grip_lat, self.state.tire_wear, self.state.compound
        )

    def _build_obs(self) -> np.ndarray:
        """ObservationV2 at the current state, with the grip indicator from the last projection."""
        grip_ind = self.conditions.grip_indicator(
            self.track, self._grip_idx, self._grip_lat, self.state.tire_wear, self.state.compound
        )
        return build_observation(
            self.track, self.state, self.obs_params, self.edge_cache, grip_indicator=grip_ind
        )

    def _check_termination(self, timing: Any, off_track_m: float) -> tuple[bool, bool, str | None]:
        """Success on target laps; failure on large off-track / wrong-way; truncate on steps."""
        if timing.completed_laps >= self.limits.target_laps:
            return True, False, "success"
        if off_track_m >= self.limits.offtrack_limit_m:
            return True, False, "offtrack"
        if self._wrong_way_count >= self.limits.wrong_way_steps:
            return True, False, "wrong_way"
        if self._step_count >= self.limits.max_steps:
            return False, True, "truncated"
        return False, False, None

    def _map_action(self, action: np.ndarray) -> tuple[float, float]:
        """Map the policy action to physics controls.

        The physics step itself maps ``steer in [-1,1]`` to ``[-max_steer, +max_steer]`` and
        ``longitudinal>=0`` to throttle / ``<0`` to brake, so we just clip to the box here.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        steer = float(np.clip(a[0], -1.0, 1.0))
        longitudinal = float(np.clip(a[1], -1.0, 1.0))
        return steer, longitudinal

    def _sample_start_index(self, options: dict[str, Any] | None) -> int:
        n = len(self.track.centerline)
        if options is not None and "start_index" in options:
            return int(options["start_index"]) % n
        if self.limits.start_randomize:
            return int(self.np_random.integers(0, n))
        return 0

    def _record(self, timing: Any) -> None:
        s = self.state
        self.recorder.append(
            self._t,
            {
                "x": round(s.x, 3),
                "y": round(s.y, 3),
                "yaw": round(s.yaw, 5),
                "speed": round(s.speed, 3),
            },
            {
                "speed_kmh": round(s.speed * 3.6),
                "lap_time": round(timing.lap_time, 3),
                "progress": round(timing.progress, 4),
            },
        )


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
