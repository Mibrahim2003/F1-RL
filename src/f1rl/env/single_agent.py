"""Single-agent racing environment + the reusable per-car step (TECHNICAL_DESIGN.md §10).

``RacingEnv`` owns the car state, the physics stepper (built through
:func:`f1rl.physics.make_physics`, never a concrete model), the track, the conditions,
the reward, the termination logic, the lap tracker, and an optional trajectory recorder.
It runs its own fixed-step loop (``substeps`` physics substeps per control step) — it does
**not** import :class:`~f1rl.sim.loop.SimLoop`, so it stays rendering-free and
recorder-optional. It passes ``gymnasium.utils.env_checker.check_env``.

**Phase 5 factoring.** The per-car update is extracted into :func:`step_one_car` (and the
per-car placement into :func:`reset_car`), keyed by a :class:`CarRuntime` (the *car* state —
``CarState``, its **own** ``LapTimer``, and the per-step projection ``prev_s``/``grip_idx``/
``grip_lat``/``wrong_way_count``) and a read-only :class:`CarStepConfig` (the shared physics /
sim / limits / obs / reward / conditions). ``RacingEnv`` is now a thin one-car wrapper over
that unit, and :class:`~f1rl.env.multi_agent.RacingParallelEnv` reuses the exact same unit per
car — so the math is written once. The read-only circuit comes from a pool
:class:`~f1rl.env.pool.CircuitEntry` (its ``track``/``edge_cache``/``track_id``/``pole_time_s``
only — never its ``lap_timer``, which is per car).

Config-driven throughout: every tunable (sim timing, physics, obs params, reward weights,
target laps, step limit, start randomization, termination thresholds) comes from config.
``step`` never renders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import gymnasium.utils.seeding as gym_seeding
import numpy as np

from f1rl.env.collisions import CollisionParams, ContactRecord
from f1rl.env.conditions import Conditions
from f1rl.env.observations import (
    ObsParams,
    build_observation,
    observation_space,
    track_query,
)
from f1rl.env.pool import CircuitEntry, CircuitPool, pool_ids_from_config
from f1rl.env.rewards import RewardWeights, reward_v1, reward_v2, reward_v3
from f1rl.physics import make_physics
from f1rl.physics.base import CarState, PhysicsModel
from f1rl.sim.timing import LapTimer, Timing
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


# ===== The reusable per-car unit (shared by RacingEnv and RacingParallelEnv) =============


@dataclass
class CarRuntime:
    """All mutable state of ONE car — never pooled, never read off the env.

    The Phase 5 trap is treating any of this as circuit/env state: the ``LapTimer`` is the
    car's own instance (``LapTimer(track, pole)``, *not* the pool entry's), and the
    projection fields (``prev_s``/``grip_idx``/``grip_lat``/``wrong_way_count``) are carried
    per car so N cars on one circuit never stomp each other's lap or grip state.
    """

    state: CarState
    lap_timer: LapTimer  # this car's OWN timer (never the pooled CircuitEntry.lap_timer)
    prev_s: float = 0.0
    grip_idx: int = 0
    grip_lat: float = 0.0
    wrong_way_count: int = 0
    t: float = 0.0
    step_count: int = 0
    last_grip: float = 1.0  # grip used by the last physics advance (carried into the step info)
    done: bool = False  # frozen after termination/truncation (black_death pads it in the field)
    # Phase 6: the contact summary written by the field collision pass each step (zero = clean,
    # and always zero for the single-agent path / a one-car field), and the race rank last step.
    contact: ContactRecord = field(default_factory=ContactRecord)
    prev_rank: int = 1


@dataclass(frozen=True)
class CarStepConfig:
    """Read-only, shared-by-the-field step configuration (built once per env/worker).

    Holds the references the per-car step reads but never mutates per car: the physics
    stepper (a pure function behind :class:`PhysicsModel`), loop timing, limits, obs params,
    reward weights, and the grip-pipeline :class:`Conditions`. ``conditions``/``physics`` are
    *shared* live objects — the curriculum mutates them in place (``apply_conditions``), which
    is intended and safe (no per-step write).
    """

    physics: PhysicsModel
    sim: SimParams
    limits: EnvLimits
    obs_params: ObsParams
    reward_weights: RewardWeights
    collision: CollisionParams
    conditions: Conditions
    use_pipeline: bool
    start_compound: int

    @classmethod
    def from_config(cls, cfg: Any) -> CarStepConfig:
        physics = make_physics(cfg)
        conditions = Conditions.from_config(cfg)
        model = str(_cfg_get(getattr(cfg, "physics", None) or cfg, "model", ""))
        return cls(
            physics=physics,
            sim=SimParams.from_config(cfg),
            limits=EnvLimits.from_config(cfg),
            obs_params=ObsParams.from_config(cfg),
            reward_weights=RewardWeights.from_config(cfg),
            collision=CollisionParams.from_config(cfg),
            conditions=conditions,
            use_pipeline=(model == "dynamic"),
            start_compound=int(conditions.tires.start_compound),
        )


def reset_car(
    entry: CircuitEntry,
    cfg: CarStepConfig,
    idx: int,
    lap_timer: LapTimer,
    *,
    lateral_m: float = 0.0,
) -> tuple[CarRuntime, np.ndarray, Timing]:
    """Place one car at centerline sample ``idx`` (optionally offset ``lateral_m`` sideways).

    Builds the ``CarState`` (heading = the centerline tangent), resets ``lap_timer`` and seeds
    its previous-s so it does not spuriously detect a lap, seeds the projection state, and
    returns ``(car, obs, timing)``. ``lateral_m`` shifts the start along the left normal (used
    by the field's ``grid`` reset to stagger columns); ``0`` reproduces the single-agent
    centerline start exactly.
    """
    track = entry.track
    c = track.centerline[idx]
    tan = track.tangent[idx]
    yaw = math.atan2(float(tan[1]), float(tan[0]))
    x = float(c[0])
    y = float(c[1])
    if lateral_m != 0.0:
        nrm = track.normal[idx]
        x += float(nrm[0]) * lateral_m
        y += float(nrm[1]) * lateral_m
    state = CarState(
        x=x,
        y=y,
        yaw=yaw,
        vx=float(cfg.limits.start_speed),
        compound=cfg.start_compound,
    )
    lap_timer.reset()
    car = CarRuntime(
        state=state,
        lap_timer=lap_timer,
        prev_s=float(track.s[idx]),
        grip_idx=int(idx),
        grip_lat=float(lateral_m),
    )
    # Seed the lap timer's internal previous-s so it does not spuriously detect a lap.
    timing = lap_timer.update(state.x, state.y, 0.0)
    obs = _build_obs(entry, cfg, car)
    return car, obs, timing


def advance_car_physics(
    entry: CircuitEntry,
    cfg: CarStepConfig,
    car: CarRuntime,
    action: np.ndarray,
) -> None:
    """Advance ONE car's physics one control step (Phase 6 split, part 1 — independent per car).

    Maps the action, takes the grip for this step from the pre-step surface/wear/weather (reusing
    the projection stored from the previous step / reset — no extra ``track_query``), runs the
    substep physics, and bumps the car's clock/step count. **No projection, reward, or obs** — the
    field collision pass runs between this and :func:`finalize_car_step`, so this stays a pure
    per-car update that never reads another car. ``car.state``/``t``/``step_count`` are mutated.
    """
    steer, longitudinal = _map_action(action)
    grip = _step_grip(entry, cfg, car)
    car.last_grip = grip
    for _ in range(cfg.sim.substeps):
        car.state = cfg.physics.step(car.state, steer, longitudinal, grip, cfg.sim.dt_physics)
    car.t += cfg.sim.dt_control
    car.step_count += 1


def finalize_car_step(
    entry: CircuitEntry,
    cfg: CarStepConfig,
    car: CarRuntime,
    *,
    neighbor_block: np.ndarray | None = None,
    places: int = 0,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Finalize ONE car after physics (+ the field collision pass) — the Gymnasium 5-tuple.

    Projects once (shared by reward + off-track + obs build + next grip), updates the lap timer,
    computes the reward (``reward_v3`` when ``reward.version >= 3`` — adding the graded contact
    penalty from ``car.contact`` and the zero-sum ``places`` term — else ``reward_v2``/``v1``),
    tracks wrong-way, terminates (with the optional contact crash-out), builds the observation
    **with the precomputed neighbor block**, and assembles the per-car info (incl. the racing
    fields). Reads the read-only circuit (never ``entry.lap_timer``). Identical math to the legacy
    single-car step when there is no contact (``car.contact`` empty) and a constant rank
    (``places = 0``) — so ``RacingEnv`` and a one-car field reproduce Phase 5 exactly.
    """
    track = entry.track
    idx, s_along, signed_lateral, half_width, _heading = track_query(
        track, car.state.x, car.state.y, car.state.yaw
    )
    off_track_m = max(0.0, abs(signed_lateral) - half_width)
    car.grip_idx = int(idx)
    car.grip_lat = float(signed_lateral)

    timing = car.lap_timer.update(car.state.x, car.state.y, car.t)

    weights = cfg.reward_weights
    if weights.version >= 3:
        slip = abs(car.state.vy) / max(abs(car.state.vx), 1.0)
        reward, terms = reward_v3(
            car.prev_s,
            s_along,
            off_track_m,
            track.length,
            weights,
            slip=slip,
            contact=car.contact,
            places=int(places),
        )
    elif weights.version >= 2:
        slip = abs(car.state.vy) / max(abs(car.state.vx), 1.0)
        reward, terms = reward_v2(car.prev_s, s_along, off_track_m, track.length, weights, slip)
    else:
        reward, terms = reward_v1(car.prev_s, s_along, off_track_m, track.length, weights)
    car.prev_s = s_along

    # Wrong-way tracking: sustained backward progress.
    if terms["ds"] < cfg.limits.wrong_way_ds_m:
        car.wrong_way_count += 1
    else:
        car.wrong_way_count = 0

    terminated, truncated, term_reason = _check_termination(cfg.limits, car, timing, off_track_m)
    # Phase 6 opt-in crash-out: a hard contact ends the car with the same failure penalty path.
    if (
        not terminated
        and cfg.collision.crashout_enabled
        and car.contact.closing_mps > cfg.collision.crashout_closing_speed_mps
    ):
        terminated, term_reason = True, "crashout"
    if terminated and term_reason in ("offtrack", "wrong_way", "crashout"):
        reward += cfg.limits.failure_reward
        terms["failure"] = cfg.limits.failure_reward
        terms["total"] = reward

    obs = _build_obs(entry, cfg, car, neighbor_block=neighbor_block)
    info = {
        "lap_time": timing.lap_time,
        "off_track": off_track_m,
        "progress": timing.progress,
        "completed_laps": timing.completed_laps,
        "last_lap": timing.last_lap,
        "best_lap": timing.best_lap,
        "delta_to_pole": timing.delta_to_pole,
        "reward_terms": terms,
        "termination": term_reason,
        "steps": car.step_count,
        "grip": car.last_grip,
        "tire_wear": car.state.tire_wear,
        "compound": car.state.compound,
        "weather": cfg.conditions.weather,
        "circuit_id": entry.track_id,
        "pole_time_s": entry.pole_time_s,
        # Phase 6 racing fields. The single-agent / one-car path leaves these neutral; the
        # field step overrides race_position / gap_ahead_s from the field ranking.
        "contact": float(car.contact.impulse),
        "contact_closing_mps": float(car.contact.closing_mps),
        "overtakes": max(0, int(places)),
        "race_position": 1,
        "gap_ahead_s": None,
    }
    return obs, float(reward), bool(terminated), bool(truncated), info


def step_one_car(
    entry: CircuitEntry,
    cfg: CarStepConfig,
    car: CarRuntime,
    action: np.ndarray,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """One car, one control step = advance physics + (no collision) + finalize.

    The single-agent contract: one car cannot collide and has a constant rank, so this is the
    exact legacy ``RacingEnv.step`` math at the new obs length (an all-zero neighbor block and
    ``reward_v3`` reducing to ``reward_v2``). ``RacingEnv`` keeps calling this unchanged.
    """
    advance_car_physics(entry, cfg, car, action)
    return finalize_car_step(entry, cfg, car)


def _map_action(action: np.ndarray) -> tuple[float, float]:
    """Map the policy action to physics controls (clip to the action box).

    The physics step maps ``steer in [-1,1]`` to ``[-max_steer, +max_steer]`` and
    ``longitudinal>=0`` to throttle / ``<0`` to brake, so we just clip here.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    steer = float(np.clip(a[0], -1.0, 1.0))
    longitudinal = float(np.clip(a[1], -1.0, 1.0))
    return steer, longitudinal


def _step_grip(entry: CircuitEntry, cfg: CarStepConfig, car: CarRuntime) -> float:
    """Grip scalar for the next physics step (pipeline when dynamic, else constant)."""
    if not cfg.use_pipeline:
        return cfg.conditions.grip
    return cfg.conditions.grip_at(
        entry.track, car.grip_idx, car.grip_lat, car.state.tire_wear, car.state.compound
    )


def _build_obs(
    entry: CircuitEntry,
    cfg: CarStepConfig,
    car: CarRuntime,
    *,
    neighbor_block: np.ndarray | None = None,
) -> np.ndarray:
    """ObservationV3 at the car state, with the grip indicator + a precomputed neighbor block.

    ``neighbor_block`` is the field's encoding of this car's K nearest neighbors (the obs builder
    stays field-agnostic). ``None`` (the single-agent / one-car path) leaves an all-zero tail.
    """
    grip_ind = cfg.conditions.grip_indicator(
        entry.track, car.grip_idx, car.grip_lat, car.state.tire_wear, car.state.compound
    )
    return build_observation(
        entry.track,
        car.state,
        cfg.obs_params,
        entry.edge_cache,
        grip_indicator=grip_ind,
        neighbor_block=neighbor_block,
    )


def _check_termination(
    limits: EnvLimits, car: CarRuntime, timing: Any, off_track_m: float
) -> tuple[bool, bool, str | None]:
    """Success on target laps; failure on large off-track / wrong-way; truncate on steps."""
    if timing.completed_laps >= limits.target_laps:
        return True, False, "success"
    if off_track_m >= limits.offtrack_limit_m:
        return True, False, "offtrack"
    if car.wrong_way_count >= limits.wrong_way_steps:
        return True, False, "wrong_way"
    if car.step_count >= limits.max_steps:
        return False, True, "truncated"
    return False, False, None


# ===== RacingEnv — a thin one-car wrapper over the unit above ============================


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

        # The shared, read-only per-car step config (physics/sim/limits/obs/reward/conditions).
        self._car_cfg = CarStepConfig.from_config(cfg)
        # Expose the shared components as attributes for callers/tests (the curriculum mutates
        # `conditions`/`physics` in place through these same references).
        self.physics = self._car_cfg.physics
        self.conditions = self._car_cfg.conditions
        self.obs_params = self._car_cfg.obs_params
        self.sim = self._car_cfg.sim
        self.limits = self._car_cfg.limits
        self.reward_weights = self._car_cfg.reward_weights
        self.use_pipeline = self._car_cfg.use_pipeline
        self.start_compound = self._car_cfg.start_compound

        # Bind an initial active circuit (the configured track_id when in the pool, else the
        # first pool id) so attributes exist before the first reset.
        initial = track_id if track_id in self.pool else self.pool.ids[0]
        self._bind_circuit(initial)

        # Weather mode: a concrete condition ("dry"/"damp"/"wet") or "sampled" (the curriculum
        # sets this; reset() then draws wet with probability weather.p_wet).
        self._weather_mode = str(self.conditions.weather)

        self.recorder = recorder

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = observation_space(self.obs_params)

        # Build an initial car so `self.state` exists before the first reset.
        self._car, _obs, _timing = reset_car(self._entry, self._car_cfg, 0, self.lap_timer)

    @property
    def state(self) -> CarState:
        """The car's current :class:`CarState` (the single car this env owns)."""
        return self._car.state

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
        self._car, obs, timing = reset_car(self._entry, self._car_cfg, idx, self.lap_timer)

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
            self._record(timing.lap_time, timing.progress)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = step_one_car(
            self._entry, self._car_cfg, self._car, action
        )
        if self.recorder is not None:
            self._record(info["lap_time"], info["progress"])
        return obs, reward, terminated, truncated, info

    # ----- circuit pool (Phase 4) -------------------------------------------------------

    def _bind_circuit(self, cid: str) -> None:
        """Rebind every per-circuit binding to pool entry ``cid`` (the swap point).

        Stores the read-only entry (``track``/``edge_cache``/``track_id``/``pole`` used by the
        per-car step) and the active ``lap_timer`` (this single env reuses the entry's timer —
        a field env instead builds one timer *per car*). The per-step projection state lives on
        the ``CarRuntime`` and is re-seeded by ``reset`` before the first ``step``, so no stale
        per-track state carries across a swap.
        """
        entry = self.pool.entries[cid]
        self._entry = entry
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
        apply_conditions(self._car_cfg, mu_base=mu_base, wear_rate=wear_rate, weather=weather)
        if weather is not None:
            self._weather_mode = str(weather)

    def apply_reward_weights(
        self, *, w_contact: float | None = None, w_overtake: float | None = None
    ) -> None:
        """Curriculum hook: ramp the racing reward weights in place (no obs-layout change).

        Reads take effect from the next ``finalize`` (the step config is read fresh each step),
        so a run can learn to coexist (contact penalty on) before it learns to fight (overtake
        reward ramped up). Pure reward-shaping change; safe mid-run.
        """
        apply_reward_weights(self._car_cfg, w_contact=w_contact, w_overtake=w_overtake)
        self.reward_weights = self._car_cfg.reward_weights

    def _resolve_weather(self) -> None:
        """Set the episode weather: a fixed condition, or sample wet with ``p_wet`` if sampled."""
        resolve_weather(self._car_cfg, self._weather_mode, self.np_random)

    # ----- helpers ----------------------------------------------------------------------

    def _sample_start_index(self, options: dict[str, Any] | None) -> int:
        n = len(self.track.centerline)
        if options is not None and "start_index" in options:
            return int(options["start_index"]) % n
        if self.limits.start_randomize:
            return int(self.np_random.integers(0, n))
        return 0

    def _record(self, lap_time: float, progress: float) -> None:
        s = self._car.state
        self.recorder.append(
            self._car.t,
            {
                "x": round(s.x, 3),
                "y": round(s.y, 3),
                "yaw": round(s.yaw, 5),
                "speed": round(s.speed, 3),
            },
            {
                "speed_kmh": round(s.speed * 3.6),
                "lap_time": round(lap_time, 3),
                "progress": round(progress, 4),
            },
        )


# ===== shared curriculum / weather helpers (used by both envs) ==========================


def apply_conditions(
    cfg: CarStepConfig,
    *,
    mu_base: float | None = None,
    wear_rate: float | None = None,
    weather: str | None = None,
) -> None:
    """Apply curriculum condition overrides to the shared :class:`CarStepConfig` in place."""
    if mu_base is not None:
        cfg.conditions.set_mu_base(float(mu_base))
    if weather is not None:
        cfg.conditions.set_weather(str(weather))
    if wear_rate is not None and hasattr(cfg.physics, "params"):
        params = cfg.physics.params
        if hasattr(params, "wear_rate"):
            import dataclasses

            cfg.physics.params = dataclasses.replace(params, wear_rate=float(wear_rate))


def apply_reward_weights(
    cfg: CarStepConfig, *, w_contact: float | None = None, w_overtake: float | None = None
) -> None:
    """Replace the shared :class:`RewardWeights` in place with the ramped racing weights.

    ``CarStepConfig`` and ``RewardWeights`` are frozen, so this rebuilds the weights with the
    overrides and swaps them via ``object.__setattr__`` (the same in-place transport the
    curriculum uses for conditions). ``None`` leaves a weight unchanged.
    """
    import dataclasses

    changes: dict[str, float] = {}
    if w_contact is not None:
        changes["w_contact"] = float(w_contact)
    if w_overtake is not None:
        changes["w_overtake"] = float(w_overtake)
    if changes:
        new_weights = dataclasses.replace(cfg.reward_weights, **changes)
        object.__setattr__(cfg, "reward_weights", new_weights)


def resolve_weather(cfg: CarStepConfig, weather_mode: str, rng: np.random.Generator) -> None:
    """Set the episode weather on the shared conditions (fixed, or sampled wet with ``p_wet``)."""
    if weather_mode == "sampled":
        p_wet = float(cfg.conditions.weather_params.p_wet)
        weather = "wet" if float(rng.random()) < p_wet else "dry"
    else:
        weather = weather_mode
    cfg.conditions.set_weather(weather)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
