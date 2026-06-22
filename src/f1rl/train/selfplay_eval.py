"""Multi-car evaluation driver for the field (Phase 5 monitoring; spec §2d owner note).

``train/evaluate.py`` drives a single :class:`~f1rl.env.single_agent.RacingEnv` (a Gymnasium
env). A :class:`~f1rl.env.multi_agent.RacingParallelEnv` is **not** a Gym env, so it needs a
dedicated driver: this module runs one deterministic field episode with the shared policy,
normalizing each car's observation by the saved ``VecNormalize`` stats (so eval matches
training), records the whole field as a multi-car ``cars[]`` trajectory (replayable in the web
app), and reports per-car + aggregate metrics (mean / per-car lap time and delta-to-pole,
off-track count, episode return).

The bar this phase is **infrastructure + the render**, not a learning gain — a non-degenerate
field return only proves the env is trainable.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from f1rl.env.multi_agent import RacingParallelEnv
from f1rl.sim.recorder import TrajectoryRecorder


@dataclass
class FieldCarMetrics:
    """Per-car metrics over one field episode (Phase 6 adds the racing stats)."""

    episode_return: float = 0.0
    length: int = 0
    completed_laps: int = 0
    best_lap_time: float | None = None
    off_track_count: int = 0
    termination: str | None = None
    # Phase 6 racing metrics.
    overtakes: int = 0  # total places gained this episode (genuine wheel-to-wheel swaps)
    contact_steps: int = 0  # steps with any contact
    contact_impulse_sum: float = 0.0  # summed contact impulse magnitude
    final_position: int = 0  # last seen race position (1 = leader)


@dataclass
class FieldResult:
    """Aggregate result of one field episode + the multi-car recorder (for the clip/replay)."""

    circuit_id: str = "unknown"
    pole_time_s: float = 0.0
    cars: dict[str, FieldCarMetrics] = field(default_factory=dict)
    recorder: TrajectoryRecorder | None = None

    @property
    def mean_return(self) -> float:
        vals = [m.episode_return for m in self.cars.values()]
        return float(np.mean(vals)) if vals else 0.0

    def summary(self) -> dict[str, Any]:
        """Mean + spread metrics across the field, ready to log as plain scalars."""
        if not self.cars:
            return {}
        ms = list(self.cars.values())
        best = [m.best_lap_time for m in ms if m.best_lap_time is not None]
        pole = self.pole_time_s
        pole_missing = pole <= 0.0
        gap = float(np.min(best) - pole) if (best and not pole_missing) else float("nan")
        total_steps = float(sum(m.length for m in ms))
        contact_steps = float(sum(m.contact_steps for m in ms))
        impulse_sum = float(sum(m.contact_impulse_sum for m in ms))
        overtakes_total = float(sum(m.overtakes for m in ms))
        return {
            "selfplay_eval/n_agents": float(len(ms)),
            "selfplay_eval/mean_return": self.mean_return,
            "selfplay_eval/return_spread": float(np.std([m.episode_return for m in ms])),
            "selfplay_eval/mean_episode_length": float(np.mean([m.length for m in ms])),
            "selfplay_eval/mean_completed_laps": float(np.mean([m.completed_laps for m in ms])),
            "selfplay_eval/mean_off_track_count": float(np.mean([m.off_track_count for m in ms])),
            "selfplay_eval/best_lap_time": float(np.min(best)) if best else float("nan"),
            "selfplay_eval/mean_best_lap_time": float(np.mean(best)) if best else float("nan"),
            "selfplay_eval/pole_time_s": float(pole),
            "selfplay_eval/best_gap_to_pole": gap,
            "selfplay_eval/pole_missing": float(pole_missing),
            # Phase 6 racing metrics: watch overtakes + contact rate TOGETHER (ramming vs timidity).
            "selfplay_eval/overtakes_total": overtakes_total,
            "selfplay_eval/overtakes_per_car": overtakes_total / len(ms),
            "selfplay_eval/contact_rate": (contact_steps / total_steps) if total_steps else 0.0,
            "selfplay_eval/mean_contact_impulse": (impulse_sum / contact_steps)
            if contact_steps
            else 0.0,
        }


def _normalize_obs(
    obs: np.ndarray, obs_rms: Any | None, clip: float, epsilon: float = 1e-8
) -> np.ndarray:
    """Apply VecNormalize-style obs normalization (mean/var) so eval matches training."""
    if obs_rms is None:
        return obs
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    var = np.asarray(obs_rms.var, dtype=np.float64)
    return np.clip((obs - mean) / np.sqrt(var + epsilon), -clip, clip)


def run_field_episode(
    model: Any,
    cfg: Any,
    *,
    n_agents: int,
    seed: int = 0,
    obs_rms: Any | None = None,
    clip_obs: float = 10.0,
    deterministic: bool = True,
    record: bool = True,
    reset_mode: str | None = None,
) -> FieldResult:
    """Run one deterministic field episode with the shared policy; return per-car + aggregate.

    ``reset_mode`` overrides ``cfg.grid.reset_mode`` for this episode (use ``"grid"`` for a tidy
    demo start). Every car steps each control step until it is done; the recorder captures the
    **whole field every frame** (a done car is frozen in place), so the clip plays continuously.
    """
    env = RacingParallelEnv(cfg, n_agents=n_agents, seed=seed)
    if reset_mode is not None:
        env.grid = replace(env.grid, reset_mode=reset_mode)

    obs, infos = env.reset(seed=seed)
    agents = list(env.possible_agents)
    team_n = len(env.grid.team_colors)
    dt = env._car_cfg.sim.dt_control
    circuit_id = env._entry.track_id
    pole = float(env._entry.pole_time_s)

    result = FieldResult(circuit_id=circuit_id, pole_time_s=pole)
    result.cars = {a: FieldCarMetrics() for a in agents}
    last_info: dict[str, dict[str, Any]] = dict(infos)

    recorder = TrajectoryRecorder(circuit_id, dt, seed, n_agents=n_agents) if record else None

    def record_frame(t: float) -> None:
        if recorder is None:
            return
        cars = []
        for i, a in enumerate(agents):
            s = env._cars[a].state
            tinfo = last_info.get(a, {})
            cars.append(
                {
                    "id": a,
                    "x": round(s.x, 3),
                    "y": round(s.y, 3),
                    "yaw": round(s.yaw, 5),
                    "speed": round(s.speed, 3),
                    "team": i % team_n,
                    "telemetry": {
                        "speed_kmh": round(s.speed * 3.6),
                        "lap_time": round(float(tinfo.get("lap_time", 0.0)), 3),
                        "progress": round(float(tinfo.get("progress", 0.0)), 4),
                        "completed_laps": int(tinfo.get("completed_laps", 0)),
                        # Phase 6 racing readouts (carried into the replayable trajectory).
                        "race_position": int(tinfo.get("race_position", 0)),
                        "gap_ahead_s": tinfo.get("gap_ahead_s"),
                        "contact": round(float(tinfo.get("contact", 0.0)), 3),
                    },
                }
            )
        recorder.append_cars(t, cars)

    t = 0.0
    record_frame(t)
    while env.agents:
        actions = {}
        for a in env.agents:
            ob = _normalize_obs(np.asarray(obs[a], dtype=np.float64), obs_rms, clip_obs)
            act, _ = model.predict(ob.astype(np.float32), deterministic=deterministic)
            actions[a] = act
        obs, rewards, _terms, _truncs, infos = env.step(actions)
        t += dt
        for a, info in infos.items():
            last_info[a] = info
            m = result.cars[a]
            m.episode_return += float(rewards[a])
            m.length += 1
            if float(info.get("off_track", 0.0)) > 0.0:
                m.off_track_count += 1
            m.completed_laps = int(info.get("completed_laps", m.completed_laps))
            if info.get("best_lap") is not None:
                m.best_lap_time = float(info["best_lap"])
            m.termination = info.get("termination")
            # Phase 6 racing metrics.
            m.overtakes += int(info.get("overtakes", 0))
            contact = float(info.get("contact", 0.0))
            if contact > 0.0:
                m.contact_steps += 1
                m.contact_impulse_sum += contact
            m.final_position = int(info.get("race_position", m.final_position))
        record_frame(t)

    result.recorder = recorder
    return result


class SelfPlayEvalCallback(BaseCallback):
    """Run a field eval episode on a schedule; log metrics, save the clip, update best."""

    def __init__(
        self,
        cfg: Any,
        n_agents: int,
        eval_freq: int,
        *,
        trajectory_dir: str | Path = "eval_trajectories",
        save_trajectory: bool = True,
        deterministic: bool = True,
        reset_mode: str = "grid",
        seed: int = 0,
        logger: Any | None = None,
        checkpoint_callback: Any | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self.n_agents = int(n_agents)
        self.eval_freq = max(1, int(eval_freq))
        self.trajectory_dir = Path(trajectory_dir)
        self.save_trajectory = bool(save_trajectory)
        self.deterministic = bool(deterministic)
        self.reset_mode = reset_mode
        self.seed = int(seed)
        self.run_logger = logger
        self.checkpoint_callback = checkpoint_callback

    def _obs_rms(self) -> Any | None:
        return getattr(self.model.get_env(), "obs_rms", None)

    def _clip_obs(self) -> float:
        env_node = getattr(self.cfg, "env", None)
        if env_node is None:
            return 10.0
        get = env_node.get if hasattr(env_node, "get") else (lambda k, d: getattr(env_node, k, d))
        return float(get("clip_obs", 10.0))

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        result = run_field_episode(
            self.model,
            self.cfg,
            n_agents=self.n_agents,
            seed=self.seed,
            obs_rms=self._obs_rms(),
            clip_obs=self._clip_obs(),
            deterministic=self.deterministic,
            record=self.save_trajectory,
            reset_mode=self.reset_mode,
        )
        summary = result.summary()
        for k, v in summary.items():
            with contextlib.suppress(Exception):
                self.logger.record(k, v)
        if self.run_logger is not None:
            self.run_logger.log(summary, step=int(self.num_timesteps))
        if self.verbose:
            print(
                f"[selfplay-eval] step={self.num_timesteps} circuit={result.circuit_id} "
                f"mean_return={result.mean_return:.3f} "
                f"best_lap={summary.get('selfplay_eval/best_lap_time')}"
            )
        if self.save_trajectory and result.recorder is not None and len(result.recorder) > 1:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            out = self.trajectory_dir / f"field_{self.num_timesteps}.json"
            with contextlib.suppress(Exception):
                result.recorder.save(out)
                if self.verbose:
                    print(f"[selfplay-eval] field trajectory -> {out}")
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.update_best(result.mean_return)
        return True
