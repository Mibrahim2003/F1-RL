"""Deterministic evaluation: run an episode, report metrics, optionally emit a clip (§12).

Shared by the eval callback (:class:`f1rl.train.callbacks.EvalVideoCallback`) and a CLI. One
deterministic episode is run on a single :class:`~f1rl.env.single_agent.RacingEnv` (so the
:class:`~f1rl.sim.recorder.TrajectoryRecorder` captures the real per-step car frames), with
the policy's observations normalized by the **saved VecNormalize stats** so eval/serve match
the training distribution. Metrics follow §12: episode return, lap time vs pole (64.3 s) and
2× pole, off-track count, steps-to-first-clean-lap.

CLI::

    .venv/Scripts/python.exe -m f1rl.train.evaluate --checkpoint checkpoints/rbr_ppo/best \
        --config experiment/rbr_ppo --episodes 3 --video out/eval.mp4 --trajectory out/eval.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.env.single_agent import RacingEnv
from f1rl.sim.recorder import TrajectoryRecorder
from f1rl.utils.seeding import seed_everything


@dataclass
class EpisodeMetrics:
    """Per-episode evaluation metrics (spec §12 logged metrics)."""

    episode_return: float = 0.0
    episode_length: int = 0
    completed_laps: int = 0
    best_lap_time: float | None = None
    last_lap_time: float | None = None
    off_track_count: int = 0  # control steps spent off the asphalt
    max_off_track_m: float = 0.0
    steps_to_first_clean_lap: int | None = None  # None if no clean lap completed
    termination: str | None = None
    # vs-pole flags (pole + 2x pole), set when a lap completes.
    beat_pole: bool = False
    beat_2x_pole: bool = False

    def as_dict(self) -> dict[str, Any]:
        nan = float("nan")
        return {
            "eval/episode_return": self.episode_return,
            "eval/episode_length": self.episode_length,
            "eval/completed_laps": self.completed_laps,
            "eval/best_lap_time": self.best_lap_time if self.best_lap_time is not None else nan,
            "eval/last_lap_time": self.last_lap_time if self.last_lap_time is not None else nan,
            "eval/off_track_count": self.off_track_count,
            "eval/max_off_track_m": self.max_off_track_m,
            "eval/steps_to_first_clean_lap": (
                self.steps_to_first_clean_lap if self.steps_to_first_clean_lap is not None else nan
            ),
            "eval/beat_pole": float(self.beat_pole),
            "eval/beat_2x_pole": float(self.beat_2x_pole),
        }


@dataclass
class EvalResult:
    """Aggregate result over ``n_episodes`` plus the recorder of the first episode."""

    episodes: list[EpisodeMetrics] = field(default_factory=list)
    recorder: TrajectoryRecorder | None = None  # first episode's trajectory (for the clip)

    @property
    def mean_return(self) -> float:
        return float(np.mean([e.episode_return for e in self.episodes])) if self.episodes else 0.0

    def summary(self, pole_time_s: float) -> dict[str, Any]:
        """Mean metrics across episodes, ready to log as plain scalars."""
        if not self.episodes:
            return {}
        best = [e.best_lap_time for e in self.episodes if e.best_lap_time is not None]
        clean = [
            e.steps_to_first_clean_lap
            for e in self.episodes
            if e.steps_to_first_clean_lap is not None
        ]
        return {
            "eval/mean_return": self.mean_return,
            "eval/mean_episode_length": float(np.mean([e.episode_length for e in self.episodes])),
            "eval/mean_off_track_count": float(np.mean([e.off_track_count for e in self.episodes])),
            "eval/completed_laps": float(np.mean([e.completed_laps for e in self.episodes])),
            "eval/best_lap_time": float(np.min(best)) if best else float("nan"),
            "eval/mean_steps_to_first_clean_lap": float(np.mean(clean)) if clean else float("nan"),
            "eval/pole_time_s": float(pole_time_s),
            "eval/two_x_pole_time_s": float(2.0 * pole_time_s),
            "eval/beat_pole_rate": float(np.mean([e.beat_pole for e in self.episodes])),
            "eval/beat_2x_pole_rate": float(np.mean([e.beat_2x_pole for e in self.episodes])),
        }


def _normalize_obs(
    obs: np.ndarray, obs_rms: Any | None, clip: float, epsilon: float = 1e-8
) -> np.ndarray:
    """Apply VecNormalize-style obs normalization (mean/var) so eval matches training."""
    if obs_rms is None:
        return obs
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    var = np.asarray(obs_rms.var, dtype=np.float64)
    normed = (obs - mean) / np.sqrt(var + epsilon)
    return np.clip(normed, -clip, clip)


def run_episode(
    model: Any,
    cfg: Any,
    *,
    seed: int = 0,
    obs_rms: Any | None = None,
    clip_obs: float = 10.0,
    deterministic: bool = True,
    record: bool = True,
    pole_time_s: float = 0.0,
) -> tuple[EpisodeMetrics, TrajectoryRecorder | None]:
    """Run one deterministic episode and return its metrics (+ recorder when ``record``).

    ``obs_rms`` is the training :class:`VecNormalize` ``obs_rms`` (mean/var); when given the
    policy sees normalized observations exactly as in training.
    """
    track_id = cfg.get("track_id") if hasattr(cfg, "get") else getattr(cfg, "track_id", "oval")
    sim_node = getattr(cfg, "sim", None)
    control_hz = int(getattr(sim_node, "control_hz", 20)) if sim_node is not None else 20
    dt = 1.0 / control_hz
    recorder = TrajectoryRecorder(track_id=str(track_id), dt=dt, seed=seed) if record else None

    env = RacingEnv(cfg, seed=seed, recorder=recorder)
    obs, _info = env.reset(seed=seed)

    metrics = EpisodeMetrics()
    if pole_time_s <= 0.0:
        pole_time_s = float(_track_pole(cfg))

    done = False
    while not done:
        norm_obs = _normalize_obs(np.asarray(obs, dtype=np.float64), obs_rms, clip_obs)
        action, _ = model.predict(norm_obs.astype(np.float32), deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        metrics.episode_return += float(reward)
        metrics.episode_length += 1
        off = float(info.get("off_track", 0.0))
        if off > 0.0:
            metrics.off_track_count += 1
            metrics.max_off_track_m = max(metrics.max_off_track_m, off)
        metrics.completed_laps = int(info.get("completed_laps", 0))
        metrics.best_lap_time = info.get("best_lap")
        metrics.last_lap_time = info.get("last_lap")
        metrics.termination = info.get("termination")

        # First clean lap = first lap completion with no off-track in this episode so far.
        if (
            metrics.steps_to_first_clean_lap is None
            and metrics.completed_laps >= 1
            and metrics.off_track_count == 0
        ):
            metrics.steps_to_first_clean_lap = metrics.episode_length

    if metrics.best_lap_time is not None and pole_time_s > 0.0:
        metrics.beat_pole = metrics.best_lap_time <= pole_time_s
        metrics.beat_2x_pole = metrics.best_lap_time <= 2.0 * pole_time_s
    return metrics, recorder


def evaluate(
    model: Any,
    cfg: Any,
    *,
    n_episodes: int = 1,
    seed: int = 0,
    obs_rms: Any | None = None,
    clip_obs: float = 10.0,
    deterministic: bool = True,
    record_first: bool = True,
    pole_time_s: float = 0.0,
) -> EvalResult:
    """Run ``n_episodes`` deterministic episodes; keep the first episode's recorder for a clip."""
    result = EvalResult()
    for i in range(n_episodes):
        rec = i == 0 and record_first
        ep, recorder = run_episode(
            model,
            cfg,
            seed=seed + i,
            obs_rms=obs_rms,
            clip_obs=clip_obs,
            deterministic=deterministic,
            record=rec,
            pole_time_s=pole_time_s,
        )
        result.episodes.append(ep)
        if rec:
            result.recorder = recorder
    return result


def _track_pole(cfg: Any) -> float:
    track = getattr(cfg, "track", None)
    if track is None:
        return 0.0
    get = track.get if hasattr(track, "get") else (lambda k, d: getattr(track, k, d))
    return float(get("pole_time_s", 0.0))


# ----- CLI ----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on deterministic episodes.")
    p.add_argument("--checkpoint", required=True, help="checkpoint directory")
    p.add_argument("--config", default="experiment/rbr_ppo", help="experiment config name")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", default=None, help="optional mp4 output path for the first episode")
    p.add_argument("--trajectory", default=None, help="optional trajectory JSON output path")
    p.add_argument("overrides", nargs="*", default=[], help="dotlist config overrides")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from f1rl.train.checkpointing import load_checkpoint
    from f1rl.train.train import load_experiment_config

    args = _parse_args(argv)
    cfg = load_experiment_config(args.config, overrides=list(args.overrides) or None)
    seed_everything(int(args.seed))

    model, _meta = load_checkpoint(args.checkpoint, env=None, device="cpu")

    # Recover obs_rms from the saved VecNormalize for matched normalization.
    obs_rms = _load_obs_rms(args.checkpoint)
    env_node = getattr(cfg, "env", None)
    clip_obs = float(getattr(env_node, "clip_obs", 10.0)) if env_node is not None else 10.0
    pole = _track_pole(cfg)

    result = evaluate(
        model,
        cfg,
        n_episodes=int(args.episodes),
        seed=int(args.seed),
        obs_rms=obs_rms,
        clip_obs=clip_obs,
        record_first=bool(args.video or args.trajectory),
        pole_time_s=pole,
    )

    summary = result.summary(pole)
    print("\nEvaluation summary:")
    for k, v in summary.items():
        print(f"  {k:<40} {v}")

    if result.recorder is not None and args.trajectory:
        path = result.recorder.save(args.trajectory)
        print(f"  trajectory -> {path}")
    if result.recorder is not None and args.video:
        from f1rl.render.renderer import render_trajectory
        from f1rl.track.loader import load_track

        track = load_track(str(cfg.get("track_id")), cfg=getattr(cfg, "track", None))
        out = render_trajectory(track, result.recorder.to_dict(), args.video)
        print(f"  video -> {out}")


def _load_obs_rms(checkpoint: str | Path) -> Any | None:
    """Load just the obs_rms from a checkpoint's vecnormalize.pkl (for eval normalization)."""
    import pickle

    vn_path = Path(checkpoint) / "vecnormalize.pkl"
    if not vn_path.exists():
        return None
    try:
        with vn_path.open("rb") as f:
            vn = pickle.load(f)
        return getattr(vn, "obs_rms", None)
    except Exception:
        return None


if __name__ == "__main__":
    main()
