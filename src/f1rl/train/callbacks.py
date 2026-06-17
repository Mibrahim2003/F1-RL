"""Training callbacks: scheduled checkpointing and periodic eval-video logging (§12, plan §C).

- :class:`CheckpointCallback` — every ``checkpoint_freq`` steps, atomically save the model +
  VecNormalize + meta via :mod:`f1rl.train.checkpointing`; keep a rolling window of the last
  ``keep_last_k`` and a separate ``best`` checkpoint by eval return.
- :class:`EvalVideoCallback` — every ``eval_freq`` steps, run one deterministic episode (via
  :mod:`f1rl.train.evaluate`) with a :class:`TrajectoryRecorder`, render an mp4 via
  :mod:`f1rl.render.renderer`, and log the clip + metrics (episode return, lap time vs pole
  64.3 s and 2× pole, off-track count, steps-to-first-clean-lap). The eval env uses the saved
  VecNormalize stats (synced obs_rms; ``training=False``, ``norm_reward=False``) so eval/serve
  match the training distribution.

No rendering happens inside the env step loop — only here, on the eval schedule.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from f1rl.train.checkpointing import save_checkpoint
from f1rl.train.evaluate import evaluate


class CheckpointCallback(BaseCallback):
    """Atomically checkpoint on a schedule; keep the last K and the best-by-eval-return."""

    def __init__(
        self,
        save_dir: str | Path,
        cfg: Any,
        checkpoint_freq: int,
        *,
        keep_last_k: int = 3,
        keep_best: bool = True,
        seed: int | None = None,
        logger: Any | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.cfg = cfg
        self.checkpoint_freq = max(1, int(checkpoint_freq))
        self.keep_last_k = int(keep_last_k)
        self.keep_best = bool(keep_best)
        self.seed = seed
        self.run_logger = logger
        self._saved: list[Path] = []
        self._best_metric: float | None = None
        self.best_path = self.save_dir / "best"
        self.latest_path = self.save_dir / "latest"

    def _vecnorm(self) -> Any:
        return self.model.get_env()

    def _save(self, name: str) -> Path:
        path = self.save_dir / name
        meta = save_checkpoint(
            path, self.model, self._vecnorm(), self.cfg, seed=self.seed, atomic=True
        )
        if self.verbose:
            print(f"[checkpoint] saved {path} @ {meta['total_timesteps']} steps")
        return path

    def _prune(self) -> None:
        """Keep only the last ``keep_last_k`` rolling step-checkpoints."""
        if self.keep_last_k <= 0:
            return
        while len(self._saved) > self.keep_last_k:
            old = self._saved.pop(0)
            _rmtree(old)

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or self.num_timesteps % self.checkpoint_freq != 0:
            return True
        path = self._save(f"step_{self.num_timesteps}")
        self._saved.append(path)
        self._prune()
        # Mirror the most recent into a stable "latest" pointer for easy --resume.
        self._save("latest")
        return True

    def update_best(self, metric: float) -> None:
        """Called by the eval callback: persist a 'best' checkpoint when ``metric`` improves."""
        if not self.keep_best:
            return
        if self._best_metric is None or metric > self._best_metric:
            self._best_metric = metric
            self._save("best")
            if self.verbose:
                print(f"[checkpoint] new best (eval return={metric:.3f})")


class EvalVideoCallback(BaseCallback):
    """Run a deterministic eval episode, render an mp4, and log clip + metrics on a schedule."""

    def __init__(
        self,
        cfg: Any,
        eval_freq: int,
        *,
        n_eval_episodes: int = 1,
        video_dir: str | Path = "eval_videos",
        record_video: bool = True,
        video_fps: int = 20,
        pole_time_s: float = 0.0,
        deterministic: bool = True,
        seed: int = 0,
        logger: Any | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.cfg = cfg
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = int(n_eval_episodes)
        self.video_dir = Path(video_dir)
        self.record_video = bool(record_video)
        self.video_fps = int(video_fps)
        self.pole_time_s = float(pole_time_s)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.run_logger = logger
        self.checkpoint_callback = checkpoint_callback
        self._track = None

    def _clip_obs(self) -> float:
        env_node = getattr(self.cfg, "env", None)
        if env_node is None:
            return 10.0
        get = env_node.get if hasattr(env_node, "get") else (lambda k, d: getattr(env_node, k, d))
        return float(get("clip_obs", 10.0))

    def _obs_rms(self) -> Any | None:
        """The training VecNormalize obs stats, so eval matches the training distribution."""
        env = self.model.get_env()
        return getattr(env, "obs_rms", None)

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        self._run_eval()
        return True

    def _run_eval(self) -> None:
        result = evaluate(
            self.model,
            self.cfg,
            n_episodes=self.n_eval_episodes,
            seed=self.seed,
            obs_rms=self._obs_rms(),
            clip_obs=self._clip_obs(),
            deterministic=self.deterministic,
            record_first=self.record_video,
            pole_time_s=self.pole_time_s,
        )
        summary = result.summary(self.pole_time_s)

        # Log scalars to SB3's logger (TensorBoard/stdout) and to the run logger (W&B + CSV).
        for k, v in summary.items():
            with contextlib.suppress(Exception):
                self.logger.record(k, v)
        if self.run_logger is not None:
            self.run_logger.log(summary, step=int(self.num_timesteps))

        if self.verbose:
            print(
                f"[eval] step={self.num_timesteps} return={result.mean_return:.3f} "
                f"best_lap={summary.get('eval/best_lap_time')}"
            )

        # Render the first episode's recorded trajectory to an mp4 and log the clip.
        if self.record_video and result.recorder is not None and len(result.recorder) > 1:
            self._render_and_log(result.recorder)

        # Tell the checkpoint callback whether this is the best policy yet.
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.update_best(result.mean_return)

    def _render_and_log(self, recorder: Any) -> None:
        from f1rl.render.renderer import render_trajectory
        from f1rl.track.loader import load_track

        if self._track is None:
            track_id = self.cfg.get("track_id") if hasattr(self.cfg, "get") else self.cfg.track_id
            self._track = load_track(str(track_id), cfg=getattr(self.cfg, "track", None))

        self.video_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.video_dir / f"eval_{self.num_timesteps}.mp4"
        try:
            render_trajectory(self._track, recorder.to_dict(), out_path, fps=self.video_fps)
        except Exception as exc:
            print(f"[eval] video render failed ({exc!r}); skipping clip this round")
            return
        if self.run_logger is not None:
            self.run_logger.log_video(
                "eval/clip", out_path, step=int(self.num_timesteps), fps=self.video_fps
            )
        if self.verbose:
            print(f"[eval] clip -> {out_path}")


def _rmtree(p: Path) -> None:
    import shutil

    shutil.rmtree(p, ignore_errors=True)
