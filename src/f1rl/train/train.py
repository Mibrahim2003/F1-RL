"""Device-agnostic PPO training entry point (spec §12, §16; plan §C).

One ``train.py`` driven entirely by config — identical locally and on the cloud, writing the
same checkpoints both ways. It loads the experiment config (+ dotlist overrides),
:func:`seed_everything`, builds the vectorized normalized env via
:func:`f1rl.env.factory.make_vec_env`, builds SB3 PPO (``MlpPolicy``, hyperparameters +
device from config), attaches the checkpoint + eval-video callbacks, and runs
``model.learn``. ``--resume <path>`` restores the model + VecNormalize + timestep count and
continues with ``reset_num_timesteps=False``.

Every hyperparameter and weight comes from ``configs/experiment/<name>.yaml`` — nothing is
hardcoded here (CLAUDE.md rule). No FastF1, no rendering in the step loop.

CLI::

    .venv/Scripts/python.exe -m f1rl.train.train --config rbr_ppo
    .venv/Scripts/python.exe -m f1rl.train.train --config rbr_ppo n_envs=2 total_timesteps=2000
    .venv/Scripts/python.exe -m f1rl.train.train --config rbr_ppo --resume <run>/checkpoints/latest
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from f1rl.env.factory import make_vec_env
from f1rl.train.callbacks import CheckpointCallback, EvalVideoCallback
from f1rl.train.checkpointing import load_checkpoint
from f1rl.train.curriculum import CurriculumCallback
from f1rl.train.wandb_logger import RunLogger
from f1rl.utils.config import load_config
from f1rl.utils.seeding import seed_everything


def load_experiment_config(name: str, overrides: list[str] | None = None) -> Any:
    """Load an experiment config by name, looking under ``configs/experiment/`` first.

    Accepts ``"rbr_ppo"``, ``"experiment/rbr_ppo"``, or a top-level config name; resolves to
    the first that exists so the CLI stays forgiving.
    """
    candidates = [name]
    if not name.startswith("experiment/"):
        candidates.insert(0, f"experiment/{name}")
    last_err: Exception | None = None
    for cand in candidates:
        try:
            return load_config(cand, overrides=overrides)
        except FileNotFoundError as exc:
            last_err = exc
    raise FileNotFoundError(
        f"experiment config not found for '{name}' (tried {candidates})"
    ) from last_err


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _ppo_kwargs(cfg: Any) -> dict[str, Any]:
    """Read PPO hyperparameters from ``cfg.ppo`` (every value from config)."""
    ppo = getattr(cfg, "ppo", None)

    def g(k: str, d: Any) -> Any:
        return _cfg_get(ppo, k, d)

    return {
        "n_steps": int(g("n_steps", 2048)),
        "batch_size": int(g("batch_size", 256)),
        "n_epochs": int(g("n_epochs", 10)),
        "gamma": float(g("gamma", 0.99)),
        "gae_lambda": float(g("gae_lambda", 0.95)),
        "ent_coef": float(g("ent_coef", 0.0)),
        "vf_coef": float(g("vf_coef", 0.5)),
        "learning_rate": float(g("learning_rate", 3e-4)),
        "clip_range": float(g("clip_range", 0.2)),
        "max_grad_norm": float(g("max_grad_norm", 0.5)),
    }


def build_model(cfg: Any, venv: Any, seed: int) -> Any:
    """Construct a PPO ``MlpPolicy`` from config hyperparameters on the config device."""
    from stable_baselines3 import PPO

    device = str(_cfg_get(cfg, "device", "cpu"))
    return PPO(
        "MlpPolicy",
        venv,
        seed=seed,
        device=device,
        verbose=1,
        **_ppo_kwargs(cfg),
    )


def _make_callbacks(cfg: Any, run_dir: Path, seed: int, logger: RunLogger) -> list[Any]:
    """Build the checkpoint + eval-video callbacks from the eval/checkpoint config blocks."""
    ckpt_node = getattr(cfg, "checkpoint", None)
    eval_node = getattr(cfg, "eval", None)

    ckpt_dir = run_dir / "checkpoints"
    checkpoint_cb = CheckpointCallback(
        save_dir=ckpt_dir,
        cfg=cfg,
        checkpoint_freq=int(_cfg_get(ckpt_node, "checkpoint_freq", 50000)),
        keep_last_k=int(_cfg_get(ckpt_node, "keep_last_k", 3)),
        keep_best=bool(_cfg_get(ckpt_node, "keep_best", True)),
        seed=seed,
        logger=logger,
    )

    pole = _track_pole(cfg)
    eval_cb = EvalVideoCallback(
        cfg=cfg,
        eval_freq=int(_cfg_get(eval_node, "eval_freq", 50000)),
        n_eval_episodes=int(_cfg_get(eval_node, "n_eval_episodes", 1)),
        video_dir=run_dir / "eval_videos",
        record_video=bool(_cfg_get(eval_node, "record_video", True)),
        video_fps=int(_cfg_get(eval_node, "video_fps", 20)),
        pole_time_s=float(_cfg_get(eval_node, "pole_time_s", pole)),
        deterministic=bool(_cfg_get(eval_node, "deterministic", True)),
        seed=seed,
        logger=logger,
        checkpoint_callback=checkpoint_cb,
    )

    callbacks: list[Any] = [checkpoint_cb, eval_cb]
    # Curriculum ramps conditions (grip -> wear -> weather) by timestep; no-op when disabled.
    curriculum_cb = CurriculumCallback(cfg, logger=logger)
    if curriculum_cb.stages:
        callbacks.append(curriculum_cb)
    return callbacks


def _track_pole(cfg: Any) -> float:
    track = getattr(cfg, "track", None)
    return float(_cfg_get(track, "pole_time_s", 0.0))


def train(
    config_name: str = "rbr_ppo",
    overrides: list[str] | None = None,
    resume: str | None = None,
    run_name: str | None = None,
    output_root: str | Path = "runs",
) -> Any:
    """Run (or resume) a PPO training job from config; returns the trained model.

    Args:
        config_name: Experiment config name (``configs/experiment/<name>.yaml``).
        overrides: Dotlist overrides applied on top of the config.
        resume: Optional checkpoint directory to resume from (continues the timestep count).
        run_name: Optional run name; defaults to ``<config>_<seed>_<timestamp>``.
        output_root: Root directory for run artifacts (checkpoints, videos, logs).
    """
    cfg = load_experiment_config(config_name, overrides=overrides)

    seed = int(_cfg_get(cfg, "seed", 42))
    seed_everything(seed)

    n_envs = int(_cfg_get(cfg, "n_envs", 4))
    total_timesteps = int(_cfg_get(cfg, "total_timesteps", 2_000_000))

    run_name = run_name or f"{config_name}_{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(cfg, n_envs=n_envs, seed=seed)

    config_snapshot = _config_container(cfg)
    logger = RunLogger(cfg, run_dir=run_dir, run_name=run_name, config_snapshot=config_snapshot)

    try:
        if resume:
            device = str(_cfg_get(cfg, "device", "cpu"))
            model, meta = load_checkpoint(resume, env=venv, device=device)
            model.set_env(venv)
            reset_num_timesteps = False
            print(
                f"[train] resumed from {resume} at {meta['total_timesteps']} steps; "
                f"continuing to {total_timesteps}"
            )
        else:
            model = build_model(cfg, venv, seed)
            reset_num_timesteps = True

        callbacks = _make_callbacks(cfg, run_dir, seed, logger)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )

        # Final atomic checkpoint so the run always ends with a resumable artifact.
        from f1rl.train.checkpointing import save_checkpoint

        final_path = run_dir / "checkpoints" / "final"
        save_checkpoint(final_path, model, venv, cfg, seed=seed, atomic=True)
        print(f"[train] final checkpoint -> {final_path} @ {model.num_timesteps} steps")
    finally:
        logger.close()
        venv.close()
    return model


def _config_container(cfg: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return dict(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        pass
    return {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for one circuit (Phase 3a).")
    p.add_argument("--config", default="rbr_ppo", help="experiment config name")
    p.add_argument("--resume", default=None, help="checkpoint directory to resume from")
    p.add_argument("--run-name", default=None, help="optional run name")
    p.add_argument("--output-root", default="runs", help="root dir for run artifacts")
    p.add_argument("overrides", nargs="*", default=[], help="dotlist overrides, e.g. n_envs=2")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    train(
        config_name=args.config,
        overrides=list(args.overrides) or None,
        resume=args.resume,
        run_name=args.run_name,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
