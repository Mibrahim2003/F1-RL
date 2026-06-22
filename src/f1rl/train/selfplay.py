"""Self-play PPO training for the field (Phase 5; spec §2b/§5, plan §C).

Mirrors :mod:`f1rl.train.train` but trains **one shared policy** across a FIELD of cars: the
:class:`~f1rl.env.multi_agent.RacingParallelEnv` is vectorized PettingZoo -> SuperSuit ->
Stable-Baselines3 by :func:`f1rl.env.factory.make_selfplay_vec_env`, so every car's transitions
update the same network (parameter sharing). The Phase 4 generalist warm-starts directly
(``--resume``) because the observation is unchanged (``OBS_VERSION = 2``) and the per-agent
obs/action spaces match across field widths — a 2-car checkpoint loads into a 4-car run.

Field size (``grid.n_agents``) is a **per-run constant** (it sets the vec-env width
``n_agents * n_copies``); grow the field by warm-starting successive runs. The circuit-pool
widening **stays** an in-place curriculum knob, broadcast to the raw field envs (SuperSuit's
``ConcatVecEnv`` has no ``env_method``, so the Phase 4 callback's transport is replaced here).

Every hyperparameter/weight comes from ``configs/experiment/<name>.yaml``; nothing hardcoded.

CLI::

    .venv/Scripts/python.exe -m f1rl.train.selfplay --config calendar_selfplay \
        --resume runs/<calendar_dynamic_run>/checkpoints/best
    .venv/Scripts/python.exe -m f1rl.train.selfplay --config calendar_selfplay --throughput
    .venv/Scripts/python.exe -m f1rl.train.selfplay --config calendar_selfplay \
        grid.n_agents=2 total_timesteps=4000 selfplay.n_copies=2
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.env.factory import make_selfplay_vec_env, make_vec_env, raw_parallel_envs
from f1rl.train.callbacks import CheckpointCallback
from f1rl.train.checkpointing import load_checkpoint, save_checkpoint
from f1rl.train.curriculum import CurriculumCallback, active_stage
from f1rl.train.selfplay_eval import SelfPlayEvalCallback
from f1rl.train.train import _cfg_get, _config_container, build_model, load_experiment_config
from f1rl.train.wandb_logger import RunLogger
from f1rl.utils.seeding import seed_everything


class SelfPlayCurriculumCallback(CurriculumCallback):
    """Curriculum for the field: broadcast conditions / pool-widening onto the raw field envs.

    SuperSuit's ``ConcatVecEnv`` does not implement ``VecEnv.env_method`` (the Phase 4
    transport), so instead of ``training_env.env_method(...)`` this reaches the underlying
    :class:`RacingParallelEnv` instances via :func:`raw_parallel_envs` and calls their hooks
    directly. Only the in-process (``num_cpus=0``) stack exposes them.
    """

    def _maybe_apply(self, *, force: bool) -> None:
        if not self.stages:
            return
        stage = active_stage(self.stages, int(self.num_timesteps))
        if stage is None:
            return
        if not force and stage.start_step == self._applied_start:
            return
        self._applied_start = stage.start_step

        raws = raw_parallel_envs(self.training_env)
        if not raws:
            if self.verbose:
                print(
                    "[curriculum] no in-process field envs reachable (num_cpus>0); "
                    "stage not applied — set selfplay.num_cpus=0 for the in-place curriculum"
                )
            return
        for r in raws:
            r.apply_conditions(
                mu_base=stage.mu_base, wear_rate=stage.wear_rate, weather=stage.weather
            )
            if stage.circuits is not None:
                r.set_track_pool(list(stage.circuits))
            # Phase 6: ramp the racing reward weights in place (coexist -> race).
            if stage.w_contact is not None or stage.w_overtake is not None:
                r.apply_reward_weights(w_contact=stage.w_contact, w_overtake=stage.w_overtake)

        nan = float("nan")
        n_circuits = float(len(stage.circuits)) if stage.circuits is not None else nan
        payload = {
            "curriculum/stage_start_step": float(stage.start_step),
            "curriculum/mu_base": float(stage.mu_base) if stage.mu_base is not None else nan,
            "curriculum/wear_rate": float(stage.wear_rate) if stage.wear_rate is not None else nan,
            "curriculum/n_circuits": n_circuits,
            "curriculum/w_contact": float(stage.w_contact) if stage.w_contact is not None else nan,
            "curriculum/w_overtake": float(stage.w_overtake)
            if stage.w_overtake is not None
            else nan,
        }
        if self.run_logger is not None:
            self.run_logger.log(payload, step=int(self.num_timesteps))
        if self.verbose:
            print(
                f"[curriculum] step={self.num_timesteps} -> stage@{stage.start_step} "
                f"mu_base={stage.mu_base} wear_rate={stage.wear_rate} weather={stage.weather} "
                f"circuits={list(stage.circuits) if stage.circuits is not None else 'unchanged'} "
                f"w_contact={stage.w_contact} w_overtake={stage.w_overtake}"
            )


def _grid_cfg(cfg: Any) -> tuple[int, int, int]:
    """Return ``(n_agents, n_copies, num_cpus)`` from the grid / selfplay config blocks."""
    grid = getattr(cfg, "grid", None)
    sp = getattr(cfg, "selfplay", None)
    n_agents = int(_cfg_get(grid, "n_agents", 1))
    n_copies = int(_cfg_get(sp, "n_copies", 2))
    num_cpus = int(_cfg_get(sp, "num_cpus", 0))
    return n_agents, n_copies, num_cpus


def _make_callbacks(
    cfg: Any, run_dir: Path, seed: int, n_agents: int, logger: RunLogger
) -> list[Any]:
    """Checkpoint + field-eval + (broadcast) curriculum callbacks from the config blocks."""
    ckpt_node = getattr(cfg, "checkpoint", None)
    eval_node = getattr(cfg, "eval", None)

    checkpoint_cb = CheckpointCallback(
        save_dir=run_dir / "checkpoints",
        cfg=cfg,
        checkpoint_freq=int(_cfg_get(ckpt_node, "checkpoint_freq", 50000)),
        keep_last_k=int(_cfg_get(ckpt_node, "keep_last_k", 3)),
        keep_best=bool(_cfg_get(ckpt_node, "keep_best", True)),
        seed=seed,
        logger=logger,
    )
    eval_cb = SelfPlayEvalCallback(
        cfg=cfg,
        n_agents=n_agents,
        eval_freq=int(_cfg_get(eval_node, "eval_freq", 50000)),
        trajectory_dir=run_dir / "eval_trajectories",
        save_trajectory=bool(_cfg_get(eval_node, "record_video", True)),
        deterministic=bool(_cfg_get(eval_node, "deterministic", True)),
        seed=seed,
        logger=logger,
        checkpoint_callback=checkpoint_cb,
    )
    callbacks: list[Any] = [checkpoint_cb, eval_cb]
    curriculum_cb = SelfPlayCurriculumCallback(cfg, logger=logger)
    if curriculum_cb.stages:
        callbacks.append(curriculum_cb)
    return callbacks


def selfplay(
    config_name: str = "calendar_selfplay",
    overrides: list[str] | None = None,
    resume: str | None = None,
    warm_start: str | None = None,
    run_name: str | None = None,
    output_root: str | Path = "runs",
) -> Any:
    """Run / resume / grow-warm-start a self-play PPO job from config; returns the trained model.

    ``--resume`` validates and continues a v3 checkpoint (timestep count carried). ``--warm-start``
    is the Phase 6 transplant: it **grows** a Phase 5 (v2) policy's input layer into a fresh v3
    policy (see :func:`f1rl.train.warmstart.grow_policy`), starting a new timestep count — the only
    sanctioned way to bring the competent driver across the ``OBS_VERSION`` 2 -> 3 bump.
    """
    cfg = load_experiment_config(config_name, overrides=overrides)
    seed = int(_cfg_get(cfg, "seed", 42))
    seed_everything(seed)

    n_agents, n_copies, num_cpus = _grid_cfg(cfg)
    total_timesteps = int(_cfg_get(cfg, "total_timesteps", 2_000_000))

    run_name = run_name or f"{config_name}_{n_agents}car_{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = make_selfplay_vec_env(cfg, n_agents, n_copies, seed=seed, num_cpus=num_cpus)
    logger = RunLogger(
        cfg, run_dir=run_dir, run_name=run_name, config_snapshot=_config_container(cfg)
    )

    try:
        device = str(_cfg_get(cfg, "device", "cpu"))
        if warm_start:
            from f1rl.train.warmstart import grow_policy

            model, meta = grow_policy(warm_start, venv, cfg, seed, device=device)
            model.set_env(venv)
            reset_num_timesteps = True  # a fresh v3 policy: start the timestep count at 0
            print(
                f"[selfplay] grown warm start from {warm_start} "
                f"(src obs_version={meta.get('obs_version')}, n_agents={meta.get('n_agents')}) "
                f"into a {n_agents}-car v3 field"
            )
        elif resume:
            model, meta = load_checkpoint(resume, env=venv, device=device)
            model.set_env(venv)
            reset_num_timesteps = False
            print(
                f"[selfplay] resumed from {resume} "
                f"(obs_version={meta.get('obs_version')}, n_agents={meta.get('n_agents')}) "
                f"into a {n_agents}-car field @ {meta['total_timesteps']} steps"
            )
        else:
            model = build_model(cfg, venv, seed)
            reset_num_timesteps = True

        callbacks = _make_callbacks(cfg, run_dir, seed, n_agents, logger)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )

        final_path = run_dir / "checkpoints" / "final"
        save_checkpoint(final_path, model, venv, cfg, seed=seed, atomic=True)
        print(f"[selfplay] final checkpoint -> {final_path} @ {model.num_timesteps} steps")
    finally:
        logger.close()
        venv.close()
    return model


def throughput_report(
    cfg: Any,
    *,
    n_agents: int,
    n_copies: int,
    num_cpus: int,
    seed: int,
    steps: int = 1500,
) -> dict[str, float]:
    """Measure field SPS vs an equal-width single-agent ``n_envs`` run (the §2 throughput check).

    Steps a random policy through both stacks at the same vector width
    (``n_agents * n_copies``) and reports agent-environment steps per second, so the
    field-size ceiling and the laptop-vs-cloud line are known before committing field sizes.
    """
    width = n_agents * n_copies

    fvec = make_selfplay_vec_env(cfg, n_agents, n_copies, seed=seed, num_cpus=num_cpus)
    fvec.reset()
    acts = np.stack([fvec.action_space.sample() for _ in range(fvec.num_envs)])
    t0 = time.perf_counter()
    for _ in range(steps):
        fvec.step(acts)
    field_sps = steps * fvec.num_envs / (time.perf_counter() - t0)
    fvec.close()

    svec = make_vec_env(cfg, n_envs=width, seed=seed)
    svec.reset()
    sacts = np.stack([svec.action_space.sample() for _ in range(svec.num_envs)])
    t0 = time.perf_counter()
    for _ in range(steps):
        svec.step(sacts)
    single_sps = steps * svec.num_envs / (time.perf_counter() - t0)
    svec.close()

    report = {
        "vector_width": float(width),
        "n_agents": float(n_agents),
        "n_copies": float(n_copies),
        "field_sps": float(field_sps),
        "single_agent_sps": float(single_sps),
        "field_vs_single_ratio": float(field_sps / single_sps) if single_sps else float("nan"),
    }
    print("\n[throughput] field vs single-agent (equal vector width):")
    for k, v in report.items():
        print(f"  {k:<28} {v:.2f}")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-play PPO training for a field (Phase 5).")
    p.add_argument("--config", default="calendar_selfplay", help="experiment config name")
    p.add_argument("--resume", default=None, help="v3 checkpoint dir to validate + resume from")
    p.add_argument(
        "--warm-start",
        default=None,
        help="Phase 5 (v2) checkpoint dir to grow into a fresh v3 policy (input-layer transplant)",
    )
    p.add_argument("--run-name", default=None, help="optional run name")
    p.add_argument("--output-root", default="runs", help="root dir for run artifacts")
    p.add_argument("--throughput", action="store_true", help="measure SPS vs single-agent and exit")
    p.add_argument(
        "overrides", nargs="*", default=[], help="dotlist overrides, e.g. grid.n_agents=4"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.throughput:
        cfg = load_experiment_config(args.config, overrides=list(args.overrides) or None)
        n_agents, n_copies, num_cpus = _grid_cfg(cfg)
        throughput_report(
            cfg,
            n_agents=n_agents,
            n_copies=n_copies,
            num_cpus=num_cpus,
            seed=int(_cfg_get(cfg, "seed", 42)),
        )
        return
    selfplay(
        config_name=args.config,
        overrides=list(args.overrides) or None,
        resume=args.resume,
        warm_start=args.warm_start,
        run_name=args.run_name,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
