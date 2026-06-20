"""Env construction seam (plan §A factory).

``make_env`` builds a single :class:`~f1rl.env.single_agent.RacingEnv` (for the Gymnasium
checker, eval, and the per-worker thunk). ``make_vec_env`` builds a ``SubprocVecEnv`` of
those workers wrapped in ``VecNormalize`` (observation normalization on, reward
normalization per config) — the seam the training code builds against.

Per-env seeding follows the plan: worker ``rank`` gets ``base_seed + rank`` so each
parallel copy explores a different start distribution while the run stays reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from f1rl.env.single_agent import RacingEnv


def make_env(cfg: Any, seed: int = 0, rank: int = 0) -> RacingEnv:
    """Build one :class:`RacingEnv` seeded at ``seed + rank``.

    Returns a bare ``RacingEnv`` (no wrappers) so callers can run the Gymnasium env checker
    and eval directly against the contract. ``make_vec_env`` adds the Monitor/VecNormalize
    stack for training.
    """
    env_seed = int(seed) + int(rank)
    env = RacingEnv(cfg, seed=env_seed)
    env.reset(seed=env_seed)
    return env


def _worker_thunk(cfg: Any, base_seed: int, rank: int) -> Callable[[], Monitor]:
    """Return a picklable thunk that builds a Monitor-wrapped env for one subprocess worker."""

    def _init() -> Monitor:
        env = make_env(cfg, seed=base_seed, rank=rank)
        return Monitor(env)

    return _init


def make_vec_env(
    cfg: Any,
    n_envs: int,
    seed: int = 0,
) -> VecNormalize:
    """Build a ``SubprocVecEnv`` of ``n_envs`` workers wrapped in ``VecNormalize``.

    Observation normalization is always on (PPO needs it for this obs vector); reward
    normalization follows ``cfg.env.norm_reward`` (default True). The ``VecNormalize`` clip
    and gamma come from config so nothing is hardcoded. Per-env seed = ``seed + rank``.
    """
    base_seed = int(seed)
    venv = SubprocVecEnv(
        [_worker_thunk(cfg, base_seed, rank) for rank in range(int(n_envs))]
    )

    env_node = getattr(cfg, "env", None)
    get = (
        env_node.get
        if env_node is not None and hasattr(env_node, "get")
        else (lambda k, d: getattr(env_node, k, d) if env_node is not None else d)
    )
    norm_reward = bool(get("norm_reward", True))
    clip_obs = float(get("clip_obs", 10.0))
    clip_reward = float(get("clip_reward", 10.0))
    gamma = float(get("norm_gamma", 0.99))

    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=gamma,
    )
