"""Env construction seam (plan §A factory).

``make_env`` builds a single :class:`~f1rl.env.single_agent.RacingEnv` (for the Gymnasium
checker, eval, and the per-worker thunk). ``make_vec_env`` builds a ``SubprocVecEnv`` of
those workers wrapped in ``VecNormalize`` (observation normalization on, reward
normalization per config) — the seam the single-agent training code builds against.

``make_selfplay_vec_env`` (Phase 5) is the parallel analogue: it wraps the
:class:`~f1rl.env.multi_agent.RacingParallelEnv` field through SuperSuit
(``black_death_v3`` -> ``pettingzoo_env_to_vec_env_v1`` -> ``concat_vec_envs_v1``) into the
same ``VecMonitor`` + ``VecNormalize`` SB3 stack, so one shared PPO policy trains across every
car (parameter sharing). The single-agent path is left untouched.

Per-env seeding follows the plan: worker ``rank`` gets ``base_seed + rank`` so each
parallel copy explores a different start distribution while the run stays reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecEnvWrapper,
    VecMonitor,
    VecNormalize,
)

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
    venv = SubprocVecEnv([_worker_thunk(cfg, base_seed, rank) for rank in range(int(n_envs))])

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


def _vecnorm_kwargs(cfg: Any) -> dict[str, Any]:
    """Read the ``VecNormalize`` knobs from ``cfg.env`` (same as the single-agent path)."""
    env_node = getattr(cfg, "env", None)
    get = (
        env_node.get
        if env_node is not None and hasattr(env_node, "get")
        else (lambda k, d: getattr(env_node, k, d) if env_node is not None else d)
    )
    return {
        "norm_reward": bool(get("norm_reward", True)),
        "clip_obs": float(get("clip_obs", 10.0)),
        "clip_reward": float(get("clip_reward", 10.0)),
        "gamma": float(get("norm_gamma", 0.99)),
    }


def make_selfplay_vec_env(
    cfg: Any,
    n_agents: int,
    n_copies: int,
    seed: int = 0,
    num_cpus: int = 0,
) -> VecNormalize:
    """Build the SuperSuit-vectorized field env (Phase 5), wrapped in ``VecNormalize``.

    The :class:`~f1rl.env.multi_agent.RacingParallelEnv` of ``n_agents`` cars is wrapped with
    ``black_death_v3`` (constant SuperSuit-visible agent width even on per-car early death),
    converted to a vector env (``pettingzoo_env_to_vec_env_v1`` — one sub-env per car), then
    replicated ``n_copies`` times (``concat_vec_envs_v1``). The resulting SB3 ``VecEnv`` has
    ``n_agents * n_copies`` sub-envs sharing one policy, and exposes the **unchanged** length-22
    obs Box and 2-D action Box (so the Phase 4 generalist warm-starts across any field width).

    ``num_cpus=0`` keeps every copy in-process (deterministic, and the only mode where the
    curriculum can reach the raw envs — see :func:`raw_parallel_envs`); raise it for throughput
    on many cores at the cost of that in-place curriculum reach.
    """
    import supersuit as ss

    from f1rl.env.multi_agent import RacingParallelEnv

    env = RacingParallelEnv(cfg, n_agents=int(n_agents), seed=int(seed))
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec = ss.concat_vec_envs_v1(
        env,
        int(n_copies),
        num_cpus=int(num_cpus),
        base_class="stable_baselines3",
    )
    vec = _SeedableSuperSuitVec(vec)  # SuperSuit's ConcatVecEnv has no .seed(); add one
    vec.seed(int(seed))  # decorrelate the copies (raw field env i gets seed + i)
    vec = VecMonitor(vec)
    return VecNormalize(vec, norm_obs=True, **_vecnorm_kwargs(cfg))


class _SeedableSuperSuitVec(VecEnvWrapper):
    """Give the SuperSuit ``ConcatVecEnv`` a working ``seed()`` (it has none).

    SB3 ``PPO(seed=...)`` calls ``env.seed(seed)`` at construction, and ``VecEnv.seed`` would
    delegate down to ``ConcatVecEnv``, which raises ``AttributeError``. This wrapper intercepts
    ``seed`` and reseeds each raw :class:`RacingParallelEnv` copy with ``seed + i`` (so the
    copies explore different circuit draws / start distributions while the run stays
    reproducible), instead of forwarding the call further down.
    """

    def reset(self) -> Any:
        return self.venv.reset()

    def step_wait(self) -> Any:
        return self.venv.step_wait()

    def seed(self, seed: int | None = None) -> list[int | None]:
        raws = raw_parallel_envs(self)
        seeds: list[int | None] = []
        for i, raw in enumerate(raws):
            s = None if seed is None else int(seed) + i
            if hasattr(raw, "reseed"):
                raw.reseed(s)
            seeds.append(s)
        return seeds


def raw_parallel_envs(vec_env: Any) -> list[Any]:
    """Return the raw :class:`RacingParallelEnv` instances inside a SuperSuit SB3 vec env.

    The Phase 4 curriculum pushes ``set_track_pool`` / ``apply_conditions`` through
    ``VecEnv.env_method``, which SuperSuit's ``ConcatVecEnv`` does **not** implement. This
    walks ``VecNormalize -> VecMonitor -> ConcatVecEnv.vec_envs[i].par_env.unwrapped`` to reach
    the underlying field envs so the curriculum can call those hooks directly. Only the
    in-process (``num_cpus=0``) stack exposes them; returns ``[]`` otherwise.
    """
    v = vec_env
    while hasattr(v, "venv"):
        v = v.venv
    out: list[Any] = []
    for ve in getattr(v, "vec_envs", []):
        par = getattr(ve, "par_env", None)
        raw = getattr(par, "unwrapped", par)
        if raw is not None:
            out.append(raw)
    return out
