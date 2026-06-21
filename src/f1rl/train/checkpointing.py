"""Checkpoint format — the single source of truth for save/resume (spec §12, plan §C).

A checkpoint is a directory holding three artifacts:

- ``model.zip``         — SB3 PPO: weights + optimizer + torch RNG (``PPO.save``).
- ``vecnormalize.pkl``  — :class:`VecNormalize` running obs/reward stats (``VecNormalize.save``).
- ``meta.json``         — sidecar with the exact schema below.

Meta sidecar schema (every key required)::

    {
      "total_timesteps": int,        # model.num_timesteps at save time
      "circuit_id": str,             # cfg.track_id — resume on the right circuit
      "obs_version": int,            # OBS_VERSION; loader refuses a mismatch
      "action_shape": [int, ...],    # Box action shape; loader refuses a mismatch
      "n_agents": int,               # Phase 5 field size the run trained on (1 = single-agent)
      "seed": int,                   # the run seed (recorded every run)
      "config_snapshot": dict,       # full resolved config, for reproducibility
      "sb3_version": str,            # stable_baselines3.__version__
      "numpy_rng_state": [...],      # np.random global RNG state, JSON-encoded
    }

Saving and resuming round-trip **exactly** (weights, optimizer, vecnorm stats, timestep
count). ``validate_checkpoint`` refuses a mismatched ``obs_version`` or action shape with a
clear, specific error before a resume can run on a stale observation layout.

The server (app-integration-engineer) imports this module — it is the one place the format
is defined. Pure save/load logic: no FastF1, no rendering.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.env.observations import OBS_VERSION

MODEL_FILE = "model.zip"
VECNORM_FILE = "vecnormalize.pkl"
META_FILE = "meta.json"

# The action contract is Box(-1, +1, shape=(2,)): [steer, longitudinal]. Stable across the
# whole phase; a mismatch means the checkpoint was trained on a different action layout.
EXPECTED_ACTION_SHAPE: tuple[int, ...] = (2,)


class CheckpointError(ValueError):
    """Raised when a checkpoint is incompatible (obs version / action shape) or malformed."""


def _config_to_container(cfg: Any) -> Any:
    """Resolve a config (OmegaConf or plain) to a JSON-serializable container."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    if isinstance(cfg, dict):
        return cfg
    return {}


def _numpy_rng_state() -> list[Any]:
    """Capture the NumPy global RNG state in a JSON-encodable form."""
    name, keys, pos, has_gauss, cached_gauss = np.random.get_state()  # noqa: NPY002
    return [name, keys.tolist(), int(pos), int(has_gauss), float(cached_gauss)]


def _restore_numpy_rng_state(state: list[Any] | None) -> None:
    """Restore the NumPy global RNG state captured by :func:`_numpy_rng_state`."""
    if not state:
        return
    name, keys, pos, has_gauss, cached_gauss = state
    np.random.set_state(  # noqa: NPY002
        (
            str(name),
            np.asarray(keys, dtype=np.uint32),
            int(pos),
            int(has_gauss),
            float(cached_gauss),
        )
    )


def build_meta(
    model: Any,
    cfg: Any,
    *,
    seed: int | None = None,
    circuit_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the meta sidecar dict for ``model`` + ``cfg`` (the §12 schema)."""
    import stable_baselines3 as sb3

    container = _config_to_container(cfg)
    if circuit_id is None:
        circuit_id = (
            container.get("track_id") if isinstance(container, dict) else None
        ) or "unknown"
    if seed is None:
        seed = container.get("seed", 0) if isinstance(container, dict) else 0

    action_space = getattr(model, "action_space", None)
    action_shape = list(getattr(action_space, "shape", EXPECTED_ACTION_SHAPE))

    # Phase 5: the constant field size this run trained on (1 for the single-agent path). Not
    # validated on resume — the per-agent obs/action spaces match across widths, so a
    # smaller-field checkpoint warm-starts a larger-field run.
    n_agents = 1
    if isinstance(container, dict):
        grid = container.get("grid")
        if isinstance(grid, dict):
            n_agents = int(grid.get("n_agents", 1) or 1)

    return {
        "total_timesteps": int(getattr(model, "num_timesteps", 0)),
        "circuit_id": str(circuit_id),
        "obs_version": int(OBS_VERSION),
        "action_shape": action_shape,
        "n_agents": int(n_agents),
        "seed": int(seed),
        "config_snapshot": container,
        "sb3_version": str(sb3.__version__),
        "numpy_rng_state": _numpy_rng_state(),
    }


def save_checkpoint(
    path: str | Path,
    model: Any,
    vecnorm: Any | None = None,
    cfg: Any | None = None,
    *,
    seed: int | None = None,
    atomic: bool = True,
) -> dict[str, Any]:
    """Save ``model`` (+ ``vecnorm`` + meta) to ``path`` and return the meta dict.

    Args:
        path: Checkpoint directory (created if missing).
        model: SB3 PPO model. Saved via ``model.save`` (weights + optimizer + torch RNG).
        vecnorm: The :class:`VecNormalize` env (obs/reward stats). When omitted, falls back
            to ``model.get_env()``; if that is not a VecNormalize, no stats file is written.
        cfg: Resolved run config; snapshotted into the meta for reproducibility.
        seed: The run seed to record (falls back to ``cfg.seed``).
        atomic: When True, write to a temp dir and atomically swap into place so a crashed
            save never leaves a half-written checkpoint (plan: atomic save).

    Returns:
        The meta sidecar dict (also written to ``meta.json``).
    """
    dest = Path(path)
    write_dir = dest.with_name(dest.name + ".tmp") if atomic else dest
    if write_dir.exists():
        _rmtree(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(write_dir / MODEL_FILE))

    vn = vecnorm
    if vn is None:
        vn = getattr(model, "get_env", lambda: None)()
    if vn is not None and hasattr(vn, "save"):
        # Not a VecNormalize (or nothing to save) — model still loads, just unnormalized.
        with contextlib.suppress(Exception):
            vn.save(str(write_dir / VECNORM_FILE))

    meta = build_meta(model, cfg, seed=seed)
    (write_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if atomic and write_dir != dest:
        if dest.exists():
            _rmtree(dest)
        write_dir.replace(dest)
    return meta


def load_meta(path: str | Path) -> dict[str, Any]:
    """Read the meta sidecar from a checkpoint directory."""
    meta_path = Path(path) / META_FILE
    if not meta_path.exists():
        raise CheckpointError(f"checkpoint meta not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_checkpoint(
    path: str | Path,
    env: Any | None = None,
    *,
    device: str = "cpu",
    restore_numpy_rng: bool = False,
    validate: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Load a checkpoint into a PPO model and return ``(model, meta)``.

    Round-trips exactly: weights, optimizer, torch RNG, ``num_timesteps``, and (when ``env``
    is a :class:`VecNormalize`) the obs/reward running stats. The timestep count is restored
    by SB3 from ``model.zip``, so ``model.num_timesteps`` equals the saved value and a
    subsequent ``learn(..., reset_num_timesteps=False)`` continues from it.

    Args:
        path: Checkpoint directory.
        env: A vectorized env to attach. If it is a :class:`VecNormalize`, the saved obs/
            reward stats are loaded into it so eval/serve match the training distribution.
        device: Torch device (``"cpu"`` here; identical local/cloud).
        restore_numpy_rng: When True, also restore the NumPy global RNG state from the meta.
        validate: When True, refuse an incompatible checkpoint (obs version / action shape)
            before returning (spec §12 validation).

    Raises:
        CheckpointError: On a missing artifact or an incompatible obs version / action shape.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    ckpt = Path(path)
    model_path = ckpt / MODEL_FILE
    if not model_path.exists():
        raise CheckpointError(f"checkpoint model not found: {model_path}")

    meta = load_meta(ckpt)
    if validate:
        validate_checkpoint(meta)

    # Attach VecNormalize stats to the provided env (so eval/serve match training).
    attach_env = env
    vecnorm_path = ckpt / VECNORM_FILE
    if env is not None and isinstance(env, VecNormalize) and vecnorm_path.exists():
        loaded = VecNormalize.load(str(vecnorm_path), env.venv)
        # Carry forward the caller's training/reward-norm intent; load() sets stats only.
        loaded.training = env.training
        loaded.norm_reward = env.norm_reward
        attach_env = loaded

    model = PPO.load(str(model_path), env=attach_env, device=device)

    if restore_numpy_rng:
        _restore_numpy_rng_state(meta.get("numpy_rng_state"))

    return model, meta


def validate_checkpoint(
    meta: dict[str, Any],
    expected_obs_version: int = OBS_VERSION,
    expected_action_shape: tuple[int, ...] = EXPECTED_ACTION_SHAPE,
) -> bool:
    """Refuse a checkpoint whose obs version or action shape does not match this build.

    Returns ``True`` on a match. Raises :class:`CheckpointError` with a specific message on
    a mismatch — the loader must never silently resume on a stale observation layout or a
    different action space (spec §12: "refused with a clear message").
    """
    obs_version = meta.get("obs_version")
    if obs_version != expected_obs_version:
        raise CheckpointError(
            f"checkpoint obs_version mismatch: checkpoint has obs_version={obs_version!r} but "
            f"this build expects OBS_VERSION={expected_obs_version}. The observation layout "
            f"changed — this checkpoint cannot be resumed and must be retrained."
        )

    action_shape = meta.get("action_shape")
    if action_shape is not None:
        got = tuple(int(v) for v in action_shape)
        want = tuple(int(v) for v in expected_action_shape)
        if got != want:
            raise CheckpointError(
                f"checkpoint action_shape mismatch: checkpoint has action_shape={got} but this "
                f"build expects {want}. The action space changed — this checkpoint cannot be "
                f"resumed."
            )
    return True


def _rmtree(p: Path) -> None:
    """Recursively remove a directory tree (best-effort, used for atomic swap)."""
    import shutil

    shutil.rmtree(p, ignore_errors=True)
