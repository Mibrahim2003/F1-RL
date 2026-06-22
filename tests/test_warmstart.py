"""Grown input-layer warm start (Phase 6 spec §2b, plan §E).

``OBS_VERSION`` 2 -> 3 means a silent ``--resume`` of a Phase 5 (v2) checkpoint is **refused**
by ``validate_checkpoint`` (covered in ``test_checkpoint.py``); the explicit transplant path is
:func:`f1rl.train.warmstart.grow_policy`. These tests lock its contract from the public signature:

- the transplant produces a v3 policy that drives **exactly like the source on a no-neighbor
  observation** — the source action on ``[0:22]`` equals the grown model's action on
  ``[0:22] ++ zeros(20)`` (the zero-initialized neighbor columns contribute nothing at step one);
- the target ``VecNormalize`` obs stats are grown to the new width (prefix copied, new dims
  mean 0 / var 1);
- a genuinely incompatible source (a different hidden layout, not a clean input-prefix grow) is
  **refused** with a ``CheckpointError``.

A real width-22 source is built by setting ``obs.k_neighbors = 0`` (ObservationV2 width) and the
width-42 target with the default ``k_neighbors = 4`` — the same growth the phase performs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from f1rl.env.factory import make_vec_env
from f1rl.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
TRACKS_DIR = REPO_ROOT / "data" / "tracks"

pytestmark = pytest.mark.skipif(
    not (TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)

PPO_KW = dict(n_steps=32, batch_size=16, n_epochs=1, device="cpu")


def _cfg(overrides):
    base = ["track_id=red_bull_ring"]
    return load_config("default", overrides=base + overrides, config_root=CONFIG_ROOT)


def _src_checkpoint(tmp_path, seed=0, net_arch=None):
    """Train a tiny width-22 (v2-width, k_neighbors=0) PPO and save it; return the ckpt dir."""
    from stable_baselines3 import PPO

    from f1rl.train.checkpointing import save_checkpoint

    src_cfg = _cfg(["obs.k_neighbors=0"])
    venv = make_vec_env(src_cfg, n_envs=1, seed=seed)
    assert venv.observation_space.shape == (22,)  # the Phase 5 width
    kw = dict(PPO_KW)
    policy_kwargs = {"net_arch": net_arch} if net_arch is not None else None
    model = PPO("MlpPolicy", venv, seed=seed, policy_kwargs=policy_kwargs, **kw)
    model.learn(total_timesteps=64)  # populate the obs-normalization running stats
    ckpt = tmp_path / "src_ckpt"
    save_checkpoint(ckpt, model, venv, src_cfg)
    return ckpt


def test_grown_warm_start_reproduces_source_driver(tmp_path):
    # The load-bearing property: the grown v3 policy drives like the Phase 5 policy on a
    # no-neighbor observation (same action on [0:22] ++ zeros).
    from stable_baselines3 import PPO

    from f1rl.train.warmstart import grow_policy

    src_ckpt = _src_checkpoint(tmp_path, seed=1)
    src_model = PPO.load(str(src_ckpt / "model.zip"), device="cpu")

    target_cfg = _cfg(["obs.k_neighbors=4"])
    target_venv = make_vec_env(target_cfg, n_envs=1, seed=1)
    assert target_venv.observation_space.shape == (42,)

    model, _meta = grow_policy(src_ckpt, target_venv, target_cfg, seed=1)

    rng = np.random.RandomState(0)
    for _ in range(5):
        o22 = rng.randn(22).astype(np.float32)
        o42 = np.concatenate([o22, np.zeros(20, dtype=np.float32)])
        a_src, _ = src_model.predict(o22[None], deterministic=True)
        a_tgt, _ = model.predict(o42[None], deterministic=True)
        np.testing.assert_allclose(a_src, a_tgt, atol=1e-5)


def test_grown_warm_start_grows_vecnormalize(tmp_path):
    from f1rl.train.warmstart import grow_policy

    src_ckpt = _src_checkpoint(tmp_path, seed=2)
    target_cfg = _cfg(["obs.k_neighbors=4"])
    target_venv = make_vec_env(target_cfg, n_envs=1, seed=2)

    grow_policy(src_ckpt, target_venv, target_cfg, seed=2)

    rms = target_venv.obs_rms
    assert np.asarray(rms.mean).shape == (42,)
    # The new neighbor dims start mean 0 / var 1 (consistent with the zero-init weight columns).
    np.testing.assert_allclose(np.asarray(rms.mean)[22:], 0.0, atol=0)
    np.testing.assert_allclose(np.asarray(rms.var)[22:], 1.0, atol=0)


def test_grown_warm_start_round_trips_v3(tmp_path):
    # The grown model saves as a normal v3 checkpoint and re-loads (resumes) without surgery.
    from f1rl.env.observations import OBS_VERSION
    from f1rl.train.checkpointing import load_checkpoint, save_checkpoint
    from f1rl.train.warmstart import grow_policy

    src_ckpt = _src_checkpoint(tmp_path, seed=3)
    target_cfg = _cfg(["obs.k_neighbors=4"])
    target_venv = make_vec_env(target_cfg, n_envs=1, seed=3)
    model, _ = grow_policy(src_ckpt, target_venv, target_cfg, seed=3)

    v3_ckpt = tmp_path / "v3_ckpt"
    meta = save_checkpoint(v3_ckpt, model, target_venv, target_cfg)
    assert meta["obs_version"] == OBS_VERSION == 3

    venv2 = make_vec_env(target_cfg, n_envs=1, seed=3)
    model2, meta2 = load_checkpoint(v3_ckpt, venv2)  # validate=True: v3 -> v3 resumes
    assert model2.observation_space.shape == (42,)
    assert meta2["obs_version"] == 3


def test_incompatible_source_is_refused(tmp_path):
    # A source with a different hidden layout is not a clean input-prefix grow => refused.
    from f1rl.train.checkpointing import CheckpointError
    from f1rl.train.warmstart import grow_policy

    src_ckpt = _src_checkpoint(tmp_path, seed=4, net_arch=[32])  # default target is (64, 64)
    target_cfg = _cfg(["obs.k_neighbors=4"])
    target_venv = make_vec_env(target_cfg, n_envs=1, seed=4)
    with pytest.raises(CheckpointError):
        grow_policy(src_ckpt, target_venv, target_cfg, seed=4)
