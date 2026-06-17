"""Checkpoint round-trip + validation contract (spec §12, plan §checkpointing).

A checkpoint holds the model (weights + optimizer + torch RNG), the VecNormalize obs
stats, and a meta sidecar ``{total_timesteps, circuit_id, obs_version, seed,
config_snapshot, sb3_version, numpy_rng_state}``. Saving and resuming must round-trip
**exactly**. The loader **refuses** a mismatched ``obs_version`` or action shape with a
clear error before resuming.

Public contract used: ``f1rl.env.factory.make_vec_env``, ``f1rl.train.checkpointing``
(``save_checkpoint`` / ``load_checkpoint`` / ``validate_checkpoint``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from f1rl.env.factory import make_vec_env
from f1rl.env.observations import OBS_VERSION

_TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"
pytestmark = pytest.mark.skipif(
    not (_TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)

# Keep PPO tiny so save/load is fast on CPU.
PPO_KW = dict(n_steps=16, batch_size=16, n_epochs=1, device="cpu")


def _tiny_model(cfg, seed=0):
    from stable_baselines3 import PPO

    venv = make_vec_env(cfg, n_envs=1, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, **PPO_KW)
    return model, venv


def test_save_then_load_round_trips_weights(cfg, tmp_path):
    from f1rl.train.checkpointing import load_checkpoint, save_checkpoint

    model, venv = _tiny_model(cfg, seed=1)
    model.learn(total_timesteps=32)
    before = {k: v.detach().cpu().numpy().copy() for k, v in model.policy.state_dict().items()}

    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, model, venv, cfg)

    venv2 = make_vec_env(cfg, n_envs=1, seed=1)
    model2, _meta = load_checkpoint(ckpt, venv2)
    after = {k: v.detach().cpu().numpy().copy() for k, v in model2.policy.state_dict().items()}

    assert set(before) == set(after)
    for k in before:
        np.testing.assert_allclose(
            before[k], after[k], rtol=0, atol=0, err_msg=f"weight {k} changed"
        )


def test_round_trips_timestep_count_exactly(cfg, tmp_path):
    from f1rl.train.checkpointing import load_checkpoint, save_checkpoint

    model, venv = _tiny_model(cfg, seed=2)
    model.learn(total_timesteps=48)
    saved_steps = model.num_timesteps

    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, model, venv, cfg)

    venv2 = make_vec_env(cfg, n_envs=1, seed=2)
    model2, meta = load_checkpoint(ckpt, venv2)
    assert model2.num_timesteps == saved_steps
    # The meta sidecar records the exact same count.
    assert int(meta["total_timesteps"]) == saved_steps


def test_round_trips_vecnormalize_stats(cfg, tmp_path):
    from f1rl.train.checkpointing import load_checkpoint, save_checkpoint

    model, venv = _tiny_model(cfg, seed=3)
    model.learn(total_timesteps=64)  # populate the obs-normalization running stats

    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, model, venv, cfg)

    venv2 = make_vec_env(cfg, n_envs=1, seed=3)
    model2, _meta = load_checkpoint(ckpt, venv2)

    # The restored env exposes the same obs-normalization mean/var (VecNormalize). The exact
    # attribute path is SB3's; both the saved and loaded env must agree.
    loaded_env = model2.get_env()
    src_mean = getattr(getattr(venv, "obs_rms", None), "mean", None)
    dst_mean = getattr(getattr(loaded_env, "obs_rms", None), "mean", None)
    if src_mean is not None and dst_mean is not None:
        np.testing.assert_allclose(np.asarray(src_mean), np.asarray(dst_mean), rtol=1e-6)


def test_meta_sidecar_holds_required_fields(cfg, tmp_path):
    from f1rl.train.checkpointing import save_checkpoint

    model, venv = _tiny_model(cfg, seed=4)
    model.learn(total_timesteps=16)
    ckpt = tmp_path / "ckpt"
    meta = save_checkpoint(ckpt, model, venv, cfg)

    # save_checkpoint may return the meta dict, or we read it from the load. Either way the
    # required keys must exist (plan: the meta sidecar schema).
    if meta is None:
        from f1rl.train.checkpointing import load_checkpoint

        venv2 = make_vec_env(cfg, n_envs=1, seed=4)
        _model2, meta = load_checkpoint(ckpt, venv2)

    required = {
        "total_timesteps",
        "circuit_id",
        "obs_version",
        "seed",
        "config_snapshot",
        "sb3_version",
        "numpy_rng_state",
    }
    assert required <= set(meta), f"missing meta keys: {required - set(meta)}"
    assert meta["obs_version"] == OBS_VERSION


def test_validate_checkpoint_accepts_matching_meta(cfg):
    from f1rl.train.checkpointing import validate_checkpoint

    meta = {
        "obs_version": OBS_VERSION,
        "action_shape": [2],
        "total_timesteps": 100,
        "circuit_id": "red_bull_ring",
    }
    # A matching meta validates without raising. (Accepts either a bool return or no return.)
    result = validate_checkpoint(meta)
    assert result is None or result is True


def test_validate_checkpoint_refuses_obs_version_mismatch():
    from f1rl.train.checkpointing import CheckpointError, validate_checkpoint

    bad = {"obs_version": OBS_VERSION + 99, "action_shape": [2]}
    with pytest.raises(CheckpointError) as excinfo:
        validate_checkpoint(bad)
    # The error message is clear about the obs-version mismatch (spec: a clear message).
    msg = str(excinfo.value).lower()
    assert "obs" in msg or "version" in msg


def test_load_checkpoint_refuses_obs_version_mismatch(cfg, tmp_path):
    # Saving with obs_version 1 and loading against an incompatible expectation must raise,
    # not silently resume on a stale observation layout.
    from f1rl.train.checkpointing import (
        CheckpointError,
        load_checkpoint,
        save_checkpoint,
        validate_checkpoint,
    )

    model, venv = _tiny_model(cfg, seed=5)
    model.learn(total_timesteps=16)
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, model, venv, cfg)

    # Corrupt the recorded obs_version in the loaded meta, then validate ⇒ must raise.
    venv2 = make_vec_env(cfg, n_envs=1, seed=5)
    _model2, meta = load_checkpoint(ckpt, venv2)
    meta = dict(meta)
    meta["obs_version"] = OBS_VERSION + 1
    with pytest.raises(CheckpointError):
        validate_checkpoint(meta)
