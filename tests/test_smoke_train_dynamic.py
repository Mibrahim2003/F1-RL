"""Smoke training on the DYNAMIC physics (spec §c integration, plan Step E + Step C gate).

The Phase-3b analogue of ``test_smoke_train``: a tiny-budget ``learn()`` on the dynamic model
(friction circle + grip pipeline, obs v2, reward v2) must run, produce a non-degenerate reward
signal, and round-trip through a checkpoint that continues the timestep count. Built from the
public factory + checkpointing contract on the real ``rbr_dynamic`` config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from f1rl.env.factory import make_vec_env

_TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"
pytestmark = pytest.mark.skipif(
    not (_TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)

PPO_KW = dict(n_steps=64, batch_size=32, n_epochs=2, device="cpu")
SMOKE_STEPS = 256


def _ppo(cfg, seed=0):
    from stable_baselines3 import PPO

    venv = make_vec_env(cfg, n_envs=1, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, **PPO_KW)
    return model, venv


def test_learn_runs_on_dynamic_physics(dyn_cfg):
    model, _venv = _ppo(dyn_cfg, seed=0)
    model.learn(total_timesteps=SMOKE_STEPS)
    assert model.num_timesteps >= SMOKE_STEPS


def test_reward_signal_non_degenerate_on_dynamic(dyn_cfg):
    venv = make_vec_env(dyn_cfg, n_envs=1, seed=1)
    venv.reset()
    rewards = []
    for _ in range(SMOKE_STEPS):
        action = np.stack([venv.action_space.sample()])
        _obs, reward, _done, _info = venv.step(action)
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
    rewards = np.array(rewards)
    assert np.all(np.isfinite(rewards))
    assert rewards.std() > 0.0


def test_observation_is_v3_length_through_vecenv(dyn_cfg):
    from f1rl.env.observations import OBS_DIM

    venv = make_vec_env(dyn_cfg, n_envs=1, seed=2)
    obs = venv.reset()
    assert np.asarray(obs).shape[-1] == OBS_DIM == 42


def test_checkpoint_resume_continues_on_dynamic(dyn_cfg, tmp_path):
    from f1rl.train.checkpointing import load_checkpoint, save_checkpoint

    model, venv = _ppo(dyn_cfg, seed=2)
    model.learn(total_timesteps=SMOKE_STEPS)
    first = model.num_timesteps

    ckpt = tmp_path / "dyn_ckpt"
    save_checkpoint(ckpt, model, venv, dyn_cfg)

    venv2 = make_vec_env(dyn_cfg, n_envs=1, seed=2)
    model2, _meta = load_checkpoint(ckpt, venv2)
    assert model2.num_timesteps == first

    model2.learn(total_timesteps=SMOKE_STEPS, reset_num_timesteps=False)
    assert model2.num_timesteps > first
