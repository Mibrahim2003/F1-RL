"""Smoke training integration (spec §c integration tests, plan Step E + Step C gate).

A short smoke run on a tiny budget must: (a) run ``learn()`` without crashing, (b) produce a
non-degenerate reward signal (finite returns, the rollout actually collects reward), and
(c) round-trip through a checkpoint so ``--resume`` continues the timestep count.

Kept intentionally tiny (a few hundred steps, 1 env, MlpPolicy on CPU) so it runs in CI on a
laptop. Built from the public factory + checkpointing contract — no train.py internals.
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


def test_learn_runs_on_tiny_budget(cfg):
    model, _venv = _ppo(cfg, seed=0)
    # The load-bearing smoke assertion: learn() completes without raising on a tiny budget.
    model.learn(total_timesteps=SMOKE_STEPS)
    assert model.num_timesteps >= SMOKE_STEPS


def test_reward_signal_is_non_degenerate(cfg):
    # Roll out the policy and confirm the env produces finite, varied rewards (the learning
    # signal is alive). We don't require the curve to go *up* in 256 steps — too noisy — only
    # that the reward is a real, non-constant, finite signal the optimizer can use.
    venv = make_vec_env(cfg, n_envs=1, seed=1)
    obs = venv.reset()
    rewards = []
    for _ in range(SMOKE_STEPS):
        action = np.stack([venv.action_space.sample()])
        obs, reward, done, _info = venv.step(action)
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
    rewards = np.array(rewards)
    assert np.all(np.isfinite(rewards))
    assert rewards.std() > 0.0  # not a constant/degenerate signal


def test_checkpoint_resume_continues_timestep_count(cfg, tmp_path):
    from f1rl.train.checkpointing import load_checkpoint, save_checkpoint

    model, venv = _ppo(cfg, seed=2)
    model.learn(total_timesteps=SMOKE_STEPS)
    first = model.num_timesteps

    ckpt = tmp_path / "smoke_ckpt"
    save_checkpoint(ckpt, model, venv, cfg)

    # Resume: load the checkpoint and continue training. The timestep count must continue
    # from where it stopped, not reset to zero (reset_num_timesteps=False semantics, spec §12).
    venv2 = make_vec_env(cfg, n_envs=1, seed=2)
    model2, meta = load_checkpoint(ckpt, venv2)
    assert model2.num_timesteps == first

    model2.learn(total_timesteps=SMOKE_STEPS, reset_num_timesteps=False)
    assert model2.num_timesteps > first
    assert model2.num_timesteps >= first + SMOKE_STEPS - PPO_KW["n_steps"]


def test_smoke_run_then_checkpoint_meta_records_progress(cfg, tmp_path):
    # After a smoke run, the saved meta records the run's total timesteps and circuit — the
    # minimum needed to resume on the right circuit (spec §12 / plan meta sidecar).
    from f1rl.train.checkpointing import load_checkpoint, save_checkpoint

    model, venv = _ppo(cfg, seed=3)
    model.learn(total_timesteps=SMOKE_STEPS)
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, model, venv, cfg)

    venv2 = make_vec_env(cfg, n_envs=1, seed=3)
    _model2, meta = load_checkpoint(ckpt, venv2)
    assert int(meta["total_timesteps"]) >= SMOKE_STEPS
    assert str(meta["circuit_id"]) == "red_bull_ring"
