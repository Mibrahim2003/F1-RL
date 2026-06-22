"""Env contract: passes the Gymnasium checker, obs in space, action shape/bounds.

Maps to spec §c ("the env API check") and §10 ("Must pass
``gymnasium.utils.env_checker.check_env``. A test enforces this."). Built from the public
contract only: ``RacingEnv(gymnasium.Env)`` with ``reset(seed) -> (obs, info)`` and
``step(action) -> (obs, reward, terminated, truncated, info)``, action ``Box(-1,+1,(2,))``.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from f1rl.env.factory import make_env
from f1rl.env.observations import OBS_DIM
from f1rl.env.single_agent import RacingEnv

_TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"
pytestmark = pytest.mark.skipif(
    not (_TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)


def test_make_env_returns_racing_env(cfg):
    env = make_env(cfg, seed=0, rank=0)
    assert isinstance(env, gym.Env)
    assert isinstance(env, RacingEnv)


def test_passes_gymnasium_env_checker(cfg):
    # The load-bearing acceptance test for the env contract (spec §10).
    env = make_env(cfg, seed=0, rank=0)
    # skip_render_check: the env never renders (rendering is offline-only, spec §11).
    check_env(env.unwrapped, skip_render_check=True)


def test_action_space_shape_and_bounds(cfg):
    env = make_env(cfg, seed=0, rank=0)
    space = env.action_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (2,)
    assert np.all(space.low == -1.0)
    assert np.all(space.high == 1.0)


def test_observation_space_is_box_of_obs_dim(cfg):
    env = make_env(cfg, seed=0, rank=0)
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (OBS_DIM,)
    assert OBS_DIM == 42  # ObservationV3 (Phase 6): v2 prefix(22) + K=4 neighbor block(4*5)


def test_dynamic_env_passes_checker(dyn_cfg):
    # The dynamic-physics env (friction circle + grip pipeline) must also pass the checker
    # at obs v2 with the unchanged action space (spec §10, plan §B gate).
    env = RacingEnv(dyn_cfg)
    check_env(env.unwrapped, skip_render_check=True)
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (OBS_DIM,)


def test_dynamic_env_random_rollout_no_nan(dyn_cfg):
    env = RacingEnv(dyn_cfg, seed=3)
    obs, _ = env.reset(seed=3)
    for _ in range(300):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            obs, _ = env.reset()


def test_reset_returns_obs_in_space(cfg):
    env = make_env(cfg, seed=123, rank=0)
    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_returns_five_tuple_and_obs_in_space(cfg):
    env = make_env(cfg, seed=7, rank=0)
    env.reset(seed=7)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float | int | np.floating)
    assert isinstance(terminated, bool | np.bool_)
    assert isinstance(truncated, bool | np.bool_)
    assert isinstance(info, dict)


def test_obs_stays_in_space_under_random_policy(cfg):
    # A random-policy smoke loop must never produce an out-of-space observation
    # (the builder clips so the checker never sees an out-of-space value — plan §A).
    env = make_env(cfg, seed=11, rank=0)
    env.reset(seed=11)
    for _ in range(200):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs), obs
        if terminated or truncated:
            env.reset()


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_per_env_seeding_offsets_rank(cfg, rank):
    # make_env(cfg, seed, rank): per-env seed = base_seed + rank (plan §factory).
    # Construction with distinct ranks must not raise and yields valid resets.
    env = make_env(cfg, seed=100, rank=rank)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
