"""Deterministic seeding contract (spec §10 determinism, §1d, §c).

"One seeding utility seeds Python, NumPy, and PyTorch. Record the seed with every run."
A fixed seed must give a reproducible rollout: same reset, same step trajectory under the
same action sequence. Built from the public ``RacingEnv`` reset/step contract via the factory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from f1rl.env.factory import make_env
from f1rl.utils.seeding import seed_everything

_TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"
pytestmark = pytest.mark.skipif(
    not (_TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)


def _rollout(cfg, seed, actions):
    seed_everything(seed)
    env = make_env(cfg, seed=seed, rank=0)
    obs0, _ = env.reset(seed=seed)
    obss = [np.asarray(obs0, dtype=np.float64).copy()]
    rewards = []
    for a in actions:
        obs, reward, terminated, truncated, _ = env.step(a)
        obss.append(np.asarray(obs, dtype=np.float64).copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return np.array(obss), np.array(rewards)


def _fixed_actions(n=120, seed=99):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(n)]


def test_same_seed_identical_reset(cfg):
    seed_everything(7)
    env_a = make_env(cfg, seed=7, rank=0)
    obs_a, _ = env_a.reset(seed=7)

    seed_everything(7)
    env_b = make_env(cfg, seed=7, rank=0)
    obs_b, _ = env_b.reset(seed=7)

    np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))


def test_same_seed_identical_rollout(cfg):
    actions = _fixed_actions()
    obss_a, rewards_a = _rollout(cfg, seed=2024, actions=actions)
    obss_b, rewards_b = _rollout(cfg, seed=2024, actions=actions)

    assert obss_a.shape == obss_b.shape
    np.testing.assert_array_equal(obss_a, obss_b)
    np.testing.assert_array_equal(rewards_a, rewards_b)


def test_different_seed_differs_somewhere(cfg):
    # Start-state randomization means a different seed should generally produce a different
    # rollout (resets at a different centerline point). Not strictly required by the contract,
    # but a sanity check that the seed actually feeds the env's randomization.
    actions = _fixed_actions()
    obss_a, _ = _rollout(cfg, seed=1, actions=actions)
    obss_b, _ = _rollout(cfg, seed=2, actions=actions)
    n = min(len(obss_a), len(obss_b))
    assert not np.array_equal(obss_a[:n], obss_b[:n])
