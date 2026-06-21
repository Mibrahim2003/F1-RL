"""Phase 5 self-play vectorization smoke (spec §c integration, §2 business logic).

SuperSuit wraps the field env into an SB3-trainable vector env: a short PPO run completes
without error and the shared-policy output is non-degenerate (this only proves the env is
trainable — no multi-agent learning gain is expected this phase). The SuperSuit-visible agent
width stays **constant** even when cars terminate early (``black_death_v3`` padding).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from f1rl.env.factory import make_selfplay_vec_env
from f1rl.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
TRACKS_DIR = REPO_ROOT / "data" / "tracks"

pytestmark = pytest.mark.skipif(
    not (TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)


def _cfg(overrides=None):
    base = ["track_id=red_bull_ring"]
    return load_config("default", overrides=base + (overrides or []), config_root=CONFIG_ROOT)


def test_supersuit_wraps_into_sb3_trainable_vec_env():
    from stable_baselines3 import PPO

    venv = make_selfplay_vec_env(_cfg(), n_agents=2, n_copies=2, seed=0, num_cpus=0)
    try:
        # The field flattens to one sub-env per car; the spaces are the unchanged single-agent
        # boxes, so one shared policy trains across every car (parameter sharing).
        assert venv.num_envs == 2 * 2
        assert venv.observation_space.shape == (22,)
        assert venv.action_space.shape == (2,)

        model = PPO("MlpPolicy", venv, seed=0, n_steps=64, batch_size=64, n_epochs=1, device="cpu")
        model.learn(total_timesteps=400)

        obs = venv.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert np.all(np.isfinite(action))  # non-degenerate
        assert action.shape == (venv.num_envs, 2)
    finally:
        venv.close()


def test_supersuit_visible_width_is_constant_under_early_death():
    # A tiny step limit makes every car truncate quickly; black_death must re-pad the removed
    # agents so the vectorizer width never changes across the death + auto-reset.
    venv = make_selfplay_vec_env(
        _cfg(["env.max_steps=5"]), n_agents=3, n_copies=1, seed=0, num_cpus=0
    )
    try:
        width = venv.num_envs
        assert width == 3
        obs = venv.reset()
        assert obs.shape == (width, 22)
        actions = np.stack([venv.action_space.sample() for _ in range(width)])
        for _ in range(15):  # crosses the 5-step truncation and the auto-reset
            obs, rewards, dones, infos = venv.step(actions)
            assert obs.shape == (width, 22)
            assert len(rewards) == width and len(dones) == width
    finally:
        venv.close()
