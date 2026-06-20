"""Per-episode circuit sampling (Phase 4 spec §2b, §c; plan §A reset change, Risks #1).

Written from the env's public surface (``reset`` returning ``info["circuit_id"]``, the
``set_track_pool`` curriculum hook, the action space) and the spec contract — not from internals.
The load-bearing checks: ``reset`` draws varying circuits, the draw is reproducible from the
seed, the rebind swaps the track/edge_cache/lap_timer/pole together with no stale per-track
state, ``set_track_pool`` changes the active draw set, and a one-circuit pool reproduces the
Phase 3b single-circuit behavior exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gymnasium.utils.env_checker import check_env

from f1rl.env.single_agent import RacingEnv
from f1rl.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
TRACKS_DIR = REPO_ROOT / "data" / "tracks"


def _have(*ids: str) -> bool:
    return all((TRACKS_DIR / f"{i}.npz").exists() for i in ids)


pytestmark = pytest.mark.skipif(
    not _have("red_bull_ring", "monza"),
    reason="cached tracks 'red_bull_ring' + 'monza' required for sampling tests",
)


def _cfg(pool: list[str]) -> object:
    cfg = load_config("default", overrides=["track_id=red_bull_ring"], config_root=CONFIG_ROOT)
    cfg.circuits.pool = list(pool)
    return cfg


def _draw_seq(seed: int, n: int = 20) -> list[str]:
    env = RacingEnv(_cfg(["red_bull_ring", "monza"]), seed=seed)
    _obs, info = env.reset(seed=seed)
    seq = [info["circuit_id"]]
    for _ in range(n - 1):
        _obs, info = env.reset()
        seq.append(info["circuit_id"])
    return seq


def test_reset_draws_varying_circuits():
    env = RacingEnv(_cfg(["red_bull_ring", "monza"]), seed=0)
    seen = {env.reset()[1]["circuit_id"] for _ in range(40)}
    assert seen == {"red_bull_ring", "monza"}  # both drawn over many resets


def test_draw_is_reproducible_from_seed():
    assert _draw_seq(7) == _draw_seq(7)  # same seed => identical circuit sequence
    assert _draw_seq(7) != _draw_seq(8)  # different seed => (almost surely) different sequence


def test_rebind_swaps_all_per_circuit_state_together():
    env = RacingEnv(_cfg(["red_bull_ring", "monza"]), seed=0)
    env.set_track_pool(["red_bull_ring"])
    env.reset()
    assert env.track_id == "red_bull_ring"
    track1, cache1, timer1, pole1 = env.track, env.edge_cache, env.lap_timer, env._pole

    env.set_track_pool(["monza"])
    env.reset()
    assert env.track_id == "monza"
    assert env.track is not track1
    assert env.edge_cache is not cache1
    assert env.lap_timer is not timer1
    assert env._pole != pole1
    assert env.lap_timer.track is env.track  # the active timer matches the active track


def test_set_track_pool_changes_active_draw_set():
    env = RacingEnv(_cfg(["red_bull_ring", "monza"]), seed=1)
    env.set_track_pool(["monza"])
    assert {env.reset()[1]["circuit_id"] for _ in range(10)} == {"monza"}
    env.set_track_pool([])  # widen back to the full configured pool
    assert {env.reset()[1]["circuit_id"] for _ in range(40)} == {"red_bull_ring", "monza"}


def test_one_circuit_pool_reproduces_single_circuit():
    env = RacingEnv(_cfg(["red_bull_ring"]), seed=0)
    for _ in range(10):
        _obs, info = env.reset()
        assert info["circuit_id"] == "red_bull_ring"
        assert info["pole_time_s"] == pytest.approx(64.3)


def test_no_stale_lap_completion_across_swap():
    env = RacingEnv(_cfg(["red_bull_ring", "monza"]), seed=0)
    env.set_track_pool(["red_bull_ring"])
    env.reset()
    for _ in range(3):
        env.step(env.action_space.sample())

    env.set_track_pool(["monza"])
    _obs, info = env.reset()
    assert info["completed_laps"] == 0  # the new circuit starts fresh, no carry-over
    _obs, _r, _term, _trunc, info2 = env.step(env.action_space.sample())
    assert info2["completed_laps"] == 0


def test_pool_env_passes_gymnasium_checker():
    env = RacingEnv(_cfg(["red_bull_ring", "monza"]))
    check_env(env.unwrapped, skip_render_check=True)
    assert env.action_space.shape == (2,)
