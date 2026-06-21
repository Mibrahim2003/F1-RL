"""Phase 5 multi-agent env contract (spec §c, §1d, §2 business logic).

Written from the public interfaces and the spec contracts: ``RacingParallelEnv`` is a
PettingZoo ``ParallelEnv`` of N homogeneous cars on one shared circuit, the observation is
**unchanged** (ObservationV2, length 22, ``OBS_VERSION = 2``, no nearby-car block), each car has
its **own** lap timer / state / done flag, the per-episode circuit draw is shared by the field
and reproducible from the seed, and both reset modes place distinct seeded starts. A non-positive
``n_agents`` and an unbuilt circuit are refused.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from f1rl.env.observations import OBS_DIM, OBS_VERSION
from f1rl.env.single_agent import RacingEnv
from f1rl.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
TRACKS_DIR = REPO_ROOT / "data" / "tracks"

pytestmark = pytest.mark.skipif(
    not (TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)

_HAS_MONZA = (TRACKS_DIR / "monza.npz").exists()


def _cfg(overrides=None):
    base = ["track_id=red_bull_ring"]
    return load_config("default", overrides=base + (overrides or []), config_root=CONFIG_ROOT)


def _make(n_agents, seed=0, overrides=None):
    from f1rl.env.multi_agent import RacingParallelEnv

    return RacingParallelEnv(_cfg(overrides), n_agents=n_agents, seed=seed)


# ----- API conformance + unchanged spaces ----------------------------------------------


def test_passes_parallel_api_test():
    from pettingzoo.test import parallel_api_test

    env = _make(3, seed=0)
    parallel_api_test(env, num_cycles=30)


def test_spaces_are_the_unchanged_boxes():
    env = _make(4)
    obs_space = env.observation_space("car_0")
    act_space = env.action_space("car_0")
    assert obs_space.shape == (OBS_DIM,)
    assert OBS_DIM == 22 and OBS_VERSION == 2  # no nearby-car block this phase
    assert act_space.shape == (2,)
    assert np.all(act_space.low == -1.0) and np.all(act_space.high == 1.0)
    # Homogeneous: every agent has the identical observation/action space.
    assert env.observation_space("car_3").shape == obs_space.shape
    assert env.action_space("car_3").shape == act_space.shape


def test_reset_returns_per_agent_obs_in_space():
    env = _make(4, seed=1)
    obs, infos = env.reset(seed=1)
    assert set(obs) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    for a, o in obs.items():
        assert o.shape == (OBS_DIM,)
        assert env.observation_space(a).contains(o), a


# ----- the observation lock (per-agent obs == single-agent obs for the same state) ------


def test_per_agent_obs_equals_single_agent_obs_for_same_state():
    cfg = _cfg()
    single = RacingEnv(cfg, seed=7)
    o_single, _ = single.reset(seed=7, options={"start_index": 100})

    field = _make(1, seed=7)
    o_field, _ = field.reset(seed=7, options={"start_indices": [100]})

    # Same circuit, same start sample, same start speed/compound -> byte-for-byte the same obs,
    # and only the 22 track-relative features (no nearby-car data).
    np.testing.assert_allclose(o_field["car_0"], o_single, rtol=1e-6, atol=1e-6)
    assert o_field["car_0"].shape == (OBS_DIM,)


# ----- per-car independence (the shared-timer trap) -------------------------------------


def test_each_car_has_its_own_lap_timer():
    env = _make(4, seed=2)
    env.reset(seed=2)
    timers = [env._cars[a].lap_timer for a in env.possible_agents]
    assert len({id(t) for t in timers}) == 4, "each car must own a distinct LapTimer instance"


def test_one_car_lap_does_not_advance_anothers_lap_state():
    env = _make(3, seed=2)
    env.reset(seed=2)
    # Force car_0 to "complete" laps; the others' lap state must be untouched.
    env._cars["car_0"].lap_timer.completed_laps = 5
    assert env._cars["car_1"].lap_timer.completed_laps == 0
    assert env._cars["car_2"].lap_timer.completed_laps == 0


def test_cars_have_independent_state_and_info():
    env = _make(2, seed=3)
    obs, _ = env.reset(seed=3, options={"start_indices": [10, 800]})
    actions = {
        "car_0": np.array([1.0, 1.0], dtype=np.float32),
        "car_1": np.array([-1.0, 1.0], dtype=np.float32),
    }
    obs, rewards, terms, truncs, infos = env.step(actions)
    # Distinct starts + distinct actions -> distinct per-car progress and observations.
    assert infos["car_0"]["progress"] != infos["car_1"]["progress"]
    assert not np.allclose(obs["car_0"], obs["car_1"])
    assert set(rewards) == set(terms) == set(truncs) == {"car_0", "car_1"}


def test_one_car_failing_does_not_end_the_others():
    # No steering + full throttle drives each car off its (curved) circuit; cars starting at
    # different samples leave at different steps, so the field empties gradually — proving the
    # per-car done flags are independent (one car's failure does not terminate the others).
    env = _make(6, seed=4)
    env.reset(seed=4)
    n = len(env.possible_agents)
    saw_partial_field = False
    for _ in range(800):
        if not env.agents:
            break
        acts = {a: np.array([0.0, 1.0], dtype=np.float32) for a in env.agents}
        env.step(acts)
        if 0 < len(env.agents) < n:
            saw_partial_field = True
    assert saw_partial_field, "some cars must finish/fail while others keep going"


# ----- reset modes ----------------------------------------------------------------------


def test_grid_mode_places_distinct_non_overlapping_slots():
    env = _make(6, seed=1)
    env.grid = replace(env.grid, reset_mode="grid")
    env.reset(seed=1)
    pts = {
        (round(env._cars[a].state.x, 2), round(env._cars[a].state.y, 2))
        for a in env.possible_agents
    }
    assert len(pts) == 6, "grid slots must be distinct / non-overlapping"


def test_scattered_mode_is_distinct_and_reproducible():
    e1 = _make(5, seed=9)
    o1, _ = e1.reset(seed=9)
    e2 = _make(5, seed=9)
    o2, _ = e2.reset(seed=9)
    # Reproducible from the seed: the same field places identically.
    for a in e1.possible_agents:
        np.testing.assert_allclose(o1[a], o2[a])
    # Distinct seeded indices (no two cars share a start sample).
    idxs = [e1._cars[a].grip_idx for a in e1.possible_agents]
    assert len(set(idxs)) == len(idxs)


# ----- per-episode circuit draw (shared by the field, reproducible) ---------------------


@pytest.mark.skipif(not _HAS_MONZA, reason="cached track 'monza' not found")
def test_circuit_draw_is_shared_by_the_field_and_reproducible():
    overrides = ["circuits.pool=[red_bull_ring,monza]"]
    e1 = _make(3, seed=5, overrides=overrides)
    _, i1 = e1.reset(seed=5)
    cids = {i1[a]["circuit_id"] for a in e1.possible_agents}
    assert len(cids) == 1, "the whole field shares one drawn circuit"

    e2 = _make(3, seed=5, overrides=overrides)
    _, i2 = e2.reset(seed=5)
    assert i2["car_0"]["circuit_id"] == i1["car_0"]["circuit_id"], "draw reproducible from seed"


# ----- error states ---------------------------------------------------------------------


def test_non_positive_n_agents_refused():
    with pytest.raises(ValueError):
        _make(0)


def test_unbuilt_circuit_refused():
    with pytest.raises(FileNotFoundError):
        _make(2, overrides=["circuits.pool=[definitely_not_a_real_circuit]"])
