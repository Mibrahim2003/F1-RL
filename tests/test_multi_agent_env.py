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
    assert OBS_DIM == 42 and OBS_VERSION == 3  # Phase 6: v2 prefix(22) + K=4 neighbor block
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


# ----- Phase 6: racing (collisions + neighbor block + zero-sum overtake) -----------------

_RACING = [
    "collision.enabled=true",
    "reward.version=3",
    "reward.w_contact=1.0",
    "reward.w_overtake=0.5",
]


def test_parallel_api_test_with_racing_on():
    # The API still conforms with collisions + the neighbor block + reward_v3 all live.
    from pettingzoo.test import parallel_api_test

    env = _make(4, seed=0, overrides=_RACING)
    parallel_api_test(env, num_cycles=30)


def test_one_car_field_reproduces_single_agent_with_racing_on():
    # A one-car field cannot collide and holds a constant rank, so reward_v3 reduces to reward_v2
    # and the all-zero neighbor block leaves the obs unchanged: it must match RacingEnv step-for-
    # step on the same start + actions, even with collisions + racing weights enabled.
    cfg = _cfg(_RACING)
    single = RacingEnv(cfg, seed=11)
    o_s, _ = single.reset(seed=11, options={"start_index": 250})

    field = _make(1, seed=11, overrides=_RACING)
    o_f, _ = field.reset(seed=11, options={"start_indices": [250]})
    np.testing.assert_allclose(o_f["car_0"], o_s, rtol=1e-6, atol=1e-6)

    rng = np.random.RandomState(0)
    for _ in range(40):
        act = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        o_s, r_s, term_s, trunc_s, _ = single.step(act)
        o_f, r_f, term_f, trunc_f, _ = field.step({"car_0": act})
        np.testing.assert_allclose(o_f["car_0"], o_s, rtol=1e-5, atol=1e-5)
        assert r_f["car_0"] == pytest.approx(r_s, abs=1e-6)
        assert term_f["car_0"] == term_s and trunc_f["car_0"] == trunc_s
        if term_s or trunc_s:
            break


def test_field_obs_carries_neighbor_block_when_cars_are_close():
    # With >1 car the tail is no longer all-zero for cars within sensing range (grid start packs
    # them together), and the racing info fields are present.
    env = _make(4, seed=1, overrides=_RACING)
    env.grid = replace(env.grid, reset_mode="grid")
    obs, infos = env.reset(seed=1)
    has_neighbors = any(np.any(obs[a][22:] != 0.0) for a in env.possible_agents)
    assert has_neighbors, "a close field has neighbors in its obs tail"
    for a in env.possible_agents:
        assert "race_position" in infos[a]
    # Distinct ranks across the grid (1..N).
    ranks = sorted(infos[a]["race_position"] for a in env.possible_agents)
    assert ranks == list(range(1, len(env.possible_agents) + 1))


def test_step_info_has_racing_fields():
    env = _make(3, seed=2, overrides=_RACING)
    env.reset(seed=2)
    acts = {a: np.array([0.0, 1.0], dtype=np.float32) for a in env.agents}
    _o, _r, _t, _tr, infos = env.step(acts)
    for a in env.agents:
        info = infos[a]
        assert "contact" in info and "race_position" in info and "gap_ahead_s" in info
        assert 1 <= info["race_position"] <= 3


def test_crashout_is_opt_in():
    # A hard contact ends a car ONLY when collision.crashout_enabled — the opt-in safety valve.
    from f1rl.env.collisions import ContactRecord
    from f1rl.env.single_agent import finalize_car_step

    over = _RACING + [
        "collision.crashout_enabled=true",
        "collision.crashout_closing_speed_mps=10.0",
    ]
    env = _make(2, seed=3, overrides=over)
    env.reset(seed=3)
    entry, cfg, car = env._entry, env._car_cfg, env._cars["car_0"]

    # A clean step: not terminated for crashout.
    car.contact = ContactRecord()
    _o, _r, term_clean, _tr, info_clean = finalize_car_step(entry, cfg, car)
    assert not (term_clean and info_clean["termination"] == "crashout")

    # A hard contact above the threshold: crashed out.
    car.contact = ContactRecord(impulse=20.0, closing_mps=25.0, count=1)
    _o, reward, term_hard, _tr, info_hard = finalize_car_step(entry, cfg, car)
    assert term_hard and info_hard["termination"] == "crashout"
    assert reward <= cfg.limits.failure_reward + 1.0  # the failure penalty was applied


def test_crashout_off_by_default_keeps_hard_contact_racing():
    from f1rl.env.collisions import ContactRecord
    from f1rl.env.single_agent import finalize_car_step

    env = _make(2, seed=4, overrides=_RACING)  # crashout_enabled defaults false
    env.reset(seed=4)
    entry, cfg, car = env._entry, env._car_cfg, env._cars["car_0"]
    car.contact = ContactRecord(impulse=20.0, closing_mps=25.0, count=1)
    _o, _r, _term, _tr, info = finalize_car_step(entry, cfg, car)
    assert info["termination"] != "crashout"  # penalized, not terminal


# ----- zero-sum overtake / rank helpers (pure functions) --------------------------------


def test_totals_to_ranks_orders_by_progress_descending():
    from f1rl.env.multi_agent import totals_to_ranks

    ranks = totals_to_ranks({"car_0": 10.0, "car_1": 30.0, "car_2": 20.0})
    assert ranks == {"car_1": 1, "car_2": 2, "car_0": 3}


def test_rank_and_overtakes_is_zero_sum_for_a_genuine_swap():
    from f1rl.env.multi_agent import rank_and_overtakes

    # car_0 was ahead (rank 1) but car_1 is now ahead by a hair (within battle range): a swap.
    prev = {"car_0": 1, "car_1": 2}
    totals = {"car_0": 100.0, "car_1": 105.0}
    ranks_now, places = rank_and_overtakes(totals, prev, battle_range_m=20.0)
    assert ranks_now == {"car_1": 1, "car_0": 2}
    assert places["car_1"] == 1 and places["car_0"] == -1  # zero-sum across the pair
    assert places["car_0"] + places["car_1"] == 0


def test_rank_and_overtakes_ignores_far_apart_shuffle():
    from f1rl.env.multi_agent import rank_and_overtakes

    # Order flipped but the cars are 80 m apart (e.g. lapping): not a wheel-to-wheel swap.
    prev = {"car_0": 1, "car_1": 2}
    totals = {"car_0": 100.0, "car_1": 180.0}
    _ranks, places = rank_and_overtakes(totals, prev, battle_range_m=20.0)
    assert places["car_0"] == 0 and places["car_1"] == 0


def test_rank_and_overtakes_no_change_no_places():
    from f1rl.env.multi_agent import rank_and_overtakes

    prev = {"car_0": 1, "car_1": 2}
    totals = {"car_0": 110.0, "car_1": 100.0}  # same order as prev
    _ranks, places = rank_and_overtakes(totals, prev, battle_range_m=20.0)
    assert places == {"car_0": 0, "car_1": 0}
