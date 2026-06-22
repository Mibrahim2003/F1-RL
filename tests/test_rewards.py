"""RewardV1 contract (spec §9, plan §A rewards).

Principle (the rule, restated): reward forward progress and speed, penalize leaving the
track and contact, **never reward centerline proximity**. Version 1 per step::

    reward = w_progress * ds - w_offtrack * offtrack_penalty(off) - w_step - w_reverse * max(0, -ds)

Public contract: ``RewardWeights.from_config(cfg)`` and
``reward_v1(prev_s, cur_s, off_track_m, length, ...) -> (float, terms: dict)``. ``ds`` is
wrap-aware signed arc-length progress; ``off`` is meters past the asphalt edge (0 on asphalt).
These tests assert the *direction* of each term, not its magnitude (the weights are config).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from f1rl.env.collisions import ContactRecord
from f1rl.env.rewards import RewardWeights, reward_v1, reward_v2, reward_v3

_TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"
pytestmark = pytest.mark.skipif(
    not (_TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)

# A representative closed-loop length (meters); the wrap-aware ds uses it. red_bull_ring ~4318 m.
LENGTH = 4318.0


def _weights(cfg):
    return RewardWeights.from_config(cfg)


def _reward(weights, prev_s, cur_s, off_track_m):
    """Call reward_v1 with the fixed positional contract; return (float, terms)."""
    r, terms = reward_v1(prev_s, cur_s, off_track_m, LENGTH, weights)
    return float(r), terms


def test_reward_rises_with_forward_progress(cfg):
    w = _weights(cfg)
    # More forward arc-length progress this step ⇒ strictly higher reward, all else equal.
    small, _ = _reward(w, prev_s=100.0, cur_s=101.0, off_track_m=0.0)
    big, _ = _reward(w, prev_s=100.0, cur_s=110.0, off_track_m=0.0)
    assert big > small


def test_no_progress_loses_to_progress(cfg):
    w = _weights(cfg)
    moving, _ = _reward(w, prev_s=100.0, cur_s=105.0, off_track_m=0.0)
    still, _ = _reward(w, prev_s=100.0, cur_s=100.0, off_track_m=0.0)
    assert moving > still


def test_off_track_is_penalized(cfg):
    w = _weights(cfg)
    # Same forward progress, but off the asphalt ⇒ lower reward than the on-asphalt case.
    on, _ = _reward(w, prev_s=100.0, cur_s=103.0, off_track_m=0.0)
    off, _ = _reward(w, prev_s=100.0, cur_s=103.0, off_track_m=3.0)
    assert off < on


def test_off_track_penalty_is_graded(cfg):
    w = _weights(cfg)
    # Penalty grows with distance past the edge (graded off-track penalty, spec §9).
    near, _ = _reward(w, prev_s=100.0, cur_s=103.0, off_track_m=1.0)
    far, _ = _reward(w, prev_s=100.0, cur_s=103.0, off_track_m=8.0)
    assert far < near


def test_on_asphalt_off_zero_no_offtrack_penalty(cfg):
    # off=0 on asphalt ⇒ the off-track term contributes nothing. The reward at off=0 equals
    # progress minus the (off-independent) step/reverse terms; adding any off>0 only lowers it.
    w = _weights(cfg)
    base, terms = _reward(w, prev_s=100.0, cur_s=103.0, off_track_m=0.0)
    worse, _ = _reward(w, prev_s=100.0, cur_s=103.0, off_track_m=2.0)
    assert worse <= base
    # If the term breakdown is exposed, the off-track component is exactly zero on asphalt.
    if isinstance(terms, dict):
        for key, val in terms.items():
            if "off" in key.lower():
                assert float(val) == pytest.approx(0.0, abs=1e-9)


def test_reverse_is_penalized(cfg):
    w = _weights(cfg)
    # Going backward (cur_s < prev_s ⇒ ds < 0) is penalized below standing still.
    still, _ = _reward(w, prev_s=100.0, cur_s=100.0, off_track_m=0.0)
    reverse, _ = _reward(w, prev_s=100.0, cur_s=95.0, off_track_m=0.0)
    assert reverse < still


def test_reverse_worse_the_further_back(cfg):
    w = _weights(cfg)
    a_bit, _ = _reward(w, prev_s=100.0, cur_s=98.0, off_track_m=0.0)
    a_lot, _ = _reward(w, prev_s=100.0, cur_s=90.0, off_track_m=0.0)
    assert a_lot < a_bit


def test_step_penalty_present_when_idle(cfg):
    # A small constant per-step penalty discourages dawdling (spec §9): with zero progress,
    # zero off-track, the reward should be <= 0 (only the step penalty remains).
    w = _weights(cfg)
    r, _ = _reward(w, prev_s=100.0, cur_s=100.0, off_track_m=0.0)
    assert r <= 0.0


def test_centerline_proximity_is_never_rewarded(cfg):
    # The load-bearing rule (spec §9, §1 non-goals, CLAUDE.md): lateral offset never enters
    # the reward. reward_v1's signature carries no lateral-offset argument, and the reward for
    # equal progress is identical regardless of where on the track width the car sits — which
    # the absence of a lateral input already guarantees. Assert the contract has no such term.
    import inspect

    sig = inspect.signature(reward_v1)
    param_names = " ".join(sig.parameters).lower()
    assert "lateral" not in param_names
    assert "offset" not in param_names
    assert "center" not in param_names

    # And the term breakdown (if present) must not contain a centerline-proximity reward.
    w = _weights(cfg)
    _, terms = _reward(w, prev_s=100.0, cur_s=103.0, off_track_m=0.0)
    if isinstance(terms, dict):
        for key in terms:
            k = key.lower()
            assert "lateral" not in k
            assert "centerline" not in k
            assert "center" not in k


# ----- Phase 6: reward_v3 = reward_v2 core + graded contact + zero-sum overtake ----------

import dataclasses  # noqa: E402


def _v3w(cfg, **over):
    return dataclasses.replace(RewardWeights.from_config(cfg), **over)


def test_reward_v3_reduces_to_reward_v2_with_no_contact_no_places(cfg):
    # The load-bearing reduction: no contact, constant rank, no gap shaping => byte-identical to
    # reward_v2 (so the single-agent path and a one-car field are unchanged).
    w = _v3w(cfg, w_contact=2.0, w_overtake=1.0)  # weights set but no event fires
    r2, t2 = reward_v2(100.0, 104.0, 0.0, LENGTH, w, slip=0.1)
    r3, t3 = reward_v3(
        100.0, 104.0, 0.0, LENGTH, w, slip=0.1, contact=ContactRecord(), places=0, gap_delta=0.0
    )
    assert r3 == pytest.approx(r2, abs=1e-12)
    assert t3["contact"] == 0.0 and t3["overtake"] == 0.0 and t3["gap"] == 0.0


def test_reward_v3_none_contact_equals_empty_contact(cfg):
    w = _v3w(cfg, w_contact=2.0)
    a, _ = reward_v3(100.0, 104.0, 0.0, LENGTH, w, contact=None)
    b, _ = reward_v3(100.0, 104.0, 0.0, LENGTH, w, contact=ContactRecord())
    assert a == pytest.approx(b)


def test_contact_subtracts_penalty_scaling_with_closing_speed(cfg):
    w = _v3w(cfg, w_contact=2.0, contact_soft_mps=5.0, contact_exp=2.0)
    clean, _ = reward_v3(100.0, 104.0, 0.0, LENGTH, w, contact=ContactRecord())
    light, _ = reward_v3(
        100.0, 104.0, 0.0, LENGTH, w, contact=ContactRecord(impulse=1.0, closing_mps=4.0, count=1)
    )
    hard, _ = reward_v3(
        100.0, 104.0, 0.0, LENGTH, w, contact=ContactRecord(impulse=8.0, closing_mps=20.0, count=1)
    )
    assert light < clean  # any contact costs
    assert hard < light  # a harder hit costs more (graded by closing speed)


def test_overtake_term_is_signed_and_zero_sum(cfg):
    w = _v3w(cfg, w_overtake=0.5)
    base, _ = reward_v3(100.0, 104.0, 0.0, LENGTH, w, places=0)
    gained, tg = reward_v3(100.0, 104.0, 0.0, LENGTH, w, places=1)
    lost, tl = reward_v3(100.0, 104.0, 0.0, LENGTH, w, places=-1)
    # +w_overtake per place gained, -w_overtake per place lost.
    assert gained - base == pytest.approx(0.5, abs=1e-9)
    assert lost - base == pytest.approx(-0.5, abs=1e-9)
    # Zero-sum across the swapping pair: +1 for the car ahead, -1 for the other.
    assert tg["overtake"] + tl["overtake"] == pytest.approx(0.0, abs=1e-9)


def test_gap_term_is_opt_in_zero_by_default(cfg):
    w = _v3w(cfg)  # w_gap defaults to 0
    with_gap, t = reward_v3(100.0, 104.0, 0.0, LENGTH, w, gap_delta=12.3)
    without, _ = reward_v3(100.0, 104.0, 0.0, LENGTH, w, gap_delta=0.0)
    assert with_gap == pytest.approx(without)
    assert t["gap"] == 0.0


def test_reward_v3_has_no_lateral_or_blame_term(cfg):
    # Still never centerline-seeking; never hand-codes blame (w_contact_fault is reserved, not a
    # per-car fault signal). The signature carries no lateral/offset input.
    import inspect

    names = " ".join(inspect.signature(reward_v3).parameters).lower()
    assert "lateral" not in names and "offset" not in names and "center" not in names
    _, terms = reward_v3(100.0, 104.0, 0.0, LENGTH, _v3w(cfg), contact=ContactRecord(), places=1)
    for key in terms:
        assert "lateral" not in key.lower() and "center" not in key.lower()
