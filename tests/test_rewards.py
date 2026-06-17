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

from f1rl.env.rewards import RewardWeights, reward_v1

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
