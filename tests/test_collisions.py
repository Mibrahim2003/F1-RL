"""Field-level collision pass contract (Phase 6 spec §2 collisions, plan §B).

``resolve_collisions(states, params)`` is the **one** place cars couple: it reads the
post-physics field, detects two-disc overlap, and applies a snapshot-then-apply contact
response (positional push-apart + a restitution/friction velocity impulse). The contract these
tests lock, all from the public signature + :class:`CollisionParams` (config), not internals:

- overlapping cars are pushed apart (penetration removed at ``push_fraction = 1``);
- a closing pair has its closing speed scrubbed (it does not keep tunnelling through);
- the response is **order-independent** — any agent order / shuffled ids gives identical
  post-step states (snapshot-then-apply);
- a hard hit records a larger impulse + closing speed than a glancing one (graded);
- only the cars passed in are touched (done/removed cars are excluded by not being in the list);
- ``enabled = False`` or fewer than two cars is a no-op.

``PhysicsModel.step`` is never involved here — physics stays the pure single-car function.
SI units (meters, m/s); world frame (x right, y up).
"""

from __future__ import annotations

import math

import pytest

from f1rl.env.collisions import CollisionParams, ContactRecord, resolve_collisions
from f1rl.physics import CarState

# Two-disc geometry the tests reason about explicitly (default radius 1.0, offset 1.25): two
# cars nose-to-tail at yaw 0 first contact when their centers are within 2*offset + 2*radius.
R = 1.0
OFFSET = 1.25
CONTACT_DX = 2.0 * OFFSET + 2.0 * R  # 4.5 m centre-to-centre on the shared axis


def _params(**over) -> CollisionParams:
    base = dict(
        enabled=True,
        disc_radius_m=R,
        disc_offset_m=OFFSET,
        restitution=0.1,
        friction=0.3,
        push_fraction=1.0,
    )
    base.update(over)
    return CollisionParams(**base)


def _car(x, y=0.0, yaw=0.0, vx=0.0, vy=0.0) -> CarState:
    return CarState(x=float(x), y=float(y), yaw=float(yaw), vx=float(vx), vy=float(vy))


def _centre_dist(a: CarState, b: CarState) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


# ----- detection + push-apart -----------------------------------------------------------


def test_overlapping_pair_is_pushed_apart():
    # Two stationary cars overlapping along x (dx < contact distance) are separated.
    a, b = _car(0.0), _car(3.0)  # 3.0 < 4.5 => overlapping
    before = _centre_dist(a, b)
    recs = resolve_collisions([a, b], _params())
    after = _centre_dist(a, b)
    assert after > before  # pushed apart
    # push_fraction = 1 removes the full penetration: they end (just) touching, gap ~ 0.
    assert after == pytest.approx(CONTACT_DX, abs=1e-6)
    assert all(r.count == 1 for r in recs)


def test_non_overlapping_pair_is_untouched():
    a, b = _car(0.0), _car(20.0)  # far apart
    recs = resolve_collisions([a, b], _params())
    assert a.x == 0.0 and b.x == 20.0
    assert all(r.count == 0 and r.impulse == 0.0 for r in recs)


def test_push_apart_is_symmetric_equal_mass():
    # Equal mass => the two cars move by equal and opposite amounts, so their midpoint is fixed.
    a, b = _car(0.0), _car(3.0)
    mid_before = 0.5 * (a.x + b.x)
    resolve_collisions([a, b], _params())
    mid_after = 0.5 * (a.x + b.x)
    assert mid_after == pytest.approx(mid_before, abs=1e-9)


# ----- velocity response: closing speed scrubbed (no tunnelling) ------------------------


def test_closing_pair_loses_closing_speed_and_separates():
    # A behind B on the +x axis, A charging into B; after the pass A must not still be closing
    # at full speed and the pair must be separated (it does not tunnel through in this pass).
    a = _car(0.0, vx=30.0)  # body vx forward; yaw 0 => world +x
    b = _car(3.0, vx=0.0)
    recs = resolve_collisions([a, b], _params())
    # Normal is +x (A -> B). Post closing speed = (a.vx - b.vx) along +x must drop sharply.
    closing_after = a.vx - b.vx
    assert closing_after < 30.0  # closing speed scrubbed
    assert b.vx > 0.0  # momentum transferred forward to the car ahead
    assert _centre_dist(a, b) == pytest.approx(CONTACT_DX, abs=1e-6)  # separated
    assert recs[0].closing_mps == pytest.approx(30.0, rel=1e-6)


def test_restitution_zero_kills_closing_speed():
    # With e = 0 the normal closing speed is fully removed (perfectly inelastic, equal mass =>
    # both end at the average normal velocity, so the relative normal speed is ~0).
    a = _car(0.0, vx=20.0)
    b = _car(3.0, vx=0.0)
    resolve_collisions([a, b], _params(restitution=0.0))
    assert (a.vx - b.vx) == pytest.approx(0.0, abs=1e-6)


# ----- order independence (snapshot-then-apply) -----------------------------------------


def test_response_is_order_independent():
    # A tight 3-car cluster resolved in two different agent orders must yield identical states.
    layout = [(0.0, 0.0, 5.0), (3.0, 0.0, 0.0), (6.0, 0.2, -5.0)]
    cars_a = [_car(x, y, vx=v) for (x, y, v) in layout]
    cars_b = [_car(x, y, vx=v) for (x, y, v) in layout]

    resolve_collisions(cars_a, _params())  # natural order 0,1,2
    # Resolve a shuffled view, then compare per original index.
    order = [2, 0, 1]
    shuffled = [cars_b[i] for i in order]
    resolve_collisions(shuffled, _params())

    for i in range(3):
        assert cars_a[i].x == pytest.approx(cars_b[i].x, abs=1e-9)
        assert cars_a[i].y == pytest.approx(cars_b[i].y, abs=1e-9)
        assert cars_a[i].vx == pytest.approx(cars_b[i].vx, abs=1e-9)
        assert cars_a[i].vy == pytest.approx(cars_b[i].vy, abs=1e-9)


# ----- impulse grading ------------------------------------------------------------------


def test_hard_hit_records_larger_impulse_than_glancing():
    soft_a, soft_b = _car(0.0, vx=4.0), _car(3.0, vx=0.0)
    hard_a, hard_b = _car(0.0, vx=40.0), _car(3.0, vx=0.0)
    soft = resolve_collisions([soft_a, soft_b], _params())
    hard = resolve_collisions([hard_a, hard_b], _params())
    assert hard[0].impulse > soft[0].impulse
    assert hard[0].closing_mps > soft[0].closing_mps


def test_no_closing_no_impulse_but_still_pushed_apart():
    # Overlapping but separating (or static) => no normal impulse, yet still un-overlapped.
    a, b = _car(0.0, vx=-5.0), _car(3.0, vx=5.0)  # moving apart on +x normal
    recs = resolve_collisions([a, b], _params())
    assert recs[0].impulse == 0.0 and recs[0].closing_mps == 0.0
    assert recs[0].count == 1  # contact geometry still recorded
    assert _centre_dist(a, b) == pytest.approx(CONTACT_DX, abs=1e-6)


# ----- dead-car exclusion + disabled / degenerate ---------------------------------------


def test_excluded_car_is_not_touched():
    # A "dead" car (not passed to the pass) is never moved, even sitting on top of a live pair.
    live_a, live_b = _car(0.0), _car(3.0)
    dead = _car(1.5)  # overlaps both, but is excluded from the field list
    resolve_collisions([live_a, live_b], _params())  # dead not passed
    assert dead.x == 1.5 and dead.y == 0.0  # untouched


def test_disabled_is_a_noop():
    a, b = _car(0.0, vx=10.0), _car(3.0)
    recs = resolve_collisions([a, b], _params(enabled=False))
    assert a.x == 0.0 and b.x == 3.0 and a.vx == 10.0
    assert all(r == ContactRecord() for r in recs)


def test_single_car_field_is_a_noop():
    a = _car(0.0, vx=10.0)
    recs = resolve_collisions([a], _params())
    assert a.x == 0.0 and a.vx == 10.0
    assert recs == [ContactRecord()]


def test_yaw_preserved_by_the_pass():
    # Contact-induced spin is a future fidelity upgrade: yaw is left unchanged.
    a, b = _car(0.0, yaw=0.3, vx=20.0), _car(3.0, yaw=-0.2)
    resolve_collisions([a, b], _params())
    assert a.yaw == 0.3 and b.yaw == -0.2


def test_param_validation_rejects_bad_geometry():
    with pytest.raises(ValueError):
        CollisionParams(disc_radius_m=0.0)
    with pytest.raises(ValueError):
        CollisionParams(restitution=1.5)
    with pytest.raises(ValueError):
        CollisionParams(friction=-0.1)


def test_rotated_pair_pushes_along_contact_normal():
    # Two cars overlapping along the y axis (stacked), yaw 0: the push must be along y, not x.
    a, b = _car(0.0, 0.0), _car(0.0, 1.0)  # within 2r on the rear/rear disc pair
    ax0, bx0 = a.x, b.x
    resolve_collisions([a, b], _params())
    assert a.x == pytest.approx(ax0, abs=1e-9) and b.x == pytest.approx(bx0, abs=1e-9)
    assert abs(a.y - b.y) > 1.0  # separated along y
