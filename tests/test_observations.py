"""ObservationV1 contract (spec §7, plan ObservationV1 table).

Layout (length 15): ``[speed_norm, heading_error, lateral_offset_norm,
curvature_lookahead[5], edge_beam[7]]``. Tests build observations from the public
functions and check the *contract* — shape, bounds, signs, and that the curvature
lookahead reads the upcoming corner — deriving expectations from the public ``Track``
arrays at runtime rather than from any implementation internals.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from f1rl.env.observations import (
    OBS_DIM,
    OBS_VERSION,
    ObsParams,
    build_observation,
    cast_beams,
    observation_space,
    sample_curvature_ahead,
    track_query,
)
from f1rl.physics import CarState

_TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"
pytestmark = pytest.mark.skipif(
    not (_TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)

# Index map for the fixed ObservationV1 layout (plan: contracts fixed before any code).
I_SPEED = 0
I_HEADING = 1
I_LATERAL = 2
I_CURV = slice(3, 8)  # 5 lookahead curvatures
I_BEAMS = slice(8, 15)  # 7 edge-distance beams
# ObservationV2 tail (Phase 3b).
I_WEAR = 15
I_COMPOUND = slice(16, 21)  # 5-way compound one-hot
I_GRIP = 21


def _params(cfg):
    return ObsParams.from_config(cfg)


def _edge_cache(track):
    """Per-track edge cache for the obs builder, or ``None`` to let it build internally.

    ``build_observation(track, state, params, edge_cache)`` accepts an optional precomputed
    ``edge_cache`` (plan signature). If the module exposes a public builder we use it;
    otherwise we pass ``None`` and the builder constructs it. Either path is contract-valid,
    and this keeps the test from assuming the cache's internal structure.
    """
    import f1rl.env.observations as obs_mod

    for name in ("build_edge_cache", "make_edge_cache"):
        fn = getattr(obs_mod, name, None)
        if callable(fn):
            return fn(track)
    return None


def _on_centerline_state(track, idx, speed=30.0):
    """A car placed exactly on the centerline at sample ``idx``, heading = tangent."""
    p = track.centerline[idx]
    t = track.tangent[idx]
    yaw = math.atan2(float(t[1]), float(t[0]))
    return CarState(x=float(p[0]), y=float(p[1]), yaw=yaw, vx=speed)


# --- shape + bounds ---------------------------------------------------------------------


def test_observation_space_shape_and_dim():
    space = observation_space()
    assert space.shape == (OBS_DIM,)
    assert OBS_DIM == 22
    assert OBS_VERSION == 2


def test_build_observation_shape_22(track, cfg):
    params = _params(cfg)
    state = _on_centerline_state(track, idx=0)
    obs = build_observation(track, state, params, _edge_cache(track))
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (22,)
    assert obs.dtype.kind == "f"


# --- ObservationV2 tail: wear, compound one-hot, grip indicator (Phase 3b) ----------------


def test_tail_wear_passes_through_and_in_unit_range(track, cfg):
    params = _params(cfg)
    p = track.centerline[0]
    t = track.tangent[0]
    yaw = math.atan2(float(t[1]), float(t[0]))
    for wear in (0.0, 0.37, 1.0):
        state = CarState(x=float(p[0]), y=float(p[1]), yaw=yaw, vx=30.0, tire_wear=wear)
        obs = build_observation(track, state, params, _edge_cache(track))
        assert obs[I_WEAR] == pytest.approx(wear, abs=1e-6)
        assert 0.0 <= obs[I_WEAR] <= 1.0


def test_tail_compound_one_hot_matches_state(track, cfg):
    params = _params(cfg)
    state = _on_centerline_state(track, idx=0)
    for compound in range(5):
        st = CarState(x=state.x, y=state.y, yaw=state.yaw, vx=20.0, compound=compound)
        obs = build_observation(track, st, params, _edge_cache(track))
        onehot = obs[I_COMPOUND]
        assert onehot.shape == (5,)
        assert onehot.sum() == pytest.approx(1.0)
        assert int(np.argmax(onehot)) == compound
        assert set(np.unique(onehot)).issubset({0.0, 1.0})


def test_tail_grip_indicator_passed_through_and_bounded(track, cfg):
    params = _params(cfg)
    state = _on_centerline_state(track, idx=0)
    obs_lo = build_observation(track, state, params, _edge_cache(track), grip_indicator=0.5)
    obs_hi = build_observation(track, state, params, _edge_cache(track), grip_indicator=1.7)
    assert obs_lo[I_GRIP] == pytest.approx(0.5)
    assert obs_hi[I_GRIP] == pytest.approx(1.7)
    # Out-of-Box values are clipped so the checker never sees them.
    space = observation_space()
    obs_clip = build_observation(track, state, params, _edge_cache(track), grip_indicator=99.0)
    assert space.contains(obs_clip.astype(space.dtype))


def test_v1_slice_unchanged_by_tail(track, cfg):
    # The v1 slice (0-14) must be byte-identical regardless of the Part 2 tail inputs.
    params = _params(cfg)
    base = _on_centerline_state(track, idx=len(track.centerline) // 3, speed=45.0)
    a = build_observation(
        track,
        CarState(x=base.x, y=base.y, yaw=base.yaw, vx=base.vx, tire_wear=0.0, compound=0),
        params,
        _edge_cache(track),
        grip_indicator=1.0,
    )
    b = build_observation(
        track,
        CarState(x=base.x, y=base.y, yaw=base.yaw, vx=base.vx, tire_wear=0.9, compound=3),
        params,
        _edge_cache(track),
        grip_indicator=0.4,
    )
    np.testing.assert_array_equal(a[0:15], b[0:15])


def test_observation_in_bounds_around_the_lap(track, cfg):
    # The builder clips so every component sits inside the declared Box (plan §A).
    params = _params(cfg)
    space = observation_space()
    n = len(track.centerline)
    for idx in range(0, n, max(1, n // 40)):
        state = _on_centerline_state(track, idx, speed=60.0)
        obs = build_observation(track, state, params, _edge_cache(track))
        assert space.contains(obs.astype(space.dtype)), (idx, obs)


# --- heading error ----------------------------------------------------------------------


def test_heading_error_zero_when_aligned_with_tangent(track, cfg):
    # On the centerline, heading == tangent ⇒ heading_error ≈ 0 (spec §7).
    params = _params(cfg)
    # Pick a low-curvature ("straight") sample so the nearest-tangent is unambiguous.
    idx = int(np.argmin(np.abs(track.curvature)))
    state = _on_centerline_state(track, idx)
    obs = build_observation(track, state, params, _edge_cache(track))
    assert obs[I_HEADING] == pytest.approx(0.0, abs=1e-3)


def test_heading_error_sign_follows_yaw_offset(track, cfg):
    # Rotating the car CCW from the tangent gives a positive heading error; CW negative.
    # (heading_error = wrap(yaw - tangent_angle), normalized by π — monotone in the offset.)
    params = _params(cfg)
    idx = int(np.argmin(np.abs(track.curvature)))
    base = _on_centerline_state(track, idx)
    plus = build_observation(
        track,
        CarState(x=base.x, y=base.y, yaw=base.yaw + 0.3, vx=base.vx),
        params,
        _edge_cache(track),
    )
    minus = build_observation(
        track,
        CarState(x=base.x, y=base.y, yaw=base.yaw - 0.3, vx=base.vx),
        params,
        _edge_cache(track),
    )
    assert plus[I_HEADING] > 0.0
    assert minus[I_HEADING] < 0.0


# --- lateral offset sign ----------------------------------------------------------------


def test_lateral_offset_zero_on_centerline(track, cfg):
    params = _params(cfg)
    idx = len(track.centerline) // 3
    state = _on_centerline_state(track, idx)
    obs = build_observation(track, state, params, _edge_cache(track))
    assert obs[I_LATERAL] == pytest.approx(0.0, abs=1e-2)


def test_lateral_offset_positive_to_the_left(track, cfg):
    # Track.normal points LEFT of travel (schema). Offsetting along +normal ⇒ positive
    # signed lateral offset; offsetting along -normal ⇒ negative (spec §7 sign convention).
    params = _params(cfg)
    idx = len(track.centerline) // 3
    p = track.centerline[idx]
    nrm = track.normal[idx]
    t = track.tangent[idx]
    yaw = math.atan2(float(t[1]), float(t[0]))
    off = 2.0  # meters, well inside the asphalt half-width

    left = CarState(x=float(p[0] + off * nrm[0]), y=float(p[1] + off * nrm[1]), yaw=yaw, vx=20.0)
    right = CarState(x=float(p[0] - off * nrm[0]), y=float(p[1] - off * nrm[1]), yaw=yaw, vx=20.0)

    obs_left = build_observation(track, left, params, _edge_cache(track))
    obs_right = build_observation(track, right, params, _edge_cache(track))
    assert obs_left[I_LATERAL] > 0.0
    assert obs_right[I_LATERAL] < 0.0


def test_track_query_lateral_sign_matches_left_normal(track, cfg):
    # track_query returns the signed lateral offset directly — assert the same sign rule.
    idx = len(track.centerline) // 2
    p = track.centerline[idx]
    nrm = track.normal[idx]
    t = track.tangent[idx]
    yaw = math.atan2(float(t[1]), float(t[0]))
    x_left = float(p[0] + 1.5 * nrm[0])
    y_left = float(p[1] + 1.5 * nrm[1])
    result = track_query(track, x_left, y_left, yaw)
    # Contract: returns (nearest_idx, s_along, signed_lateral, half_width, heading_error).
    signed_lateral = result[2]
    assert signed_lateral > 0.0


# --- edge beams -------------------------------------------------------------------------


def test_edge_beams_in_unit_range(track, cfg):
    params = _params(cfg)
    n = len(track.centerline)
    for idx in range(0, n, max(1, n // 30)):
        state = _on_centerline_state(track, idx)
        obs = build_observation(track, state, params, _edge_cache(track))
        beams = obs[I_BEAMS]
        assert beams.shape == (7,)
        assert np.all(beams >= 0.0)
        assert np.all(beams <= 1.0)


def test_cast_beams_normalized_and_count(track, cfg):
    params = _params(cfg)
    angles = np.radians([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    idx = len(track.centerline) // 4
    p = track.centerline[idx]
    t = track.tangent[idx]
    yaw = math.atan2(float(t[1]), float(t[0]))
    beam_max = getattr(params, "beam_max", 50.0)
    raw = cast_beams(track, float(p[0]), float(p[1]), yaw, angles, beam_max)
    raw = np.asarray(raw)
    assert raw.shape == (7,)
    # Distances are finite and non-negative (raw meters or already-normalized — both bounded).
    assert np.all(np.isfinite(raw))
    assert np.all(raw >= 0.0)


# --- curvature lookahead ----------------------------------------------------------------


def test_curvature_lookahead_has_five_entries(track, cfg):
    params = _params(cfg)
    state = _on_centerline_state(track, idx=0)
    obs = build_observation(track, state, params, _edge_cache(track))
    assert obs[I_CURV].shape == (5,)


def test_curvature_lookahead_picks_upcoming_corner_sign(track, cfg):
    # Find a clear corner (peak |curvature|), then place the car ~40 m before it on a
    # straighter sample. The lookahead should report the corner's signed curvature ahead,
    # matching the sign of the real curvature at the corner (spec §7: "how the car sees the
    # upcoming corners"). We read curvature from the public Track array — not from the env.
    params = _params(cfg)
    curv = track.curvature
    s = track.s
    corner = int(np.argmax(np.abs(curv)))
    corner_sign = math.copysign(1.0, float(curv[corner]))
    assert abs(curv[corner]) > 1e-3  # the circuit really has a corner

    # A sample roughly 40 m of arc length before the corner.
    target_s = float(s[corner]) - 40.0
    if target_s < 0.0:
        target_s += float(s[-1])
    approach = int(np.argmin(np.abs(s - target_s)))

    state = _on_centerline_state(track, approach)
    obs = build_observation(track, state, params, _edge_cache(track))
    lookahead = obs[I_CURV]

    # At least one of the 5 lookahead samples should see the corner and share its sign,
    # with non-trivial magnitude (the corner is within the {10,25,50,100,150} m horizon).
    same_sign = [v for v in lookahead if math.copysign(1.0, v) == corner_sign and abs(v) > 1e-4]
    assert same_sign, (corner_sign, lookahead.tolist())


def test_sample_curvature_ahead_matches_track_curvature_here(track):
    # A 0 m lookahead at arc length s should report ~the curvature at that point
    # (interpolation contract on track.s / track.curvature, plan §A).
    idx = len(track.centerline) // 2
    s_here = float(track.s[idx])
    val = sample_curvature_ahead(track, s_here, 0.0)
    val = float(np.asarray(val).reshape(-1)[0]) if np.ndim(val) else float(val)
    assert val == pytest.approx(float(track.curvature[idx]), abs=5e-3)


def test_speed_norm_increases_with_speed(track, cfg):
    # speed_norm = vx / ref_speed: monotone in speed, ~0 at rest (spec §7).
    params = _params(cfg)
    idx = len(track.centerline) // 5
    slow = build_observation(
        track, _on_centerline_state(track, idx, speed=5.0), params, _edge_cache(track)
    )
    fast = build_observation(
        track, _on_centerline_state(track, idx, speed=80.0), params, _edge_cache(track)
    )
    rest = build_observation(
        track, _on_centerline_state(track, idx, speed=0.0), params, _edge_cache(track)
    )
    assert rest[I_SPEED] == pytest.approx(0.0, abs=1e-6)
    assert fast[I_SPEED] > slow[I_SPEED] > 0.0
