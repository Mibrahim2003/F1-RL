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
    assert OBS_DIM == 15


def test_build_observation_shape_15(track, cfg):
    params = _params(cfg)
    state = _on_centerline_state(track, idx=0)
    obs = build_observation(track, state, params, _edge_cache(track))
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (15,)
    assert obs.dtype.kind == "f"


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
