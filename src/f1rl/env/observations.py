"""ObservationV1 builder — pure NumPy (TECHNICAL_DESIGN.md §7, plan ObservationV1 table).

This module is the single source of the observation the policy sees, and it is reused
**verbatim** by the server's live-policy path. To keep the training stack out of the
server's hot path it imports **no torch and no gymnasium in the heavy path** —
``observation_space()`` imports ``gymnasium.spaces`` lazily so the builder itself stays a
plain NumPy function.

Layout (length 22), fixed for ``OBS_VERSION = 2``::

    [ speed_norm,             # 0    vx / ref_speed
      heading_error,          # 1    wrap(yaw - tangent_angle) / pi
      lateral_offset_norm,    # 2    (signed lateral) / half_width_on_that_side  (input only)
      curvature_lookahead[5], # 3-7  signed curvature at s + {10,25,50,100,150} m, x curvature_scale
      edge_beam[7],           # 8-14 rangefinder distance to asphalt edge / beam_max, clipped [0,1]
      tire_wear,              # 15   state.tire_wear, already [0,1]               (Part 2)
      compound_onehot[5],     # 16-20 one-hot of state.compound (0 soft .. 4 wet) (Part 2)
      grip_indicator ]        # 21   effective grip at the car / mu_base, clipped (Part 2)

The v1 slice (0-14) is **byte-identical** to ``OBS_VERSION = 1``; only the tail is new. Every
tunable (reference speed, lookahead distances, beam angles, beam range, curvature scale) comes
from :class:`ObsParams`, built from config — no magic constant in logic. SI units throughout
(meters, m/s, radians).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from f1rl.track.schema import Track

OBS_VERSION = 2
OBS_DIM = 22
N_COMPOUNDS = 5  # soft, medium, hard, intermediate, wet — one-hot of CarState.compound

# Default obs parameters (all overridable from config under the ``obs:`` block).
_DEFAULT_REF_SPEED = 92.0  # m/s — kinematic top speed; speed_norm = vx / ref_speed
_DEFAULT_LOOKAHEAD_M = (10.0, 25.0, 50.0, 100.0, 150.0)  # 5 curvature-lookahead distances
_DEFAULT_BEAM_ANGLES_DEG = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)  # 7 rangefinder beams
_DEFAULT_BEAM_MAX = 60.0  # m — beam range; distances normalize by this and clip to [0, 1]
_DEFAULT_CURVATURE_SCALE = 50.0  # multiplies signed curvature (1/m) into a bounded range

# Generous Box bounds: the builder clips so the checker never sees an out-of-space value.
_SPEED_HI = 2.0
_HEADING_HI = 1.0  # heading_error / pi already in [-1, 1]
_LATERAL_HI = 3.0  # signed lateral / half_width; >1 means off the asphalt on that side
_CURV_HI = 3.0
_BEAM_LO = 0.0
_BEAM_HI = 1.0
# Part 2 tail bounds.
_WEAR_LO, _WEAR_HI = 0.0, 1.0
_ONEHOT_LO, _ONEHOT_HI = 0.0, 1.0
_GRIP_IND_LO, _GRIP_IND_HI = 0.0, 2.0  # normalized grip / mu_base, clipped (matches Conditions)


@dataclass(frozen=True)
class ObsParams:
    """Tunable ObservationV1 parameters, all SI (meters, m/s, radians-from-degrees)."""

    ref_speed: float = _DEFAULT_REF_SPEED
    lookahead_m: tuple[float, ...] = _DEFAULT_LOOKAHEAD_M
    beam_angles_deg: tuple[float, ...] = _DEFAULT_BEAM_ANGLES_DEG
    beam_max: float = _DEFAULT_BEAM_MAX
    curvature_scale: float = _DEFAULT_CURVATURE_SCALE
    # Beam angles in radians, derived once at construction (not read from config directly).
    beam_angles_rad: np.ndarray = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        rad = np.radians(np.asarray(self.beam_angles_deg, dtype=np.float64))
        object.__setattr__(self, "beam_angles_rad", rad)

    @classmethod
    def from_config(cls, cfg: Any) -> ObsParams:
        """Build from an ``obs`` config node (mapping/OmegaConf) or fall back to defaults.

        Accepts either the root config (reads ``cfg.obs``) or the ``obs`` node directly.
        """
        node = cfg
        if hasattr(cfg, "obs") and cfg.obs is not None:
            node = cfg.obs
        get = node.get if hasattr(node, "get") else (lambda k, d: getattr(node, k, d))

        lookahead = _as_float_tuple(get("lookahead_m", cls.lookahead_m))
        if len(lookahead) != 5:
            raise ValueError(f"obs.lookahead_m must have 5 entries, got {len(lookahead)}")
        angles = _as_float_tuple(get("beam_angles_deg", cls.beam_angles_deg))
        if len(angles) != 7:
            raise ValueError(f"obs.beam_angles_deg must have 7 entries, got {len(angles)}")

        return cls(
            ref_speed=float(get("ref_speed", cls.ref_speed)),
            lookahead_m=lookahead,
            beam_angles_deg=angles,
            beam_max=float(get("beam_max", cls.beam_max)),
            curvature_scale=float(get("curvature_scale", cls.curvature_scale)),
        )


@dataclass(frozen=True)
class EdgeCache:
    """Per-track precompute for the rangefinder beams.

    The two asphalt edges are polylines ``centerline +/- normal * half_width`` (the left
    normal points left of travel). The beams are cast against these as line segments. The
    polylines are computed once per track and reused every step.
    """

    left_edge: np.ndarray  # (N, 2) centerline + normal * half_width_left
    right_edge: np.ndarray  # (N, 2) centerline - normal * half_width_right
    # Segment start points and direction vectors, precomputed for vectorized ray casts.
    seg_p: np.ndarray  # (M, 2) segment start points (both edges concatenated)
    seg_d: np.ndarray  # (M, 2) segment direction vectors (end - start)


def observation_space():
    """Return the ObservationV1 :class:`gymnasium.spaces.Box` (lazy gymnasium import)."""
    import gymnasium.spaces as spaces

    low = np.empty(OBS_DIM, dtype=np.float32)
    high = np.empty(OBS_DIM, dtype=np.float32)
    low[0], high[0] = -_SPEED_HI, _SPEED_HI
    low[1], high[1] = -_HEADING_HI, _HEADING_HI
    low[2], high[2] = -_LATERAL_HI, _LATERAL_HI
    low[3:8], high[3:8] = -_CURV_HI, _CURV_HI
    low[8:15], high[8:15] = _BEAM_LO, _BEAM_HI
    low[15], high[15] = _WEAR_LO, _WEAR_HI
    low[16:21], high[16:21] = _ONEHOT_LO, _ONEHOT_HI
    low[21], high[21] = _GRIP_IND_LO, _GRIP_IND_HI
    return spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32)


def build_edge_cache(track: Track) -> EdgeCache:
    """Precompute the two asphalt-edge polylines and their segments for beam casting."""
    c = np.asarray(track.centerline, dtype=np.float64)
    n = np.asarray(track.normal, dtype=np.float64)
    hl = np.asarray(track.half_width_left, dtype=np.float64).reshape(-1, 1)
    hr = np.asarray(track.half_width_right, dtype=np.float64).reshape(-1, 1)
    left_edge = c + n * hl
    right_edge = c - n * hr

    seg_p_list = []
    seg_d_list = []
    for edge in (left_edge, right_edge):
        p0 = edge
        if track.closed:
            p1 = np.roll(edge, -1, axis=0)  # wrap the seam
        else:
            p0 = edge[:-1]
            p1 = edge[1:]
        seg_p_list.append(p0)
        seg_d_list.append(p1 - p0)
    seg_p = np.concatenate(seg_p_list, axis=0)
    seg_d = np.concatenate(seg_d_list, axis=0)
    return EdgeCache(left_edge=left_edge, right_edge=right_edge, seg_p=seg_p, seg_d=seg_d)


def track_query(
    track: Track, x: float, y: float, yaw: float
) -> tuple[int, float, float, float, float]:
    """One nearest-point projection, shared by the obs builder, the reward, and the server.

    Returns ``(nearest_idx, s_along, signed_lateral, half_width, heading_error)``:

    - ``nearest_idx``  : centerline sample closest to ``(x, y)``.
    - ``s_along``      : arc length at that sample (meters).
    - ``signed_lateral``: offset along the **left** normal (positive = left of travel), meters.
    - ``half_width``   : asphalt half-width on the car's side (left if ``signed_lateral>=0``).
    - ``heading_error``: ``wrap(yaw - tangent_angle)`` at the sample, radians in (-pi, pi].
    """
    idx = track.nearest_index(x, y)
    c = track.centerline[idx]
    nrm = track.normal[idx]
    tan = track.tangent[idx]

    dx = x - float(c[0])
    dy = y - float(c[1])
    signed_lateral = dx * float(nrm[0]) + dy * float(nrm[1])
    s_along = float(track.s[idx])

    half_width = (
        float(track.half_width_left[idx])
        if signed_lateral >= 0.0
        else float(track.half_width_right[idx])
    )

    tangent_angle = math.atan2(float(tan[1]), float(tan[0]))
    heading_error = _wrap_pi(yaw - tangent_angle)
    return idx, s_along, signed_lateral, half_width, heading_error


def sample_curvature_ahead(track: Track, s: float, lookahead_m: Any) -> np.ndarray | float:
    """Signed curvature interpolated at ``s + lookahead`` along the centerline.

    ``lookahead_m`` may be a scalar or an iterable of distances (meters). On a closed loop
    the lookahead wraps around the seam; on an open path it clamps to the last sample.
    Returns a float for a scalar lookahead, else a NumPy array.
    """
    s_arr = np.asarray(track.s, dtype=np.float64)
    kappa = np.asarray(track.curvature, dtype=np.float64)
    length = track.length

    scalar = np.ndim(lookahead_m) == 0
    look = np.atleast_1d(np.asarray(lookahead_m, dtype=np.float64))
    target = float(s) + look

    if track.closed:
        target = np.mod(target, length)
        # Extend the table by one wrapped point so np.interp covers [s[-1], length).
        xp = np.concatenate([s_arr, [length]])
        fp = np.concatenate([kappa, kappa[:1]])
        vals = np.interp(target, xp, fp)
    else:
        vals = np.interp(target, s_arr, kappa)

    return float(vals[0]) if scalar else vals


def cast_beams(
    track: Track,
    x: float,
    y: float,
    yaw: float,
    angles: Any,
    beam_max: float,
    edge_cache: EdgeCache | None = None,
) -> np.ndarray:
    """Cast rangefinder beams from ``(x, y)`` to the asphalt edges; return raw distances (m).

    Each beam is a ray at ``yaw + angle``; its value is the distance to the nearest
    asphalt-edge segment, clipped to ``beam_max`` (and ``beam_max`` when nothing is hit
    within range). The edge polylines are precomputed (``edge_cache``) and the ray/segment
    intersection is vectorized over all segments per beam. Returns shape ``(len(angles),)``.
    """
    if edge_cache is None:
        edge_cache = build_edge_cache(track)

    angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    seg_p = edge_cache.seg_p  # (M, 2)
    seg_d = edge_cache.seg_d  # (M, 2)
    origin = np.array([float(x), float(y)], dtype=np.float64)

    out = np.empty(angles.shape[0], dtype=np.float64)
    for k, ang in enumerate(angles):
        theta = yaw + ang
        r = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        out[k] = _ray_min_distance(origin, r, seg_p, seg_d, beam_max)
    return out


def build_observation(
    track: Track,
    state: Any,
    params: ObsParams,
    edge_cache: EdgeCache | None = None,
    grip_indicator: float | None = None,
) -> np.ndarray:
    """Assemble the 22-dim ObservationV2 from a car ``state`` (clipped to the Box).

    ``state`` is a :class:`~f1rl.physics.base.CarState` (uses ``x``, ``y``, ``yaw``, ``vx``,
    ``tire_wear``, ``compound``). ``grip_indicator`` is the normalized grip at the car
    (``Conditions.grip_indicator``, already ``/ mu_base``); it defaults to ``1.0`` (full grip)
    for callers without a conditions provider. The v1 slice (0-14) is byte-identical to v1.
    """
    if edge_cache is None:
        edge_cache = build_edge_cache(track)

    x = float(state.x)
    y = float(state.y)
    yaw = float(state.yaw)
    vx = float(state.vx)

    _idx, s_along, signed_lateral, half_width, heading_error = track_query(track, x, y, yaw)

    speed_norm = vx / params.ref_speed if params.ref_speed > 0.0 else 0.0
    heading_norm = heading_error / math.pi
    lateral_norm = signed_lateral / half_width if half_width > 0.0 else 0.0

    curv = sample_curvature_ahead(track, s_along, params.lookahead_m)
    curv = np.asarray(curv, dtype=np.float64) * params.curvature_scale

    beams_raw = cast_beams(
        track, x, y, yaw, params.beam_angles_rad, params.beam_max, edge_cache
    )
    if params.beam_max > 0.0:
        beams_norm = np.clip(beams_raw / params.beam_max, 0.0, 1.0)
    else:
        beams_norm = beams_raw

    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[0] = speed_norm
    obs[1] = heading_norm
    obs[2] = lateral_norm
    obs[3:8] = curv
    obs[8:15] = beams_norm

    # Part 2 tail: tire wear, compound one-hot, grip indicator.
    obs[15] = float(getattr(state, "tire_wear", 0.0))
    obs[16:21] = 0.0
    compound = int(getattr(state, "compound", 0))
    if 0 <= compound < N_COMPOUNDS:
        obs[16 + compound] = 1.0
    obs[21] = 1.0 if grip_indicator is None else float(grip_indicator)

    # Clip to the declared Box so the env_checker never sees an out-of-space value.
    np.clip(obs[0:1], -_SPEED_HI, _SPEED_HI, out=obs[0:1])
    np.clip(obs[1:2], -_HEADING_HI, _HEADING_HI, out=obs[1:2])
    np.clip(obs[2:3], -_LATERAL_HI, _LATERAL_HI, out=obs[2:3])
    np.clip(obs[3:8], -_CURV_HI, _CURV_HI, out=obs[3:8])
    np.clip(obs[8:15], _BEAM_LO, _BEAM_HI, out=obs[8:15])
    np.clip(obs[15:16], _WEAR_LO, _WEAR_HI, out=obs[15:16])
    np.clip(obs[21:22], _GRIP_IND_LO, _GRIP_IND_HI, out=obs[21:22])
    return obs


# --- internals --------------------------------------------------------------------------


def _ray_min_distance(
    origin: np.ndarray,
    r: np.ndarray,
    seg_p: np.ndarray,
    seg_d: np.ndarray,
    beam_max: float,
) -> float:
    """Min distance along ray ``origin + t*r`` (t>=0) to any segment, capped at ``beam_max``.

    Solves ``origin + t*r = seg_p + u*seg_d`` for each segment via 2D cross products,
    vectorized over all segments. Valid hits need ``t >= 0`` and ``0 <= u <= 1``.
    """
    # denom = r x seg_d  (scalar per segment)
    denom = r[0] * seg_d[:, 1] - r[1] * seg_d[:, 0]
    diff = seg_p - origin  # (M, 2)
    # t = (diff x seg_d) / denom ; u = (diff x r) / denom
    diff_cross_d = diff[:, 0] * seg_d[:, 1] - diff[:, 1] * seg_d[:, 0]
    diff_cross_r = diff[:, 0] * r[1] - diff[:, 1] * r[0]

    nonparallel = np.abs(denom) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(nonparallel, diff_cross_d / denom, np.inf)
        u = np.where(nonparallel, diff_cross_r / denom, np.inf)

    valid = nonparallel & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    if not np.any(valid):
        return float(beam_max)
    return float(min(np.min(t[valid]), beam_max))


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    """Coerce a config list/sequence (incl. OmegaConf ListConfig) to a tuple of floats."""
    return tuple(float(v) for v in value)


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))
