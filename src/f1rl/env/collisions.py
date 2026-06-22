"""Field-level collision pass (Phase 6; spec §2 collisions, TECHNICAL_DESIGN.md §5/§10).

Cars are solid bodies. Between every car's substep physics and any car's finalize, the field
env runs **one** collision pass over the live field: it detects overlapping cars and applies a
contact response (a positional push-apart + a restitution/friction velocity impulse) that
separates them and scrubs their closing speed. This is the **one** place cars are coupled.

The car body is **two discs** (front + rear), each of radius ``disc_radius_m``, centered at
``+-disc_offset_m`` along the body axis from the car center — a far better fit for a long thin
car than a single fat circle, and cheaper than an oriented box (the later fidelity upgrade).

**``PhysicsModel.step`` is never imported or touched** — physics stays the pure single-car
function of §5; the coupling lives only here. The response is **snapshot-then-apply**: all
post-physics states are read into a snapshot, every live pair's correction is computed against
that snapshot, then the *summed* corrections are applied — so the result does not depend on
agent iteration order and is reproducible from the seed. Velocities are resolved in the world
frame and mapped back into each car's body frame (``vx``/``vy``); ``yaw``/``yaw_rate`` are left
unchanged (contact-induced spin is a future fidelity upgrade). Equal mass (homogeneous field).

Every geometry/response constant comes from :class:`CollisionParams` (config), never logic.
SI units (meters, m/s); world frame (x right, y up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Defaults mirror the ``collision:`` block in configs/default.yaml (all overridable from config).
_DEFAULT_DISC_RADIUS_M = 1.0
_DEFAULT_DISC_OFFSET_M = 1.25
_DEFAULT_RESTITUTION = 0.1
_DEFAULT_FRICTION = 0.3
_DEFAULT_PUSH_FRACTION = 1.0
_DEFAULT_CRASHOUT_CLOSING_MPS = 25.0


@dataclass(frozen=True)
class CollisionParams:
    """Two-disc collision geometry + the equal-mass contact response, all from config."""

    enabled: bool = False
    body: str = "two_disc"  # two_disc (default) | disc (single center disc); obb = future
    disc_radius_m: float = _DEFAULT_DISC_RADIUS_M
    disc_offset_m: float = _DEFAULT_DISC_OFFSET_M
    restitution: float = _DEFAULT_RESTITUTION
    friction: float = _DEFAULT_FRICTION
    push_fraction: float = _DEFAULT_PUSH_FRACTION
    crashout_enabled: bool = False
    crashout_closing_speed_mps: float = _DEFAULT_CRASHOUT_CLOSING_MPS

    def __post_init__(self) -> None:
        if self.disc_radius_m <= 0.0:
            raise ValueError(f"collision.disc_radius_m must be > 0, got {self.disc_radius_m}")
        if self.disc_offset_m < 0.0:
            raise ValueError(f"collision.disc_offset_m must be >= 0, got {self.disc_offset_m}")
        if not (0.0 <= self.restitution <= 1.0):
            raise ValueError(f"collision.restitution must be in [0,1], got {self.restitution}")
        if not (0.0 <= self.friction <= 1.0):
            raise ValueError(f"collision.friction must be in [0,1], got {self.friction}")
        if not (0.0 <= self.push_fraction <= 1.0):
            raise ValueError(f"collision.push_fraction must be in [0,1], got {self.push_fraction}")

    @property
    def disc_offsets(self) -> tuple[float, ...]:
        """Body-axis offsets of the discs from the car center (two discs, or one for ``disc``)."""
        if self.body == "disc":
            return (0.0,)
        return (self.disc_offset_m, -self.disc_offset_m)

    @classmethod
    def from_config(cls, cfg: Any) -> CollisionParams:
        """Build from a ``collision`` config node (mapping/OmegaConf) or fall back to defaults.

        Accepts either the root config (reads ``cfg.collision``) or the node directly. A missing
        block yields the disabled default (so a pre-Phase-6 config keeps the blind parade).
        """
        node = cfg
        if hasattr(cfg, "collision"):
            node = cfg.collision
        if node is None:
            return cls()
        get = node.get if hasattr(node, "get") else (lambda k, d: getattr(node, k, d))
        return cls(
            enabled=bool(get("enabled", cls.enabled)),
            body=str(get("body", cls.body)),
            disc_radius_m=float(get("disc_radius_m", cls.disc_radius_m)),
            disc_offset_m=float(get("disc_offset_m", cls.disc_offset_m)),
            restitution=float(get("restitution", cls.restitution)),
            friction=float(get("friction", cls.friction)),
            push_fraction=float(get("push_fraction", cls.push_fraction)),
            crashout_enabled=bool(get("crashout_enabled", cls.crashout_enabled)),
            crashout_closing_speed_mps=float(
                get("crashout_closing_speed_mps", cls.crashout_closing_speed_mps)
            ),
        )


@dataclass
class ContactRecord:
    """Per-car contact summary for ONE step (zero = clean), read by ``reward_v3`` + the info."""

    impulse: float = 0.0  # aggregate normal-impulse magnitude this step (m/s, equal mass)
    closing_mps: float = 0.0  # max closing speed of any contact this step
    count: int = 0  # number of contacts this step


@dataclass
class _Snapshot:
    """Read-only post-physics state of one car for the order-independent collision solve."""

    cx: float
    cy: float
    yaw: float
    vx_w: float  # world-frame velocity x
    vy_w: float  # world-frame velocity y
    discs: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))  # (n_disc, 2) centers


def _snapshot(state: Any, offsets: tuple[float, ...]) -> _Snapshot:
    yaw = float(state.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy = float(state.x), float(state.y)
    vx_w = float(state.vx) * c - float(state.vy) * s
    vy_w = float(state.vx) * s + float(state.vy) * c
    discs = np.array([[cx + off * c, cy + off * s] for off in offsets], dtype=np.float64)
    return _Snapshot(cx=cx, cy=cy, yaw=yaw, vx_w=vx_w, vy_w=vy_w, discs=discs)


def resolve_collisions(states: list[Any], params: CollisionParams) -> list[ContactRecord]:
    """Detect + resolve contact across the live field; mutate ``states``, return per-car records.

    Args:
        states: The live cars' :class:`~f1rl.physics.base.CarState` (post-physics, body-frame
            ``vx``/``vy``). Mutated in place: ``x``/``y`` pushed apart, ``vx``/``vy`` impulsed.
            Done/removed cars must not be passed (they leave the field).
        params: The two-disc geometry + response, from config.

    Returns:
        One :class:`ContactRecord` per input car (same order); all-zero when ``enabled`` is
        false, fewer than two cars, or no overlap. Order-independent and reproducible: every
        correction is computed against a frozen snapshot and applied as a sum.
    """
    n = len(states)
    records = [ContactRecord() for _ in range(n)]
    if not params.enabled or n < 2:
        return records

    offsets = params.disc_offsets
    r = params.disc_radius_m
    two_r = 2.0 * r
    snaps = [_snapshot(s, offsets) for s in states]

    # Accumulate corrections against the snapshot (order-independent), then apply the sums.
    dpos = np.zeros((n, 2), dtype=np.float64)  # world position correction per car
    dvel = np.zeros((n, 2), dtype=np.float64)  # world velocity correction per car

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = snaps[i], snaps[j]
            # Closest disc-disc pair between car i and car j.
            best_gap = math.inf
            best_nx = best_ny = 0.0
            for di in si.discs:
                for dj in sj.discs:
                    dx = dj[0] - di[0]
                    dy = dj[1] - di[1]
                    dist = math.hypot(dx, dy)
                    gap = dist - two_r
                    if gap < best_gap:
                        best_gap = gap
                        if dist > 1e-9:
                            best_nx, best_ny = dx / dist, dy / dist
                        else:  # coincident centers: push along i->j car-center axis (or +x)
                            ax = sj.cx - si.cx
                            ay = sj.cy - si.cy
                            an = math.hypot(ax, ay)
                            best_nx, best_ny = (ax / an, ay / an) if an > 1e-9 else (1.0, 0.0)
            if best_gap >= 0.0:
                continue  # no overlap on the closest disc pair

            # Normal points from i to j. Positional push-apart: split the penetration evenly.
            penetration = -best_gap  # > 0
            push = 0.5 * params.push_fraction * penetration
            dpos[i, 0] -= push * best_nx
            dpos[i, 1] -= push * best_ny
            dpos[j, 0] += push * best_nx
            dpos[j, 1] += push * best_ny

            # Relative velocity of i w.r.t. j along the normal (>0 => approaching / closing).
            rvx = si.vx_w - sj.vx_w
            rvy = si.vy_w - sj.vy_w
            v_n = rvx * best_nx + rvy * best_ny
            closing = v_n if v_n > 0.0 else 0.0
            if v_n > 0.0:
                # Equal-mass normal impulse (1/m_i + 1/m_j = 2 with m = 1): j = (1+e) * v_n / 2.
                jimp = (1.0 + params.restitution) * v_n / 2.0
                dvel[i, 0] -= jimp * best_nx
                dvel[i, 1] -= jimp * best_ny
                dvel[j, 0] += jimp * best_nx
                dvel[j, 1] += jimp * best_ny

                # Tangential friction damping: remove a fraction of the relative tangential
                # velocity, split evenly between the pair.
                vt_x = rvx - v_n * best_nx
                vt_y = rvy - v_n * best_ny
                damp = 0.5 * params.friction
                dvel[i, 0] -= damp * vt_x
                dvel[i, 1] -= damp * vt_y
                dvel[j, 0] += damp * vt_x
                dvel[j, 1] += damp * vt_y

                imp_mag = abs(jimp)
            else:
                imp_mag = 0.0

            for idx in (i, j):
                rec = records[idx]
                rec.impulse += imp_mag
                rec.closing_mps = max(rec.closing_mps, closing)
                rec.count += 1

    # Apply the summed corrections; map the world-frame velocity change back into body frame.
    for idx, st in enumerate(states):
        if not (dpos[idx].any() or dvel[idx].any()):
            continue
        st.x = float(st.x) + float(dpos[idx, 0])
        st.y = float(st.y) + float(dpos[idx, 1])
        new_vx_w = snaps[idx].vx_w + float(dvel[idx, 0])
        new_vy_w = snaps[idx].vy_w + float(dvel[idx, 1])
        yaw = float(st.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        # world -> body: vx = w . forward, vy = w . left
        st.vx = new_vx_w * c + new_vy_w * s
        st.vy = -new_vx_w * s + new_vy_w * c
    return records
