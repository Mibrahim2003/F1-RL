"""RewardV1 — progress-based, never centerline-seeking (TECHNICAL_DESIGN.md §9, plan §A).

The rule, restated: reward forward progress, penalize leaving the track and going
backward, and add a small per-step cost to discourage dawdling. **The signed lateral
offset never enters the reward** — the racing line must emerge from speed and progress,
not be hand-fed. Every weight and the off-track penalty shape live in :class:`RewardWeights`,
built from config. SI units (meters, seconds).

Per step::

    ds     = wrap-aware signed arc-length progress since the last step (m)
    off    = meters past the asphalt edge, 0 on asphalt
    reward = w_progress * ds
           - w_offtrack * offtrack_penalty(off)
           - w_step
           - w_reverse * max(0, -ds)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from f1rl.env.collisions import ContactRecord

# Default reward weights and off-track shape (all overridable from the ``reward:`` config).
_DEFAULT_W_PROGRESS = 1.0  # per meter of forward arc-length progress
_DEFAULT_W_OFFTRACK = 1.0  # scales the graded off-track penalty
_DEFAULT_W_STEP = 0.05  # small constant per step, discourages dawdling
_DEFAULT_W_REVERSE = 2.0  # extra penalty per meter of backward progress
_DEFAULT_OFFTRACK_SOFT_M = 2.0  # meters past the edge over which the penalty ramps to ~linear
_DEFAULT_OFFTRACK_EXP = 2.0  # >1 makes small excursions cheap and far ones costly
_DEFAULT_VERSION = 1  # 1 = reward_v1 (Phase 1) | 2 = reward_v2 (Part 2) | 3 = reward_v3 (Phase 6)
_DEFAULT_W_SLIP = 0.0  # v2 only: penalty on excessive slip/spin; 0 keeps behavior ~v1
_DEFAULT_SLIP_THRESHOLD = 0.20  # v2 only: slip metric above this (rad-ish) is penalized
# Phase 6 racing terms (reward_v3); all weights default 0 => reward_v3 reduces to reward_v2.
_DEFAULT_W_CONTACT = 0.0  # contact penalty weight
_DEFAULT_CONTACT_SOFT_MPS = 5.0  # closing-speed scale; a light brush << a hard hit
_DEFAULT_CONTACT_EXP = 2.0  # >1 makes light contact cheap, heavy contact costly
_DEFAULT_W_CONTACT_FAULT = 0.0  # extra share for the at-fault car (reserved; default 0)
_DEFAULT_W_OVERTAKE = 0.0  # + per place gained / - per place lost (zero-sum, genuine swaps)
_DEFAULT_OVERTAKE_BATTLE_RANGE_M = 20.0  # both cars within this total-progress gap to count a swap
_DEFAULT_W_GAP = 0.0  # opt-in dense gap-closing shaping (like w_slip; default 0)


@dataclass(frozen=True)
class RewardWeights:
    """Tunable reward weights and off-track penalty shape, all from config (v1 and v2)."""

    w_progress: float = _DEFAULT_W_PROGRESS
    w_offtrack: float = _DEFAULT_W_OFFTRACK
    w_step: float = _DEFAULT_W_STEP
    w_reverse: float = _DEFAULT_W_REVERSE
    offtrack_soft_m: float = _DEFAULT_OFFTRACK_SOFT_M
    offtrack_exp: float = _DEFAULT_OFFTRACK_EXP
    version: int = _DEFAULT_VERSION
    w_slip: float = _DEFAULT_W_SLIP
    slip_threshold: float = _DEFAULT_SLIP_THRESHOLD
    # Phase 6 racing weights (reward_v3).
    w_contact: float = _DEFAULT_W_CONTACT
    contact_soft_mps: float = _DEFAULT_CONTACT_SOFT_MPS
    contact_exp: float = _DEFAULT_CONTACT_EXP
    w_contact_fault: float = _DEFAULT_W_CONTACT_FAULT
    w_overtake: float = _DEFAULT_W_OVERTAKE
    overtake_battle_range_m: float = _DEFAULT_OVERTAKE_BATTLE_RANGE_M
    w_gap: float = _DEFAULT_W_GAP

    @classmethod
    def from_config(cls, cfg: Any) -> RewardWeights:
        """Build from a ``reward`` config node (mapping/OmegaConf) or fall back to defaults.

        Accepts either the root config (reads ``cfg.reward``) or the ``reward`` node directly.
        """
        node = cfg
        if hasattr(cfg, "reward") and cfg.reward is not None:
            node = cfg.reward
        get = node.get if hasattr(node, "get") else (lambda k, d: getattr(node, k, d))
        return cls(
            w_progress=float(get("w_progress", cls.w_progress)),
            w_offtrack=float(get("w_offtrack", cls.w_offtrack)),
            w_step=float(get("w_step", cls.w_step)),
            w_reverse=float(get("w_reverse", cls.w_reverse)),
            offtrack_soft_m=float(get("offtrack_soft_m", cls.offtrack_soft_m)),
            offtrack_exp=float(get("offtrack_exp", cls.offtrack_exp)),
            version=int(get("version", cls.version)),
            w_slip=float(get("w_slip", cls.w_slip)),
            slip_threshold=float(get("slip_threshold", cls.slip_threshold)),
            w_contact=float(get("w_contact", cls.w_contact)),
            contact_soft_mps=float(get("contact_soft_mps", cls.contact_soft_mps)),
            contact_exp=float(get("contact_exp", cls.contact_exp)),
            w_contact_fault=float(get("w_contact_fault", cls.w_contact_fault)),
            w_overtake=float(get("w_overtake", cls.w_overtake)),
            overtake_battle_range_m=float(
                get("overtake_battle_range_m", cls.overtake_battle_range_m)
            ),
            w_gap=float(get("w_gap", cls.w_gap)),
        )


def signed_progress(prev_s: float, cur_s: float, length: float) -> float:
    """Wrap-aware signed arc-length progress ``ds`` (meters), positive = forward.

    Folds the value into ``(-length/2, length/2]`` so a lap wrap across the start/finish
    line reads as a small forward step, not a near-full-lap jump backward.
    """
    ds = float(cur_s) - float(prev_s)
    if length <= 0.0:
        return ds
    half = 0.5 * length
    # Wrap into (-half, half].
    ds = (ds + half) % length - half
    return ds


def offtrack_penalty(off_track_m: float, weights: RewardWeights) -> float:
    """Graded penalty for being off the asphalt: 0 on asphalt, growing with distance.

    Zero while on asphalt (``off <= 0``). Past the edge it grows as
    ``(off / soft_m) ** exp`` so a slight excursion costs little and a deep one costs a lot
    (grass/gravel already bleed time on their own; this is the shaped signal on top).
    """
    off = float(off_track_m)
    if off <= 0.0:
        return 0.0
    soft = weights.offtrack_soft_m if weights.offtrack_soft_m > 0.0 else 1.0
    return (off / soft) ** weights.offtrack_exp


def reward_v1(
    prev_s: float,
    cur_s: float,
    off_track_m: float,
    length: float,
    weights: RewardWeights,
) -> tuple[float, dict[str, float]]:
    """Compute the per-step RewardV1 and its term breakdown.

    Args:
        prev_s: Arc length at the previous step (m).
        cur_s: Arc length at the current step (m).
        off_track_m: Meters past the asphalt edge (0 on asphalt).
        length: Total lap length (m), for wrap-aware progress.
        weights: Tunable :class:`RewardWeights`.

    Returns:
        ``(reward, terms)`` where ``terms`` carries each signed contribution plus the raw
        ``ds`` and ``off`` for logging. ``lateral`` deliberately never appears.
    """
    ds = signed_progress(prev_s, cur_s, length)

    progress_term = weights.w_progress * ds
    offtrack_term = -weights.w_offtrack * offtrack_penalty(off_track_m, weights)
    step_term = -weights.w_step
    reverse_term = -weights.w_reverse * max(0.0, -ds)

    reward = progress_term + offtrack_term + step_term + reverse_term
    terms = {
        "progress": progress_term,
        "offtrack": offtrack_term,
        "step": step_term,
        "reverse": reverse_term,
        "ds": ds,
        "off": float(off_track_m),
        "total": reward,
    }
    return float(reward), terms


def slip_penalty(slip: float, weights: RewardWeights) -> float:
    """Penalty magnitude for slip past the threshold: ``max(0, |slip| - slip_threshold)``.

    Zero below the threshold (clean cornering is free); grows linearly past it to discourage
    overdriving past the grip limit. Scaled by ``w_slip`` at the call site.
    """
    excess = abs(float(slip)) - weights.slip_threshold
    return excess if excess > 0.0 else 0.0


def reward_v2(
    prev_s: float,
    cur_s: float,
    off_track_m: float,
    length: float,
    weights: RewardWeights,
    slip: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """RewardV2 = the v1 progress core plus an optional, config-gated slip/spin penalty.

    With ``w_slip = 0`` (the default) this is numerically identical to :func:`reward_v1` — the
    extra shaping is opt-in. Still **never centerline-seeking**: the racing line must emerge
    from progress and speed. ``slip`` is a slip metric from the env (e.g. ``|vy / vx|``).

    Returns the same ``(reward, terms)`` breakdown shape as v1 (with an extra ``slip`` term).
    """
    reward, terms = reward_v1(prev_s, cur_s, off_track_m, length, weights)
    slip_term = -weights.w_slip * slip_penalty(slip, weights)
    reward += slip_term
    terms["slip"] = slip_term
    terms["total"] = reward
    return float(reward), terms


def contact_cost(contact: ContactRecord, weights: RewardWeights) -> float:
    """Graded contact cost from a step's :class:`ContactRecord`: ``(closing / soft) ** exp``.

    Zero on a clean step (no contact). Scales with the closing speed of the contact so a light
    brush costs little and a hard hit costs a lot; ``contact_exp > 1`` sharpens that. Multiplied
    by ``w_contact`` at the call site. Never references lateral offset or a racing line.
    """
    if contact.count <= 0 or contact.closing_mps <= 0.0:
        return 0.0
    soft = weights.contact_soft_mps if weights.contact_soft_mps > 0.0 else 1.0
    return (contact.closing_mps / soft) ** weights.contact_exp


def reward_v3(
    prev_s: float,
    cur_s: float,
    off_track_m: float,
    length: float,
    weights: RewardWeights,
    slip: float = 0.0,
    contact: ContactRecord | None = None,
    places: int = 0,
    gap_delta: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """RewardV3 = the reward_v2 core + a graded contact penalty + a zero-sum position term.

    The racing terms (Phase 6):

    - **contact**: ``-w_contact * contact_cost(contact)`` — charges for touching another car,
      graded by closing speed (clean racing pays, dirty racing does not). Symmetric by default;
      ``w_contact_fault`` is reserved for charging the at-fault car more (default 0, no fault
      data is recorded yet).
    - **overtake/defend**: ``+w_overtake * places`` where ``places`` is the field-computed net
      places gained this step, **already gated to genuine wheel-to-wheel swaps** and zero-sum
      across the swapping pair (gaining a place pays; losing one costs — the same weight drives
      both overtaking and defending). The maneuver is never encoded; it emerges.
    - **gap**: ``+w_gap * gap_delta`` — opt-in dense gap-closing shaping (``w_gap = 0`` default).

    With no contact (``contact`` empty / None) and a constant rank (``places = 0``, ``gap_delta
    = 0``) this is **numerically identical to** :func:`reward_v2` — so the single-agent path and
    a one-car field are unchanged. Still never centerline-seeking; never hand-codes blame or a
    racing line. ``places`` is an int (signed); ``gap_delta`` is meters of gap closed this step.
    """
    reward, terms = reward_v2(prev_s, cur_s, off_track_m, length, weights, slip)

    rec = contact if contact is not None else ContactRecord()
    cost = contact_cost(rec, weights)
    # Symmetric contact penalty. `w_contact_fault` is reserved (no per-car fault is recorded
    # yet); when the collision pass attributes fault it adds an asymmetric share on top.
    contact_term = -weights.w_contact * cost
    overtake_term = weights.w_overtake * float(places)
    gap_term = weights.w_gap * float(gap_delta)

    reward += contact_term + overtake_term + gap_term
    terms["contact"] = contact_term
    terms["overtake"] = overtake_term
    terms["gap"] = gap_term
    terms["total"] = reward
    return float(reward), terms
