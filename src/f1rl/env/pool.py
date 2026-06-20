"""Circuit pool for multi-circuit training (Phase 4 spec §2b, plan §A).

Phase 3b bound one track for the life of the env. Phase 4 turns that single binding into a
**config-driven pool**: every circuit id is loaded once, its edge cache precomputed once, and
its pole resolved once; the env then *samples* a circuit per ``reset`` and rebinds the active
track/edge-cache/lap-timer/pole from the prebuilt entries. The per-reset cost is a dict lookup
and a draw — never a rebuild — so sampling stays cheap (the new cost is RAM, not steps).

This module is **runtime-safe** like :mod:`f1rl.track.loader`: it never imports FastF1 and
never touches the network. It reads cached ``data/tracks/<id>.npz`` (via ``load_track``) and the
per-circuit pole from ``configs/track/<id>.yaml`` (YAML only). An id with no built ``.npz``
raises the loader's ``FileNotFoundError`` with the build hint — the pool never silently shrinks.

The observation is unchanged: this is a *sampling* change only (TECHNICAL_DESIGN §7 — local,
relative features generalize across the calendar), so the Phase 3b checkpoint warm-starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.env.observations import EdgeCache, build_edge_cache
from f1rl.sim.timing import LapTimer
from f1rl.track.loader import DEFAULT_TRACKS_DIR, load_track
from f1rl.track.schema import Track


def _get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def resolve_pole(track_id: str, cfg: Any, config_root: Path | None = None) -> tuple[float, bool]:
    """Resolve a circuit's pole time from its config (never from the geometry ``.npz``).

    The pole lives in ``configs/track/<id>.yaml`` (``pole_time_s``); the ``.npz`` stores only
    geometry (``official_length_m``). When ``cfg`` already carries the merged ``track`` node for
    this id, read it directly; otherwise load the per-circuit YAML. The node is matched on its
    own ``id`` field, not ``cfg.track_id`` — a ``track_id=...`` override changes ``cfg.track_id``
    without re-merging ``cfg.track``, so trusting ``cfg.track_id`` would read a stale pole.

    Returns ``(pole_time_s, pole_missing)`` where ``pole_missing`` is ``True`` for a missing or
    non-positive pole, so the benchmark flags that circuit and skips its delta (never compares
    against zero). Runtime-safe: YAML only, no FastF1, no network.
    """
    ctrack = getattr(cfg, "track", None)
    if ctrack is not None and str(_get(ctrack, "id", None)) == str(track_id):
        pole = float(_get(ctrack, "pole_time_s", 0.0) or 0.0)
        return pole, pole <= 0.0

    # Late import keeps the module import-light; CONFIG_ROOT is this repo's configs/.
    from omegaconf import OmegaConf

    from f1rl.utils.config import CONFIG_ROOT

    root = config_root or CONFIG_ROOT
    path = Path(root) / "track" / f"{track_id}.yaml"
    if not path.exists():
        return 0.0, True
    node = OmegaConf.load(path)
    pole = float(_get(node, "pole_time_s", 0.0) or 0.0)
    return pole, pole <= 0.0


def pool_ids_from_config(cfg: Any, fallback_id: str) -> list[str]:
    """The configured circuit pool ids, or ``[fallback_id]`` when no pool is set.

    An empty or absent ``circuits.pool`` reproduces the Phase 3b single-circuit behavior
    exactly (a one-circuit pool), so the change is backward compatible.
    """
    node = getattr(cfg, "circuits", None)
    pool = _get(node, "pool", None) or []
    ids = [str(c) for c in pool]
    return ids if ids else [str(fallback_id)]


@dataclass
class CircuitEntry:
    """One prebuilt circuit in the pool: geometry, edge cache, lap timer, and pole."""

    track_id: str
    track: Track
    edge_cache: EdgeCache
    lap_timer: LapTimer
    pole_time_s: float
    pole_missing: bool


class CircuitPool:
    """A set of prebuilt circuits an env samples from on ``reset`` (built once per worker).

    Each id is loaded and precomputed at construction. ``sample`` draws an id from the
    **active** set with the given RNG (``self.np_random`` in the env, so the draw is
    reproducible from the run seed). ``set_active`` narrows the active set for the curriculum
    (pool widening) without rebuilding anything.
    """

    def __init__(
        self,
        ids: list[str],
        cfg: Any,
        *,
        preloaded: dict[str, Track] | None = None,
        config_root: Path | None = None,
        tracks_dir: Path = DEFAULT_TRACKS_DIR,
    ) -> None:
        if not ids:
            raise ValueError("CircuitPool requires at least one circuit id")
        preloaded = preloaded or {}
        self.ids: list[str] = list(dict.fromkeys(ids))  # de-dup, preserve order
        self.entries: dict[str, CircuitEntry] = {}
        for cid in self.ids:
            track = preloaded.get(cid)
            if track is None:
                # Raises FileNotFoundError + build hint for an unbuilt circuit (no silent drop).
                track = load_track(cid, tracks_dir=tracks_dir)
            pole, missing = resolve_pole(cid, cfg, config_root)
            self.entries[cid] = CircuitEntry(
                track_id=cid,
                track=track,
                edge_cache=build_edge_cache(track),
                lap_timer=LapTimer(track, pole),
                pole_time_s=pole,
                pole_missing=missing,
            )

        node = getattr(cfg, "circuits", None)
        self.sampling = str(_get(node, "sampling", "uniform"))
        weights = _get(node, "weights", None)
        self.weights: dict[str, float] | None = (
            {str(k): float(v) for k, v in dict(weights).items()} if weights else None
        )
        self.pin_per_worker = bool(_get(node, "pin_per_worker", False))

        self._active: list[str] = list(self.ids)

    @property
    def active(self) -> list[str]:
        """The ids currently eligible to be drawn (narrowed by the curriculum)."""
        return list(self._active)

    def set_active(self, circuits: list[str] | str | None) -> None:
        """Set the active sampling set (curriculum pool-widening); empty/None/"all" => full pool.

        Validates every id is in the built pool (a stage can only narrow to circuits already
        loaded). Takes effect on the next ``sample`` (next ``reset``); never rebuilds entries.
        """
        if not circuits or circuits == "all":
            self._active = list(self.ids)
            return
        if isinstance(circuits, str):
            circuits = [circuits]
        for cid in circuits:
            if cid not in self.entries:
                raise KeyError(
                    f"circuit '{cid}' not in built pool {sorted(self.entries)}; "
                    f"add it to circuits.pool so it is loaded before the curriculum selects it"
                )
        self._active = [str(c) for c in circuits]

    def sample(self, rng: np.random.Generator) -> str:
        """Draw one active circuit id with ``rng`` (uniform, or weighted when configured)."""
        active = self._active
        if len(active) == 1:
            return active[0]
        if self.sampling == "weighted" and self.weights:
            w = np.array([self.weights.get(c, 1.0) for c in active], dtype=np.float64)
            total = w.sum()
            if total > 0.0:
                return active[int(rng.choice(len(active), p=w / total))]
        return active[int(rng.integers(0, len(active)))]

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, cid: str) -> bool:
        return cid in self.entries
