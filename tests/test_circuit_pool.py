"""Circuit-pool contract (Phase 4 spec §2b, §c; plan §A "Circuit pool + per-episode sampling").

Written from the public pool surface (``CircuitPool`` / ``resolve_pole`` /
``pool_ids_from_config``) and the spec, not from env internals: the pool loads every configured
id once, resolves each circuit's pole from its ``configs/track/<id>.yaml`` (never the ``.npz``),
flags a missing pole instead of scoring against zero, refuses an unbuilt id with the build hint,
and only narrows the active set to circuits it has already built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from f1rl.env.pool import CircuitPool, pool_ids_from_config, resolve_pole
from f1rl.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
TRACKS_DIR = REPO_ROOT / "data" / "tracks"


def _have(*ids: str) -> bool:
    return all((TRACKS_DIR / f"{i}.npz").exists() for i in ids)


pytestmark = pytest.mark.skipif(
    not _have("red_bull_ring", "monza"),
    reason="cached tracks 'red_bull_ring' + 'monza' required for pool tests",
)


def _cfg(pool: list[str]) -> object:
    cfg = load_config("default", overrides=["track_id=red_bull_ring"], config_root=CONFIG_ROOT)
    cfg.circuits.pool = list(pool)
    return cfg


def test_pool_loads_every_configured_id():
    pool = CircuitPool(["red_bull_ring", "monza"], _cfg(["red_bull_ring", "monza"]))
    assert set(pool.entries) == {"red_bull_ring", "monza"}
    for cid in ("red_bull_ring", "monza"):
        e = pool.entries[cid]
        assert e.track is not None
        assert e.edge_cache is not None
        assert e.lap_timer.track is e.track  # each circuit gets its own lap timer


def test_unbuilt_id_raises_with_build_hint():
    with pytest.raises(FileNotFoundError) as exc:
        CircuitPool(["no_such_circuit_xyz"], _cfg(["red_bull_ring"]))
    assert "build" in str(exc.value).lower()  # the loader's build hint, never a silent drop


def test_per_circuit_pole_resolves_from_config():
    cfg = _cfg(["red_bull_ring", "monza"])
    pole_rbr, missing_rbr = resolve_pole("red_bull_ring", cfg, config_root=CONFIG_ROOT)
    assert pole_rbr == pytest.approx(64.3)
    assert missing_rbr is False
    pole_monza, missing_monza = resolve_pole("monza", cfg, config_root=CONFIG_ROOT)
    assert pole_monza == pytest.approx(79.8)
    assert missing_monza is False


def test_missing_pole_is_flagged_not_zero():
    pole, missing = resolve_pole("definitely_not_a_circuit", _cfg(["red_bull_ring"]), CONFIG_ROOT)
    assert pole == 0.0
    assert missing is True


def test_pool_binds_resolved_pole_to_lap_timer():
    pool = CircuitPool(["red_bull_ring", "monza"], _cfg(["red_bull_ring", "monza"]))
    assert pool.entries["red_bull_ring"].lap_timer.pole == pytest.approx(64.3)
    assert pool.entries["monza"].lap_timer.pole == pytest.approx(79.8)
    assert pool.entries["red_bull_ring"].pole_missing is False


def test_pool_ids_fallback_to_track_id_when_empty():
    cfg = load_config("default", overrides=["track_id=red_bull_ring"], config_root=CONFIG_ROOT)
    assert pool_ids_from_config(cfg, "red_bull_ring") == ["red_bull_ring"]  # default pool empty
    cfg.circuits.pool = ["monza", "red_bull_ring"]
    assert pool_ids_from_config(cfg, "red_bull_ring") == ["monza", "red_bull_ring"]


def test_set_active_narrows_then_restores_full_pool():
    pool = CircuitPool(["red_bull_ring", "monza"], _cfg(["red_bull_ring", "monza"]))
    pool.set_active(["monza"])
    assert pool.active == ["monza"]
    pool.set_active([])  # empty => full configured pool (curriculum widening)
    assert set(pool.active) == {"red_bull_ring", "monza"}


def test_set_active_refuses_id_not_in_built_pool():
    pool = CircuitPool(["red_bull_ring", "monza"], _cfg(["red_bull_ring", "monza"]))
    # catalunya is built, but it was never loaded into THIS pool -> the curriculum cannot select it
    with pytest.raises(KeyError):
        pool.set_active(["catalunya"])
