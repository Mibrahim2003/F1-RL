"""Shared fixtures for the Phase 3a training-core tests.

These tests are written from the **spec contracts and public signatures only**
(`.claude/specs/phase-3a-training-core.md` §c/§1d, the plan's fixed-contract section,
`TECHNICAL_DESIGN.md` §7–10). They deliberately do not read env/train implementation
internals, so they catch the implementation being wrong rather than mirroring it.

Fixtures here load the real `red_bull_ring` circuit and build a config rooted at this
repo's `configs/`. They are auto-discovered by pytest (no import needed), matching the
import-free style of ``tests/test_physics_kinematic.py``. The per-file skip marker for a
missing cache is defined locally in each test module via :data:`TRACKS_DIR`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from f1rl.track.loader import load_track
from f1rl.utils.config import load_config

# The Phase 3a test circuit: built, cached, pole 64.3 s (plan: "Circuit = red_bull_ring").
TEST_TRACK_ID = "red_bull_ring"

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
TRACKS_DIR = REPO_ROOT / "data" / "tracks"


@pytest.fixture
def track():
    """The built `red_bull_ring` Track (public schema only)."""
    return load_track(TEST_TRACK_ID, tracks_dir=TRACKS_DIR)


@pytest.fixture
def cfg():
    """Base config with ``track_id`` pointed at the test circuit.

    Built from the repo's real ``configs/`` so obs/reward/physics parameters come from
    config, never hardcoded in the tests (the project's config-driven rule).
    """
    return load_config(
        "default",
        overrides=[f"track_id={TEST_TRACK_ID}"],
        config_root=CONFIG_ROOT,
    )
