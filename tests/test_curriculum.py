"""Curriculum scheduler contract (spec §2 curriculum, plan §C).

The scheduler reads a config stage table, finds the active stage for the current timestep,
and pushes the stage's conditions into every worker via ``env_method('apply_conditions')`` —
conditions only, never the obs layout. Built from the public functions
(``parse_stages`` / ``active_stage`` / ``CurriculumCallback``) and the env's public
``apply_conditions`` hook.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from f1rl.train.curriculum import (
    CurriculumCallback,
    CurriculumStage,
    active_stage,
    parse_stages,
)

_TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"
pytestmark = pytest.mark.skipif(
    not (_TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)


def test_parse_stages_reads_and_sorts(dyn_cfg):
    stages = parse_stages(dyn_cfg)
    assert len(stages) == 4
    starts = [s.start_step for s in stages]
    assert starts == sorted(starts)
    assert stages[0].start_step == 0
    assert stages[-1].weather == "sampled"


def test_parse_stages_empty_when_disabled(cfg):
    # default.yaml has curriculum.enabled: false.
    assert parse_stages(cfg) == []


def test_active_stage_selection_at_thresholds():
    stages = [
        CurriculumStage(0, mu_base=2.3),
        CurriculumStage(600_000, mu_base=1.95),
        CurriculumStage(1_200_000, mu_base=1.95, wear_rate=0.02),
    ]
    assert active_stage(stages, 0).start_step == 0
    assert active_stage(stages, 599_999).start_step == 0
    assert active_stage(stages, 600_000).start_step == 600_000
    assert active_stage(stages, 1_500_000).start_step == 1_200_000


class _StubVecEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, name, **kwargs):
        self.calls.append((name, kwargs))


class _StubModel:
    """Minimal stand-in so ``callback.training_env`` (= model.get_env()) returns the stub."""

    def __init__(self, env):
        self._env = env

    def get_env(self):
        return self._env


def test_callback_pushes_active_stage_into_workers(dyn_cfg):
    cb = CurriculumCallback(dyn_cfg, verbose=0)
    stub = _StubVecEnv()
    cb.model = _StubModel(stub)
    cb.num_timesteps = 1_250_000  # inside the wear stage
    cb._maybe_apply(force=True)

    assert stub.calls, "callback did not push conditions to the workers"
    name, kwargs = stub.calls[-1]
    assert name == "apply_conditions"
    assert kwargs["wear_rate"] == pytest.approx(0.02)
    assert kwargs["mu_base"] == pytest.approx(1.95)


def test_callback_pushes_again_only_on_stage_change(dyn_cfg):
    cb = CurriculumCallback(dyn_cfg, verbose=0)
    stub = _StubVecEnv()
    cb.model = _StubModel(stub)

    cb.num_timesteps = 0
    cb._maybe_apply(force=True)
    cb.num_timesteps = 10_000  # same stage -> no new push
    cb._maybe_apply(force=False)
    assert len(stub.calls) == 1

    cb.num_timesteps = 600_000  # next stage -> one more push
    cb._maybe_apply(force=False)
    assert len(stub.calls) == 2


# --- env-side hook ----------------------------------------------------------------------


def test_apply_conditions_changes_grip_and_wear(dyn_cfg):
    from f1rl.env.single_agent import RacingEnv

    env = RacingEnv(dyn_cfg, seed=0)
    env.reset(seed=0)
    g_before = env.conditions.grip_at(env.track, 0, 0.0, wear=0.0, compound=0)

    env.apply_conditions(mu_base=1.0, wear_rate=0.05, weather="wet")
    g_after = env.conditions.grip_at(env.track, 0, 0.0, wear=0.0, compound=0)

    assert g_after < g_before  # lower mu_base + wet weather -> less grip
    assert env._weather_mode == "wet"
    assert env.physics.params.wear_rate == pytest.approx(0.05)
