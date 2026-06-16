"""Config loading: default.yaml merges in its track_id's track/<id>.yaml under cfg.track."""

from __future__ import annotations

import pytest

from f1rl.utils.config import load_config


def test_default_merges_track_config():
    cfg = load_config("default")
    assert cfg.track_id == "oval"
    assert cfg.track.id == "oval"
    assert cfg.track.pole_time_s == pytest.approx(47.5)
    assert cfg.track.total_laps == 30


def test_overrides_apply_after_track_merge():
    cfg = load_config("default", overrides=["track.pole_time_s=10.0", "physics.mass=820"])
    assert cfg.track.pole_time_s == pytest.approx(10.0)
    assert cfg.physics.mass == pytest.approx(820)


def test_missing_track_config_raises(tmp_path):
    (tmp_path / "default.yaml").write_text("track_id: nope\n")
    with pytest.raises(FileNotFoundError):
        load_config("default", config_root=tmp_path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config("default", config_root=tmp_path)
