"""Config loading with OmegaConf (TECHNICAL_DESIGN.md §14).

Configs live as YAML under ``configs/``. ``load_config`` reads ``default.yaml``, merges in
the per-circuit file named by its ``track_id`` (``configs/track/<track_id>.yaml``) under the
``track`` key, then applies command-line-style dotlist overrides. No tuning constant lives in
logic; everything tunable is here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def load_config(
    name: str = "default",
    overrides: Sequence[str] | None = None,
    config_root: Path | None = None,
) -> DictConfig:
    """Load a YAML config by name, merge in its track config, and apply dotlist overrides.

    Args:
        name: File stem under ``configs/`` (without ``.yaml``).
        overrides: Dotlist overrides, e.g. ``["physics.mass=820"]``.
        config_root: Override the config directory (used in tests).

    Returns:
        The merged :class:`omegaconf.DictConfig`, with track geometry under ``cfg.track``.
    """
    root = config_root or CONFIG_ROOT
    path = root / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)

    track_id = cfg.get("track_id")
    if track_id is not None:
        track_path = root / "track" / f"{track_id}.yaml"
        if not track_path.exists():
            raise FileNotFoundError(f"Track config not found: {track_path}")
        cfg.track = OmegaConf.load(track_path)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(cfg, DictConfig)
    return cfg


def load_track_config(
    track_id: str,
    name: str = "default",
    config_root: Path | None = None,
) -> DictConfig:
    """Load the base config but with ``cfg.track`` set to a chosen circuit.

    Phase 2 needs to load any circuit's meta (``pole_time_s``, ``total_laps``, widths) at
    runtime for the track selector, independent of ``default.yaml``'s ``track_id``. This
    loads ``configs/track/<track_id>.yaml`` into ``cfg.track`` and sets ``cfg.track_id``.

    Raises:
        FileNotFoundError: If the circuit config does not exist.
    """
    root = config_root or CONFIG_ROOT
    cfg = load_config(name, config_root=config_root)
    track_path = root / "track" / f"{track_id}.yaml"
    if not track_path.exists():
        raise FileNotFoundError(f"Track config not found: {track_path}")
    cfg.track_id = track_id
    cfg.track = OmegaConf.load(track_path)
    return cfg
