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
