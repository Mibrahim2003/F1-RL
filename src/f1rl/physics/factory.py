"""Physics-model factory (TECHNICAL_DESIGN.md §5).

One construction point for the swappable :class:`~f1rl.physics.base.PhysicsModel`.
The env, sim loop, and server all build physics through :func:`make_physics`, so the
Part 2 dynamic-model swap is a single config change (``physics.model``) with no caller
edits. Every constant comes from config; SI units throughout.
"""

from __future__ import annotations

from typing import Any

from f1rl.physics.base import PhysicsModel
from f1rl.physics.dynamic import DynamicBicycle, DynamicParams
from f1rl.physics.kinematic import KinematicBicycle, KinematicParams


def make_physics(cfg: Any) -> PhysicsModel:
    """Construct the configured physics model.

    Selects on ``cfg.physics.model``:

    - ``"kinematic"`` (Phase 1, default) → :class:`KinematicBicycle` built from
      :meth:`KinematicParams.from_config`.
    - ``"dynamic"`` (Part 2) → :class:`DynamicBicycle` (friction-circle dynamic model) built
      from :meth:`DynamicParams.from_config`.

    Args:
        cfg: Root config node (mapping or OmegaConf) carrying a ``physics`` block.
            ``physics.model`` selects the model; the remaining ``physics`` keys are the
            tunable SI constants read by the selected model's ``from_config``.

    Returns:
        A :class:`~f1rl.physics.base.PhysicsModel` instance.

    Raises:
        ValueError: when ``physics.model`` is unrecognized.
    """
    physics_cfg = cfg.physics
    model = str(_get(physics_cfg, "model", "kinematic"))

    if model == "kinematic":
        return KinematicBicycle(KinematicParams.from_config(physics_cfg))
    if model == "dynamic":
        return DynamicBicycle(DynamicParams.from_config(physics_cfg))
    raise ValueError(f"unknown physics.model {model!r}; expected 'kinematic' or 'dynamic'.")


def _get(cfg: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a mapping/OmegaConf node or an attribute-style object."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
