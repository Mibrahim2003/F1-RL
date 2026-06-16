"""Track loading.

Phase 1 only has the procedural oval, so this is a thin dispatcher. The cached-``.npz``
path for real circuits (Phase 2) is stubbed here so callers have a stable entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from f1rl.track.oval import OvalParams, build_oval
from f1rl.track.schema import Track


def load_track(track_id: str = "oval", cfg: Any | None = None) -> Track:
    """Load a track by id.

    Args:
        track_id: ``"oval"`` in Phase 1. Real-circuit ids load from ``data/tracks`` later.
        cfg: Optional config node with oval geometry overrides.

    Raises:
        NotImplementedError: For non-oval ids until the Phase 2 pipeline lands.
    """
    if track_id == "oval":
        params = OvalParams.from_config(cfg) if cfg is not None else None
        return build_oval(params, name="oval")

    cache = Path("data/tracks") / f"{track_id}.npz"
    raise NotImplementedError(
        f"Real-circuit loading from {cache} arrives in Phase 2 (FastF1 pipeline). "
        f"Phase 1 only provides track_id='oval'."
    )
