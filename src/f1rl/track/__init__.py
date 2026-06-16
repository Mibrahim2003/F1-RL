"""Track representation and builders. Geometry is always in SI meters.

Phase 1 ships a procedurally generated oval (:func:`~f1rl.track.oval.build_oval`).
The real-circuit FastF1 pipeline arrives in Phase 2 behind the same :class:`Track` schema.
"""

from f1rl.track.oval import build_oval
from f1rl.track.schema import Track

__all__ = ["Track", "build_oval"]
