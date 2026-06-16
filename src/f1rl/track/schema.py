"""The :class:`Track` schema (TECHNICAL_DESIGN.md §6).

All geometry is in SI meters in the world frame. The same schema is produced by the
procedural oval (Phase 1) and the FastF1 build pipeline (Phase 2), so everything
downstream — env, timing, renderer — depends only on this structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Track:
    """A closed (or open) racing circuit sampled along its centerline.

    Arrays are parallel and indexed by centerline sample ``i`` (``N`` samples). Tangent
    and normal are unit vectors; the normal points to the **left** of travel.
    """

    name: str
    centerline: np.ndarray  # (N, 2) meters
    tangent: np.ndarray  # (N, 2) unit vectors
    normal: np.ndarray  # (N, 2) unit vectors, point left
    s: np.ndarray  # (N,) cumulative arc length, meters
    curvature: np.ndarray  # (N,) signed, 1/m
    half_width_left: np.ndarray  # (N,) meters to the asphalt edge
    half_width_right: np.ndarray  # (N,) meters to the asphalt edge
    runoff_width: np.ndarray  # (N,) meters of grass/gravel beyond the edge
    gradient: np.ndarray  # (N,) slope, default zeros
    closed: bool

    @property
    def length(self) -> float:
        """Total lap length in meters (arc length back to the start for a closed loop)."""
        if self.closed:
            last = self.centerline[-1]
            first = self.centerline[0]
            return float(self.s[-1] + np.hypot(*(first - last)))
        return float(self.s[-1])

    def nearest_index(self, x: float, y: float) -> int:
        """Index of the centerline sample closest to ``(x, y)`` (brute force)."""
        dx = self.centerline[:, 0] - x
        dy = self.centerline[:, 1] - y
        return int(np.argmin(dx * dx + dy * dy))

    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box ``(min_x, min_y, max_x, max_y)`` in meters."""
        mn = self.centerline.min(axis=0)
        mx = self.centerline.max(axis=0)
        return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the ``GET /track`` endpoint.

        Arrays become nested lists; the renderer reconstructs the asphalt ribbon, kerbs,
        and start/finish line from these.
        """
        mn_x, mn_y, mx_x, mx_y = self.bounds()
        return {
            "name": self.name,
            "closed": self.closed,
            "length": self.length,
            "centerline": self.centerline.round(3).tolist(),
            "tangent": self.tangent.round(5).tolist(),
            "normal": self.normal.round(5).tolist(),
            "half_width_left": self.half_width_left.round(3).tolist(),
            "half_width_right": self.half_width_right.round(3).tolist(),
            "runoff_width": self.runoff_width.round(3).tolist(),
            "bounds": {"min_x": mn_x, "min_y": mn_y, "max_x": mx_x, "max_y": mx_y},
            "start_finish": {
                "point": self.centerline[0].round(3).tolist(),
                "tangent": self.tangent[0].round(5).tolist(),
                "normal": self.normal[0].round(5).tolist(),
            },
        }
