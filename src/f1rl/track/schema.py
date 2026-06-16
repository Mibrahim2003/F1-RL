"""The :class:`Track` schema (TECHNICAL_DESIGN.md §6).

All geometry is in SI meters in the world frame. The same schema is produced by the
procedural oval (Phase 1) and the FastF1/OSM build pipeline (Phase 2), so everything
downstream — env, timing, renderer — depends only on this structure.

Phase 2 splits the single Phase-1 ``runoff_width`` band into explicit
``kerb_width`` / ``grass_width`` / ``gravel_width`` bands (each measured outward past the
asphalt edge) and adds the build metadata needed for the scale check and the low-confidence
badge: ``country``, ``official_length_m``, and ``source``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Default scale-check tolerance: computed arc length must sit within this fraction of the
# published lap length, else the circuit is flagged low-confidence. Overridable per-track
# (build pipeline reads it from config) but kept here so the schema is self-contained.
DEFAULT_LENGTH_TOLERANCE = 0.05


@dataclass
class Track:
    """A closed (or open) racing circuit sampled along its centerline.

    Arrays are parallel and indexed by centerline sample ``i`` (``N`` samples). Tangent
    and normal are unit vectors; the normal points to the **left** of travel. Surface bands
    are widths measured outward from the asphalt edge: kerb first, then grass, then gravel.
    """

    name: str
    centerline: np.ndarray  # (N, 2) meters
    tangent: np.ndarray  # (N, 2) unit vectors
    normal: np.ndarray  # (N, 2) unit vectors, point left
    s: np.ndarray  # (N,) cumulative arc length, meters
    curvature: np.ndarray  # (N,) signed, 1/m
    half_width_left: np.ndarray  # (N,) meters to the asphalt edge
    half_width_right: np.ndarray  # (N,) meters to the asphalt edge
    kerb_width: np.ndarray  # (N,) red/white band past the asphalt edge
    grass_width: np.ndarray  # (N,) green band past the kerb
    gravel_width: np.ndarray  # (N,) sand/gravel band, where present
    gradient: np.ndarray  # (N,) slope, default zeros
    closed: bool
    country: str = ""
    official_length_m: float = 0.0  # published length for the scale check; <=0 = unknown
    source: str = "manual"  # "fastf1" | "osm" | "fastf1+osm" | "manual" | "procedural"
    # Optional per-sample runoff label (0 = grass, 1 = gravel). ``None`` when uniform bands.
    surface_zones: np.ndarray | None = None
    length_tolerance: float = DEFAULT_LENGTH_TOLERANCE
    # Build pipeline can force-flag a track (self-intersection, missing/poor OSM, manual).
    low_confidence_override: bool = field(default=False)

    @property
    def length(self) -> float:
        """Total lap length in meters (arc length back to the start for a closed loop)."""
        if self.closed:
            last = self.centerline[-1]
            first = self.centerline[0]
            return float(self.s[-1] + np.hypot(*(first - last)))
        return float(self.s[-1])

    @property
    def length_error(self) -> float | None:
        """Relative error of the computed length vs ``official_length_m`` (``None`` if unknown)."""
        if self.official_length_m <= 0:
            return None
        return abs(self.length - self.official_length_m) / self.official_length_m

    @property
    def low_confidence(self) -> bool:
        """True when the track needs a human eye: manual source, build flag, or length miss."""
        if self.low_confidence_override or self.source == "manual":
            return True
        err = self.length_error
        return err is not None and err > self.length_tolerance

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

        Arrays become nested lists; the renderer reconstructs the asphalt ribbon, the kerb,
        grass, and gravel bands, and the start/finish line from these.
        """
        mn_x, mn_y, mx_x, mx_y = self.bounds()
        err = self.length_error
        return {
            "name": self.name,
            "country": self.country,
            "closed": self.closed,
            "length": self.length,
            "official_length_m": self.official_length_m,
            "length_error": round(err, 4) if err is not None else None,
            "source": self.source,
            "low_confidence": self.low_confidence,
            "centerline": self.centerline.round(3).tolist(),
            "tangent": self.tangent.round(5).tolist(),
            "normal": self.normal.round(5).tolist(),
            "half_width_left": self.half_width_left.round(3).tolist(),
            "half_width_right": self.half_width_right.round(3).tolist(),
            "kerb_width": self.kerb_width.round(3).tolist(),
            "grass_width": self.grass_width.round(3).tolist(),
            "gravel_width": self.gravel_width.round(3).tolist(),
            "bounds": {"min_x": mn_x, "min_y": mn_y, "max_x": mx_x, "max_y": mx_y},
            "start_finish": {
                "point": self.centerline[0].round(3).tolist(),
                "tangent": self.tangent[0].round(5).tolist(),
                "normal": self.normal[0].round(5).tolist(),
            },
        }

    # ----- cached-file round trip -------------------------------------------------------

    def save_npz(self, path: Any) -> None:
        """Save to ``data/tracks/<name>.npz``. Round-trips exactly via :meth:`from_npz`."""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        has_zones = self.surface_zones is not None
        np.savez(
            p,
            centerline=self.centerline,
            tangent=self.tangent,
            normal=self.normal,
            s=self.s,
            curvature=self.curvature,
            half_width_left=self.half_width_left,
            half_width_right=self.half_width_right,
            kerb_width=self.kerb_width,
            grass_width=self.grass_width,
            gravel_width=self.gravel_width,
            gradient=self.gradient,
            surface_zones=(self.surface_zones if has_zones else np.empty(0, dtype=np.int8)),
            has_surface_zones=np.array(has_zones),
            name=np.array(self.name),
            country=np.array(self.country),
            source=np.array(self.source),
            closed=np.array(self.closed),
            official_length_m=np.array(self.official_length_m, dtype=float),
            length_tolerance=np.array(self.length_tolerance, dtype=float),
            low_confidence_override=np.array(self.low_confidence_override),
        )

    @classmethod
    def from_npz(cls, path: Any) -> Track:
        """Load a cached track. Inverse of :meth:`save_npz`."""
        with np.load(path, allow_pickle=False) as d:
            has_zones = bool(d["has_surface_zones"].item())
            return cls(
                name=str(d["name"].item()),
                centerline=d["centerline"],
                tangent=d["tangent"],
                normal=d["normal"],
                s=d["s"],
                curvature=d["curvature"],
                half_width_left=d["half_width_left"],
                half_width_right=d["half_width_right"],
                kerb_width=d["kerb_width"],
                grass_width=d["grass_width"],
                gravel_width=d["gravel_width"],
                gradient=d["gradient"],
                surface_zones=d["surface_zones"] if has_zones else None,
                closed=bool(d["closed"].item()),
                country=str(d["country"].item()),
                source=str(d["source"].item()),
                official_length_m=float(d["official_length_m"].item()),
                length_tolerance=float(d["length_tolerance"].item()),
                low_confidence_override=bool(d["low_confidence_override"].item()),
            )
