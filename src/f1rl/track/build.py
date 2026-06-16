"""Offline circuit-build pipeline (TECHNICAL_DESIGN.md §6, Phase 2).

**Build-time only.** FastF1, Shapely, and ``requests`` are imported lazily inside the
acquisition helpers so this module never drags a network dependency into the runtime/training
path. Nothing here is imported by the env or the server hot path — the server loads cached
``.npz`` files via :func:`f1rl.track.loader.load_track`.

Pipeline per circuit (run by :mod:`scripts.build_all_tracks`):

1. Acquire shape — :func:`shape_from_fastf1` reads a clean fast lap's X/Y trace.
2. Acquire width — :func:`width_from_osm` offsets the OSM asphalt edges against the
   centerline (Shapely); falls back to the per-circuit config constant when OSM is
   missing/poor.
3. Resample + smooth — periodic SciPy spline at uniform ~2–3 m spacing.
4. Recenter to the centroid origin.
5. Geometry — shared :mod:`f1rl.track.geometry`.
6. Bands — kerb / grass / gravel from config; gravel only on flagged corner zones.
7. Validate — arc length vs official length, Shapely self-intersection, bounded widths.
8. Save — ``data/tracks/<id>.npz`` + a row in ``data/tracks/_build_report.json``.

The pure processing (steps 3–8) lives in :func:`build_from_points`, which takes raw points
plus config and needs no network — that is the unit-tested path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.track.geometry import arc_length, frames, signed_curvature
from f1rl.track.schema import DEFAULT_LENGTH_TOLERANCE, Track

DEFAULT_CACHE_DIR = Path("data/tracks")
DEFAULT_OSM_CACHE_DIR = Path("data/raw_telemetry/osm")
DEFAULT_FASTF1_CACHE_DIR = Path("data/raw_telemetry/fastf1")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class BuildError(RuntimeError):
    """Raised when a circuit cannot be built (acquisition failed, no usable shape)."""


@dataclass
class BuildConfig:
    """Per-circuit build parameters, parsed from ``configs/track/<id>.yaml``."""

    id: str
    country: str = ""
    official_length_m: float = 0.0
    source: str = "fastf1"  # requested source; actual source recorded on the Track
    spacing: float = 3.0  # target centerline sample spacing, m
    smoothing: float = 0.0  # spline smoothing factor (0 = interpolate)
    half_width: float = 6.0  # default half-width per side (no OSM), m
    half_width_straight: float = 7.0  # widen toward this on low-curvature straights, m
    min_half_width: float = 3.0
    max_half_width: float = 12.0
    kerb_width: float = 1.0
    grass_width: float = 8.0
    gravel_width: float = 6.0
    gravel_on_corners: bool = True  # place gravel runoff at corner zones
    corner_kappa: float = 0.004  # |curvature| above this counts as a corner (1/m)
    length_tolerance: float = DEFAULT_LENGTH_TOLERANCE
    # FastF1
    fastf1_year: int = 2024
    fastf1_event: str | int | None = None  # GP name or round number
    fastf1_session: str = "Q"  # qualifying gives a clean fast lap
    # OSM
    osm_enabled: bool = False
    osm_query: str | None = None  # raw Overpass QL; if None a default raceway query is built
    osm_relation: int | None = None
    osm_bbox: list[float] | None = None  # [south, west, north, east]
    osm_min_inside_frac: float = 0.6  # reject OSM width if fewer samples land on asphalt

    @classmethod
    def from_config(cls, cfg: Any) -> BuildConfig:
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d: getattr(cfg, k, d))
        ff1 = get("fastf1", {}) or {}
        osm = get("osm", {}) or {}
        ff1_get = ff1.get if hasattr(ff1, "get") else (lambda k, d: getattr(ff1, k, d))
        osm_get = osm.get if hasattr(osm, "get") else (lambda k, d: getattr(osm, k, d))
        bbox = osm_get("bbox", None)
        return cls(
            id=str(get("id", "unknown")),
            country=str(get("country", "")),
            official_length_m=float(get("official_length_m", 0.0)),
            source=str(get("source", "fastf1")),
            spacing=float(get("spacing", 3.0)),
            smoothing=float(get("smoothing", 0.0)),
            half_width=float(get("half_width", 6.0)),
            half_width_straight=float(get("half_width_straight", get("half_width", 6.0) + 1.0)),
            min_half_width=float(get("min_half_width", 3.0)),
            max_half_width=float(get("max_half_width", 12.0)),
            kerb_width=float(get("kerb_width", 1.0)),
            grass_width=float(get("grass_width", 8.0)),
            gravel_width=float(get("gravel_width", 6.0)),
            gravel_on_corners=bool(get("gravel_on_corners", True)),
            corner_kappa=float(get("corner_kappa", 0.004)),
            length_tolerance=float(get("length_tolerance", DEFAULT_LENGTH_TOLERANCE)),
            fastf1_year=int(ff1_get("year", 2024)),
            fastf1_event=ff1_get("event", None),
            fastf1_session=str(ff1_get("session", "Q")),
            osm_enabled=bool(osm_get("enabled", False)),
            osm_query=osm_get("query", None),
            osm_relation=osm_get("relation", None),
            osm_bbox=list(bbox) if bbox is not None else None,
            osm_min_inside_frac=float(osm_get("min_inside_frac", 0.6)),
        )


# ---------------------------------------------------------------------------------------
# Step 1: shape acquisition (FastF1) — network, lazy import
# ---------------------------------------------------------------------------------------


def shape_from_fastf1(cfg: BuildConfig, cache_dir: Path = DEFAULT_FASTF1_CACHE_DIR) -> np.ndarray:
    """Return the fastest lap's X/Y trace as an ``(M, 2)`` array in meters.

    FastF1 position/telemetry X/Y are in 1/10 m, so the trace is divided by 10. The result
    is the driven racing line (a faithful, recognizable shape and lap distance), not the exact
    centerline — accepted per the spec for visualization.
    """
    import fastf1  # build-time only

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    event: Any = cfg.fastf1_event if cfg.fastf1_event is not None else cfg.id
    session = fastf1.get_session(cfg.fastf1_year, event, cfg.fastf1_session)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    lap = session.laps.pick_fastest()
    if lap is None or (hasattr(lap, "empty") and lap.empty):
        raise BuildError(
            f"{cfg.id}: no fastest lap in {cfg.fastf1_year} {event} {cfg.fastf1_session}"
        )
    tel = lap.get_telemetry()
    xy = np.column_stack([tel["X"].to_numpy(float), tel["Y"].to_numpy(float)]) / 10.0
    return _clean_trace(xy)


def _clean_trace(xy: np.ndarray) -> np.ndarray:
    """Drop NaNs and consecutive duplicate points from a raw trace."""
    finite = xy[np.isfinite(xy).all(axis=1)]
    if len(finite) < 8:
        raise BuildError("trace has too few finite points")
    keep = np.ones(len(finite), dtype=bool)
    keep[1:] = np.hypot(*(finite[1:] - finite[:-1]).T) > 1e-6
    return finite[keep]


def shape_from_osm(cfg: BuildConfig, cache_dir: Path = DEFAULT_OSM_CACHE_DIR) -> np.ndarray:
    """Shape a circuit with no FastF1 telemetry (e.g. Madrid) from OSM raceway centerlines.

    OSM frequently maps the racing surface centerline as ``highway=raceway`` polylines. This
    fetches those ways, projects to meters, and greedily stitches them into one closed loop.
    Raises :class:`BuildError` if nothing usable is returned (caller flags + skips, per plan).
    """
    data = _overpass_fetch(cfg, cache_dir)
    rings_ll = _parse_osm_polygons(data)
    if not rings_ll:
        raise BuildError(f"{cfg.id}: OSM returned no usable ways")
    segments = [_project_equirect_global(r) for r in rings_ll]
    loop = _stitch_loop(segments)
    if loop is None or len(loop) < 16:
        raise BuildError(f"{cfg.id}: could not stitch OSM ways into a loop")
    return _clean_trace(loop)


def _project_equirect_global(lonlat: np.ndarray, lat0_deg: float | None = None) -> np.ndarray:
    """Equirectangular projection sharing one origin so multiple ways stay registered."""
    lon = lonlat[:, 0]
    lat = lonlat[:, 1]
    lat0 = np.deg2rad(lat0_deg if lat0_deg is not None else lat.mean())
    r = 6_378_137.0
    return np.column_stack([np.deg2rad(lon) * np.cos(lat0) * r, np.deg2rad(lat) * r])


def _stitch_loop(segments: list[np.ndarray], tol: float = 50.0) -> np.ndarray | None:
    """Greedily join polyline segments end-to-end into a single closed loop."""
    segs = [s for s in segments if len(s) >= 2]
    if not segs:
        return None
    chain = list(segs.pop(0))
    progressed = True
    while segs and progressed:
        progressed = False
        tail = np.asarray(chain[-1])
        for i, s in enumerate(segs):
            if np.hypot(*(s[0] - tail)) < tol:
                chain.extend(list(s[1:]))
                segs.pop(i)
                progressed = True
                break
            if np.hypot(*(s[-1] - tail)) < tol:
                chain.extend(list(s[::-1][1:]))
                segs.pop(i)
                progressed = True
                break
    return np.asarray(chain, dtype=float)


# ---------------------------------------------------------------------------------------
# Step 2: width acquisition (OSM / Overpass) — network, lazy import
# ---------------------------------------------------------------------------------------


def width_from_osm(
    cfg: BuildConfig, centerline: np.ndarray, cache_dir: Path = DEFAULT_OSM_CACHE_DIR
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-sample ``(half_width_left, half_width_right)`` from the OSM asphalt outline.

    Fetches (and disk-caches) the Overpass response, projects the asphalt ways to a local
    meter frame, registers them to the FastF1 ``centerline`` with a similarity transform, and
    casts each sample's ±normal ray to the asphalt boundary. Returns ``None`` when OSM is
    disabled, missing, or too poor to register (caller falls back to config widths).
    """
    if not cfg.osm_enabled:
        return None
    try:
        data = _overpass_fetch(cfg, cache_dir)
        rings_ll = _parse_osm_polygons(data)
        if not rings_ll:
            return None
        rings_m = [_project_equirect(r) for r in rings_ll]
        poly = _aligned_asphalt_polygon(rings_m, centerline)
        if poly is None:
            return None
        tangent, normal = frames(centerline, closed=True)
        return _measure_half_widths(centerline, normal, poly, cfg)
    except Exception:
        # Any acquisition/parse/registration failure → config fallback. Never crash a build.
        return None


def _overpass_fetch(cfg: BuildConfig, cache_dir: Path) -> dict:
    """Fetch the Overpass JSON for this circuit, caching the raw response (rate-limit safe)."""
    import requests  # build-time only

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cfg.id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    query = cfg.osm_query or _default_overpass_query(cfg)
    if query is None:
        raise BuildError(f"{cfg.id}: OSM enabled but no query/relation/bbox configured")
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _default_overpass_query(cfg: BuildConfig) -> str | None:
    """Build a default Overpass query for raceway asphalt from a relation id or bbox."""
    if cfg.osm_relation is not None:
        return f"[out:json][timeout:180];relation({cfg.osm_relation});(way(r);>;);out geom;"
    if cfg.osm_bbox is not None:
        s, w, n, e = cfg.osm_bbox
        bbox = f"{s},{w},{n},{e}"
        return (
            "[out:json][timeout:180];"
            f'(way["highway"="raceway"]({bbox});way["area:highway"]({bbox});'
            f'way["leisure"="track"]["sport"="motor"]({bbox}););'
            "(._;>;);out geom;"
        )
    return None


def _parse_osm_polygons(data: dict) -> list[np.ndarray]:
    """Extract way geometries as ``(K, 2)`` lon/lat rings from an Overpass ``out geom`` JSON."""
    rings: list[np.ndarray] = []
    for el in data.get("elements", []):
        geom = el.get("geometry")
        if el.get("type") == "way" and geom:
            ring = np.array([[p["lon"], p["lat"]] for p in geom], dtype=float)
            if len(ring) >= 4:
                rings.append(ring)
    return rings


def _project_equirect(lonlat: np.ndarray) -> np.ndarray:
    """Equirectangular lon/lat → local meters about the ring's mean latitude."""
    lon = lonlat[:, 0]
    lat = lonlat[:, 1]
    lat0 = np.deg2rad(lat.mean())
    r = 6_378_137.0
    x = np.deg2rad(lon) * np.cos(lat0) * r
    y = np.deg2rad(lat) * r
    return np.column_stack([x - x.mean(), y - y.mean()])


def _aligned_asphalt_polygon(rings_m: list[np.ndarray], centerline: np.ndarray):
    """Merge OSM rings into one Shapely polygon and align it to the FastF1 centerline.

    Uses a PCA-based similarity transform (translate, scale, rotate), resolving the
    rotation/flip ambiguity by maximizing the fraction of centerline samples inside the
    asphalt. Returns the aligned Shapely polygon, or ``None`` if registration is poor.
    """
    from shapely.geometry import MultiLineString, Point, Polygon
    from shapely.ops import polygonize, unary_union

    lines = MultiLineString([r for r in rings_m if len(r) >= 2])
    polys = list(polygonize(unary_union(lines)))
    if not polys:
        return None
    osm = unary_union(polys)
    osm_pts = np.asarray(osm.convex_hull.exterior.coords)

    best = None
    best_frac = -1.0
    c_mean = centerline.mean(axis=0)
    c_centered = centerline - c_mean
    c_scale = np.sqrt((c_centered**2).sum(axis=1).mean())
    o_mean = osm_pts.mean(axis=0)
    o_centered = osm_pts - o_mean
    o_scale = np.sqrt((o_centered**2).sum(axis=1).mean())
    if o_scale < 1e-6:
        return None
    scale = c_scale / o_scale

    # Principal axis of each cloud.
    c_ang = _principal_angle(c_centered)
    o_ang = _principal_angle(o_centered)
    for flip in (1.0, -1.0):
        for half in (0.0, np.pi):
            theta = (c_ang - o_ang) + half
            cos, sin = np.cos(theta), np.sin(theta)
            rot = np.array([[cos, -sin], [sin, cos]])
            sx = np.array([[flip, 0.0], [0.0, 1.0]])
            transformed = _apply_similarity(
                np.asarray(osm.exterior.coords), o_mean, rot @ sx, scale, c_mean
            )
            poly = Polygon(transformed)
            if not poly.is_valid:
                poly = poly.buffer(0)
            inside = sum(
                poly.contains(Point(p)) for p in centerline[:: max(1, len(centerline) // 200)]
            )
            n = len(centerline[:: max(1, len(centerline) // 200)])
            frac = inside / max(1, n)
            if frac > best_frac:
                best_frac, best = frac, poly
    return best


def _principal_angle(centered: np.ndarray) -> float:
    cov = np.cov(centered.T)
    w, v = np.linalg.eigh(cov)
    major = v[:, int(np.argmax(w))]
    return float(np.arctan2(major[1], major[0]))


def _apply_similarity(
    pts: np.ndarray, src_mean: np.ndarray, rot: np.ndarray, scale: float, dst_mean: np.ndarray
) -> np.ndarray:
    return (pts - src_mean) @ rot.T * scale + dst_mean


def _measure_half_widths(
    centerline: np.ndarray, normal: np.ndarray, poly, cfg: BuildConfig
) -> tuple[np.ndarray, np.ndarray] | None:
    """Cast ±normal rays from each sample to the asphalt boundary; clamp to config bounds."""
    from shapely.geometry import LineString, Point

    n = len(centerline)
    hl = np.full(n, cfg.half_width)
    hr = np.full(n, cfg.half_width)
    boundary = poly.boundary
    probe = cfg.max_half_width * 1.5
    inside_hits = 0
    for i in range(n):
        c = centerline[i]
        nv = normal[i]
        if not poly.contains(Point(c)):
            continue
        inside_hits += 1
        for sign, arr in ((+1.0, hl), (-1.0, hr)):
            ray = LineString([c, c + sign * nv * probe])
            hit = ray.intersection(boundary)
            d = _nearest_hit_distance(c, hit)
            if d is not None and cfg.min_half_width <= d <= cfg.max_half_width:
                arr[i] = d
    if inside_hits / n < cfg.osm_min_inside_frac:
        return None
    return hl, hr


def _nearest_hit_distance(origin: np.ndarray, hit) -> float | None:
    """Closest distance from ``origin`` to a Shapely intersection result, or ``None``."""
    if hit.is_empty:
        return None
    coords: list = []
    if hit.geom_type == "Point":
        coords = [(hit.x, hit.y)]
    elif hasattr(hit, "geoms"):
        for g in hit.geoms:
            if g.geom_type == "Point":
                coords.append((g.x, g.y))
            else:
                coords.extend(list(g.coords))
    else:
        coords = list(getattr(hit, "coords", []))
    if not coords:
        return None
    d = [float(np.hypot(*(np.asarray(p) - origin))) for p in coords]
    return min(d)


# ---------------------------------------------------------------------------------------
# Steps 3–8: pure processing (no network) — the unit-tested path
# ---------------------------------------------------------------------------------------


def resample_smooth(
    points: np.ndarray, spacing: float, smoothing: float, closed: bool
) -> np.ndarray:
    """Resample to uniform ~``spacing`` m via a (periodic, when closed) SciPy spline."""
    from scipy.interpolate import splev, splprep

    pts = np.asarray(points, dtype=float)
    if closed and np.hypot(*(pts[0] - pts[-1])) > 1e-6:
        pts = np.vstack([pts, pts[:1]])
    # Rough length for sample count.
    seg = np.hypot(*np.diff(pts, axis=0).T)
    length = float(seg.sum())
    n_out = max(16, int(round(length / max(0.5, spacing))))
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing, per=1 if closed else 0)
    u = np.linspace(0.0, 1.0, n_out, endpoint=not closed)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def recenter(points: np.ndarray) -> np.ndarray:
    """Translate so the centroid sits at the origin."""
    return points - points.mean(axis=0)


def assign_bands(
    curvature: np.ndarray, cfg: BuildConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Kerb / grass / gravel band widths from config. Gravel only on flagged corner zones."""
    n = len(curvature)
    kerb = np.full(n, cfg.kerb_width)
    grass = np.full(n, cfg.grass_width)
    gravel = np.zeros(n)
    zones: np.ndarray | None = None
    if cfg.gravel_on_corners and cfg.gravel_width > 0:
        is_corner = np.abs(curvature) >= cfg.corner_kappa
        gravel[is_corner] = cfg.gravel_width
        zones = is_corner.astype(np.int8)
    return kerb, grass, gravel, zones


def default_half_widths(curvature: np.ndarray, cfg: BuildConfig) -> tuple[np.ndarray, np.ndarray]:
    """Config-default half-widths: widen toward ``half_width_straight`` on low-curvature samples."""
    n = len(curvature)
    kappa = np.abs(curvature)
    # 0 on the sharpest corner in this track → 1 on a straight.
    span = max(kappa.max(), 1e-9)
    straightness = 1.0 - np.clip(kappa / span, 0.0, 1.0)
    width = cfg.half_width + (cfg.half_width_straight - cfg.half_width) * straightness
    width = np.clip(width, cfg.min_half_width, cfg.max_half_width)
    return np.full(n, 0.0) + width, np.full(n, 0.0) + width


def validate(track: Track, cfg: BuildConfig) -> tuple[bool, list[str]]:
    """Run the scale check, self-intersection check, and bounded-width check.

    Returns ``(low_confidence, notes)``. Never raises; a failed geometric check is a flag.
    """
    notes: list[str] = []
    low = False

    err = track.length_error
    if err is None:
        notes.append("no official length; scale check skipped")
        low = True
    elif err > cfg.length_tolerance:
        notes.append(f"length error {err:.1%} > tol {cfg.length_tolerance:.1%}")
        low = True

    widths = np.concatenate([track.half_width_left, track.half_width_right])
    if not np.all(np.isfinite(widths)) or widths.min() <= 0:
        notes.append("non-positive half-width")
        low = True

    if not _edges_simple(track):
        notes.append("self-intersecting asphalt edge")
        low = True

    if cfg.source == "manual":
        notes.append("manual source")
        low = True

    return low, notes


def _edges_simple(track: Track) -> bool:
    """True if both offset edges are simple (no self-intersection). Degrades True if no Shapely."""
    try:
        from shapely.geometry import LinearRing
    except Exception:
        return True
    try:
        left = track.centerline + track.normal * track.half_width_left[:, None]
        right = track.centerline - track.normal * track.half_width_right[:, None]
        return bool(LinearRing(left).is_simple and LinearRing(right).is_simple)
    except Exception:
        return False


def build_from_points(
    points: np.ndarray,
    cfg: BuildConfig,
    *,
    half_widths: tuple[np.ndarray, np.ndarray] | None = None,
    source: str | None = None,
    closed: bool = True,
) -> tuple[Track, dict[str, Any]]:
    """Pure build: raw ``(M, 2)`` points + config → ``(Track, report_row)``. No network.

    ``half_widths`` (per-sample left/right, post-resample length) override the config-default
    widths — pass the OSM-measured widths here. ``source`` records the actual data origin.
    """
    centerline = recenter(resample_smooth(points, cfg.spacing, cfg.smoothing, closed))
    tangent, normal = frames(centerline, closed)
    s, seg_len = arc_length(centerline, closed)
    curvature = signed_curvature(tangent, seg_len, closed)

    has_osm_widths = half_widths is not None and len(half_widths[0]) == len(centerline)
    if has_osm_widths:
        hl, hr = half_widths
    else:
        hl, hr = default_half_widths(curvature, cfg)
    used_source = source if source is not None else (cfg.source or "fastf1")

    kerb, grass, gravel, zones = assign_bands(curvature, cfg)

    track = Track(
        name=cfg.id,
        centerline=centerline,
        tangent=tangent,
        normal=normal,
        s=s,
        curvature=curvature,
        half_width_left=hl,
        half_width_right=hr,
        kerb_width=kerb,
        grass_width=grass,
        gravel_width=gravel,
        gradient=np.zeros(len(centerline)),
        closed=closed,
        country=cfg.country,
        official_length_m=cfg.official_length_m,
        source=used_source,
        surface_zones=zones,
        length_tolerance=cfg.length_tolerance,
    )
    low, notes = validate(track, cfg)
    track.low_confidence_override = low

    report = {
        "id": cfg.id,
        "country": cfg.country,
        "source": used_source,
        "points": int(len(centerline)),
        "length_m": round(track.length, 1),
        "official_length_m": cfg.official_length_m,
        "length_error": round(track.length_error, 4) if track.length_error is not None else None,
        "low_confidence": track.low_confidence,
        "notes": notes,
    }
    return track, report


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------


def build_track(
    cfg: BuildConfig,
    *,
    fastf1_cache: Path = DEFAULT_FASTF1_CACHE_DIR,
    osm_cache: Path = DEFAULT_OSM_CACHE_DIR,
) -> tuple[Track, dict[str, Any]]:
    """Acquire shape + width over the network, then run the pure build.

    Shape comes from FastF1 when an event is configured, else from OSM raceway centerlines
    (for circuits without telemetry, e.g. Madrid). Width comes from the OSM asphalt outline
    when available, else the per-circuit config default.
    """
    if cfg.fastf1_event is not None:
        points = shape_from_fastf1(cfg, fastf1_cache)
        shape_src = "fastf1"
    elif cfg.osm_enabled:
        points = shape_from_osm(cfg, osm_cache)
        shape_src = "osm"
    else:
        raise BuildError(f"{cfg.id}: no shape source (set fastf1.event or osm.enabled)")

    # Provisional centerline registers OSM width, then we rebuild with the measured widths.
    provisional = recenter(resample_smooth(points, cfg.spacing, cfg.smoothing, True))
    widths = width_from_osm(cfg, provisional, osm_cache) if shape_src == "fastf1" else None
    # widths only present for the fastf1 path; otherwise source is the shape source ("osm").
    source = "fastf1+osm" if widths is not None else shape_src
    return build_from_points(provisional, cfg, half_widths=widths, source=source, closed=True)


def save_track(
    track: Track,
    report: dict[str, Any],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    report_path: Path | None = None,
) -> Path:
    """Save the ``.npz`` and append the report row to ``data/tracks/_build_report.json``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{track.name}.npz"
    track.save_npz(npz_path)
    rp = report_path or (cache_dir / "_build_report.json")
    rows: dict[str, Any] = {}
    if rp.exists():
        try:
            rows = json.loads(rp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows = {}
    rows[track.name] = report
    rp.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return npz_path
