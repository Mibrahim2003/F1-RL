"""Offline build pipeline (pure path): geometry, bands, validation, npz round-trip.

The network acquisition (FastF1/Overpass) is build-time only and not exercised here — these
tests cover the pure processing in :func:`f1rl.track.build.build_from_points` plus the
Shapely-backed width measurement, using a synthetic ellipse "circuit".
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from f1rl.track.build import (
    BuildConfig,
    build_from_points,
    default_half_widths,
    resample_smooth,
    save_track,
)
from f1rl.track.geometry import frames
from f1rl.track.schema import Track


def _ellipse(a: float = 500.0, b: float = 300.0, n: int = 400) -> tuple[np.ndarray, float]:
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([a * np.cos(th), b * np.sin(th)])
    # Ramanujan perimeter approximation.
    peri = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    return pts, peri


def _cfg(peri: float, **kw) -> BuildConfig:
    base = dict(
        id="ellipse",
        country="Testland",
        official_length_m=peri,
        spacing=3.0,
        half_width=6.0,
        half_width_straight=8.0,
        kerb_width=1.0,
        grass_width=8.0,
        gravel_width=6.0,
        gravel_on_corners=True,
    )
    base.update(kw)
    return BuildConfig(**base)


def test_arc_length_within_tolerance():
    pts, peri = _ellipse()
    track, report = build_from_points(pts, _cfg(peri), source="fastf1")
    assert track.length_error is not None
    assert track.length_error < 0.02
    assert report["low_confidence"] is False


def test_centerline_closed_and_uniform():
    pts, peri = _ellipse()
    track, _ = build_from_points(pts, _cfg(peri, spacing=3.0), source="fastf1")
    assert track.closed is True
    # Resampled spacing should be near the target, with no big gaps.
    seg = np.hypot(*np.diff(track.centerline, axis=0).T)
    assert seg.max() < 6.0
    assert abs(seg.mean() - 3.0) < 1.0


def test_edges_non_crossing_and_widths_positive():
    pts, peri = _ellipse()
    track, report = build_from_points(pts, _cfg(peri), source="fastf1")
    # validate() runs the Shapely is_simple check; a clean ellipse must not be flagged for it.
    assert "self-intersecting asphalt edge" not in report["notes"]
    assert track.half_width_left.min() > 0
    assert track.half_width_right.min() > 0


def test_default_half_widths_widen_on_straights():
    pts, _ = _ellipse(a=800.0, b=200.0)
    cl = resample_smooth(pts, 3.0, 0.0, True)
    _, _ = frames(cl, True)
    from f1rl.track.geometry import arc_length, signed_curvature

    _, seg = arc_length(cl, True)
    tan, _ = frames(cl, True)
    kappa = signed_curvature(tan, seg, True)
    hl, _ = default_half_widths(kappa, _cfg(1.0, half_width=6.0, half_width_straight=9.0))
    # Straights (low |curvature|) are wider than the tight ends.
    assert hl.max() > hl.min()
    assert hl.max() <= 9.0 + 1e-6 and hl.min() >= 6.0 - 1e-6


def test_gravel_only_on_corners():
    pts, peri = _ellipse()
    track, _ = build_from_points(pts, _cfg(peri, gravel_on_corners=True), source="fastf1")
    assert track.surface_zones is not None
    assert (track.gravel_width > 0).sum() > 0
    # Gravel only where a corner zone is flagged.
    assert np.all((track.gravel_width > 0) == (track.surface_zones == 1))


def test_no_gravel_for_street_circuit():
    pts, peri = _ellipse()
    track, _ = build_from_points(
        pts, _cfg(peri, gravel_on_corners=False, gravel_width=0.0), source="fastf1"
    )
    assert np.all(track.gravel_width == 0.0)
    assert track.surface_zones is None


def test_length_mismatch_flags_low_confidence():
    pts, peri = _ellipse()
    # Claim a wildly wrong official length → scale check fails → flagged.
    track, report = build_from_points(pts, _cfg(peri * 2.0), source="fastf1")
    assert track.low_confidence is True
    assert any("length error" in note for note in report["notes"])


def test_npz_round_trip_exact(tmp_path):
    pts, peri = _ellipse()
    track, report = build_from_points(pts, _cfg(peri), source="fastf1")
    save_track(track, report, cache_dir=tmp_path)
    loaded = Track.from_npz(tmp_path / "ellipse.npz")
    for attr in (
        "centerline",
        "tangent",
        "normal",
        "s",
        "curvature",
        "half_width_left",
        "half_width_right",
        "kerb_width",
        "grass_width",
        "gravel_width",
        "gradient",
    ):
        assert np.allclose(getattr(track, attr), getattr(loaded, attr)), attr
    assert np.array_equal(track.surface_zones, loaded.surface_zones)
    assert loaded.name == track.name
    assert loaded.country == track.country
    assert loaded.source == track.source
    assert loaded.official_length_m == pytest.approx(track.official_length_m)
    assert (tmp_path / "_build_report.json").exists()


def test_save_track_bakes_faithful_api_payload(tmp_path):
    # save_track writes <id>.api.json; it must equal the .npz's recomputed API dict so the web
    # server can serve it verbatim.
    import json

    pts, peri = _ellipse()
    track, report = build_from_points(pts, _cfg(peri), source="fastf1")
    save_track(track, report, cache_dir=tmp_path)
    baked = json.loads((tmp_path / "ellipse.api.json").read_text(encoding="utf-8"))
    assert baked == Track.from_npz(tmp_path / "ellipse.npz").to_api_dict()


def test_bake_all_writes_payloads_and_catalog(tmp_path):
    import json

    from f1rl.track.loader import bake_all

    pts, peri = _ellipse()
    track, report = build_from_points(pts, _cfg(peri), source="fastf1")
    track.save_npz(tmp_path / "ellipse.npz")  # raw .npz only, no baked payload yet
    baked = bake_all(tmp_path)
    assert baked == ["ellipse"]
    assert (tmp_path / "ellipse.api.json").is_file()
    catalog = json.loads((tmp_path / "_catalog.json").read_text(encoding="utf-8"))
    ids = {row["id"] for row in catalog}
    assert {"oval", "ellipse"} <= ids


def test_built_track_api_dict_json_serializable():
    import json

    pts, peri = _ellipse()
    track, _ = build_from_points(pts, _cfg(peri), source="fastf1")
    d = track.to_api_dict()
    json.dumps(d)  # must not raise
    assert d["source"] == "fastf1"
    assert d["low_confidence"] is False
    assert len(d["kerb_width"]) == len(d["centerline"])


def test_width_from_osm_measures_synthetic_band():
    """The Shapely ray-cast width measurement recovers a known offset band."""
    from shapely.geometry import Polygon

    from f1rl.track.build import _measure_half_widths

    a, b, n = 500.0, 300.0, 300
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    cl = np.column_stack([a * np.cos(th), b * np.sin(th)])
    _, normal = frames(cl, True)
    outer = cl + normal * 7.0
    inner = (cl - normal * 5.0)[::-1]
    poly = Polygon(np.vstack([outer, inner])).buffer(0)
    cfg = _cfg(1.0, min_half_width=2.0, max_half_width=12.0)
    res = _measure_half_widths(cl, normal, poly, cfg)
    assert res is not None
    hl, hr = res
    assert float(np.median(hl)) == pytest.approx(7.0, abs=0.2)
    assert float(np.median(hr)) == pytest.approx(5.0, abs=0.2)
