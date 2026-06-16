"""Procedural oval: closed loop, arc length, curvature, unit frames, widths."""

from __future__ import annotations

import math

import numpy as np
import pytest

from f1rl.track.oval import OvalParams, build_oval


def test_shape_and_closed():
    track = build_oval()
    assert track.closed is True
    assert track.centerline.ndim == 2 and track.centerline.shape[1] == 2
    n = len(track.centerline)
    for arr in (track.tangent, track.normal, track.s, track.curvature):
        assert len(arr) == n


def test_arc_length_monotonic_and_total():
    p = OvalParams()
    track = build_oval(p)
    assert np.all(np.diff(track.s) > 0)
    expected = 2 * p.straight_length + 2 * math.pi * p.corner_radius
    assert track.length == pytest.approx(expected, rel=0.01)


def test_tangent_is_unit_and_normal_perpendicular():
    track = build_oval()
    norms = np.linalg.norm(track.tangent, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    dots = np.sum(track.tangent * track.normal, axis=1)
    assert np.allclose(dots, 0.0, atol=1e-6)


def test_curvature_zero_on_straights_and_positive_on_arcs():
    p = OvalParams()
    track = build_oval(p)
    expected_kappa = 1.0 / p.corner_radius
    near_arc = np.abs(np.abs(track.curvature) - expected_kappa) < 0.2 * expected_kappa
    near_straight = np.abs(track.curvature) < 0.05 * expected_kappa
    # Most samples are clearly on a straight or clearly on an arc.
    assert near_arc.sum() > 0
    assert near_straight.sum() > 0
    # Corners are left-handers (CCW traversal) => curvature positive on the arcs.
    assert track.curvature[near_arc].mean() > 0


def test_widths_constant():
    p = OvalParams(half_width=6.5, kerb_width=1.2, runoff_width=10.0)
    track = build_oval(p)
    assert np.allclose(track.half_width_left, 6.5)
    assert np.allclose(track.half_width_right, 6.5)
    # Phase-2 bands: runoff_width maps to grass; the oval has a thin kerb and no gravel.
    assert np.allclose(track.kerb_width, 1.2)
    assert np.allclose(track.grass_width, 10.0)
    assert np.allclose(track.gravel_width, 0.0)


def test_oval_metadata_and_confidence():
    track = build_oval()
    # The oval's official length equals its analytic length, so it is not flagged.
    assert track.source == "procedural"
    assert track.length_error is not None and track.length_error < 0.01
    assert track.low_confidence is False


def test_api_dict_is_json_friendly():
    track = build_oval()
    d = track.to_api_dict()
    assert d["closed"] is True
    assert isinstance(d["centerline"], list)
    assert len(d["centerline"][0]) == 2
    assert "start_finish" in d and "point" in d["start_finish"]
    for key in ("kerb_width", "grass_width", "gravel_width", "country", "source", "low_confidence"):
        assert key in d
