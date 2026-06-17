"""Grip-pipeline contract (spec §4, plan "Grip pipeline").

``effective_grip = mu_base * tire(compound, wear) * weather * surface`` — one scalar gating
the friction circle. Tests assert the product form and the monotone direction of each factor
(wear up -> grip down; soft > medium > hard; wet < dry; gravel < grass < kerb < asphalt),
from the public signatures only.
"""

from __future__ import annotations

import pytest

from f1rl.physics.tires import (
    ASPHALT,
    GRASS,
    GRAVEL,
    KERB,
    SurfaceParams,
    TireParams,
    WeatherParams,
    effective_grip,
    surface_factor,
    tire_factor,
    weather_factor,
)

SOFT, MEDIUM, HARD, INTER, WET = 0, 1, 2, 3, 4


def _params():
    return TireParams(), WeatherParams(), SurfaceParams()


def test_effective_grip_is_product_of_four_factors():
    tires, weather, surface = _params()
    compound, wear, w, zone = MEDIUM, 0.4, "damp", KERB
    expected = (
        tires.mu_base
        * tire_factor(compound, wear, tires)
        * weather_factor(w, weather)
        * surface_factor(zone, surface)
    )
    got = effective_grip(compound, wear, w, zone, tires, weather, surface)
    assert got == pytest.approx(expected)


def test_wear_lowers_grip_monotonically():
    tires, _, _ = _params()
    grips = [tire_factor(SOFT, wear, tires) for wear in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b <= a for a, b in zip(grips, grips[1:], strict=False)), grips
    assert grips[-1] < grips[0]


def test_compound_ordering_soft_gt_medium_gt_hard_at_equal_wear():
    tires, _, _ = _params()
    soft = tire_factor(SOFT, 0.0, tires)
    medium = tire_factor(MEDIUM, 0.0, tires)
    hard = tire_factor(HARD, 0.0, tires)
    assert soft > medium > hard


def test_weather_wet_lower_than_dry():
    _, weather, _ = _params()
    assert (
        weather_factor("wet", weather)
        < weather_factor("damp", weather)
        <= weather_factor("dry", weather)
    )
    assert weather_factor("dry", weather) == pytest.approx(1.0)


def test_surface_offtrack_lower_than_asphalt():
    _, _, surface = _params()
    asphalt = surface_factor(ASPHALT, surface)
    kerb = surface_factor(KERB, surface)
    grass = surface_factor(GRASS, surface)
    gravel = surface_factor(GRAVEL, surface)
    assert asphalt == pytest.approx(1.0)
    assert gravel < grass < kerb <= asphalt


def test_grip_bounds_sane():
    tires, weather, surface = _params()
    # Best case (fresh soft, dry, asphalt) ~ mu_base; worst case (worn, wet, gravel) > 0.
    best = effective_grip(SOFT, 0.0, "dry", ASPHALT, tires, weather, surface)
    worst = effective_grip(HARD, 1.0, "wet", GRAVEL, tires, weather, surface)
    assert best == pytest.approx(tires.mu_base * tires.compound_grip[SOFT])
    assert 0.0 <= worst < best


def test_tire_factor_never_negative_at_full_wear():
    tires = TireParams(compound_wear_falloff=(2.0, 2.0, 2.0, 2.0, 2.0))  # exaggerated falloff
    assert tire_factor(SOFT, 1.0, tires) == 0.0  # clamped at 0, not negative


def test_from_config_reads_blocks():
    cfg = {
        "tires": {"mu_base": 1.4, "compound_grip": [1.0, 0.9, 0.8, 0.7, 0.6]},
        "weather": {"wet_factor": 0.5},
        "surface": {"gravel": 0.2},
    }
    tires = TireParams.from_config(cfg["tires"])
    weather = WeatherParams.from_config(cfg["weather"])
    surface = SurfaceParams.from_config(cfg["surface"])
    assert tires.mu_base == pytest.approx(1.4)
    assert weather_factor("wet", weather) == pytest.approx(0.5)
    assert surface_factor(GRAVEL, surface) == pytest.approx(0.2)
