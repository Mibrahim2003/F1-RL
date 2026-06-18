"""Deterministic, agent-free lap-time calibration (spec §1d/§4, plan §C "Calibration").

The benchmark is only fair if a clean *optimal* lap on the configured car lands near the real
pole. This estimates that optimal lap time as a **pure function of the physics config + track
geometry** — no policy, no training compute — via a velocity-profile model:

1. **Lateral grip limit** per centerline sample: the fastest speed that the friction circle
   can hold through curvature ``kappa`` (the aero term makes the limit speed-dependent, so we
   solve for ``v`` in closed form).
2. **Forward-backward pass**: bound acceleration by engine force and braking by brake force,
   each within the *remaining* friction after cornering, to get a feasible speed profile.
3. Integrate ``dt = ds / v`` around the lap for the clean optimal time.

Sweep a lever (``downforce_coeff``, then ``mu_base``, then ``max_engine_force``) and print a
table of estimate vs ``pole_time_s``; bake the chosen value into ``rbr_dynamic.yaml``.

CLI::

    .venv/Scripts/python.exe -m f1rl.train.calibrate --config experiment/rbr_dynamic \\
        --sweep downforce_coeff --values 2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from f1rl.physics.dynamic import DynamicParams, G
from f1rl.physics.tires import TireParams
from f1rl.track.loader import load_track
from f1rl.track.schema import Track


@dataclass(frozen=True)
class CalibrationInputs:
    """The physics levers the lap-time estimate depends on (all SI)."""

    grip: float  # mu_base * compound_grip[start] on clean dry asphalt (friction coefficient)
    mass: float
    downforce_coeff: float
    max_engine_force: float
    max_brake_force: float
    drag_coeff: float
    rolling_coeff: float
    v_max: float  # top speed cap (engine vs drag), m/s


def inputs_from_cfg(cfg: Any) -> CalibrationInputs:
    """Assemble the calibration levers from a config (dynamic physics + tires blocks)."""
    p = DynamicParams.from_config(cfg.physics)
    tires = TireParams.from_config(cfg)
    grip = tires.mu_base * tires.compound_grip[tires.start_compound]
    v_max = float(np.sqrt(p.max_engine_force / p.drag_coeff)) if p.drag_coeff > 0 else 120.0
    return CalibrationInputs(
        grip=grip,
        mass=p.mass,
        downforce_coeff=p.downforce_coeff,
        max_engine_force=p.max_engine_force,
        max_brake_force=p.max_brake_force,
        drag_coeff=p.drag_coeff,
        rolling_coeff=p.rolling_coeff,
        v_max=v_max,
    )


def _segment_ds(track: Track) -> np.ndarray:
    """Arc-length of each centerline segment (m), wrap-aware for a closed loop."""
    c = np.asarray(track.centerline, dtype=np.float64)
    nxt = np.roll(c, -1, axis=0) if track.closed else np.vstack([c[1:], c[-1:]])
    return np.hypot(nxt[:, 0] - c[:, 0], nxt[:, 1] - c[:, 1])


def _lateral_limit(kappa: np.ndarray, ins: CalibrationInputs) -> np.ndarray:
    """Pure-cornering speed limit per sample from the friction circle (aero-aware).

    Balance: ``m v^2 |kappa| = grip*m*g + downforce_coeff*v^2``  ->
    ``v^2 = grip*m*g / (m|kappa| - downforce_coeff)``. When the denominator is <= 0 the aero
    grip alone covers the corner, so the limit is the straight-line top speed ``v_max``.
    """
    denom = ins.mass * np.abs(kappa) - ins.downforce_coeff
    v2 = np.full_like(kappa, ins.v_max**2, dtype=np.float64)
    pos = denom > 1e-9
    v2[pos] = (ins.grip * ins.mass * G) / denom[pos]
    v2 = np.clip(v2, 0.0, ins.v_max**2)
    return np.sqrt(v2)


def _long_grip_force(v: float, kappa: float, ins: CalibrationInputs) -> float:
    """Longitudinal tire force still available after cornering uses part of the grip circle."""
    f_max = ins.grip * ins.mass * G + ins.downforce_coeff * v * v
    f_lat = ins.mass * v * v * abs(kappa)
    return float(np.sqrt(max(0.0, f_max * f_max - f_lat * f_lat)))


def estimate_lap_time(track: Track, ins: CalibrationInputs) -> float:
    """Clean optimal lap time (s) from the forward-backward velocity profile."""
    kappa = np.asarray(track.curvature, dtype=np.float64)
    ds = _segment_ds(track)  # ds[i] = distance from sample i to i+1
    n = len(kappa)

    v = _lateral_limit(kappa, ins)

    closed = track.closed
    # Forward (accel) then backward (brake), iterated so the closed-loop seam converges.
    for _ in range(3):
        for i in range(n):
            j = (i + 1) % n if closed else min(i + 1, n - 1)
            f_long = min(ins.max_engine_force, _long_grip_force(v[i], kappa[i], ins))
            a = (f_long - ins.drag_coeff * v[i] ** 2 - ins.rolling_coeff * ins.mass * G) / ins.mass
            v_next = np.sqrt(max(0.0, v[i] ** 2 + 2.0 * a * ds[i]))
            v[j] = min(v[j], v_next, ins.v_max)
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n if closed else min(i + 1, n - 1)
            f_long = min(ins.max_brake_force, _long_grip_force(v[j], kappa[j], ins))
            a = (f_long + ins.drag_coeff * v[j] ** 2 + ins.rolling_coeff * ins.mass * G) / ins.mass
            v_prev = np.sqrt(max(0.0, v[j] ** 2 + 2.0 * a * ds[i]))
            v[i] = min(v[i], v_prev)

    v = np.maximum(v, 1.0)  # avoid div-by-zero on a degenerate sample
    v_avg = 0.5 * (v + (np.roll(v, -1) if closed else np.append(v[1:], v[-1])))
    return float(np.sum(ds / v_avg))


def sweep(track: Track, cfg: Any, lever: str, values: list[float]) -> list[tuple[float, float]]:
    """Recompute the lap-time estimate for each value of ``lever``; return (value, lap_time)."""
    out: list[tuple[float, float]] = []
    base = inputs_from_cfg(cfg)
    for val in values:
        ins = base
        if lever == "downforce_coeff":
            ins = _replace(base, downforce_coeff=val)
        elif lever == "mu_base":
            # mu_base scales grip directly (compound_grip[start] folded into base.grip already).
            tires = TireParams.from_config(cfg)
            ins = _replace(base, grip=val * tires.compound_grip[tires.start_compound])
        elif lever == "max_engine_force":
            v_max = float(np.sqrt(val / base.drag_coeff)) if base.drag_coeff > 0 else base.v_max
            ins = _replace(base, max_engine_force=val, v_max=v_max)
        else:
            raise ValueError(f"unknown sweep lever {lever!r}")
        out.append((val, estimate_lap_time(track, ins)))
    return out


def _replace(ins: CalibrationInputs, **kw: float) -> CalibrationInputs:
    import dataclasses

    return dataclasses.replace(ins, **kw)


# ----- CLI ----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate the car so a clean lap lands near pole.")
    p.add_argument("--config", default="experiment/rbr_dynamic", help="experiment config name")
    p.add_argument(
        "--sweep",
        default="downforce_coeff",
        choices=["downforce_coeff", "mu_base", "max_engine_force"],
        help="lever to sweep",
    )
    p.add_argument("--values", default=None, help="comma-separated lever values to try")
    p.add_argument("overrides", nargs="*", default=[], help="dotlist config overrides")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from f1rl.train.train import load_experiment_config

    args = _parse_args(argv)
    cfg = load_experiment_config(args.config, overrides=list(args.overrides) or None)
    track = load_track(str(cfg.get("track_id")), cfg=getattr(cfg, "track", None))
    pole = float(getattr(cfg.track, "pole_time_s", 0.0)) if hasattr(cfg, "track") else 0.0

    base_time = estimate_lap_time(track, inputs_from_cfg(cfg))
    print(f"\nTrack: {track.name}  pole_time_s={pole}  length={track.length:.1f} m")
    print(f"Baseline estimate (current config): {base_time:.2f} s  delta={base_time - pole:+.2f}\n")

    if args.values:
        values = [float(v) for v in args.values.split(",")]
    else:
        defaults = {
            "downforce_coeff": [2, 3, 4, 5, 6, 7, 8],
            "mu_base": [0.95, 1.05, 1.15, 1.25, 1.35, 1.45],
            "max_engine_force": [6000, 7500, 9000, 10500, 12000],
        }
        values = [float(v) for v in defaults[args.sweep]]

    rows = sweep(track, cfg, args.sweep, values)
    print(f"Sweep '{args.sweep}'  (target pole = {pole} s):")
    print(f"  {'value':>12} {'lap_est_s':>12} {'delta_to_pole':>14}")
    best = min(rows, key=lambda r: abs(r[1] - pole)) if pole > 0 else None
    for val, t in rows:
        mark = "  <- closest" if best is not None and val == best[0] else ""
        print(f"  {val:>12.3f} {t:>12.2f} {t - pole:>+14.2f}{mark}")
    if best is not None:
        print(f"\nClosest to pole: {args.sweep} = {best[0]} -> {best[1]:.2f} s")


if __name__ == "__main__":
    main()
