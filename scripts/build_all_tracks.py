"""Build every real circuit from its config into a cached ``Track`` (.npz).

Build-time only — needs the ``trackbuild`` extra (FastF1, Shapely, requests) and network
access for FastF1/Overpass on the first run (responses are then disk-cached).

    .venv/Scripts/python.exe -m pip install -e ".[dev,trackbuild]"
    .venv/Scripts/python.exe scripts/build_all_tracks.py            # all circuits
    .venv/Scripts/python.exe scripts/build_all_tracks.py monza spa  # a subset

Each circuit is isolated: one failure is reported and never blocks the rest. A summary
table prints at the end and the per-circuit report lands in ``data/tracks/_build_report.json``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from f1rl.track.build import DEFAULT_CACHE_DIR, BuildConfig, build_track, save_track
from f1rl.track.loader import write_catalog

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "track"
SKIP = {"oval"}  # the procedural oval is not a real-circuit build


def _circuit_ids(argv: list[str]) -> list[str]:
    if argv:
        return argv
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem not in SKIP)


def main(argv: list[str]) -> int:
    ids = _circuit_ids(argv)
    if not ids:
        print(f"no circuit configs found in {CONFIG_DIR}")
        return 1

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for cid in ids:
        cfg_path = CONFIG_DIR / f"{cid}.yaml"
        if not cfg_path.exists():
            failures.append((cid, "config not found"))
            continue
        cfg = BuildConfig.from_config(OmegaConf.load(cfg_path))
        print(f"building {cid} ...", flush=True)
        try:
            track, report = build_track(cfg)
            save_track(track, report)
            rows.append(report)
            flag = "  LOW-CONF" if report["low_confidence"] else ""
            err = report["length_error"]
            err_s = f"{err:.1%}" if err is not None else "n/a"
            print(f"  ok: {report['points']} pts, {report['length_m']} m, err {err_s}{flag}")
        except Exception as e:  # isolate: one bad circuit never blocks the rest
            failures.append((cid, str(e)))
            print(f"  FAILED: {e}")
            traceback.print_exc()

    # Refresh the pre-baked selector catalog so a new/rebuilt circuit shows up preloaded.
    if rows:
        write_catalog(DEFAULT_CACHE_DIR)

    _print_summary(rows, failures)
    return 0 if rows and not failures else (0 if rows else 1)


def _print_summary(rows: list[dict], failures: list[tuple[str, str]]) -> None:
    print("\n=== build summary ===")
    print(f"{'circuit':<16}{'source':<12}{'pts':>6}{'length':>10}{'err':>8}  flag")
    for r in sorted(rows, key=lambda x: x["id"]):
        err = r["length_error"]
        err_s = f"{err:.1%}" if err is not None else "n/a"
        flag = "LOW-CONF" if r["low_confidence"] else ""
        print(
            f"{r['id']:<16}{r['source']:<12}{r['points']:>6}"
            f"{r['length_m']:>10.0f}{err_s:>8}  {flag}"
        )
    print(f"\nbuilt {len(rows)} circuit(s); {len(failures)} failure(s)")
    for cid, msg in failures:
        print(f"  FAILED {cid}: {msg}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
