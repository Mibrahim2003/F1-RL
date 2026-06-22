"""Bake the web backend's static map payloads from already-built ``.npz`` caches.

Writes one ``data/tracks/<id>.api.json`` per circuit (the verbatim ``GET /track/<id>`` body)
plus ``data/tracks/_catalog.json`` (the verbatim ``GET /api/tracks`` body), so the server
serves a file read instead of loading numpy and re-serializing JSON on every request — the
maps are "preloaded" on the next app open.

Runtime-safe: reads only existing ``.npz`` files (no FastF1, no network), unlike
``build_all_tracks.py`` which rebuilds geometry from the network. Run it after editing caches
out of band, or to backfill payloads for circuits built before this bake existed:

    .venv/Scripts/python.exe scripts/bake_track_json.py
"""

from __future__ import annotations

from f1rl.track.loader import DEFAULT_TRACKS_DIR, bake_all


def main() -> int:
    baked = bake_all(DEFAULT_TRACKS_DIR)
    print(f"baked {len(baked)} circuit payload(s) + _catalog.json into {DEFAULT_TRACKS_DIR}/")
    for cid in baked:
        print(f"  {cid}{'.api.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
