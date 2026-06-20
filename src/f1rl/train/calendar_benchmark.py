"""Calendar lap-time table: one policy scored vs the pole on every pool circuit (Phase 4 §2b).

The phase artifact. A deterministic, agent-driven sweep over the configured circuit pool: for
each circuit it builds a one-circuit config (so the env binds just that circuit and ``evaluate``
resolves the right pole), runs the existing :func:`f1rl.train.evaluate.evaluate` for K
deterministic episodes with the **saved VecNormalize stats**, and collects best lap / pole /
delta / 2x-pole flag. It then assembles a table (one row per circuit) plus aggregates (mean and
worst delta, beat-2x-pole rate), prints it, optionally logs per-circuit scalars to Weights &
Biases, and saves the table as JSON + CSV.

This adds **no** metric logic — it reuses ``evaluate`` verbatim per circuit. The new code is the
loop, the per-circuit pole/config resolution, and the table assembly/save. Runtime-safe: no
FastF1, no network.

CLI::

    .venv/Scripts/python.exe -m f1rl.train.calendar_benchmark \
        --checkpoint runs/<run>/checkpoints/best --config calendar_dynamic --episodes 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.env.pool import pool_ids_from_config, resolve_pole
from f1rl.train.evaluate import _load_obs_rms, evaluate
from f1rl.utils.seeding import seed_everything


def _cfg_for_circuit(base_cfg: Any, circuit_id: str) -> Any:
    """A deep copy of the run config bound to a single circuit (one-circuit pool + its track)."""
    from omegaconf import OmegaConf

    from f1rl.utils.config import CONFIG_ROOT

    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.track_id = circuit_id
    track_path = CONFIG_ROOT / "track" / f"{circuit_id}.yaml"
    cfg.track = OmegaConf.load(track_path)
    # Bind the eval env to exactly this circuit (so the draw is deterministic, pole is its own).
    cfg.circuits = OmegaConf.create(
        {"pool": [circuit_id], "sampling": "uniform", "weights": None, "pin_per_worker": False}
    )
    return cfg


def benchmark_circuit(
    model: Any,
    base_cfg: Any,
    circuit_id: str,
    *,
    episodes: int,
    seed: int,
    obs_rms: Any | None,
    clip_obs: float,
) -> dict[str, Any]:
    """Run ``evaluate`` on one circuit and return its table row (best lap / pole / delta)."""
    cfg = _cfg_for_circuit(base_cfg, circuit_id)
    pole, missing = resolve_pole(circuit_id, cfg)
    result = evaluate(
        model,
        cfg,
        n_episodes=episodes,
        seed=seed,
        obs_rms=obs_rms,
        clip_obs=clip_obs,
        deterministic=True,
        record_first=False,
        pole_time_s=pole,
    )
    summary = result.summary(pole)
    return {
        "circuit": circuit_id,
        "best_lap_time": summary.get("eval/best_lap_time", float("nan")),
        "pole_time_s": float(pole),
        "delta_to_pole": summary.get("eval/gap_to_pole", float("nan")),
        "beat_pole_rate": summary.get("eval/beat_pole_rate", 0.0),
        "beat_2x_pole_rate": summary.get("eval/beat_2x_pole_rate", 0.0),
        "off_track_count": summary.get("eval/mean_off_track_count", float("nan")),
        "completed_laps": summary.get("eval/completed_laps", 0.0),
        "pole_missing": bool(missing),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool-level aggregates: completion, mean/worst delta, beat-pole rates (skip missing poles)."""
    nan = float("nan")
    completed = [r for r in rows if not _isnan(r["best_lap_time"])]
    deltas = [
        r["delta_to_pole"] for r in rows if not (r["pole_missing"] or _isnan(r["delta_to_pole"]))
    ]
    return {
        "n_circuits": len(rows),
        "n_completed": len(completed),
        "mean_delta_to_pole": float(np.mean(deltas)) if deltas else nan,
        "worst_delta_to_pole": float(np.max(deltas)) if deltas else nan,
        "worst_circuit": (
            max(
                (r for r in rows if not (r["pole_missing"] or _isnan(r["delta_to_pole"]))),
                key=lambda r: r["delta_to_pole"],
                default={"circuit": None},
            )["circuit"]
            if deltas
            else None
        ),
        "beat_pole_rate": float(np.mean([r["beat_pole_rate"] for r in rows])) if rows else nan,
        "beat_2x_pole_rate": (
            float(np.mean([r["beat_2x_pole_rate"] for r in rows])) if rows else nan
        ),
    }


def run_calendar_benchmark(
    checkpoint: str | Path,
    config_name: str,
    *,
    episodes: int = 2,
    seed: int = 0,
    out_dir: str | Path = "out",
    logger: Any | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Sweep the configured pool, build the table, save JSON+CSV, and (optionally) log to W&B."""
    from f1rl.train.checkpointing import load_checkpoint
    from f1rl.train.train import load_experiment_config

    base_cfg = load_experiment_config(config_name)
    seed_everything(int(seed))

    model, _meta = load_checkpoint(checkpoint, env=None, device=device)
    obs_rms = _load_obs_rms(checkpoint)
    env_node = getattr(base_cfg, "env", None)
    clip_obs = float(getattr(env_node, "clip_obs", 10.0)) if env_node is not None else 10.0

    pool = pool_ids_from_config(base_cfg, base_cfg.get("track_id"))
    rows = [
        benchmark_circuit(
            model,
            base_cfg,
            cid,
            episodes=int(episodes),
            seed=int(seed),
            obs_rms=obs_rms,
            clip_obs=clip_obs,
        )
        for cid in pool
    ]
    aggregates = aggregate(rows)
    table = {"rows": rows, "aggregates": aggregates}

    _print_table(rows, aggregates)
    paths = _save_table(table, out_dir)
    print(f"\n  table -> {paths['json']}\n  table -> {paths['csv']}")

    if logger is not None:
        _log_to_wandb(logger, rows, aggregates)

    return table


# ----- helpers ----------------------------------------------------------------------------


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _fmt(v: Any) -> str:
    return "  --  " if _isnan(v) else f"{float(v):6.2f}"


def _print_table(rows: list[dict[str, Any]], aggregates: dict[str, Any]) -> None:
    print("\nCalendar lap-time table (achieved vs pole):")
    print(f"  {'circuit':<16} {'best':>7} {'pole':>7} {'delta':>7} {'2xpole':>7} {'offtrk':>7}")
    for r in rows:
        twox = "yes" if r["beat_2x_pole_rate"] > 0.0 else "no"
        flag = "  (no pole)" if r["pole_missing"] else ""
        print(
            f"  {r['circuit']:<16} {_fmt(r['best_lap_time'])} {_fmt(r['pole_time_s'])} "
            f"{_fmt(r['delta_to_pole'])} {twox:>7} {_fmt(r['off_track_count'])}{flag}"
        )
    print(
        f"\n  circuits={aggregates['n_circuits']} completed={aggregates['n_completed']} "
        f"mean_delta={_fmt(aggregates['mean_delta_to_pole'])} "
        f"worst_delta={_fmt(aggregates['worst_delta_to_pole'])} "
        f"(@{aggregates['worst_circuit']}) "
        f"beat_2x_pole_rate={aggregates['beat_2x_pole_rate']:.2f}"
    )


def _save_table(table: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "calendar_benchmark.json"
    csv_path = out / "calendar_benchmark.csv"
    json_path.write_text(json.dumps(table, indent=2), encoding="utf-8")

    rows = table["rows"]
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"json": str(json_path), "csv": str(csv_path)}


def _log_to_wandb(logger: Any, rows: list[dict[str, Any]], aggregates: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    for r in rows:
        cid = r["circuit"]
        payload[f"calendar/{cid}/best_lap_time"] = r["best_lap_time"]
        payload[f"calendar/{cid}/delta_to_pole"] = r["delta_to_pole"]
        payload[f"calendar/{cid}/beat_2x_pole"] = r["beat_2x_pole_rate"]
    payload["calendar/mean_delta_to_pole"] = aggregates["mean_delta_to_pole"]
    payload["calendar/worst_delta_to_pole"] = aggregates["worst_delta_to_pole"]
    payload["calendar/beat_2x_pole_rate"] = aggregates["beat_2x_pole_rate"]
    with __import__("contextlib").suppress(Exception):
        logger.log(payload)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a checkpoint vs the pole on every pool circuit.")
    p.add_argument("--checkpoint", required=True, help="checkpoint directory")
    p.add_argument("--config", default="calendar_dynamic", help="experiment config name")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="out", help="directory for the saved JSON/CSV table")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_calendar_benchmark(
        args.checkpoint,
        args.config,
        episodes=int(args.episodes),
        seed=int(args.seed),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
