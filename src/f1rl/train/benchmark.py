"""Steps-per-second benchmark (spec §1d, §g; plan Step C — built FIRST).

Measures environment throughput (control steps / second) for a sweep of ``n_envs`` using
the real :func:`f1rl.env.factory.make_vec_env` seam — the same vectorized stack training
uses. The result drives the local-vs-cloud decision with data, per the spec, before any
budget is tuned.

Throughput here is the *environment* rate (random actions, no policy forward pass), which
is the dominant cost for this CPU-bound, beam-casting env. Reported as total steps/sec
across all parallel workers (the number that matters for filling the PPO rollout buffer).

Run it:

    .venv/Scripts/python.exe -m f1rl.train.benchmark
    .venv/Scripts/python.exe -m f1rl.train.benchmark --n-envs 1 2 4 8 --steps 2000

Never imports FastF1, never renders.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from f1rl.env.factory import make_vec_env

_DEFAULT_N_ENVS = (1, 2, 4, 8)
_DEFAULT_STEPS = 1500
_DEFAULT_WARMUP = 50


@dataclass(frozen=True)
class BenchResult:
    """One row of the SPS table for a given ``n_envs``."""

    n_envs: int
    total_steps: int
    elapsed_s: float
    steps_per_s: float  # total control steps/sec across all workers

    @property
    def per_env_steps_per_s(self) -> float:
        return self.steps_per_s / self.n_envs if self.n_envs else 0.0


def benchmark_n_envs(
    cfg: Any,
    n_envs: int,
    steps: int = _DEFAULT_STEPS,
    warmup: int = _DEFAULT_WARMUP,
    seed: int = 0,
) -> BenchResult:
    """Measure total steps/sec for one ``n_envs`` over ``steps`` vectorized control steps.

    ``warmup`` steps are run and discarded first (subprocess spin-up, import warmth) so the
    timed window measures steady-state throughput.
    """
    venv = make_vec_env(cfg, n_envs=n_envs, seed=seed)
    rng = np.random.default_rng(seed)
    try:
        venv.reset()
        action_shape = (n_envs, *venv.action_space.shape)
        low = float(venv.action_space.low.min())
        high = float(venv.action_space.high.max())

        def _step_once() -> None:
            actions = rng.uniform(low=low, high=high, size=action_shape).astype(np.float32)
            venv.step(actions)

        for _ in range(warmup):
            _step_once()

        start = time.perf_counter()
        for _ in range(steps):
            _step_once()
        elapsed = time.perf_counter() - start
    finally:
        venv.close()

    total_control_steps = steps * n_envs
    sps = total_control_steps / elapsed if elapsed > 0 else 0.0
    return BenchResult(
        n_envs=n_envs,
        total_steps=total_control_steps,
        elapsed_s=elapsed,
        steps_per_s=sps,
    )


def run_benchmark(
    config_name: str = "rbr_ppo",
    overrides: list[str] | None = None,
    n_envs_list: tuple[int, ...] = _DEFAULT_N_ENVS,
    steps: int = _DEFAULT_STEPS,
    warmup: int = _DEFAULT_WARMUP,
    seed: int = 0,
) -> list[BenchResult]:
    """Run the full SPS sweep and print the table. Returns the rows for programmatic use."""
    from f1rl.train.train import load_experiment_config

    cfg = load_experiment_config(config_name, overrides=overrides)
    results: list[BenchResult] = []
    print(
        f"\nSPS benchmark — config '{config_name}', track '{cfg.get('track_id')}', "
        f"{steps} steps/env (+{warmup} warmup), device=cpu"
    )
    print("-" * 64)
    print(f"{'n_envs':>7} | {'total sps':>12} | {'per-env sps':>12} | {'elapsed s':>10}")
    print("-" * 64)
    for n in n_envs_list:
        res = benchmark_n_envs(cfg, n_envs=n, steps=steps, warmup=warmup, seed=seed)
        results.append(res)
        print(
            f"{res.n_envs:>7} | {res.steps_per_s:>12,.0f} | "
            f"{res.per_env_steps_per_s:>12,.0f} | {res.elapsed_s:>10.2f}"
        )
    print("-" * 64)
    best = max(results, key=lambda r: r.steps_per_s)
    print(f"Best total throughput: n_envs={best.n_envs} at {best.steps_per_s:,.0f} steps/s")
    print(
        "Rule of thumb: if best total sps stalls or drops past the core count, you are "
        "CPU-bound — that is the local-vs-cloud signal.\n"
    )
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Env steps-per-second benchmark.")
    p.add_argument("--config", default="rbr_ppo", help="experiment config name")
    p.add_argument(
        "--n-envs",
        type=int,
        nargs="+",
        default=list(_DEFAULT_N_ENVS),
        help="n_envs values to sweep",
    )
    p.add_argument("--steps", type=int, default=_DEFAULT_STEPS, help="timed steps per env")
    p.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP, help="discarded warmup steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="dotlist config overrides, e.g. obs.beam_max=50",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_benchmark(
        config_name=args.config,
        overrides=list(args.overrides) or None,
        n_envs_list=tuple(args.n_envs),
        steps=args.steps,
        warmup=args.warmup,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
