"""Thin CLI wrapper for the steps-per-second benchmark (plan Step C).

Run it before tuning the training budget to make the local-vs-cloud call with data:

    .venv/Scripts/python.exe scripts/benchmark_sps.py
    .venv/Scripts/python.exe scripts/benchmark_sps.py --n-envs 1 2 4 8 --steps 2000

All logic lives in :mod:`f1rl.train.benchmark`; this is just the entry point.
"""

from __future__ import annotations

from f1rl.train.benchmark import main

if __name__ == "__main__":
    main()
