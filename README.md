# F1-RL

A 2D top-down Formula 1 simulator where **every car is driven by a learned policy**. Twenty-two cars race real 2026 circuits, teaching themselves the racing line, fighting for position, overtaking and defending — a living race that runs on its own and surprises you every time.

You watch from directly above. The whole circuit fits on screen, drawn faithfully, with cars as small, true-proportion shapes. The beauty comes from clarity: every car, every position, every move, all at a glance.

## Status

Early stage — design complete, build starting. See the planning docs:

- **[Vision](.claude/PROJECT_VISION.md)** — what this is and why.
- **[Technical Design](.claude/TECHNICAL_DESIGN.md)** — the authoritative engineering spec: stack, physics, env contract, reward design, and the phased build order.

## The build, in stages

Each phase ends with something watchable in the app.

1. **App & viewer** — Pygame top-down viewer, manual drive, replay (oval track, kinematic physics).
2. **Tracks** — offline FastF1 pipeline to build every real 2026 circuit, plus an in-app surface/condition editor.
3. **One car, one circuit** — observations + rewards + PPO, then dynamic physics with the grip pipeline; benchmark lap time vs. the real pole.
4. **One car, many circuits** — a single track-agnostic policy across the whole calendar.
5. **Many cars** — multi-agent self-play, scaling 2 → 4 → 22 cars.
6. **Racing for real** — nearby-car awareness, collisions, overtaking and defending.
7. **Pit stops & polish** — pit strategy, team colors, showcase videos.

## Stack

Python 3.10+ · NumPy · Gymnasium / PettingZoo / SuperSuit · Stable-Baselines3 (PPO) · PyTorch · FastF1 · SciPy · Pygame · imageio · Weights & Biases · OmegaConf · Ruff · Pytest.

> Develop locally with Claude Code, push to GitHub; training runs on Colab/Kaggle GPUs that clone this repo. See the [technical design](.claude/TECHNICAL_DESIGN.md) for the full rationale behind every choice.
