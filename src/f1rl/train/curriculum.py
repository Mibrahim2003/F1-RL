"""Curriculum scheduler (spec §2 curriculum, plan §C, "Contracts fixed").

A single training run ramps the *conditions* (grip -> wear -> weather) from easy to hard at
timestep thresholds, so the agent learns the racing line before it has to manage low grip,
wear, and rain. The schedule is a **config table** (``cfg.curriculum.stages``); the callback
finds the active stage for ``num_timesteps`` and pushes its overrides into every worker via
:meth:`VecEnv.env_method` (``apply_conditions``). It only ever touches *conditions* — never
the observation layout — so there is no mid-run retrain.

No tuning constant lives here: every stage value comes from config (CLAUDE.md rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum stage: conditions (and Phase-4 circuit pool) from ``start_step`` onward."""

    start_step: int
    mu_base: float | None = None
    wear_rate: float | None = None
    weather: str | None = None
    # Phase 4: the active circuit set from this step on. ``None`` => leave the pool unchanged;
    # an empty list => the full configured pool (widen). Tuple for the frozen dataclass.
    circuits: tuple[str, ...] | None = None
    # Phase 6: ramp the racing reward weights in place (coexist -> race). ``None`` => unchanged.
    w_contact: float | None = None
    w_overtake: float | None = None


def parse_stages(cfg: Any) -> list[CurriculumStage]:
    """Read and sort the ``curriculum.stages`` table from config (empty when disabled/absent)."""
    node = getattr(cfg, "curriculum", None)
    if node is None:
        return []
    get = node.get if hasattr(node, "get") else (lambda k, d: getattr(node, k, d))
    if not bool(get("enabled", False)):
        return []
    raw = get("stages", []) or []
    stages: list[CurriculumStage] = []
    for s in raw:
        sget = s.get if hasattr(s, "get") else (lambda k, d, _s=s: getattr(_s, k, d))
        mu = sget("mu_base", None)
        wear = sget("wear_rate", None)
        weather = sget("weather", None)
        # Phase 4: a `circuits` key (even an empty list) means "set the pool"; absent => None.
        circuits = sget("circuits", None)
        circuits = None if circuits is None else tuple(str(c) for c in circuits)
        # Phase 6: optional racing reward-weight ramps.
        w_contact = sget("w_contact", None)
        w_overtake = sget("w_overtake", None)
        stages.append(
            CurriculumStage(
                start_step=int(sget("start_step", 0)),
                mu_base=None if mu is None else float(mu),
                wear_rate=None if wear is None else float(wear),
                weather=None if weather is None else str(weather),
                circuits=circuits,
                w_contact=None if w_contact is None else float(w_contact),
                w_overtake=None if w_overtake is None else float(w_overtake),
            )
        )
    stages.sort(key=lambda st: st.start_step)
    return stages


def active_stage(stages: list[CurriculumStage], num_timesteps: int) -> CurriculumStage | None:
    """The last stage whose ``start_step`` is ``<= num_timesteps`` (None if none qualify)."""
    current: CurriculumStage | None = None
    for st in stages:
        if num_timesteps >= st.start_step:
            current = st
        else:
            break
    return current


class CurriculumCallback(BaseCallback):
    """Apply the active curriculum stage's conditions to the workers as training progresses."""

    def __init__(self, cfg: Any, *, logger: Any | None = None, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.stages = parse_stages(cfg)
        self.run_logger = logger
        self._applied_start: int | None = None

    def _on_training_start(self) -> None:
        # Apply the stage active at the resume/start timestep before the first rollout.
        self._maybe_apply(force=True)

    def _on_step(self) -> bool:
        self._maybe_apply(force=False)
        return True

    def _maybe_apply(self, *, force: bool) -> None:
        if not self.stages:
            return
        stage = active_stage(self.stages, int(self.num_timesteps))
        if stage is None:
            return
        if not force and stage.start_step == self._applied_start:
            return
        self._applied_start = stage.start_step
        self.training_env.env_method(
            "apply_conditions",
            mu_base=stage.mu_base,
            wear_rate=stage.wear_rate,
            weather=stage.weather,
        )
        # Phase 4: widen/narrow the active circuit pool when the stage sets `circuits`
        # (None => leave it; [] => the full configured pool). Sampling-side only, no obs change.
        if stage.circuits is not None:
            self.training_env.env_method("set_track_pool", circuits=list(stage.circuits))
        nan = float("nan")
        n_circuits = float(len(stage.circuits)) if stage.circuits is not None else nan
        payload = {
            "curriculum/stage_start_step": float(stage.start_step),
            "curriculum/mu_base": float(stage.mu_base) if stage.mu_base is not None else nan,
            "curriculum/wear_rate": float(stage.wear_rate) if stage.wear_rate is not None else nan,
            "curriculum/n_circuits": n_circuits,
        }
        if self.run_logger is not None:
            self.run_logger.log(payload, step=int(self.num_timesteps))
        if self.verbose:
            print(
                f"[curriculum] step={self.num_timesteps} -> stage@{stage.start_step} "
                f"mu_base={stage.mu_base} wear_rate={stage.wear_rate} weather={stage.weather} "
                f"circuits={list(stage.circuits) if stage.circuits is not None else 'unchanged'}"
            )
