"""A learned-policy pilot for the watch-live mode — the live counterpart to the autopilot.

:class:`PolicyPilot` loads a trained PPO checkpoint and drives a car around the live track,
mirroring :class:`~f1rl.sim.autopilot.CenterlineAutopilot.control` so the server drops it into
the exact same per-session slot. Each step it builds **ObservationV1 from the same pure
``env/observations.py`` builder the training loop uses** (no reimplemented obs — train/serve
skew is the bug this avoids), normalizes it with the checkpoint's saved VecNormalize stats,
and queries the policy deterministically.

The checkpoint format and its loader live in :mod:`f1rl.train.checkpointing` (the single
source for the format). That module is pure save/load — no FastF1, no rendering — so importing
it here keeps the server clear of the training hot path and any renderer.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.env.conditions import Conditions
from f1rl.env.observations import (
    EdgeCache,
    ObsParams,
    build_edge_cache,
    build_observation,
    track_query,
)
from f1rl.physics.base import CarState
from f1rl.track.schema import Track
from f1rl.train.checkpointing import (
    VECNORM_FILE,
    CheckpointError,
    load_checkpoint,
    validate_checkpoint,
)


class PolicyPilot:
    """Runs a trained PPO checkpoint live, producing ``(steer, longitudinal)`` commands.

    Same call interface as :class:`~f1rl.sim.autopilot.CenterlineAutopilot`, so the server
    can assign it to ``sim.autopilot`` and the send loop keeps calling ``.control(state)``.

    Construction validates the checkpoint's ``obs_version`` / action shape against this build
    and raises :class:`~f1rl.train.checkpointing.CheckpointError` on a mismatch, so a stale
    observation layout never silently drives the car. The observation builder, the obs params,
    and the per-track edge cache are built once; ``control`` is then a per-step lookup.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        track: Track,
        cfg: Any,
        *,
        device: str = "cpu",
    ) -> None:
        """Load and validate a checkpoint, ready to drive ``track``.

        Args:
            checkpoint_path: Checkpoint directory (``model.zip`` + ``vecnormalize.pkl`` +
                ``meta.json``).
            track: The live circuit the pilot drives — the edge cache is built from it.
            cfg: Resolved run config; ``cfg.obs`` builds the :class:`ObsParams` and
                ``cfg.env.clip_obs`` sets the VecNormalize obs clip (defaults to 10).
            device: Torch device for inference (``"cpu"`` here; identical local/cloud).

        Raises:
            CheckpointError: On a missing artifact or an incompatible obs version /
                action shape (re-raised from the loader / validator).
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.track = track

        # Load the model + meta; load_checkpoint validates obs_version / action shape and
        # raises CheckpointError on a mismatch (env=None: no VecNormalize attached here).
        self.model, self.meta = load_checkpoint(self.checkpoint_path, env=None, device=device)
        # Belt-and-braces: validate again explicitly (cheap, surfaces a clear message).
        validate_checkpoint(self.meta)

        # Recover the saved obs normalization stats (mean/var) so live inference matches the
        # training distribution exactly — the same _normalize_obs path used by evaluate.py.
        self._obs_rms = _load_obs_rms(self.checkpoint_path)

        # Build the obs params and the per-track edge cache once (beams reuse it every step).
        self._params: ObsParams = ObsParams.from_config(cfg)
        self._edge_cache: EdgeCache = build_edge_cache(track)
        # The shared grip provider so the obs grip-indicator matches training (ObservationV2).
        self._conditions = Conditions.from_config(cfg)

        env_node = getattr(cfg, "env", None)
        self._clip_obs = (
            float(getattr(env_node, "clip_obs", 10.0)) if env_node is not None else 10.0
        )

    def control(self, state: CarState) -> tuple[float, float]:
        """Return the policy's ``(steer, longitudinal)`` command for the live car ``state``.

        Builds ObservationV1 from ``state`` via the shared builder, normalizes it with the
        saved VecNormalize stats (clipped to ``clip_obs``), and queries the policy
        deterministically. Both outputs are in ``[-1, 1]`` (the action space).
        """
        idx, _s, signed_lateral, _hw, _heading = track_query(
            self.track, state.x, state.y, state.yaw
        )
        grip_ind = self._conditions.grip_indicator(
            self.track, idx, signed_lateral, state.tire_wear, state.compound
        )
        obs = build_observation(
            self.track, state, self._params, self._edge_cache, grip_indicator=grip_ind
        )
        norm_obs = _normalize_obs(np.asarray(obs, dtype=np.float64), self._obs_rms, self._clip_obs)
        action, _ = self.model.predict(norm_obs.astype(np.float32), deterministic=True)
        steer = float(np.clip(action[0], -1.0, 1.0))
        longitudinal = float(np.clip(action[1], -1.0, 1.0))
        return steer, longitudinal


def _normalize_obs(
    obs: np.ndarray, obs_rms: Any | None, clip: float, epsilon: float = 1e-8
) -> np.ndarray:
    """Apply VecNormalize obs normalization (mean/var) so serve matches training.

    Mirrors :func:`f1rl.train.evaluate._normalize_obs` exactly: ``(obs - mean) /
    sqrt(var + epsilon)`` then clip to ``+/- clip``. Returns ``obs`` unchanged when no stats
    are available (the model still runs, just on unnormalized inputs).
    """
    if obs_rms is None:
        return obs
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    var = np.asarray(obs_rms.var, dtype=np.float64)
    normed = (obs - mean) / np.sqrt(var + epsilon)
    return np.clip(normed, -clip, clip)


def _load_obs_rms(checkpoint: str | Path) -> Any | None:
    """Load just the ``obs_rms`` (mean/var) from a checkpoint's ``vecnormalize.pkl``.

    Returns ``None`` if the file is missing or unreadable — inference then runs on
    unnormalized observations rather than failing.
    """
    vn_path = Path(checkpoint) / VECNORM_FILE
    if not vn_path.exists():
        return None
    try:
        with vn_path.open("rb") as f:
            vn = pickle.load(f)
        return getattr(vn, "obs_rms", None)
    except Exception:
        return None


__all__ = ["PolicyPilot", "CheckpointError"]
