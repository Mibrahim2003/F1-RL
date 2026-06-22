"""Grown input-layer warm start (Phase 6; spec §2b, plan §E).

Phase 6 bumps ``OBS_VERSION`` 2 -> 3 (the neighbor block is appended at the tail), so
``validate_checkpoint`` **refuses** a silent ``--resume`` of a Phase 5 (v2) checkpoint — correct,
the observation layout changed. This module is the deliberate, explicit transplant instead: it
builds the fresh Phase 6 policy, copies every Phase 5 weight **except the policy/value input
layer**, copies that layer's columns for the unchanged inputs ``0:22`` and **zero-initializes**
the new neighbor columns ``22:``, and grows the ``VecNormalize`` obs statistics to the new width
(new dims start mean 0 / var 1).

The result drives **exactly as well as Phase 5 on step one** (a no-neighbor observation produces
the same action, because the neighbor columns contribute 0) and only has to learn to use the
neighbor block on top. Training from scratch is the documented fallback.

The transplant targets ``mlp_extractor.policy_net.0`` / ``mlp_extractor.value_net.0`` (the first
``Linear`` of each head for SB3 ``MlpPolicy`` + ``FlattenExtractor``); a source whose non-neighbor
layout is not a clean prefix of the target (any other shape mismatch) is **refused** so a
genuinely incompatible policy cannot be transplanted.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.train.checkpointing import VECNORM_FILE, CheckpointError, load_checkpoint
from f1rl.train.train import build_model


def grow_policy(
    src_ckpt: str | Path,
    target_venv: Any,
    cfg: Any,
    seed: int,
    *,
    base_dim: int = 22,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Transplant a Phase 5 (v2) policy into a fresh Phase 6 (v3) PPO; return ``(model, meta)``.

    Args:
        src_ckpt: The Phase 5 / Phase 4 checkpoint directory (loaded with ``validate=False`` so a
            v2 ``obs_version`` is accepted here — this is the explicit transplant path).
        target_venv: The Phase 6 :class:`VecNormalize` field env (width ``22 + K*5``); its
            ``obs_rms`` is grown in place.
        cfg: The Phase 6 run config (same PPO hyperparameters / device).
        seed: The run seed for the fresh model.
        base_dim: The unchanged-prefix width (the ObservationV2 length, 22).

    Returns:
        ``(model, src_meta)`` — a v3 PPO whose weights reproduce the source driver on a
        no-neighbor observation, with the source meta for logging.

    Raises:
        CheckpointError: If a non-input layer shape does not match (an incompatible layout).
    """
    import torch

    src_model, meta = load_checkpoint(src_ckpt, env=None, device=device, validate=False)
    model = build_model(cfg, target_venv, seed)

    src_sd = src_model.policy.state_dict()
    dst_sd = model.policy.state_dict()
    new_sd: dict[str, Any] = {}
    grown: list[str] = []
    for key, dst_tensor in dst_sd.items():
        if key not in src_sd:
            new_sd[key] = dst_tensor.clone()
            continue
        src_tensor = src_sd[key]
        if src_tensor.shape == dst_tensor.shape:
            new_sd[key] = src_tensor.clone()
        elif (
            src_tensor.ndim == 2
            and src_tensor.shape[0] == dst_tensor.shape[0]
            and src_tensor.shape[1] < dst_tensor.shape[1]
        ):
            # Grow the input-layer weight: copy the unchanged columns, zero the new ones.
            w = dst_tensor.clone()
            n_src = src_tensor.shape[1]
            w[:, :n_src] = src_tensor
            w[:, n_src:] = 0.0
            new_sd[key] = w
            grown.append(key)
        else:
            raise CheckpointError(
                f"warm-start layout incompatible at '{key}': source {tuple(src_tensor.shape)} "
                f"cannot grow into target {tuple(dst_tensor.shape)}. The non-neighbor layout is "
                f"not a prefix of the target — this policy cannot be transplanted."
            )

    with torch.no_grad():
        model.policy.load_state_dict(new_sd)

    grow_vecnormalize(target_venv, src_ckpt, base_dim=base_dim)
    print(f"[warmstart] transplanted {len(new_sd)} tensors (grew input layers: {grown})")
    return model, meta


def grow_vecnormalize(target_venv: Any, src_ckpt: str | Path, *, base_dim: int = 22) -> None:
    """Grow the target ``VecNormalize`` obs stats from the source checkpoint's stats.

    Copies the source running mean/var/count into the unchanged prefix ``0:base_dim`` and leaves
    the new neighbor dims at mean 0 / var 1 — consistent with the zero-initialized weight columns
    (the neighbor block contributes 0 at step one regardless). No-op if either side has no stats.
    """
    vn_path = Path(src_ckpt) / VECNORM_FILE
    tgt_rms = getattr(target_venv, "obs_rms", None)
    if not vn_path.exists() or tgt_rms is None:
        return
    with open(vn_path, "rb") as f:
        src_vn = pickle.load(f)
    src_rms = getattr(src_vn, "obs_rms", None)
    if src_rms is None:
        return

    n = int(np.asarray(src_rms.mean).shape[0])
    n = min(n, base_dim, int(np.asarray(tgt_rms.mean).shape[0]))
    tgt_rms.mean[:n] = np.asarray(src_rms.mean)[:n]
    tgt_rms.var[:n] = np.asarray(src_rms.var)[:n]
    tgt_rms.mean[n:] = 0.0
    tgt_rms.var[n:] = 1.0
    tgt_rms.count = float(src_rms.count)
