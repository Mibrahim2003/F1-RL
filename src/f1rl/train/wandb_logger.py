"""Weights & Biases logging with an offline / local-CSV fallback (spec §d, plan §C).

The contract: an outage, a missing ``wandb`` install, ``WANDB_MODE=offline``/``disabled``,
or an init failure must **never lose the curves**. So every scalar logged through
:class:`RunLogger` is always appended to a local CSV under the run directory, regardless of
whether the W&B upload succeeds. W&B is best-effort on top of that durable local log.

Reads the ``wandb`` config block (``project, entity, mode, group, tags``). The ``WANDB_MODE``
environment variable, if set, overrides ``cfg.wandb.mode`` (so CI can force offline).
"""

from __future__ import annotations

import contextlib
import csv
import os
from pathlib import Path
from typing import Any


def _wandb_cfg(cfg: Any) -> dict[str, Any]:
    node = getattr(cfg, "wandb", None)
    if node is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(node):
            return dict(OmegaConf.to_container(node, resolve=True))
    except Exception:
        pass
    if isinstance(node, dict):
        return dict(node)
    return {}


class RunLogger:
    """A run logger that mirrors every scalar to a local CSV and (best-effort) to W&B.

    Use as a context manager or call :meth:`close` explicitly::

        with RunLogger(cfg, run_dir, run_name="rbr_ppo_0") as logger:
            logger.log({"train/return": 12.3}, step=1000)
            logger.log_video("eval/clip", "eval.mp4", step=1000)
    """

    def __init__(
        self,
        cfg: Any,
        run_dir: str | Path,
        run_name: str | None = None,
        *,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or self.run_dir.name
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_fields: list[str] = ["step"]
        self._csv_rows: list[dict[str, Any]] = []
        self.run = None  # the W&B run handle, or None on fallback

        wb = _wandb_cfg(cfg)
        mode = os.environ.get("WANDB_MODE", wb.get("mode", "online"))
        self._mode = str(mode)
        self._init_wandb(wb, config_snapshot, mode)

    def _init_wandb(
        self, wb: dict[str, Any], config_snapshot: dict[str, Any] | None, mode: str
    ) -> None:
        if str(mode) == "disabled":
            return
        try:
            import wandb

            self.run = wandb.init(
                project=wb.get("project", "f1rl"),
                entity=wb.get("entity") or None,
                group=wb.get("group") or None,
                tags=list(wb.get("tags") or []),
                name=self.run_name,
                mode=str(mode),
                dir=str(self.run_dir),
                config=config_snapshot or {},
                reinit=True,
            )
        except Exception as exc:  # import error, network error, bad creds — never fatal.
            self.run = None
            print(f"[wandb] init failed ({exc!r}); falling back to local CSV at {self._csv_path}")

    # ----- logging ----------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of scalars. Always written to the local CSV; uploaded to W&B if live."""
        row: dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            row[k] = v
            if k not in self._csv_fields:
                self._csv_fields.append(k)
        self._csv_rows.append(row)
        self._flush_csv()

        if self.run is not None:
            try:
                self.run.log(metrics, step=step)
            except Exception as exc:
                print(f"[wandb] log failed ({exc!r}); curve preserved in {self._csv_path}")

    def log_video(self, key: str, path: str | Path, step: int | None = None, fps: int = 20) -> None:
        """Log an mp4 eval clip to W&B (best-effort). The file is kept on disk regardless."""
        p = Path(path)
        # Record the clip path in the CSV so the local log references it even without W&B.
        self.log({f"{key}_path": str(p)}, step=step)
        if self.run is None or not p.exists():
            return
        try:
            import wandb

            self.run.log({key: wandb.Video(str(p), fps=fps, format="mp4")}, step=step)
        except Exception as exc:
            print(f"[wandb] video log failed ({exc!r}); clip kept at {p}")

    def _flush_csv(self) -> None:
        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields)
            writer.writeheader()
            for row in self._csv_rows:
                writer.writerow(row)

    def close(self) -> None:
        self._flush_csv()
        if self.run is not None:
            with contextlib.suppress(Exception):
                self.run.finish()
            self.run = None

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
