"""CSV/JSON run logging (Session B)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

from ppo.config import Config

PROGRESS_COLUMNS = ("global_step", "episodic_return", "episodic_length")
METRICS_COLUMNS = ("global_step", "policy_loss", "value_loss", "entropy", "approx_kl",
                   "clipfrac", "explained_variance", "learning_rate", "sps")


class RunLogger:
    """Writes results/<env_id>/<run_name>/seed_<seed>/{config.json,progress.csv,metrics.csv}.

    Attribute `run_dir` (pathlib.Path) is the seed directory. Existing logs in that
    directory are overwritten. Every row is flushed to disk immediately so that a
    killed run still leaves readable, consistent files.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run_dir = Path(cfg.out_dir) / cfg.env_id / cfg.run_name / f"seed_{cfg.seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

        self._progress_f = open(self.run_dir / "progress.csv", "w", newline="")
        self._metrics_f = open(self.run_dir / "metrics.csv", "w", newline="")
        self._progress = csv.writer(self._progress_f)
        self._metrics = csv.writer(self._metrics_f)
        self._progress.writerow(PROGRESS_COLUMNS)
        self._metrics.writerow(METRICS_COLUMNS)
        self._progress_f.flush()
        self._metrics_f.flush()
        self._closed = False

    def log_episode(self, global_step: int, episodic_return: float, episodic_length: int) -> None:
        self._progress.writerow([int(global_step), float(episodic_return), int(episodic_length)])
        self._progress_f.flush()

    def log_metrics(self, global_step: int, metrics: Dict[str, float]) -> None:
        """metrics keys: policy_loss, value_loss, entropy, approx_kl, clipfrac,
        explained_variance, learning_rate, sps."""
        # Missing keys are written as NaN (extra keys are ignored) so a metrics
        # mismatch never kills a long training run.
        row = [int(global_step)] + [float(metrics.get(k, float("nan"))) for k in METRICS_COLUMNS[1:]]
        self._metrics.writerow(row)
        self._metrics_f.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._progress_f.close()
        self._metrics_f.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
