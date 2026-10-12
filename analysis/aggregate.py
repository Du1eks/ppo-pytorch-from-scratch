"""Aggregate training logs into learning curves on a common step grid and final-return tables.

Reads the log layout written by ``ppo.logger.RunLogger``::

    <results_dir>/<env_id>/<run_name>/seed_<n>/{config.json, progress.csv, metrics.csv}

and provides

* ``discover_runs``     -- find every (env, run_name) with its seed directories
* ``load_progress``     -- episodic returns of one seed
* ``seed_curve``        -- one seed's smoothed return interpolated onto a step grid
* ``final_return``      -- mean of the last N episodes of one seed
* ``bootstrap_ci``      -- percentile bootstrap confidence interval of a mean
* ``aggregate_env``     -- per-config curves (mean and 95% CI over seeds) for one env
* ``summarize``         -- final return table (mean +/- CI over seeds, delta vs baseline)

Run as a script to write the tables to ``--out-dir``::

    python analysis/aggregate.py --results-dir results --out-dir analysis/output

Conventions (see README): the learning curve of a seed at step ``s`` is the mean return of the
last ``window`` episodes that finished at or before ``s``; the final return of a seed is the mean
return of its last 10 episodes; the interval over seeds is a 95% percentile bootstrap of the mean.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASELINE = "baseline"
FINISHED_FRACTION = 0.95  # a run is finished when its last logged step >= this * total_timesteps
DEFAULT_WINDOW = 20       # episodes in the trailing smoothing window of learning curves
DEFAULT_LAST_N = 10       # episodes averaged for the final return of one seed
DEFAULT_GRID_POINTS = 200
DEFAULT_N_BOOT = 10_000
DEFAULT_CI = 0.95

_SEED_RE = re.compile(r"^seed_(\d+)$")


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
@dataclass
class SeedRun:
    """The logs of one (env, config, seed)."""
    env: str
    config: str
    seed: int
    path: Path
    steps: np.ndarray                 # global_step of each finished episode (non-decreasing)
    returns: np.ndarray               # episodic_return
    lengths: np.ndarray               # episodic_length
    total_timesteps: Optional[int]    # from config.json (None if unavailable)
    last_step: int                    # last logged step (metrics.csv, else progress.csv)

    @property
    def finished(self) -> bool:
        if self.total_timesteps is None:
            return True
        return self.last_step >= FINISHED_FRACTION * self.total_timesteps


def _seed_of(path: Path) -> Optional[int]:
    m = _SEED_RE.match(path.name)
    return int(m.group(1)) if m else None


def discover_runs(results_dir) -> Dict[str, Dict[str, List[Path]]]:
    """{env_id: {run_name: [seed_dir, ...]}} for every seed directory that has a progress.csv."""
    root = Path(results_dir)
    found: Dict[str, Dict[str, List[Path]]] = {}
    if not root.is_dir():
        return found
    for env_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            seeds = sorted((p for p in run_dir.iterdir()
                            if p.is_dir() and _seed_of(p) is not None and (p / "progress.csv").is_file()),
                           key=_seed_of)
            if seeds:
                found.setdefault(env_dir.name, {})[run_dir.name] = seeds
    return found


def _last_step(csv_path: Path) -> Optional[int]:
    if not csv_path.is_file():
        return None
    try:
        col = pd.read_csv(csv_path, usecols=["global_step"])["global_step"]
    except (ValueError, pd.errors.EmptyDataError):
        return None
    return int(col.iloc[-1]) if len(col) else None


def load_seed(seed_dir, env: Optional[str] = None, config: Optional[str] = None) -> SeedRun:
    """Load one seed directory (progress.csv is required, the rest is optional)."""
    seed_dir = Path(seed_dir)
    env = env or seed_dir.parent.parent.name
    config = config or seed_dir.parent.name
    seed = _seed_of(seed_dir)
    if seed is None:
        raise ValueError(f"not a seed directory: {seed_dir}")
    try:
        df = pd.read_csv(seed_dir / "progress.csv")
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["global_step", "episodic_return", "episodic_length"])
    df = df.dropna()
    df = df.sort_values("global_step", kind="stable")
    total = None
    cfg_path = seed_dir / "config.json"
    if cfg_path.is_file():
        try:
            total = int(json.loads(cfg_path.read_text())["total_timesteps"])
        except (ValueError, KeyError, TypeError):
            total = None
    steps = df["global_step"].to_numpy(dtype=np.int64)
    candidates = [s for s in (_last_step(seed_dir / "metrics.csv"),
                              int(steps[-1]) if len(steps) else None) if s is not None]
    return SeedRun(env=env, config=config, seed=seed, path=seed_dir, steps=steps,
                   returns=df["episodic_return"].to_numpy(dtype=np.float64),
                   lengths=df["episodic_length"].to_numpy(dtype=np.int64),
                   total_timesteps=total, last_step=max(candidates) if candidates else 0)


def load_runs(results_dir, envs: Optional[Sequence[str]] = None, configs: Optional[Sequence[str]] = None,
              include_unfinished: bool = False, verbose: bool = True
              ) -> Dict[str, Dict[str, List[SeedRun]]]:
    """{env: {config: [SeedRun, ...]}}. Unfinished seeds are dropped unless include_unfinished."""
    out: Dict[str, Dict[str, List[SeedRun]]] = {}
    for env, by_cfg in discover_runs(results_dir).items():
        if envs and env not in envs:
            continue
        for cfg_name, seed_dirs in by_cfg.items():
            if configs and cfg_name not in configs:
                continue
            runs = [load_seed(p, env, cfg_name) for p in seed_dirs]
            kept = [r for r in runs if include_unfinished or r.finished]
            if verbose and len(kept) < len(runs):
                dropped = [r.seed for r in runs if r not in kept]
                warnings.warn(f"{env}/{cfg_name}: ignoring unfinished seeds {dropped} "
                              f"(pass include_unfinished=True / --include-unfinished to keep them)")
            if kept:
                out.setdefault(env, {})[cfg_name] = kept
    return out


# --------------------------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------------------------
def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last `window` elements (fewer at the start); same length as x."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    window = max(1, int(window))
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    return (csum[idx] - csum[lo]) / (idx - lo)


def make_grid(max_step: float, n_points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
    return np.linspace(0.0, float(max_step), int(n_points))


def seed_curve(steps: np.ndarray, returns: np.ndarray, grid: np.ndarray,
               window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Smoothed episodic return of one seed on `grid`.

    The returns are smoothed with a trailing window over episodes, then linearly interpolated at
    the grid steps. Grid points before the first finished episode or after the last one are NaN
    (no extrapolation).
    """
    grid = np.asarray(grid, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    if len(steps) == 0:
        return np.full(grid.shape, np.nan)
    smooth = trailing_mean(returns, window)
    curve = np.interp(grid, steps, smooth)
    curve[(grid < steps[0]) | (grid > steps[-1])] = np.nan
    return curve


def final_return(run: SeedRun, last_n: int = DEFAULT_LAST_N) -> float:
    """Mean return of the last `last_n` episodes of a seed (NaN if it finished no episode)."""
    if len(run.returns) == 0:
        return float("nan")
    return float(np.mean(run.returns[-last_n:]))


def bootstrap_ci(values: np.ndarray, n_boot: int = DEFAULT_N_BOOT, ci: float = DEFAULT_CI,
                 rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bootstrap CI of the mean over axis 0 (the seed axis).

    Returns (mean, low, high), each with the shape of ``values.shape[1:]``. With a single sample
    the interval collapses to the mean (there is nothing to resample).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    n = values.shape[0]
    mean = values.mean(axis=0)
    if n < 2:
        lo = hi = mean.copy()
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = rng.integers(0, n, size=(int(n_boot), n))
        boot = values[idx].mean(axis=1)  # [n_boot, ...]
        alpha = (1.0 - ci) / 2.0
        lo, hi = np.quantile(boot, [alpha, 1.0 - alpha], axis=0)
    if squeeze:
        return mean[0], lo[0], hi[0]
    return mean, lo, hi


@dataclass
class ConfigCurve:
    env: str
    config: str
    seeds: List[int]
    grid: np.ndarray
    per_seed: np.ndarray            # [S, G], NaN outside a seed's coverage
    mean: np.ndarray                # [G], NaN where not every seed has data
    lo: np.ndarray
    hi: np.ndarray
    n_seeds: int = field(init=False)

    def __post_init__(self):
        self.n_seeds = len(self.seeds)


def aggregate_config(runs: Sequence[SeedRun], grid: np.ndarray, window: int = DEFAULT_WINDOW,
                     n_boot: int = DEFAULT_N_BOOT, ci: float = DEFAULT_CI, seed: int = 0) -> ConfigCurve:
    """Mean and bootstrap CI over seeds, only where *every* seed has data (common-grid rule)."""
    per_seed = np.stack([seed_curve(r.steps, r.returns, grid, window) for r in runs])
    valid = np.all(np.isfinite(per_seed), axis=0)
    mean = np.full(grid.shape, np.nan)
    lo = np.full(grid.shape, np.nan)
    hi = np.full(grid.shape, np.nan)
    if valid.any():
        m, l, h = bootstrap_ci(per_seed[:, valid], n_boot, ci, np.random.default_rng(seed))
        mean[valid], lo[valid], hi[valid] = m, l, h
    first = runs[0]
    return ConfigCurve(first.env, first.config, [r.seed for r in runs], grid, per_seed, mean, lo, hi)


def aggregate_env(runs_by_config: Dict[str, List[SeedRun]], window: int = DEFAULT_WINDOW,
                  n_points: int = DEFAULT_GRID_POINTS, n_boot: int = DEFAULT_N_BOOT,
                  ci: float = DEFAULT_CI, seed: int = 0) -> Dict[str, ConfigCurve]:
    """Curves of every config of one env on one shared grid (0 .. largest step logged)."""
    max_step = max((r.steps[-1] for runs in runs_by_config.values() for r in runs if len(r.steps)),
                   default=0)
    grid = make_grid(max_step, n_points)
    return {name: aggregate_config(runs, grid, window, n_boot, ci, seed)
            for name, runs in runs_by_config.items()}


# --------------------------------------------------------------------------------------------
# final returns and tables
# --------------------------------------------------------------------------------------------
# Display order: baseline, then the ablations in the order of the study; unknown names go last.
CONFIG_ORDER = (BASELINE, "no_gae", "no_adv_norm", "no_obs_norm", "no_vf_clip", "no_lr_anneal",
                "clip_0.1", "clip_0.3")


def config_order(names: Iterable[str]) -> List[str]:
    """Run names in display order (see CONFIG_ORDER), unknown names alphabetically at the end."""
    rank = {n: i for i, n in enumerate(CONFIG_ORDER)}
    return sorted(set(names), key=lambda n: (rank.get(n, len(rank)), n))


def final_returns_table(runs: Dict[str, Dict[str, List[SeedRun]]], last_n: int = DEFAULT_LAST_N) -> pd.DataFrame:
    """One row per (env, config, seed)."""
    rows = []
    for env, by_cfg in runs.items():
        for cfg_name in config_order(by_cfg):
            for r in by_cfg[cfg_name]:
                rows.append(dict(env=env, config=cfg_name, seed=r.seed, final_return=final_return(r, last_n),
                                 episodes=len(r.returns), last_step=r.last_step,
                                 total_timesteps=r.total_timesteps, finished=r.finished))
    cols = ["env", "config", "seed", "final_return", "episodes", "last_step", "total_timesteps", "finished"]
    return pd.DataFrame(rows, columns=cols)


def _diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator
             ) -> Tuple[float, float]:
    """Percentile bootstrap CI of mean(a) - mean(b), resampling both groups independently."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(ma - mb, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def summarize(per_seed: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT, ci: float = DEFAULT_CI,
              seed: int = 0) -> pd.DataFrame:
    """One row per (env, config): final return mean and CI over seeds, delta vs the env's baseline."""
    rows = []
    for env, env_df in per_seed.groupby("env", sort=True):
        base_vals = env_df.loc[env_df.config == BASELINE, "final_return"].dropna().to_numpy()
        for cfg_name in config_order(env_df.config.unique()):
            vals = env_df.loc[env_df.config == cfg_name, "final_return"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            rng = np.random.default_rng(seed)
            mean, lo, hi = bootstrap_ci(vals, n_boot, ci, rng)
            row = dict(env=env, config=cfg_name, n_seeds=len(vals), final_mean=float(mean),
                       ci_low=float(lo), ci_high=float(hi), std=float(vals.std(ddof=1)) if len(vals) > 1 else np.nan)
            if len(base_vals) and cfg_name != BASELINE:
                d_lo, d_hi = _diff_ci(vals, base_vals, n_boot, ci, np.random.default_rng(seed + 1))
                row.update(delta=float(mean - base_vals.mean()),
                           delta_pct=float(100.0 * (mean - base_vals.mean()) / abs(base_vals.mean()))
                           if base_vals.mean() != 0 else np.nan,
                           delta_ci_low=d_lo, delta_ci_high=d_hi,
                           differs=bool(np.isfinite(d_lo) and (d_lo > 0 or d_hi < 0)))
            else:
                row.update(delta=np.nan, delta_pct=np.nan, delta_ci_low=np.nan, delta_ci_high=np.nan,
                           differs=np.nan)
            rows.append(row)
    cols = ["env", "config", "n_seeds", "final_mean", "ci_low", "ci_high", "std", "delta", "delta_pct",
            "delta_ci_low", "delta_ci_high", "differs"]
    return pd.DataFrame(rows, columns=cols)


def _fmt(x, digits=1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{digits}f}"


def to_markdown(summary: pd.DataFrame, ci: float = DEFAULT_CI) -> str:
    """One Markdown table per environment."""
    pct = int(round(ci * 100))
    parts: List[str] = []
    for env, df in summary.groupby("env", sort=True):
        parts.append(f"### {env}\n")
        parts.append(f"| Config | Final return (mean) | {pct}% CI | Delta vs baseline | Delta {pct}% CI | Seeds |")
        parts.append("|---|---:|---:|---:|---:|---:|")
        for _, r in df.iterrows():
            if r.config == BASELINE:
                delta, delta_ci = "-", "-"
            else:
                pct_s = f" ({r.delta_pct:+.0f}%)" if np.isfinite(r.delta_pct) else ""
                delta = f"{r.delta:+,.1f}{pct_s}" if np.isfinite(r.delta) else "n/a"
                delta_ci = (f"[{r.delta_ci_low:+,.1f}, {r.delta_ci_high:+,.1f}]"
                            if np.isfinite(r.delta_ci_low) else "n/a")
                if bool(r.differs):
                    delta_ci += " *"
            parts.append(f"| {r.config} | {_fmt(r.final_mean)} | [{_fmt(r.ci_low)}, {_fmt(r.ci_high)}] | "
                         f"{delta} | {delta_ci} | {int(r.n_seeds)} |")
        parts.append("")
    parts.append(f"Final return = mean of the last {DEFAULT_LAST_N} training episodes per seed; "
                 f"mean and {pct}% percentile-bootstrap CI over seeds. "
                 f"`*` marks a difference to the baseline whose bootstrap CI excludes zero.")
    return "\n".join(parts) + "\n"


def curves_table(curves: Dict[str, Dict[str, ConfigCurve]]) -> pd.DataFrame:
    """Long-format table of all aggregated curves (valid grid points only)."""
    rows = []
    for env, by_cfg in curves.items():
        for name in config_order(by_cfg):
            c = by_cfg[name]
            ok = np.isfinite(c.mean)
            for step, m, l, h in zip(c.grid[ok], c.mean[ok], c.lo[ok], c.hi[ok]):
                rows.append((env, name, c.n_seeds, step, m, l, h))
    return pd.DataFrame(rows, columns=["env", "config", "n_seeds", "global_step", "mean", "ci_low", "ci_high"])


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="analysis/output")
    p.add_argument("--envs", nargs="*", default=None, help="restrict to these env ids")
    p.add_argument("--configs", nargs="*", default=None, help="restrict to these run names")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="smoothing window in episodes")
    p.add_argument("--last-n", type=int, default=DEFAULT_LAST_N, help="episodes in the final return")
    p.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--include-unfinished", action="store_true",
                   help="keep seeds whose run did not reach 95%% of total_timesteps")
    return p


def run(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    runs = load_runs(args.results_dir, args.envs, args.configs, args.include_unfinished)
    if not runs:
        print(f"no finished runs found under {args.results_dir}", file=sys.stderr)
        return None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_seed = final_returns_table(runs, args.last_n)
    summary = summarize(per_seed, args.n_boot)
    curves = {env: aggregate_env(by_cfg, args.window, args.grid_points, args.n_boot)
              for env, by_cfg in runs.items()}
    per_seed.to_csv(out / "final_returns_per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary))
    curves_table(curves).to_csv(out / "curves.csv", index=False)
    print(to_markdown(summary))
    print(f"wrote summary.csv, summary.md, final_returns_per_seed.csv, curves.csv to {out}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
