"""Learning-curve figures and the ablation table.

For every environment found under ``--results-dir`` this writes

* ``<env>_baseline.png``   mean learning curve of the default config, 95% bootstrap CI over seeds
* ``<env>_ablations.png``  one panel per ablation, each drawn against the baseline on the same axes
                           (the two clip coefficients share one panel together with the baseline)

and, through ``analysis.aggregate``, ``summary.csv`` / ``summary.md`` (the ablation table),
``final_returns_per_seed.csv`` and ``curves.csv``::

    python analysis/plot_curves.py --results-dir results --out-dir analysis/output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python analysis/plot_curves.py`
from analysis import aggregate as agg  # noqa: E402

# ---- style: light surface, recessive grid, text in ink colours (never the series colour) -------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {  # fixed colour per configuration (colour follows the entity, not its rank)
    "baseline": "#2a78d6",
    "no_gae": "#eb6834",
    "no_adv_norm": "#1baf7a",
    "no_obs_norm": "#6250d6",
    "no_vf_clip": "#e87ba4",
    "no_lr_anneal": "#eda100",
    "clip_0.1": "#e34948",
    "clip_0.3": "#008300",
}
FALLBACK = ["#6250d6", "#e87ba4", "#eda100", "#e34948", "#008300"]
LABELS = {
    "baseline": "baseline (clip 0.2)",
    "no_gae": "no GAE (Monte-Carlo returns)",
    "no_adv_norm": "no advantage normalisation",
    "no_obs_norm": "no observation normalisation",
    "no_vf_clip": "no value-loss clipping",
    "no_lr_anneal": "no LR annealing",
    "clip_0.1": "clip 0.1",
    "clip_0.3": "clip 0.3",
}
PANEL_TITLES = {"clip": "clip coefficient"}
LINE_WIDTH = 2.0


def color_of(name: str, _extra: Dict[str, str] = {}) -> str:
    if name in COLORS:
        return COLORS[name]
    if name not in _extra:
        _extra[name] = FALLBACK[len(_extra) % len(FALLBACK)]
    return _extra[name]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_steps))


def _fmt_steps(x, _pos=None) -> str:
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def _draw(ax, curve: agg.ConfigCurve, name: str, label: Optional[str] = None) -> bool:
    ok = np.isfinite(curve.mean)
    if not ok.any():
        return False
    c = color_of(name)
    ax.fill_between(curve.grid[ok], curve.lo[ok], curve.hi[ok], color=c, alpha=0.18, linewidth=0)
    ax.plot(curve.grid[ok], curve.mean[ok], color=c, linewidth=LINE_WIDTH,
            label=f"{label or LABELS.get(name, name)} (n={curve.n_seeds})")
    return True


def _threshold(env: str) -> Optional[float]:
    """The environment's registered solved-score, if gymnasium defines one."""
    try:
        import gymnasium as gym
        return gym.spec(env).reward_threshold
    except Exception:  # unknown env / gymnasium missing
        return None


def _finish(fig, axes, env: str, window: int, n_note: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(f"{env}  |  mean and 95% bootstrap CI over seeds{n_note}", color=INK, fontsize=11, x=0.01,
                 ha="left")
    for ax in axes:
        _style_axes(ax)


def plot_baseline(curves: Dict[str, agg.ConfigCurve], env: str, path: Path, window: int) -> Optional[Path]:
    """Baseline-only figure (falls back to every config if the env has no baseline run)."""
    names = [agg.BASELINE] if agg.BASELINE in curves else agg.config_order(curves)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    drawn = [n for n in names if _draw(ax, curves[n], n)]
    if not drawn:
        plt.close(fig)
        return None
    thr = _threshold(env)
    if thr is not None:
        ax.axhline(thr, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(f"solved: {thr:g}", xy=(0.01, thr), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", color=INK_2, fontsize=8)
    ax.set_xlabel("environment steps", color=INK_2)
    ax.set_ylabel(f"episodic return ({window}-episode trailing mean)", color=INK_2, fontsize=9)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    _finish(fig, [ax], env, window, "")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def ablation_groups(names: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """[(panel title, [config names to draw next to the baseline])], clip variants share a panel."""
    others = [n for n in agg.config_order(names) if n != agg.BASELINE]
    groups: List[Tuple[str, List[str]]] = []
    clips = [n for n in others if n.startswith("clip_")]
    for n in others:
        if n in clips:
            continue
        groups.append((LABELS.get(n, n), [n]))
    if clips:
        groups.append((PANEL_TITLES["clip"], clips))
    return groups


def plot_ablations(curves: Dict[str, agg.ConfigCurve], env: str, path: Path, window: int,
                   ncols: int = 3) -> Optional[Path]:
    """Small multiples: each panel shows one ablation (or the clip family) against the baseline."""
    groups = ablation_groups(list(curves))
    if not groups:
        return None
    ncols = min(ncols, len(groups))
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    flat = list(axes.ravel())
    for ax, (title, members) in zip(flat, groups):
        if agg.BASELINE in curves:
            _draw(ax, curves[agg.BASELINE], agg.BASELINE, "baseline")
        for n in members:
            _draw(ax, curves[n], n)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    for ax in flat[len(groups):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("environment steps", color=INK_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"episodic return ({window}-ep. mean)", color=INK_2, fontsize=9)
    flat[0].set_xlim(left=0)
    _finish(fig, [ax for ax in flat if ax.get_visible()], env, window, "")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="analysis/output")
    p.add_argument("--envs", nargs="*", default=None)
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--window", type=int, default=agg.DEFAULT_WINDOW)
    p.add_argument("--last-n", type=int, default=agg.DEFAULT_LAST_N)
    p.add_argument("--grid-points", type=int, default=agg.DEFAULT_GRID_POINTS)
    p.add_argument("--n-boot", type=int, default=agg.DEFAULT_N_BOOT)
    p.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    p.add_argument("--include-unfinished", action="store_true")
    return p


def run(args: argparse.Namespace) -> List[Path]:
    runs = agg.load_runs(args.results_dir, args.envs, args.configs, args.include_unfinished)
    if not runs:
        print(f"no finished runs found under {args.results_dir}", file=sys.stderr)
        return []
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    all_curves = {}
    for env, by_cfg in runs.items():
        curves = agg.aggregate_env(by_cfg, args.window, args.grid_points, args.n_boot)
        all_curves[env] = curves
        for fn, suffix in ((plot_baseline, "baseline"), (plot_ablations, "ablations")):
            p = fn(curves, env, fig_dir / f"{env}_{suffix}.{args.format}", args.window)
            if p is not None:
                written.append(p)
    # tables (CSV + Markdown), same conventions as analysis/aggregate.py
    per_seed = agg.final_returns_table(runs, args.last_n)
    summary = agg.summarize(per_seed, args.n_boot)
    per_seed.to_csv(out / "final_returns_per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(agg.to_markdown(summary))
    agg.curves_table(all_curves).to_csv(out / "curves.csv", index=False)
    written += [out / n for n in ("summary.csv", "summary.md", "final_returns_per_seed.csv", "curves.csv")]
    for p in written:
        print("wrote", p)
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
