"""Smoke tests for analysis/plot_curves.py on small synthetic logs (temp dir only)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import plot_curves as pc  # noqa: E402


def write_runs(root, env, configs, seeds=3, total=2000, episodes=80):
    rng = np.random.default_rng(0)
    for k, cfg in enumerate(configs):
        for seed in range(seeds):
            d = Path(root) / env / cfg / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            steps = np.linspace(total / episodes, total, episodes).astype(int)
            rets = (100 - 20 * k) * (1 - np.exp(-3 * steps / total)) + rng.normal(0, 3, episodes)
            pd.DataFrame({"global_step": steps, "episodic_return": rets,
                          "episodic_length": 25}).to_csv(d / "progress.csv", index=False)
            pd.DataFrame({"global_step": [total]}).to_csv(d / "metrics.csv", index=False)
            (d / "config.json").write_text(json.dumps({"total_timesteps": total}))


def test_ablation_groups_put_clip_variants_in_one_panel():
    groups = pc.ablation_groups(["clip_0.3", "baseline", "no_gae", "clip_0.1", "no_obs_norm"])
    titles = [t for t, _ in groups]
    assert [m for _, m in groups] == [["no_gae"], ["no_obs_norm"], ["clip_0.1", "clip_0.3"]]
    assert titles[-1] == "clip coefficient"
    assert pc.ablation_groups(["baseline"]) == []


def test_colors_are_fixed_per_config():
    assert pc.color_of("baseline") == pc.COLORS["baseline"]
    assert len({pc.COLORS[k] for k in pc.COLORS}) == len(pc.COLORS)
    assert pc.color_of("some_new_ablation") == pc.color_of("some_new_ablation")


def test_end_to_end_figures_and_tables(tmp_path):
    res, out = tmp_path / "results", tmp_path / "out"
    write_runs(res, "HalfCheetah-v5", ["baseline", "no_gae", "clip_0.1", "clip_0.3"])
    write_runs(res, "CartPole-v1", ["baseline"])
    written = pc.run(pc.build_parser().parse_args(
        ["--results-dir", str(res), "--out-dir", str(out), "--n-boot", "200", "--grid-points", "60"]))
    names = {p.name for p in written}
    assert {"HalfCheetah-v5_baseline.png", "HalfCheetah-v5_ablations.png", "CartPole-v1_baseline.png",
            "summary.csv", "summary.md"} <= names
    assert "CartPole-v1_ablations.png" not in names  # nothing to compare against
    for p in written:
        assert p.is_file() and p.stat().st_size > 0, p
    png = (out / "figures" / "HalfCheetah-v5_ablations.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary.config[summary.env == "HalfCheetah-v5"]) == {"baseline", "no_gae", "clip_0.1", "clip_0.3"}
    assert "### HalfCheetah-v5" in (out / "summary.md").read_text()


def test_no_runs_writes_nothing(tmp_path):
    assert pc.run(pc.build_parser().parse_args(
        ["--results-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path / "out")])) == []
    assert not (tmp_path / "out").exists()
