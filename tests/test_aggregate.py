"""Tests for analysis/aggregate.py on small synthetic logs (written to a temp dir, never to results/)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import aggregate as agg  # noqa: E402


def write_seed(root, env, config, seed, steps, returns, total=1000, last_metrics_step=None, with_config=True):
    d = Path(root) / env / config / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"global_step": steps, "episodic_return": returns,
                  "episodic_length": np.full(len(steps), 10)}).to_csv(d / "progress.csv", index=False)
    last = last_metrics_step if last_metrics_step is not None else (steps[-1] if len(steps) else 0)
    pd.DataFrame({"global_step": [last]}).assign(
        policy_loss=0, value_loss=0, entropy=0, approx_kl=0, clipfrac=0, explained_variance=0,
        learning_rate=0, sps=0).to_csv(d / "metrics.csv", index=False)
    if with_config:
        (d / "config.json").write_text(json.dumps({"total_timesteps": total}))
    return d


def test_trailing_mean():
    out = agg.trailing_mean(np.array([1.0, 3.0, 5.0, 7.0]), 2)
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(agg.trailing_mean(np.array([2.0, 4.0]), 10), [2.0, 3.0])
    assert agg.trailing_mean(np.array([]), 3).shape == (0,)


def test_seed_curve_interpolates_and_masks_outside_coverage():
    steps = np.array([100, 200, 300])
    returns = np.array([10.0, 20.0, 30.0])
    grid = np.array([0.0, 100.0, 150.0, 300.0, 400.0])
    curve = agg.seed_curve(steps, returns, grid, window=1)
    assert np.isnan(curve[0]) and np.isnan(curve[-1])  # before first / after last episode
    np.testing.assert_allclose(curve[1:4], [10.0, 15.0, 30.0])
    # window=2 smooths: [10, 15, 25]
    curve2 = agg.seed_curve(steps, returns, grid, window=2)
    np.testing.assert_allclose(curve2[1:4], [10.0, 12.5, 25.0])


def test_seed_curve_empty_run_is_all_nan():
    assert np.isnan(agg.seed_curve(np.array([]), np.array([]), np.linspace(0, 10, 5))).all()


def test_bootstrap_ci_properties():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 10, size=50)
    mean, lo, hi = agg.bootstrap_ci(x, n_boot=4000, rng=np.random.default_rng(1))
    assert mean == pytest.approx(x.mean())
    assert lo < mean < hi
    # close to the normal-theory interval for a mean
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.2)
    # constant data -> zero width, single sample -> collapses to the value
    m, l, h = agg.bootstrap_ci(np.full(5, 3.0))
    assert (m, l, h) == (3.0, 3.0, 3.0)
    m, l, h = agg.bootstrap_ci(np.array([7.0]))
    assert (m, l, h) == (7.0, 7.0, 7.0)


def test_bootstrap_ci_is_reproducible_and_vectorised():
    x = np.random.default_rng(0).normal(size=(5, 12))
    a = agg.bootstrap_ci(x, n_boot=500, rng=np.random.default_rng(3))
    b = agg.bootstrap_ci(x, n_boot=500, rng=np.random.default_rng(3))
    for u, v in zip(a, b):
        assert u.shape == (12,)
        np.testing.assert_array_equal(u, v)
    assert np.all(a[1] <= a[0]) and np.all(a[0] <= a[2])


def test_discover_and_load(tmp_path):
    write_seed(tmp_path, "EnvA", "baseline", 0, [10, 20], [1.0, 2.0])
    write_seed(tmp_path, "EnvA", "baseline", 10, [10, 20], [1.0, 2.0])
    write_seed(tmp_path, "EnvA", "baseline", 2, [10, 20], [1.0, 2.0])
    write_seed(tmp_path, "EnvA", "no_gae", 0, [10, 20], [1.0, 2.0])
    (tmp_path / "EnvA" / "empty_run" / "seed_0").mkdir(parents=True)  # no progress.csv -> ignored
    found = agg.discover_runs(tmp_path)
    assert list(found) == ["EnvA"]
    assert set(found["EnvA"]) == {"baseline", "no_gae"}
    assert [p.name for p in found["EnvA"]["baseline"]] == ["seed_0", "seed_2", "seed_10"]  # numeric order
    assert agg.discover_runs(tmp_path / "does_not_exist") == {}


def test_unfinished_runs_are_dropped_by_default(tmp_path):
    write_seed(tmp_path, "EnvA", "baseline", 0, [100, 960], [1.0, 2.0], total=1000)               # finished
    write_seed(tmp_path, "EnvA", "baseline", 1, [100, 400], [1.0, 2.0], total=1000)               # unfinished
    write_seed(tmp_path, "EnvA", "baseline", 2, [100, 400], [1.0, 2.0], total=1000,
               last_metrics_step=960)                                                             # metrics say done
    write_seed(tmp_path, "EnvA", "baseline", 3, [100, 400], [1.0, 2.0], with_config=False)        # unknown total
    with pytest.warns(UserWarning, match="unfinished"):
        runs = agg.load_runs(tmp_path)
    assert [r.seed for r in runs["EnvA"]["baseline"]] == [0, 2, 3]
    all_runs = agg.load_runs(tmp_path, include_unfinished=True, verbose=False)
    assert [r.seed for r in all_runs["EnvA"]["baseline"]] == [0, 1, 2, 3]


def test_final_return_is_mean_of_last_n_episodes(tmp_path):
    returns = np.arange(30, dtype=float)
    d = write_seed(tmp_path, "EnvA", "baseline", 0, np.arange(1, 31) * 10, returns, total=300)
    run = agg.load_seed(d)
    assert agg.final_return(run) == pytest.approx(np.mean(returns[-10:]))
    assert agg.final_return(run, last_n=3) == pytest.approx(28.0)
    short = agg.load_seed(write_seed(tmp_path, "EnvA", "baseline", 1, [10, 20], [4.0, 6.0], total=20))
    assert agg.final_return(short) == pytest.approx(5.0)  # fewer than 10 episodes -> mean of all
    empty = agg.load_seed(write_seed(tmp_path, "EnvA", "baseline", 2, [], [], total=20))
    assert np.isnan(agg.final_return(empty))


def test_aggregate_config_uses_common_grid_only(tmp_path):
    # seed 0 covers steps 100..300, seed 1 covers 100..200 -> mean only valid up to 200
    a = agg.load_seed(write_seed(tmp_path, "E", "baseline", 0, [100, 200, 300], [10.0, 20.0, 30.0], total=300))
    b = agg.load_seed(write_seed(tmp_path, "E", "baseline", 1, [100, 200], [30.0, 40.0], total=200))
    grid = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0])
    curve = agg.aggregate_config([a, b], grid, window=1, n_boot=200)
    assert curve.n_seeds == 2 and curve.per_seed.shape == (2, 6)
    np.testing.assert_array_equal(np.isfinite(curve.mean), [False, True, True, True, False, False])
    np.testing.assert_allclose(curve.mean[1:4], [20.0, 25.0, 30.0])
    ok = np.isfinite(curve.mean)
    assert np.all(curve.lo[ok] <= curve.mean[ok]) and np.all(curve.mean[ok] <= curve.hi[ok])


def test_summarize_delta_and_significance(tmp_path):
    rng = np.random.default_rng(0)
    for seed in range(5):
        write_seed(tmp_path, "E", "baseline", seed, [10, 20, 30], 100 + rng.normal(0, 1, 3), total=30)
        write_seed(tmp_path, "E", "no_gae", seed, [10, 20, 30], 50 + rng.normal(0, 1, 3), total=30)
        write_seed(tmp_path, "E", "clip_0.1", seed, [10, 20, 30], 100 + rng.normal(0, 5, 3), total=30)
    runs = agg.load_runs(tmp_path)
    per_seed = agg.final_returns_table(runs)
    assert len(per_seed) == 15
    assert set(per_seed.columns) >= {"env", "config", "seed", "final_return"}
    summary = agg.summarize(per_seed, n_boot=2000).set_index("config")
    assert list(summary.index) == ["baseline", "no_gae", "clip_0.1"]  # display order
    assert summary.loc["baseline", "n_seeds"] == 5
    assert summary.loc["no_gae", "delta"] == pytest.approx(summary.loc["no_gae", "final_mean"]
                                                           - summary.loc["baseline", "final_mean"])
    assert summary.loc["no_gae", "delta_pct"] == pytest.approx(-50, abs=3)
    assert bool(summary.loc["no_gae", "differs"]) is True
    assert bool(summary.loc["clip_0.1", "differs"]) is False
    assert np.isnan(summary.loc["baseline", "delta"])
    for _, r in summary.iterrows():
        assert r.ci_low <= r.final_mean <= r.ci_high


def test_markdown_table(tmp_path):
    for seed in range(3):
        write_seed(tmp_path, "E", "baseline", seed, [10, 20], [100.0 + seed, 100.0], total=20)
        write_seed(tmp_path, "E", "no_gae", seed, [10, 20], [10.0 + seed, 10.0], total=20)
    runs = agg.load_runs(tmp_path)
    md = agg.to_markdown(agg.summarize(agg.final_returns_table(runs), n_boot=500))
    assert "### E" in md
    assert md.count("| baseline |") == 1 and md.count("| no_gae |") == 1
    assert "|---|" in md and "95% CI" in md and "*" in md


def test_cli_end_to_end(tmp_path):
    res, out = tmp_path / "results", tmp_path / "out"
    rng = np.random.default_rng(0)
    for cfg in ("baseline", "no_gae"):
        for seed in range(3):
            n = 40
            write_seed(res, "E", cfg, seed, np.arange(1, n + 1) * 25, np.linspace(0, 100, n) + rng.normal(0, 2, n),
                       total=1000)
    summary = agg.run(agg.build_parser().parse_args(
        ["--results-dir", str(res), "--out-dir", str(out), "--n-boot", "300", "--grid-points", "50"]))
    assert len(summary) == 2
    for name in ("summary.csv", "summary.md", "final_returns_per_seed.csv", "curves.csv"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    curves = pd.read_csv(out / "curves.csv")
    assert set(curves.config) == {"baseline", "no_gae"}
    assert (curves.ci_low <= curves["mean"]).all() and (curves["mean"] <= curves.ci_high).all()


def test_cli_without_runs_returns_none(tmp_path):
    assert agg.run(agg.build_parser().parse_args(
        ["--results-dir", str(tmp_path / "nothing"), "--out-dir", str(tmp_path / "out")])) is None
