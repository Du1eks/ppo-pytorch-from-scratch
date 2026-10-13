"""Tests for RunLogger (file layout, column names) and the Config helpers it depends on."""
import csv
import json

import pytest

from ppo.config import Config
from ppo.logger import RunLogger

PROGRESS_COLUMNS = ["global_step", "episodic_return", "episodic_length"]
METRICS_COLUMNS = ["global_step", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
                   "explained_variance", "learning_rate", "sps"]


def make_cfg(tmp_path, **kw):
    return Config(env_id="CartPole-v1", seed=3, run_name="unit", out_dir=str(tmp_path), **kw)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def metrics(i=0):
    return {"policy_loss": 0.1 * i, "value_loss": 1.0 + i, "entropy": 0.69, "approx_kl": 0.01,
            "clipfrac": 0.1, "explained_variance": 0.5, "learning_rate": 2.5e-4, "sps": 1000.0}


def test_run_dir_layout(tmp_path):
    logger = RunLogger(make_cfg(tmp_path))
    expected = tmp_path / "CartPole-v1" / "unit" / "seed_3"
    assert logger.run_dir == expected
    logger.close()
    for name in ("config.json", "progress.csv", "metrics.csv"):
        assert (expected / name).is_file(), name


def test_config_json_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path, lr=1e-3, hidden_sizes=(32, 16), normalize_obs=True)
    logger = RunLogger(cfg)
    logger.close()
    data = json.loads((logger.run_dir / "config.json").read_text())
    assert data == cfg.to_dict()
    assert Config.from_dict(data) == cfg


def test_column_names_even_without_rows(tmp_path):
    logger = RunLogger(make_cfg(tmp_path))
    logger.close()
    assert read_csv(logger.run_dir / "progress.csv") == [PROGRESS_COLUMNS]
    assert read_csv(logger.run_dir / "metrics.csv") == [METRICS_COLUMNS]


def test_rows_are_written_in_order(tmp_path):
    logger = RunLogger(make_cfg(tmp_path))
    logger.log_episode(100, 21.0, 21)
    logger.log_episode(250, 35.5, 30)
    for i in range(3):
        logger.log_metrics(512 * (i + 1), metrics(i))
    logger.close()

    progress = read_csv(logger.run_dir / "progress.csv")
    assert progress[1:] == [["100", "21.0", "21"], ["250", "35.5", "30"]]
    rows = read_csv(logger.run_dir / "metrics.csv")
    assert len(rows) == 4
    assert [int(r[0]) for r in rows[1:]] == [512, 1024, 1536]
    last = dict(zip(rows[0], rows[3]))
    assert float(last["value_loss"]) == pytest.approx(3.0)
    assert float(last["learning_rate"]) == pytest.approx(2.5e-4)
    assert float(last["sps"]) == pytest.approx(1000.0)


def test_rows_are_readable_before_close(tmp_path):
    # a killed run must still leave consistent files behind
    logger = RunLogger(make_cfg(tmp_path))
    logger.log_episode(10, 9.0, 9)
    logger.log_metrics(128, metrics())
    assert len(read_csv(logger.run_dir / "progress.csv")) == 2
    assert len(read_csv(logger.run_dir / "metrics.csv")) == 2
    logger.close()


def test_numpy_scalars_are_accepted(tmp_path):
    np = pytest.importorskip("numpy")
    logger = RunLogger(make_cfg(tmp_path))
    logger.log_episode(np.int64(5), np.float32(2.5), np.int32(2))
    logger.log_metrics(np.int64(128), {k: np.float32(v) for k, v in metrics().items()})
    logger.close()
    assert read_csv(logger.run_dir / "progress.csv")[1] == ["5", "2.5", "2"]


def test_close_is_idempotent(tmp_path):
    logger = RunLogger(make_cfg(tmp_path))
    logger.close()
    logger.close()


def test_runs_with_different_seeds_do_not_collide(tmp_path):
    a = RunLogger(make_cfg(tmp_path))
    b = RunLogger(Config(env_id="CartPole-v1", seed=4, run_name="unit", out_dir=str(tmp_path)))
    a.log_episode(1, 1.0, 1)
    b.close()
    a.close()
    assert a.run_dir != b.run_dir
    assert len(read_csv(a.run_dir / "progress.csv")) == 2
    assert len(read_csv(b.run_dir / "progress.csv")) == 1


# ---- Config helpers -------------------------------------------------------------

def test_config_derived_sizes():
    cfg = Config(total_timesteps=10_000, num_envs=4, num_steps=128, num_minibatches=4)
    assert cfg.batch_size == 512
    assert cfg.minibatch_size == 128
    assert cfg.num_iterations == 19


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Config.from_dict({"lr": 1e-3, "not_a_field": 1})


def test_config_hidden_sizes_list_becomes_tuple():
    cfg = Config.from_dict({"hidden_sizes": [8, 8]})
    assert cfg.hidden_sizes == (8, 8)
    assert cfg.to_dict()["hidden_sizes"] == [8, 8]
