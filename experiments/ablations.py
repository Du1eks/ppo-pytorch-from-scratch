"""Ablation definitions: name -> Config overrides applied on top of the per-env default config."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python experiments/ablations.py`
from ppo.config import Config  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# env_id -> YAML file with its default config
ENV_CONFIGS: Dict[str, str] = {
    "CartPole-v1": "cartpole.yaml",
    "LunarLander-v3": "lunarlander.yaml",
    "HalfCheetah-v5": "halfcheetah.yaml",
    "Hopper-v5": "hopper.yaml",
}

# Environments on which the ablation study is run (main runs use all of ENV_CONFIGS).
ABLATION_ENVS = ("HalfCheetah-v5", "Hopper-v5")

BASELINE = "baseline"

ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_gae": {"use_gae": False},          # Monte-Carlo discounted returns, advantage = return - value
    "no_adv_norm": {"norm_adv": False},
    "no_obs_norm": {"normalize_obs": False},
    "no_vf_clip": {"clip_vloss": False},
    "no_lr_anneal": {"anneal_lr": False},
    "clip_0.1": {"clip_coef": 0.1},
    "clip_0.3": {"clip_coef": 0.3},
}
# clip_coef = 0.2 is the default, i.e. the baseline run, so it is not repeated here.


def config_names() -> list:
    """All run names for one env: the baseline plus every ablation."""
    return [BASELINE, *ABLATIONS]


def load_base(env_id: str) -> Dict[str, Any]:
    """Load the default YAML for env_id as a plain dict (validated through Config)."""
    if env_id not in ENV_CONFIGS:
        raise KeyError(f"no config for env {env_id!r}; known: {sorted(ENV_CONFIGS)}")
    with open(CONFIG_DIR / ENV_CONFIGS[env_id]) as f:
        d = yaml.safe_load(f)
    Config.from_dict(d)  # raises on unknown keys
    return d


def build_config(env_id: str, name: str = BASELINE, seed: int = 0, **extra: Any) -> Config:
    """Config for (env, baseline-or-ablation name, seed); run_name is the config name."""
    if name != BASELINE and name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; known: {[BASELINE, *ABLATIONS]}")
    d = load_base(env_id)
    d.update(ABLATIONS.get(name, {}))
    d.update(extra)
    d["seed"] = seed
    d["run_name"] = name
    return Config.from_dict(d)


if __name__ == "__main__":
    for n, o in ABLATIONS.items():
        print(f"{n:14s} {o}")
