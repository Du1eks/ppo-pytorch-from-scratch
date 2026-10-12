"""Frozen hyperparameter container shared by every module."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple


@dataclass
class Config:
    # run identity
    env_id: str = "CartPole-v1"
    seed: int = 0
    run_name: str = "ppo"
    out_dir: str = "results"
    device: str = "cpu"

    # rollout / optimisation sizes
    total_timesteps: int = 500_000
    num_envs: int = 4
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4

    # optimiser
    lr: float = 2.5e-4
    anneal_lr: bool = True

    # advantage estimation
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_gae: bool = True

    # PPO loss
    clip_coef: float = 0.2
    clip_vloss: bool = True
    norm_adv: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.0  # 0.0 disables early stopping

    # normalisation (continuous control)
    normalize_obs: bool = False
    normalize_reward: bool = False

    # network
    hidden_sizes: Tuple[int, ...] = (64, 64)

    # ---- derived helpers (not hyperparameters) ----
    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return self.total_timesteps // self.batch_size

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        d = dict(d)
        if "hidden_sizes" in d:
            d["hidden_sizes"] = tuple(d["hidden_sizes"])
        return cls(**d)
