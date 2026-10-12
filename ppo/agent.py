"""Actor-critic network (Session A)."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _layer_init(layer: nn.Linear, gain: float = math.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    """Orthogonal weight init with the given gain, constant bias."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int, out_gain: float) -> nn.Sequential:
    """tanh MLP: hidden layers use gain sqrt(2), the output layer uses `out_gain`."""
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(_layer_init(nn.Linear(prev, h), math.sqrt(2.0)))
        layers.append(nn.Tanh())
        prev = h
    layers.append(_layer_init(nn.Linear(prev, out_dim), out_gain))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (tanh, orthogonal init).

    Discrete: actor outputs logits of a Categorical (head gain 0.01).
    Continuous: actor outputs the mean of a diagonal Normal (head gain 0.01) with a
    state-independent learnable log-std parameter (initialised to 0, i.e. std = 1).
    The critic head has gain 1.0.
    """

    def __init__(self, obs_dim: int, act_dim: int, discrete: bool,
                 hidden_sizes: Sequence[int] = (64, 64)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.discrete = bool(discrete)
        hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.critic = _mlp(self.obs_dim, hidden_sizes, 1, out_gain=1.0)
        self.actor = _mlp(self.obs_dim, hidden_sizes, self.act_dim, out_gain=0.01)
        if not self.discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [B, obs_dim] -> value [B]."""
        return self.critic(obs).squeeze(-1)

    def _distribution(self, obs: torch.Tensor):
        out = self.actor(obs)
        if self.discrete:
            return Categorical(logits=out)
        std = torch.exp(self.actor_logstd.expand_as(out))
        return Normal(out, std)

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, logprob [B], entropy [B], value [B]).

        If `action` is None one is sampled (discrete: long [B]; continuous: float [B, act_dim],
        unclipped - environment-side clipping is the caller's job). Otherwise `action` is evaluated.
        For continuous actions log-prob and entropy are summed over action dimensions.
        """
        dist = self._distribution(obs)
        if action is None:
            action = dist.sample()
        if self.discrete:
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            logprob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.get_value(obs)
