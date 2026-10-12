"""Rollout storage, GAE and minibatching (Session A)."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size on-policy storage, tensors shaped [num_steps, num_envs, ...].

    Done-flag convention (CleanRL): ``dones[t]`` is the done flag that is in effect when
    ``obs[t]`` is observed, i.e. it is 1 if the step taken *before* ``obs[t]`` ended an episode
    (``terminated or truncated``), so ``obs[t]`` is the first observation of a new episode.
    The flag produced by the step taken from ``obs[t]`` is ``dones[t + 1]`` (or ``last_done``
    for the final stored step). GAE therefore uses ``1 - dones[t + 1]`` as the "next is not
    terminal" mask for step ``t`` and ``1 - last_done`` for the last step.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_shape: Tuple[int, ...],
                 act_shape: Tuple[int, ...], device: str = "cpu"):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.obs_shape = tuple(obs_shape)
        self.act_shape = tuple(act_shape)
        self.device = torch.device(device)
        n, e = self.num_steps, self.num_envs
        dev = self.device
        self.obs = torch.zeros((n, e) + self.obs_shape, dtype=torch.float32, device=dev)
        # Scalar action shape -> discrete (long); the dtype is re-synced on the first add().
        act_dtype = torch.long if self.act_shape == () else torch.float32
        self.actions = torch.zeros((n, e) + self.act_shape, dtype=act_dtype, device=dev)
        self.logprobs = torch.zeros((n, e), dtype=torch.float32, device=dev)
        self.rewards = torch.zeros((n, e), dtype=torch.float32, device=dev)
        self.dones = torch.zeros((n, e), dtype=torch.float32, device=dev)
        self.values = torch.zeros((n, e), dtype=torch.float32, device=dev)
        self.advantages = torch.zeros((n, e), dtype=torch.float32, device=dev)
        self.returns = torch.zeros((n, e), dtype=torch.float32, device=dev)
        self.pos = 0

    # ------------------------------------------------------------------ storage
    def _to(self, x, like: torch.Tensor, dtype=None) -> torch.Tensor:
        dtype = like.dtype if dtype is None else dtype
        return torch.as_tensor(x).detach().to(device=self.device, dtype=dtype)

    def add(self, obs, action, logprob, reward, done, value) -> None:
        """Store one timestep for all envs. `done` = terminated or truncated of the step just taken."""
        if self.pos >= self.num_steps:
            raise RuntimeError("RolloutBuffer is full; call compute_returns and consume it, "
                               "then reset() before adding more")
        t = self.pos
        action = torch.as_tensor(action).detach()
        if self.pos == 0 and action.dtype != self.actions.dtype:
            # adopt the dtype the policy really produces (long for discrete, float for Box)
            self.actions = self.actions.to(action.dtype)
        self.obs[t] = self._to(obs, self.obs).reshape((self.num_envs,) + self.obs_shape)
        self.actions[t] = action.to(device=self.device, dtype=self.actions.dtype).reshape(
            (self.num_envs,) + self.act_shape)
        self.logprobs[t] = self._to(logprob, self.logprobs).reshape(self.num_envs)
        self.rewards[t] = self._to(reward, self.rewards).reshape(self.num_envs)
        self.dones[t] = self._to(done, self.dones, dtype=torch.float32).reshape(self.num_envs)
        self.values[t] = self._to(value, self.values).reshape(self.num_envs)
        self.pos += 1

    def reset(self) -> None:
        """Start a new rollout (storage is simply overwritten)."""
        self.pos = 0

    # ------------------------------------------------------------------ returns
    @torch.no_grad()
    def compute_returns(self, last_value: torch.Tensor, last_done: torch.Tensor,
                        gamma: float, gae_lambda: float, use_gae: bool = True) -> None:
        """Fill advantages and returns (CleanRL convention). use_gae=False -> discounted MC returns bootstrapped with last_value."""
        last_value = self._to(last_value, self.values).reshape(self.num_envs)
        last_done = self._to(last_done, self.dones, dtype=torch.float32).reshape(self.num_envs)
        T = self.num_steps
        if use_gae:
            lastgae = torch.zeros(self.num_envs, device=self.device)
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - last_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                lastgae = delta + gamma * gae_lambda * nextnonterminal * lastgae
                self.advantages[t] = lastgae
            self.returns = self.advantages + self.values
        else:
            nextreturn = last_value
            for t in reversed(range(T)):
                nextnonterminal = 1.0 - (last_done if t == T - 1 else self.dones[t + 1])
                nextreturn = self.rewards[t] + gamma * nextnonterminal * nextreturn
                self.returns[t] = nextreturn
            self.advantages = self.returns - self.values

    # ------------------------------------------------------------------ batching
    def get_minibatches(self, num_minibatches: int, rng: np.random.Generator) -> Iterator[dict]:
        """Yield dicts with keys obs, actions, logprobs, advantages, returns, values (flattened, shuffled)."""
        batch_size = self.num_steps * self.num_envs
        flat = {
            "obs": self.obs.reshape((batch_size,) + self.obs_shape),
            "actions": self.actions.reshape((batch_size,) + self.act_shape),
            "logprobs": self.logprobs.reshape(batch_size),
            "advantages": self.advantages.reshape(batch_size),
            "returns": self.returns.reshape(batch_size),
            "values": self.values.reshape(batch_size),
        }
        perm = torch.as_tensor(rng.permutation(batch_size), dtype=torch.long, device=self.device)
        # array_split keeps every sample even when batch_size % num_minibatches != 0
        for idx in np.array_split(perm.cpu().numpy(), num_minibatches):
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
            yield {k: v[idx] for k, v in flat.items()}
