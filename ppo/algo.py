"""PPO update step (Session A)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ppo.agent import ActorCritic
from ppo.buffer import RolloutBuffer


def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true, unbiased=False)
    if var_y.item() == 0.0:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y)


def ppo_update(agent: ActorCritic, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer,
               rng: np.random.Generator, *, update_epochs: int, num_minibatches: int,
               clip_coef: float, clip_vloss: bool, norm_adv: bool, ent_coef: float,
               vf_coef: float, max_grad_norm: float, target_kl: float = 0.0) -> Dict[str, float]:
    """Returns dict with keys policy_loss, value_loss, entropy, approx_kl, clipfrac, explained_variance.

    Expects `buffer.compute_returns` to have been called. Reported losses/statistics are
    averaged over all minibatches that were processed; explained variance is computed once
    over the whole rollout (old value predictions vs. returns), as in CleanRL.
    If target_kl > 0, the update stops after the first epoch whose mean approx_kl exceeds it.
    """
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
    n_total = 0

    for _ in range(update_epochs):
        epoch_kl, n_epoch = 0.0, 0
        for mb in buffer.get_minibatches(num_minibatches, rng):
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb["obs"], mb["actions"])
            logratio = newlogprob - mb["logprobs"]
            ratio = logratio.exp()

            with torch.no_grad():
                # k3 estimator of KL(old || new), low variance and non-negative
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            adv = mb["advantages"]
            if norm_adv and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # clipped surrogate (maximise min(...) == minimise max of the negatives)
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if clip_vloss:
                v_unclipped = (newvalue - mb["returns"]) ** 2
                v_clipped = mb["values"] + torch.clamp(newvalue - mb["values"], -clip_coef, clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - mb["returns"]) ** 2).mean()
            else:
                v_loss = 0.5 * ((newvalue - mb["returns"]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            sums["policy_loss"] += pg_loss.item()
            sums["value_loss"] += v_loss.item()
            sums["entropy"] += entropy_loss.item()
            sums["approx_kl"] += approx_kl.item()
            sums["clipfrac"] += clipfrac.item()
            epoch_kl += approx_kl.item()
            n_epoch += 1
            n_total += 1

        if target_kl and target_kl > 0.0 and n_epoch > 0 and epoch_kl / n_epoch > target_kl:
            break

    stats = {k: v / max(n_total, 1) for k, v in sums.items()}
    stats["explained_variance"] = _explained_variance(buffer.values.reshape(-1),
                                                      buffer.returns.reshape(-1))
    return stats
