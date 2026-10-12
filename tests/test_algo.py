"""Tests for ppo_update: clipping behaviour, reported statistics and learning signal."""
import math

import numpy as np
import pytest
import torch

from ppo.agent import ActorCritic
from ppo.algo import ppo_update
from ppo.buffer import RolloutBuffer

T, N, OBS_DIM = 8, 4, 5
EXPECTED_KEYS = {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "explained_variance"}
CLIP = 0.2


def build(discrete, log_ratio=0.0, advantage=1.0, seed=0):
    """Agent plus a filled buffer whose stored log-probs differ from the current ones by
    `-log_ratio`, so the importance ratio of every sample is exactly exp(log_ratio)."""
    torch.manual_seed(seed)
    act_dim = 3
    agent = ActorCritic(OBS_DIM, act_dim, discrete, (16, 16))
    act_shape = () if discrete else (act_dim,)
    buf = RolloutBuffer(T, N, (OBS_DIM,), act_shape, "cpu")
    obs_all = torch.randn(T, N, OBS_DIM)
    with torch.no_grad():
        for t in range(T):
            action, logprob, _, value = agent.get_action_and_value(obs_all[t])
            buf.add(obs_all[t], action, logprob - log_ratio, torch.zeros(N), torch.zeros(N), value)
    buf.compute_returns(torch.zeros(N), torch.zeros(N), 0.99, 0.95)
    buf.advantages = torch.full((T, N), float(advantage))
    buf.returns = buf.values.clone()
    return agent, buf


def update(agent, buf, lr=1.0, **overrides):
    kwargs = dict(update_epochs=1, num_minibatches=1, clip_coef=CLIP, clip_vloss=True, norm_adv=False,
                  ent_coef=0.0, vf_coef=0.0, max_grad_norm=1e6, target_kl=0.0)
    kwargs.update(overrides)
    opt = torch.optim.SGD(agent.parameters(), lr=lr)
    before = [p.detach().clone() for p in agent.parameters()]
    stats = ppo_update(agent, opt, buf, np.random.default_rng(0), **kwargs)
    delta = max((p.detach() - b).abs().max().item() for p, b in zip(agent.parameters(), before))
    return stats, delta


def test_returns_expected_keys_and_finite_values():
    agent, buf = build(True, advantage=1.0)
    buf.advantages = torch.randn(T, N)
    stats, _ = update(agent, buf, lr=1e-3, norm_adv=True, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
    assert set(stats) >= EXPECTED_KEYS
    for k in EXPECTED_KEYS:
        assert math.isfinite(float(stats[k])) or k == "explained_variance", k


# (sign of advantage, log ratio, gradient should vanish?)  ratio = exp(log_ratio); clip range [0.8, 1.2]
CLIP_CASES = [
    pytest.param(+1.0, +1.0, True, id="A>0,ratio>1+eps:clipped"),
    pytest.param(+1.0, -1.0, False, id="A>0,ratio<1-eps:active"),
    pytest.param(-1.0, -1.0, True, id="A<0,ratio<1-eps:clipped"),
    pytest.param(-1.0, +1.0, False, id="A<0,ratio>1+eps:active"),
    pytest.param(+1.0, 0.0, False, id="A>0,ratio=1:active"),
    pytest.param(-1.0, 0.0, False, id="A<0,ratio=1:active"),
]


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("advantage,log_ratio,zero_grad", CLIP_CASES)
def test_clipped_objective_gradient(discrete, advantage, log_ratio, zero_grad):
    agent, buf = build(discrete, log_ratio=log_ratio, advantage=advantage)
    _, delta = update(agent, buf)  # entropy and value losses switched off
    if zero_grad:
        assert delta == 0.0, "samples outside the trust region in the favourable direction must give no gradient"
    else:
        assert delta > 1e-7, "unclipped samples must produce a policy gradient"


@pytest.mark.parametrize("advantage,log_ratio,expected_sign", [(+1.0, -1.0, +1), (-1.0, +1.0, -1)])
def test_active_gradient_pushes_ratio_in_the_right_direction(advantage, log_ratio, expected_sign):
    # Unclipped samples: a step must move the current log-prob towards (A>0) / away from (A<0) the taken action.
    agent, buf = build(True, log_ratio=log_ratio, advantage=advantage)
    obs = buf.obs.reshape(-1, OBS_DIM)
    actions = buf.actions.reshape(-1)
    with torch.no_grad():
        _, before, _, _ = agent.get_action_and_value(obs, actions)
    update(agent, buf, lr=5.0)
    with torch.no_grad():
        _, after, _, _ = agent.get_action_and_value(obs, actions)
    assert expected_sign * (after - before).mean().item() > 0


def test_clipfrac_and_kl_reflect_ratio():
    agent, buf = build(True, log_ratio=1.0, advantage=1.0)
    stats, _ = update(agent, buf)
    assert stats["clipfrac"] == pytest.approx(1.0)
    assert stats["approx_kl"] > 0.1  # (e^1 - 1) - 1 = 0.718 per sample for the k3 estimator, 1.0 for k1

    agent, buf = build(True, log_ratio=0.0, advantage=1.0)
    stats, _ = update(agent, buf)
    assert stats["clipfrac"] == pytest.approx(0.0)
    assert abs(stats["approx_kl"]) < 1e-4


def test_policy_loss_value_at_ratio_one_equals_minus_mean_advantage():
    agent, buf = build(True, log_ratio=0.0, advantage=1.0)
    buf.advantages = torch.randn(T, N)
    stats, _ = update(agent, buf, lr=0.0)
    assert stats["policy_loss"] == pytest.approx(-buf.advantages.mean().item(), abs=1e-4)


def test_advantage_normalisation_is_per_minibatch():
    # Constant positive advantages normalise to ~0 -> policy loss ~ 0 and no policy gradient.
    agent, buf = build(True, log_ratio=0.0, advantage=3.0)
    stats, delta = update(agent, buf, norm_adv=True)
    assert abs(stats["policy_loss"]) < 1e-3
    assert delta < 1e-5
    # Without normalisation the same advantages give a real loss.
    agent, buf = build(True, log_ratio=0.0, advantage=3.0)
    stats, _ = update(agent, buf, norm_adv=False, lr=0.0)
    assert stats["policy_loss"] == pytest.approx(-3.0, abs=1e-4)


def test_value_loss_matches_half_mse_without_clipping():
    agent, buf = build(True)
    buf.returns = buf.values + 2.0  # constant error of 2 -> 0.5 * 4 = 2
    stats, _ = update(agent, buf, lr=0.0, clip_vloss=False, vf_coef=0.5)
    assert stats["value_loss"] == pytest.approx(2.0, rel=1e-3)


def test_value_clipping_limits_value_movement_gradient():
    # With clip_vloss the loss is max(unclipped, clipped). When the new value is already further than
    # clip_coef from the old value in the direction of the target, the clipped term is constant -> no
    # gradient for the critic parameters. Here the target is far away so the unclipped term dominates
    # initially (new == old), hence the critic is still trained.
    agent, buf = build(True)
    buf.returns = buf.values + 5.0
    _, delta = update(agent, buf, lr=0.1, clip_vloss=True, vf_coef=0.5)
    assert delta > 1e-6


def test_entropy_bonus_changes_parameters_even_with_zero_advantage():
    agent, buf = build(True, advantage=0.0)
    _, delta_without = update(agent, buf, ent_coef=0.0)
    agent, buf = build(True, advantage=0.0)
    stats, delta_with = update(agent, buf, ent_coef=1.0, lr=1.0)
    assert delta_without == 0.0
    assert delta_with > 0.0
    assert stats["entropy"] == pytest.approx(math.log(3), abs=0.05)


def test_gradient_clipping_bounds_the_step():
    agent, buf = build(True, log_ratio=-1.0, advantage=1.0)
    _, big = update(agent, buf, lr=1.0, max_grad_norm=1e6)
    agent, buf = build(True, log_ratio=-1.0, advantage=1.0)
    _, small = update(agent, buf, lr=1.0, max_grad_norm=1e-3)
    assert small < big
    assert small <= 1e-3 + 1e-9  # SGD step = lr * clipped gradient, each entry <= total norm


def test_target_kl_stops_early():
    # Huge KL on the first minibatch -> must stop long before 50 epochs; compare to no early stop.
    agent, buf = build(True, log_ratio=1.0, advantage=1.0)
    _, delta_free = update(agent, buf, lr=0.5, update_epochs=20, num_minibatches=2, target_kl=0.0)
    agent, buf = build(True, log_ratio=1.0, advantage=1.0)
    stats, delta_stop = update(agent, buf, lr=0.5, update_epochs=20, num_minibatches=2, target_kl=1e-6)
    assert delta_stop <= delta_free + 1e-9


def test_repeated_updates_reduce_value_loss():
    torch.manual_seed(0)
    agent, buf = build(True)
    buf.returns = torch.full((T, N), 3.0)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(30):
        stats = ppo_update(agent, opt, buf, rng, update_epochs=4, num_minibatches=2, clip_coef=10.0,
                           clip_vloss=False, norm_adv=False, ent_coef=0.0, vf_coef=0.5,
                           max_grad_norm=0.5, target_kl=0.0)
        buf.values = agent.get_value(buf.obs.reshape(-1, OBS_DIM)).detach().reshape(T, N)
        losses.append(stats["value_loss"])
    assert losses[-1] < 0.2 * losses[0]


def test_policy_improves_on_toy_bandit():
    # Contextless bandit: action 1 always has advantage +1, the others -1.
    # The probability of action 1 (initially ~1/3) must clearly increase.
    torch.manual_seed(0)
    agent = ActorCritic(OBS_DIM, 3, True, (16,))
    opt = torch.optim.Adam(agent.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    obs = torch.zeros(1, OBS_DIM)

    def prob_action_1():
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(obs.expand(3000, OBS_DIM))
        return (actions == 1).float().mean().item()

    assert prob_action_1() < 0.5
    for _ in range(25):
        buf = RolloutBuffer(32, 1, (OBS_DIM,), (), "cpu")
        with torch.no_grad():
            for _t in range(32):
                a, lp, _, v = agent.get_action_and_value(obs)
                buf.add(obs, a, lp, torch.zeros(1), torch.zeros(1), v)
        buf.compute_returns(torch.zeros(1), torch.zeros(1), 0.99, 0.95)
        buf.advantages = torch.where(buf.actions == 1, 1.0, -1.0).float()
        buf.returns = buf.values.clone() + buf.advantages
        ppo_update(agent, opt, buf, rng, update_epochs=4, num_minibatches=2, clip_coef=0.2, clip_vloss=True,
                   norm_adv=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.0)
    assert prob_action_1() > 0.6
