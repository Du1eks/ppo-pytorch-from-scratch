"""Tests for ActorCritic: output shapes, initialisation and distribution behaviour."""
import math

import pytest
import torch

from ppo.agent import ActorCritic

OBS_DIM = 5
BATCH = 7


def make_agent(discrete, act_dim=3, hidden=(32, 32)):
    torch.manual_seed(0)
    return ActorCritic(OBS_DIM, act_dim, discrete, hidden)


@pytest.mark.parametrize("discrete,act_dim", [(True, 4), (False, 3)])
def test_output_shapes(discrete, act_dim):
    agent = make_agent(discrete, act_dim)
    obs = torch.randn(BATCH, OBS_DIM)
    value = agent.get_value(obs)
    assert value.shape == (BATCH,)
    action, logprob, entropy, value2 = agent.get_action_and_value(obs)
    assert logprob.shape == (BATCH,)
    assert entropy.shape == (BATCH,)
    assert value2.shape == (BATCH,)
    if discrete:
        assert action.shape == (BATCH,)
        assert not action.is_floating_point()
        assert int(action.min()) >= 0 and int(action.max()) < act_dim
    else:
        assert action.shape == (BATCH, act_dim)
        assert action.is_floating_point()


@pytest.mark.parametrize("discrete", [True, False])
def test_value_consistent_between_methods(discrete):
    agent = make_agent(discrete)
    obs = torch.randn(BATCH, OBS_DIM)
    _, _, _, value = agent.get_action_and_value(obs)
    torch.testing.assert_close(value, agent.get_value(obs))


@pytest.mark.parametrize("discrete", [True, False])
def test_given_action_is_returned_and_logprob_is_deterministic(discrete):
    agent = make_agent(discrete)
    obs = torch.randn(BATCH, OBS_DIM)
    action, logprob, _, _ = agent.get_action_and_value(obs)
    action2, logprob2, _, _ = agent.get_action_and_value(obs, action)
    torch.testing.assert_close(action2, action)
    torch.testing.assert_close(logprob2, logprob)
    assert torch.all(logprob <= 0) if discrete else torch.isfinite(logprob).all()


def test_discrete_initial_policy_is_near_uniform():
    # policy head gain 0.01 -> logits ~ 0 -> entropy ~ log(n_actions)
    agent = make_agent(True, act_dim=4)
    obs = torch.randn(64, OBS_DIM)
    _, _, entropy, _ = agent.get_action_and_value(obs)
    assert entropy.mean().item() == pytest.approx(math.log(4), abs=0.02)
    # and it must be able to represent a non-uniform policy
    actions = torch.zeros(64, dtype=torch.long)
    _, logprob, _, _ = agent.get_action_and_value(obs, actions)
    assert logprob.mean().item() == pytest.approx(-math.log(4), abs=0.05)


def test_continuous_initial_policy_has_unit_std_and_state_independent_entropy():
    act_dim = 3
    agent = make_agent(False, act_dim)
    obs = torch.randn(64, OBS_DIM) * 3
    _, _, entropy, _ = agent.get_action_and_value(obs)
    # standard normal: 0.5 * log(2 pi e) per dimension, summed over action dimensions
    expected = act_dim * 0.5 * math.log(2 * math.pi * math.e)
    assert entropy.mean().item() == pytest.approx(expected, abs=0.05)
    # state-independent log-std: entropy identical for every observation
    torch.testing.assert_close(entropy, entropy[0].expand_as(entropy))


def test_continuous_samples_have_unit_spread():
    agent = make_agent(False, act_dim=2)
    obs = torch.zeros(4000, OBS_DIM)
    action, _, _, _ = agent.get_action_and_value(obs)
    assert action.std(dim=0).mean().item() == pytest.approx(1.0, abs=0.1)
    assert action.mean(dim=0).abs().max().item() < 0.2  # head gain 0.01 -> mean ~ 0


def test_continuous_logprob_matches_closed_form_standard_normal():
    agent = make_agent(False, act_dim=2)
    obs = torch.randn(BATCH, OBS_DIM)
    action = torch.randn(BATCH, 2)
    _, logprob, _, _ = agent.get_action_and_value(obs, action)
    # mean ~ 0 and std = 1 at init, so logprob ~ standard-normal logpdf (loose: head gain is 0.01)
    expected = (-0.5 * action**2 - 0.5 * math.log(2 * math.pi)).sum(-1)
    torch.testing.assert_close(logprob, expected, atol=0.1, rtol=0)


@pytest.mark.parametrize("discrete", [True, False])
def test_gradients_flow_to_all_parameters(discrete):
    agent = make_agent(discrete)
    obs = torch.randn(BATCH, OBS_DIM)
    _, logprob, entropy, value = agent.get_action_and_value(obs)
    (logprob.sum() + entropy.sum() + value.sum()).backward()
    for name, p in agent.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    # at least the vast majority of parameters must receive a nonzero gradient
    nonzero = sum(1 for p in agent.parameters() if p.grad.abs().sum() > 0)
    assert nonzero >= len(list(agent.parameters())) - 1


def test_single_observation_batch():
    agent = make_agent(True)
    obs = torch.randn(1, OBS_DIM)
    assert agent.get_value(obs).shape == (1,)
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    assert action.shape == (1,) and logprob.shape == (1,)
    assert entropy.shape == (1,) and value.shape == (1,)


def test_weights_are_orthogonal_with_expected_gain():
    # All 2-D weights must be (semi-)orthogonal: W W^T or W^T W is a scaled identity.
    agent = make_agent(True, act_dim=4, hidden=(64, 64))
    gains = []
    for p in agent.parameters():
        if p.ndim != 2:
            continue
        w = p.detach()
        gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
        gain_sq = gram.diagonal().mean().item()
        gains.append(math.sqrt(gain_sq))
        torch.testing.assert_close(gram, gain_sq * torch.eye(gram.shape[0]), atol=1e-4, rtol=1e-3)
    # hidden layers sqrt(2), policy head 0.01, value head 1.0
    assert any(g == pytest.approx(math.sqrt(2), rel=1e-2) for g in gains)
    assert any(g == pytest.approx(0.01, rel=1e-2) for g in gains)
    assert any(g == pytest.approx(1.0, rel=1e-2) for g in gains)


def test_biases_initialised_to_zero():
    agent = make_agent(True)
    for name, p in agent.named_parameters():
        if p.ndim == 1 and "log_std" not in name:
            assert torch.count_nonzero(p) == 0, name
