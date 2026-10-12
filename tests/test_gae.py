"""Tests for RolloutBuffer: storage, GAE / Monte-Carlo returns and minibatching.

Convention under test (CleanRL): ``dones[t]`` is the done flag that came out of the
environment step *before* ``obs[t]`` (i.e. ``obs[t]`` is the first observation of a new
episode when it is 1). The bootstrap for step ``t`` is therefore masked with
``1 - dones[t + 1]``, and for the last step with ``1 - last_done``.
"""
import numpy as np
import pytest
import torch

from ppo.buffer import RolloutBuffer

GAMMA = 0.9
LAM = 0.8


def make_buffer(rewards, values, dones, num_envs=1, obs_dim=2, act_shape=()):
    """Fill a buffer from [T, N] arrays; obs[t, n, 0] encodes a unique id."""
    rewards = np.asarray(rewards, dtype=np.float32).reshape(len(rewards), num_envs)
    values = np.asarray(values, dtype=np.float32).reshape(len(values), num_envs)
    dones = np.asarray(dones, dtype=np.float32).reshape(len(dones), num_envs)
    T = rewards.shape[0]
    buf = RolloutBuffer(T, num_envs, (obs_dim,), act_shape, "cpu")
    for t in range(T):
        obs = torch.zeros(num_envs, obs_dim)
        obs[:, 0] = torch.arange(num_envs) + 100 * t
        action = torch.zeros((num_envs,) + tuple(act_shape))
        buf.add(obs, action, torch.zeros(num_envs), torch.from_numpy(rewards[t]),
                torch.from_numpy(dones[t]), torch.from_numpy(values[t]))
    return buf, rewards, values, dones


def reference_gae(rewards, values, dones, last_value, last_done, gamma, lam):
    """Straightforward numpy reference of the CleanRL recursion."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float64)
    last = np.zeros(N, dtype=np.float64)
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal, next_value = 1.0 - last_done, last_value
        else:
            nonterminal, next_value = 1.0 - dones[t + 1], values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def as_np(x):
    return x.detach().cpu().numpy().reshape(x.shape)


def test_gae_hand_computed_with_done_mid_rollout():
    # obs[2] starts a new episode: the transition out of step 1 was terminal.
    buf, *_ = make_buffer(rewards=[1.0, 2.0, 3.0], values=[0.5, 1.0, 1.5], dones=[0.0, 0.0, 1.0])
    buf.compute_returns(torch.tensor([2.0]), torch.tensor([0.0]), GAMMA, LAM, use_gae=True)

    # t=2: delta = 3 + 0.9*2.0 - 1.5 = 3.3                        -> A2 = 3.3
    # t=1: next is a new episode, no bootstrap: delta = 2 - 1.0   -> A1 = 1.0
    # t=0: delta = 1 + 0.9*1.0 - 0.5 = 1.4, A0 = 1.4 + 0.9*0.8*1.0 = 2.12
    expected_adv = np.array([[2.12], [1.0], [3.3]])
    expected_ret = expected_adv + np.array([[0.5], [1.0], [1.5]])
    np.testing.assert_allclose(as_np(buf.advantages), expected_adv, atol=1e-5)
    np.testing.assert_allclose(as_np(buf.returns), expected_ret, atol=1e-5)


def test_gae_last_done_masks_final_bootstrap():
    buf, *_ = make_buffer(rewards=[1.0, 2.0, 3.0], values=[0.5, 1.0, 1.5], dones=[0.0, 0.0, 1.0])
    buf.compute_returns(torch.tensor([2.0]), torch.tensor([1.0]), GAMMA, LAM, use_gae=True)
    # t=2: no bootstrap: delta = 3 - 1.5 = 1.5 -> A2 = 1.5; A1 = 1.0 (unchanged);
    # A0 = 1.4 + 0.72 * 1.0 = 2.12 (unchanged because the episode boundary blocks t=1 -> t=2)
    np.testing.assert_allclose(as_np(buf.advantages), [[2.12], [1.0], [1.5]], atol=1e-5)


def test_gae_without_any_done_matches_closed_form():
    # Single step: A = r + gamma * V(s') - V(s).
    buf, *_ = make_buffer(rewards=[1.0], values=[0.25], dones=[0.0])
    buf.compute_returns(torch.tensor([2.0]), torch.tensor([0.0]), GAMMA, LAM, use_gae=True)
    np.testing.assert_allclose(as_np(buf.advantages), [[1.0 + 0.9 * 2.0 - 0.25]], atol=1e-6)
    np.testing.assert_allclose(as_np(buf.returns), [[1.0 + 0.9 * 2.0]], atol=1e-6)


def test_gae_matches_reference_random_multi_env():
    rng = np.random.default_rng(0)
    T, N = 16, 3
    rewards = rng.normal(size=(T, N))
    values = rng.normal(size=(T, N))
    dones = (rng.random((T, N)) < 0.2).astype(np.float32)
    last_value = rng.normal(size=N).astype(np.float32)
    last_done = (rng.random(N) < 0.5).astype(np.float32)
    buf, r, v, d = make_buffer(rewards, values, dones, num_envs=N)
    buf.compute_returns(torch.from_numpy(last_value), torch.from_numpy(last_done), 0.99, 0.95)
    adv, ret = reference_gae(r.astype(np.float64), v.astype(np.float64), d.astype(np.float64),
                             last_value.astype(np.float64), last_done.astype(np.float64), 0.99, 0.95)
    np.testing.assert_allclose(as_np(buf.advantages), adv, atol=1e-4)
    np.testing.assert_allclose(as_np(buf.returns), ret, atol=1e-4)
    assert buf.advantages.shape == (T, N) and buf.returns.shape == (T, N)


def test_gae_lambda_one_equals_discounted_returns_minus_value():
    rng = np.random.default_rng(1)
    T, N = 10, 2
    rewards = rng.normal(size=(T, N))
    values = rng.normal(size=(T, N))
    dones = (rng.random((T, N)) < 0.15).astype(np.float32)
    buf_gae, *_ = make_buffer(rewards, values, dones, num_envs=N)
    buf_mc, *_ = make_buffer(rewards, values, dones, num_envs=N)
    last_value, last_done = torch.tensor([0.7, -0.3]), torch.tensor([0.0, 0.0])
    buf_gae.compute_returns(last_value, last_done, 0.97, 1.0, use_gae=True)
    buf_mc.compute_returns(last_value, last_done, 0.97, 0.95, use_gae=False)
    np.testing.assert_allclose(as_np(buf_gae.returns), as_np(buf_mc.returns), atol=1e-4)
    np.testing.assert_allclose(as_np(buf_gae.advantages), as_np(buf_mc.advantages), atol=1e-4)


def test_monte_carlo_hand_computed():
    buf, *_ = make_buffer(rewards=[1.0, 2.0, 3.0], values=[0.5, 1.0, 1.5], dones=[0.0, 0.0, 1.0])
    buf.compute_returns(torch.tensor([2.0]), torch.tensor([0.0]), GAMMA, LAM, use_gae=False)
    # R2 = 3 + 0.9*2.0 = 4.8 ; R1 = 2 (episode ends) ; R0 = 1 + 0.9*2 = 2.8
    np.testing.assert_allclose(as_np(buf.returns), [[2.8], [2.0], [4.8]], atol=1e-5)
    np.testing.assert_allclose(as_np(buf.advantages), [[2.3], [1.0], [3.3]], atol=1e-5)


def test_monte_carlo_ignores_gae_lambda():
    buf_a, *_ = make_buffer([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [0.0, 0.0, 0.0])
    buf_b, *_ = make_buffer([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [0.0, 0.0, 0.0])
    buf_a.compute_returns(torch.tensor([1.0]), torch.tensor([0.0]), 0.9, 0.1, use_gae=False)
    buf_b.compute_returns(torch.tensor([1.0]), torch.tensor([0.0]), 0.9, 0.99, use_gae=False)
    np.testing.assert_allclose(as_np(buf_a.returns), as_np(buf_b.returns), atol=1e-6)


def test_envs_are_independent():
    # env 0 has a done in the middle, env 1 never; each column must match a single-env run.
    rewards = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    values = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    dones = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    both, *_ = make_buffer(rewards, values, dones, num_envs=2)
    both.compute_returns(torch.tensor([0.5, 0.5]), torch.tensor([0.0, 0.0]), 0.9, 0.9)
    for n in range(2):
        single, *_ = make_buffer(rewards[:, n], values[:, n], dones[:, n], num_envs=1)
        single.compute_returns(torch.tensor([0.5]), torch.tensor([0.0]), 0.9, 0.9)
        np.testing.assert_allclose(as_np(both.advantages)[:, n], as_np(single.advantages)[:, 0], atol=1e-6)
    # the done in env 0 must change its advantages relative to env 1
    assert not np.allclose(as_np(both.advantages)[:, 0], as_np(both.advantages)[:, 1])


def test_add_stores_in_order():
    buf, rewards, values, dones = make_buffer([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(as_np(buf.rewards), rewards)
    np.testing.assert_allclose(as_np(buf.values), values)
    np.testing.assert_allclose(as_np(buf.dones), dones)
    assert buf.obs.shape == (3, 1, 2)
    assert buf.obs[2, 0, 0].item() == 200.0


@pytest.mark.parametrize("act_shape", [(), (3,)])
def test_minibatches_cover_batch_exactly_once(act_shape):
    T, N, M = 8, 4, 4
    rng = np.random.default_rng(0)
    buf, *_ = make_buffer(rng.normal(size=(T, N)), rng.normal(size=(T, N)),
                          np.zeros((T, N)), num_envs=N, act_shape=act_shape)
    buf.compute_returns(torch.zeros(N), torch.zeros(N), 0.99, 0.95)
    batches = list(buf.get_minibatches(M, np.random.default_rng(0)))
    assert len(batches) == M
    seen = []
    for mb in batches:
        assert set(mb) >= {"obs", "actions", "logprobs", "advantages", "returns", "values"}
        assert mb["obs"].shape == ((T * N) // M, 2)
        assert mb["actions"].shape == ((T * N) // M,) + tuple(act_shape)
        for key in ("logprobs", "advantages", "returns", "values"):
            assert mb[key].shape == ((T * N) // M,)
        seen.extend(mb["obs"][:, 0].tolist())
    expected = sorted((np.arange(N)[None, :] + 100 * np.arange(T)[:, None]).ravel().tolist())
    assert sorted(seen) == expected  # every sample exactly once


def test_minibatches_keep_rows_aligned():
    T, N = 6, 2
    rewards = np.arange(T * N, dtype=np.float32).reshape(T, N)
    buf, *_ = make_buffer(rewards, rewards * 0.5, np.zeros((T, N)), num_envs=N)
    buf.compute_returns(torch.zeros(N), torch.zeros(N), 0.9, 0.9)
    for mb in buf.get_minibatches(3, np.random.default_rng(3)):
        # obs[...,0] = n + 100 t ; values were set to 0.5 * reward = 0.5 * (t * N + n)
        t = torch.div(mb["obs"][:, 0], 100, rounding_mode="floor")
        n = mb["obs"][:, 0] - 100 * t
        torch.testing.assert_close(mb["values"], 0.5 * (t * N + n))


def test_minibatches_seeded_by_rng():
    T, N = 8, 2
    buf, *_ = make_buffer(np.ones((T, N)), np.zeros((T, N)), np.zeros((T, N)), num_envs=N)
    buf.compute_returns(torch.zeros(N), torch.zeros(N), 0.9, 0.9)
    first = [mb["obs"][:, 0].tolist() for mb in buf.get_minibatches(2, np.random.default_rng(7))]
    again = [mb["obs"][:, 0].tolist() for mb in buf.get_minibatches(2, np.random.default_rng(7))]
    other = [mb["obs"][:, 0].tolist() for mb in buf.get_minibatches(2, np.random.default_rng(8))]
    assert first == again
    assert first != other
