"""Tests for make_envs / env_spec: spaces, seeding determinism, normalisation, episode statistics."""
import numpy as np
import pytest

from ppo.envs import env_spec, make_envs

GAMMA = 0.99


def episode_stats(info):
    """Return (returns, lengths) of episodes that finished in this step's info dict.

    Handles both info layouts used by gymnasium vector envs (``info['episode']`` directly, or inside
    ``info['final_info']`` with SAME_STEP autoreset)."""
    for src in (info.get("final_info"), info):
        if isinstance(src, dict) and "episode" in src:
            ep = src["episode"]
            mask = src.get("_episode", np.ones_like(np.asarray(ep["r"]), dtype=bool))
            idx = np.flatnonzero(mask)
            return [float(ep["r"][i]) for i in idx], [int(ep["l"][i]) for i in idx]
    return [], []


def rollout(env_id, num_envs, seed, steps, normalize_obs=False, normalize_reward=False, action_fn=None, reset_seed=True):
    envs = make_envs(env_id, num_envs, seed, normalize_obs, normalize_reward, GAMMA)
    obs, _ = envs.reset(seed=seed) if reset_seed else envs.reset()
    rng = np.random.default_rng(123)
    observations, rewards, returns, lengths = [np.asarray(obs)], [], [], []
    for _ in range(steps):
        if action_fn is None:
            action = envs.action_space.sample()
        else:
            action = action_fn(rng, envs)
        obs, reward, terminated, truncated, info = envs.step(action)
        observations.append(np.asarray(obs))
        rewards.append(np.asarray(reward))
        r, l = episode_stats(info)
        returns += r
        lengths += l
    envs.close()
    return np.stack(observations), np.stack(rewards), returns, lengths


def test_env_spec_discrete():
    envs = make_envs("CartPole-v1", 2, 0, False, False, GAMMA)
    assert env_spec(envs) == {"obs_dim": 4, "act_dim": 2, "discrete": True}
    envs.close()


def test_env_spec_continuous():
    envs = make_envs("Pendulum-v1", 2, 0, False, False, GAMMA)
    assert env_spec(envs) == {"obs_dim": 3, "act_dim": 1, "discrete": False}
    envs.close()


def test_env_spec_mujoco_if_available():
    gym = pytest.importorskip("gymnasium")
    try:
        gym.make("HalfCheetah-v5").close()
    except Exception as exc:  # pragma: no cover - depends on local mujoco install
        pytest.skip(f"MuJoCo unavailable: {exc}")
    envs = make_envs("HalfCheetah-v5", 2, 0, True, True, GAMMA)
    assert env_spec(envs) == {"obs_dim": 17, "act_dim": 6, "discrete": False}
    envs.close()


def test_vector_shapes():
    envs = make_envs("CartPole-v1", 3, 0, False, False, GAMMA)
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (3, 4)
    obs, reward, terminated, truncated, _ = envs.step(envs.action_space.sample())
    assert obs.shape == (3, 4)
    assert np.asarray(reward).shape == (3,)
    assert np.asarray(terminated).shape == (3,) and np.asarray(truncated).shape == (3,)
    envs.close()


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_same_seed_gives_identical_trajectories(env_id):
    a = rollout(env_id, 2, seed=5, steps=60)
    b = rollout(env_id, 2, seed=5, steps=60)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_different_seeds_give_different_trajectories(env_id):
    a = rollout(env_id, 2, seed=1, steps=20)
    b = rollout(env_id, 2, seed=2, steps=20)
    assert not np.array_equal(a[0], b[0])


def test_sub_environments_have_different_seeds():
    envs = make_envs("CartPole-v1", 4, 0, False, False, GAMMA)
    obs, _ = envs.reset(seed=0)
    assert len({tuple(np.round(o, 6)) for o in obs}) == 4
    envs.close()


def test_action_space_sampling_is_seeded():
    a = make_envs("CartPole-v1", 3, 11, False, False, GAMMA)
    b = make_envs("CartPole-v1", 3, 11, False, False, GAMMA)
    np.testing.assert_array_equal([a.action_space.sample() for _ in range(5)],
                                  [b.action_space.sample() for _ in range(5)])
    a.close()
    b.close()


def test_normalised_runs_are_deterministic_too():
    a = rollout("Pendulum-v1", 2, seed=3, steps=50, normalize_obs=True, normalize_reward=True)
    b = rollout("Pendulum-v1", 2, seed=3, steps=50, normalize_obs=True, normalize_reward=True)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_observation_normalisation_uses_running_mean_and_std_and_clips():
    # Zero torque -> identical raw dynamics in both runs, so the normalised stream must equal the raw
    # stream standardised with the running statistics of everything seen so far.
    zero = lambda rng, envs: np.zeros(envs.action_space.shape, dtype=np.float32)
    raw, *_ = rollout("Pendulum-v1", 2, seed=0, steps=300, action_fn=zero)
    norm, *_ = rollout("Pendulum-v1", 2, seed=0, steps=300, normalize_obs=True, action_fn=zero)
    assert norm.shape == raw.shape
    assert np.isfinite(norm).all()
    assert np.abs(norm).max() <= 10.0 + 1e-5
    for k in (50, 150, 300):
        seen = raw[: k + 1].reshape(-1, raw.shape[-1]).astype(np.float64)
        expected = (raw[k] - seen.mean(axis=0)) / np.sqrt(seen.var(axis=0) + 1e-8)
        np.testing.assert_allclose(norm[k], np.clip(expected, -10, 10), atol=2e-2)
    assert not np.allclose(norm, raw)


def test_reward_normalisation_changes_rewards_but_not_reported_returns():
    # Pendulum episodes last exactly 200 steps; zero torque keeps both runs on the same dynamics.
    zero = lambda rng, envs: np.zeros(envs.action_space.shape, dtype=np.float32)
    _, rew_plain, ret_plain, len_plain = rollout("Pendulum-v1", 1, seed=0, steps=400, action_fn=zero)
    _, rew_norm, ret_norm, len_norm = rollout("Pendulum-v1", 1, seed=0, steps=400, normalize_reward=True,
                                              action_fn=zero)
    assert len(ret_plain) == len(ret_norm) >= 2
    assert len_plain == len_norm == [200] * len(len_plain)
    # learning curves must show the real environment return, not the normalised one
    np.testing.assert_allclose(ret_norm, ret_plain, rtol=1e-5)
    assert not np.allclose(rew_plain, rew_norm)
    assert np.abs(rew_norm).max() <= 10.0 + 1e-6
    assert np.isfinite(rew_norm).all()


def test_episode_statistics_are_reported_for_cartpole():
    _, rewards, returns, lengths = rollout("CartPole-v1", 2, seed=0, steps=300)
    assert len(returns) >= 2
    # CartPole gives +1 per step so return == length
    np.testing.assert_allclose(returns, lengths)
    # every reward in a CartPole step is 1
    assert np.all(rewards == 1.0)


def test_continuous_actions_are_clipped_to_the_action_bounds():
    # Pendulum torque bounds are [-2, 2]: an out-of-range action must behave like the bound itself.
    outs = []
    for torque in (100.0, 2.0):
        envs = make_envs("Pendulum-v1", 1, 0, False, False, GAMMA)
        envs.reset(seed=0)
        obs, reward, *_ = envs.step(np.array([[torque]], dtype=np.float32))
        outs.append((np.asarray(obs), np.asarray(reward)))
        envs.close()
    np.testing.assert_allclose(outs[0][0], outs[1][0])
    np.testing.assert_allclose(outs[0][1], outs[1][1])


def test_episode_ends_are_flagged_and_obs_stays_valid():
    envs = make_envs("CartPole-v1", 2, 0, False, False, GAMMA)
    envs.reset(seed=0)
    ended = 0
    for _ in range(400):
        obs, _, terminated, truncated, _ = envs.step(envs.action_space.sample())
        ended += int(np.sum(np.logical_or(terminated, truncated)))
        assert np.isfinite(obs).all()
    assert ended > 0  # random CartPole policy always fails well within 400 steps
    envs.close()
