"""Environment construction.

Vector env layout (innermost to outermost):

    per-env:  gym.make -> RecordEpisodeStatistics (raw, unnormalised returns)
              -> FlattenObservation (only for multi-dimensional Box observations)
              -> ClipAction (Box action spaces only)
    vector:   SyncVectorEnv (SAME_STEP autoreset)
              -> NormalizeClipObs (running mean/std, clip [-10, 10])   (optional)
              -> NormalizeReward      -> clip to [-10, 10]   (optional)

SAME_STEP autoreset means that when an episode ends, the observation returned by
``step`` is already the first observation of the next episode (the CleanRL / old
gym convention), and the terminal observation is in ``info["final_obs"]``.
Episode statistics of finished episodes are therefore read from
``info["final_info"]["episode"]`` (see ``extract_episodes``).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorWrapper
from gymnasium.wrappers import vector as vector_wrappers
from gymnasium.wrappers.utils import RunningMeanStd

NORM_CLIP = 10.0


class NormalizeClipObs(VectorWrapper):
    """Running mean/std observation normalisation followed by clipping to [-clip, clip].

    gymnasium's own vector NormalizeObservation (and VectorObservationWrapper) refuse SAME_STEP
    autoreset even though normalisation does not depend on the autoreset mode, so this small equivalent is used instead.
    Every observation returned by reset/step (including the first obs of a new episode)
    updates the statistics, as in CleanRL.
    """

    def __init__(self, env: gym.vector.VectorEnv, epsilon: float = 1e-8, clip: float = NORM_CLIP):
        super().__init__(env)
        shape = self.single_observation_space.shape
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.obs_rms = RunningMeanStd(shape=shape, dtype=np.float64)
        self.epsilon = epsilon
        self.clip = clip
        self.update_running_mean = True

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._normalize(obs), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self._normalize(obs), reward, terminated, truncated, info

    def _normalize(self, observations: np.ndarray) -> np.ndarray:
        if self.update_running_mean:
            self.obs_rms.update(observations)
        out = (observations - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)


def _make_thunk(env_id: str):
    def thunk() -> gym.Env:
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) > 1:
            env = gym.wrappers.FlattenObservation(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def make_envs(env_id: str, num_envs: int, seed: int, normalize_obs: bool,
              normalize_reward: bool, gamma: float) -> gym.vector.VectorEnv:
    """Build a seeded synchronous vector env.

    Seeding is deterministic: env ``i`` is seeded with ``seed + i`` and the
    (single and batched) action spaces are seeded with ``seed``. The seeding
    reset is done on the raw vector env, before the normalisation wrappers are
    attached, so it does not pollute the running statistics. Calling
    ``envs.reset(seed=seed)`` afterwards restarts the exact same trajectories.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    envs = gym.vector.SyncVectorEnv(
        [_make_thunk(env_id) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs.reset(seed=seed)
    envs.action_space.seed(seed)
    envs.single_action_space.seed(seed)
    envs.single_observation_space.seed(seed)

    if normalize_obs:
        envs = NormalizeClipObs(envs)
    if normalize_reward:
        envs = vector_wrappers.NormalizeReward(envs, gamma=gamma)
        envs = vector_wrappers.TransformReward(
            envs, lambda r: np.clip(r, -NORM_CLIP, NORM_CLIP))
    return envs


def env_spec(envs: gym.vector.VectorEnv) -> Dict[str, object]:
    """Returns {'obs_dim': int, 'act_dim': int, 'discrete': bool}."""
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    if not isinstance(obs_space, gym.spaces.Box):
        raise NotImplementedError(f"only Box observation spaces are supported, got {obs_space}")
    obs_dim = int(np.prod(obs_space.shape))
    if isinstance(act_space, gym.spaces.Discrete):
        return {"obs_dim": obs_dim, "act_dim": int(act_space.n), "discrete": True}
    if isinstance(act_space, gym.spaces.Box):
        return {"obs_dim": obs_dim, "act_dim": int(np.prod(act_space.shape)), "discrete": False}
    raise NotImplementedError(f"unsupported action space {act_space}")


def extract_episodes(info: dict) -> List[Tuple[float, int]]:
    """Finished episodes (return, length) contained in one ``envs.step`` info dict.

    Returns raw (unnormalised) values because RecordEpisodeStatistics sits below the
    normalisation wrappers.
    """
    out: List[Tuple[float, int]] = []
    sources = []
    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        sources.append(final_info)
    sources.append(info)
    for src in sources:
        ep = src.get("episode")
        if ep is None:
            continue
        mask = src.get("_episode")
        if mask is None:
            mask = np.ones_like(np.asarray(ep["r"]), dtype=bool)
        for i in np.flatnonzero(mask):
            out.append((float(ep["r"][i]), int(ep["l"][i])))
        break
    return out


def find_obs_rms(envs: gym.vector.VectorEnv):
    """Return the observation running-mean-std object of a wrapped vector env, or None."""
    env = envs
    while env is not None:
        if hasattr(env, "obs_rms"):
            return env.obs_rms
        env = getattr(env, "env", None)
    return None
