"""Training entry point (Session B): python -m ppo.train --env-id CartPole-v1 ..."""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import fields
from typing import List, Optional, Sequence

import numpy as np
import torch

from ppo.agent import ActorCritic
from ppo.algo import ppo_update
from ppo.buffer import RolloutBuffer
from ppo.config import Config
from ppo.envs import env_spec, extract_episodes, find_obs_rms, make_envs
from ppo.logger import RunLogger


def _int_list(tokens: Sequence[str]) -> List[int]:
    """Accepts `--hidden-sizes 64 64` as well as `--hidden-sizes 64,64`."""
    out: List[int] = []
    for tok in tokens:
        out.extend(int(t) for t in str(tok).split(",") if t.strip())
    if not out:
        raise argparse.ArgumentTypeError("hidden sizes must not be empty")
    return out


def build_parser() -> argparse.ArgumentParser:
    """One flag per Config field (`--env-id`, `--total-timesteps`, ...); bools are `--x/--no-x`."""
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train PPO from scratch.")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction,
                                default=default, help=f"(default: {default})")
        elif isinstance(default, tuple):
            parser.add_argument(flag, dest=f.name, nargs="+", type=str, default=list(default),
                                help=f"ints, space or comma separated (default: {list(default)})")
        else:
            parser.add_argument(flag, dest=f.name, type=type(default), default=default,
                                help=f"(default: {default})")
    return parser


def config_from_args(argv: Optional[Sequence[str]] = None) -> Config:
    args = vars(build_parser().parse_args(argv))
    args["hidden_sizes"] = tuple(_int_list(args["hidden_sizes"]))
    return Config.from_dict(args)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: Config) -> None:
    if cfg.minibatch_size < 1:
        raise ValueError(f"num_minibatches={cfg.num_minibatches} exceeds batch size {cfg.batch_size}")
    if cfg.num_iterations < 1:
        raise ValueError(f"total_timesteps={cfg.total_timesteps} is smaller than one batch ({cfg.batch_size})")

    torch.set_num_threads(1)
    _seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    envs = make_envs(cfg.env_id, cfg.num_envs, cfg.seed, cfg.normalize_obs,
                     cfg.normalize_reward, cfg.gamma)
    logger = RunLogger(cfg)
    try:
        spec = env_spec(envs)
        discrete = bool(spec["discrete"])
        obs_dim, act_dim = int(spec["obs_dim"]), int(spec["act_dim"])

        agent = ActorCritic(obs_dim, act_dim, discrete, cfg.hidden_sizes).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
        act_shape = () if discrete else (act_dim,)

        def to_obs(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)

        global_step = 0
        start_time = time.time()
        next_obs_np, _ = envs.reset(seed=cfg.seed)
        next_obs = to_obs(next_obs_np)
        next_done = torch.zeros(cfg.num_envs, device=device)

        for iteration in range(1, cfg.num_iterations + 1):
            lr_now = cfg.lr
            if cfg.anneal_lr:
                lr_now = (1.0 - (iteration - 1.0) / cfg.num_iterations) * cfg.lr
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            # ---- rollout (fresh buffer per iteration: only the frozen contract is used) ----
            buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, (obs_dim,), act_shape, cfg.device)
            for _ in range(cfg.num_steps):
                global_step += cfg.num_envs
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                next_obs_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                # Buffer convention (CleanRL): the done flag stored with obs[t] is the one that
                # was in effect when obs[t] was observed, i.e. the outcome of the step *before*
                # it; the outcome of the step taken here is stored at t + 1 (or passed as
                # last_done to compute_returns after the final step).
                buffer.add(next_obs, action, logprob,
                           torch.as_tensor(np.asarray(reward, dtype=np.float32), device=device),
                           next_done, value)
                done = np.logical_or(terminated, truncated)
                next_obs = to_obs(next_obs_np)
                next_done = torch.as_tensor(done.astype(np.float32), device=device)

                for ep_return, ep_length in extract_episodes(info):
                    logger.log_episode(global_step, ep_return, ep_length)

            # ---- bootstrap, advantages, update ----
            with torch.no_grad():
                last_value = agent.get_value(next_obs)
            buffer.compute_returns(last_value, next_done, cfg.gamma, cfg.gae_lambda, cfg.use_gae)
            stats = ppo_update(
                agent, optimizer, buffer, rng,
                update_epochs=cfg.update_epochs, num_minibatches=cfg.num_minibatches,
                clip_coef=cfg.clip_coef, clip_vloss=cfg.clip_vloss, norm_adv=cfg.norm_adv,
                ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
                target_kl=cfg.target_kl)

            sps = global_step / max(time.time() - start_time, 1e-8)
            logger.log_metrics(global_step, {**stats, "learning_rate": lr_now, "sps": sps})

        # ---- save final model (+ observation statistics for later evaluation) ----
        ckpt = {"model": agent.state_dict(), "config": cfg.to_dict(),
                "obs_dim": obs_dim, "act_dim": act_dim, "discrete": discrete}
        obs_rms = find_obs_rms(envs)
        if obs_rms is not None:
            ckpt["obs_rms"] = {"mean": np.asarray(obs_rms.mean), "var": np.asarray(obs_rms.var),
                               "count": float(obs_rms.count)}
        torch.save(ckpt, logger.run_dir / "model.pt")
    finally:
        logger.close()
        envs.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    train(config_from_args(argv))


if __name__ == "__main__":
    main()
