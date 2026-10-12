# PPO from scratch in PyTorch

A from-scratch implementation of Proximal Policy Optimization (PPO) in PyTorch. The only libraries
used are `torch` and `gymnasium` (environments); there is no RL library. The code is validated with
unit tests on the numerically sensitive parts, a CartPole sanity check, and a multi-seed study on
discrete and continuous-control environments, including an ablation of the implementation details
that are usually left out of PPO descriptions.

> **Status.** The implementation, the tests and the analysis pipeline are done. The full-length
> runs are not part of this commit: every number in the results section below is a placeholder
> (`TBD`) that is filled in from `analysis/output/` after the sweep has run. No result is quoted
> from memory or from another project.

## What is implemented

| Component | File | Notes |
|---|---|---|
| Actor-critic | `ppo/agent.py` | separate tanh MLPs, orthogonal init (gain sqrt(2) hidden, 0.01 policy head, 1.0 value head), `Categorical` policy for discrete actions, diagonal `Normal` with a state-independent log-std for continuous actions |
| Rollout buffer | `ppo/buffer.py` | GAE(lambda) and a Monte-Carlo-return variant, `terminated`/`truncated` handled through the done flags, seeded shuffled minibatches |
| PPO update | `ppo/algo.py` | clipped surrogate, optional clipped value loss, per-minibatch advantage normalisation, entropy bonus, gradient-norm clipping, optional `target_kl` early stop; reports policy/value loss, entropy, approximate KL, clip fraction, explained variance |
| Environments | `ppo/envs.py` | seeded `SyncVectorEnv`, raw episode statistics, action clipping, running observation normalisation and reward scaling (both clipped to [-10, 10]) |
| Training loop | `ppo/train.py` | rollout, bootstrap, update, learning-rate annealing, seeding, CSV logging; every `Config` field is a command-line flag |
| Logging | `ppo/logger.py` | `progress.csv` (one row per finished episode) and `metrics.csv` (one row per update) per run |
| Experiments | `experiments/` | default YAML config per environment, ablation definitions, multi-process sweep runner |
| Analysis | `analysis/` | aggregation on a common step grid, bootstrap confidence intervals, learning-curve plots, ablation table |

The details that decide whether PPO works are the ones listed in *The 37 Implementation Details of
Proximal Policy Optimization* (Huang et al., 2022) and *Implementation Matters in Deep Policy
Gradients* (Engstrom et al., 2020); the ablations below test a subset of them.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls in `gymnasium[mujoco,box2d]`, which is needed for LunarLander, HalfCheetah and
Hopper. CartPole and Pendulum need neither extra. If the MuJoCo wheel does not install locally, the
same commands work on a Colab CPU runtime.

## Quick start

```bash
# train one agent
python -m ppo.train --env-id CartPole-v1 --seed 0 --run-name demo --total-timesteps 100000

# short end-to-end check: one CartPole run and two short HalfCheetah runs
bash scripts/smoke_test.sh

# unit tests
pytest -q
```

Logs are written to `results/<env_id>/<run_name>/seed_<n>/` (`config.json`, `progress.csv`,
`metrics.csv`). Run `python -m ppo.train --help` for all hyperparameters.

## Reproduce the study

```bash
python experiments/run_sweep.py --all --dry-run     # print the 90 commands
python experiments/run_sweep.py --all --jobs 4      # run them (long; finished runs are skipped on restart)
python analysis/plot_curves.py                      # figures + tables in analysis/output/
```

* **Main runs:** the default config of CartPole-v1, LunarLander-v3, HalfCheetah-v5 and Hopper-v5,
  seeds 0 to 4 (`experiments/configs/*.yaml`).
* **Ablations (HalfCheetah-v5 and Hopper-v5, seeds 0 to 4):** no GAE (Monte-Carlo returns), no advantage
  normalisation, no observation normalisation, no value-loss clipping, no learning-rate annealing, and
  clip coefficient 0.1 and 0.3 (0.2 is the baseline). Each ablation changes exactly one setting of the
  baseline config (`experiments/ablations.py`).
* To reduce the compute budget, use fewer seeds (`--seeds 0 1 2`) or fewer steps (`--max-steps`), and
  state that next to the results. Runs with a shortened budget are not comparable to the numbers below.

### How the analysis computes its numbers

* **Learning curve of one seed:** at step `s`, the mean return of the last 20 episodes that finished at or
  before `s` (the window is `--window`), linearly interpolated onto a grid of 200 steps. Grid points
  before a seed's first finished episode or after its last one are left empty rather than extrapolated.
* **Curve over seeds:** the mean over seeds with a 95% percentile-bootstrap confidence interval
  (10,000 resamples of the seeds), drawn only where every seed has data.
* **Final return:** per seed, the mean return of the last 10 training episodes; then the mean over seeds
  with a 95% bootstrap interval. The table also gives the difference to the baseline with its own
  bootstrap interval; a difference whose interval excludes zero is starred.
* Runs that did not reach 95% of their `total_timesteps` are excluded unless `--include-unfinished` is passed.

Outputs: `analysis/output/figures/<env>_baseline.png`, `<env>_ablations.png`, and `summary.md`,
`summary.csv`, `final_returns_per_seed.csv`, `curves.csv`.

## Results

All returns are training-episode returns of the stochastic policy in the unnormalised environment, so
they include exploration noise.

### Sanity and discrete control

| Environment | Steps | Solved threshold | Final return (mean, 95% CI over seeds) | Seeds |
|---|---|---|---|---|
| CartPole-v1 | TBD | 475 | TBD | TBD |
| LunarLander-v3 | TBD | 200 | TBD | TBD |

### Continuous control

The reference column is to be filled with numbers read from the published CleanRL and Stable-Baselines3
results for PPO at 1M steps, with the exact source and version cited; it is left empty until they have been
checked against the sources.

| Environment | Steps | This implementation (mean, 95% CI) | Published PPO reference | Seeds |
|---|---|---|---|---|
| HalfCheetah-v5 | TBD | TBD | TBD (source) | TBD |
| Hopper-v5 | TBD | TBD | TBD (source) | TBD |

Learning curves: `analysis/output/figures/<env>_baseline.png` (placeholder, generated by the sweep).

### Ablations

Final return of every ablation next to the baseline, same seeds, same budget (`analysis/output/summary.md`
is the source of this table):

| Config | HalfCheetah-v5 final return | Delta vs baseline | Hopper-v5 final return | Delta vs baseline |
|---|---|---|---|---|
| baseline | TBD | - | TBD | - |
| no GAE | TBD | TBD | TBD | TBD |
| no advantage normalisation | TBD | TBD | TBD | TBD |
| no observation normalisation | TBD | TBD | TBD | TBD |
| no value-loss clipping | TBD | TBD | TBD | TBD |
| no LR annealing | TBD | TBD | TBD | TBD |
| clip 0.1 | TBD | TBD | TBD | TBD |
| clip 0.3 | TBD | TBD | TBD | TBD |

Per-ablation learning curves against the baseline: `analysis/output/figures/<env>_ablations.png`.

### Discussion

To be written once the runs are complete. It will cover which implementation details matter on which
environment, where the baseline agrees or disagrees with the published reference range, and which
differences are larger than the seed-to-seed spread.

## Tests

`pytest -q` runs the suite in `tests/` (it does not need MuJoCo or Box2D):

| File | Covers |
|---|---|
| `test_gae.py` | GAE and Monte-Carlo returns against hand-computed values, including a done flag in the middle of a rollout and a done at the bootstrap step; independence of environments; minibatch coverage and seeding |
| `test_algo.py` | clipped objective: zero gradient when the ratio is outside the clip range in the favourable direction, non-zero gradient otherwise (all four sign/ratio cases, discrete and continuous); clip fraction and KL; value loss; entropy bonus; gradient clipping; `target_kl`; a toy bandit and value-regression convergence check |
| `test_agent.py` | output shapes, near-uniform initial policy, unit-variance initial Gaussian, state-independent entropy, orthogonal initialisation with the expected gains, gradient flow |
| `test_envs.py` | environment specs, seeding determinism, observation normalisation against a running mean/std computed independently, reward normalisation without changing the reported episode returns, action clipping |
| `test_logger.py` | directory layout, CSV column names, row order, `config.json` round trip, `Config` helpers |
| `test_aggregate.py`, `test_plot_curves.py` | the analysis pipeline on small synthetic logs |

## Limitations

* **Five seeds are few.** A bootstrap interval over five seeds is itself noisy and tends to be too narrow;
  treat the starred differences in the ablation table as suggestive, not as significance tests, and
  prefer large effects over small ones. More seeds are cheap to add: `--seeds 5 6 7 8 9`.
* **Hyperparameters were not tuned per ablation.** Each ablation changes one setting of the baseline
  config only; an ablated component may need a different learning rate or clip range to work well.
* **Training returns, not evaluation returns.** Final returns are averaged over the last 10 training
  episodes of the stochastic policy. A separate deterministic evaluation could score higher.
* **Reference numbers** come from other codebases, other library versions and other seeds; a gap to them is
  a hint, not a proof of a bug or of a better implementation.
* **CPU only.** Runs use the CPU with a single torch thread each. Wall-clock times depend on the machine
  and are not reported here.
* **Scope.** Feed-forward policies, a single synchronous vector environment per run, no recurrent
  policies, no image observations, no multi-GPU or asynchronous collection.

## Repository layout

```
ppo/           config.py agent.py buffer.py algo.py envs.py logger.py train.py
experiments/   configs/*.yaml  ablations.py  run_sweep.py
analysis/      aggregate.py  plot_curves.py
tests/         test_gae.py test_algo.py test_agent.py test_envs.py test_logger.py
               test_aggregate.py test_plot_curves.py
scripts/       smoke_test.sh
results/       <env>/<run_name>/seed_<n>/{config.json,progress.csv,metrics.csv}   (generated, not tracked)
```

## References

* Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
* Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015)
* Huang et al., *The 37 Implementation Details of Proximal Policy Optimization*, ICLR Blog Track (2022)
* Engstrom et al., *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO* (2020)
* CleanRL `ppo.py` and `ppo_continuous_action.py`, used as a point of comparison for hyperparameters and results
