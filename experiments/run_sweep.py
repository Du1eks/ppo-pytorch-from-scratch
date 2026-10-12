"""Launch `python -m ppo.train` for env x config x seed, N processes in parallel.

Examples
  python experiments/run_sweep.py --all --dry-run              # full study, print commands only
  python experiments/run_sweep.py --all --jobs 4               # run everything (long)
  python experiments/run_sweep.py --envs CartPole-v1 --seeds 0 --max-steps 20000
  python experiments/run_sweep.py --envs Hopper-v5 --configs baseline no_gae --seeds 0 1 2

A run counts as finished (and is skipped) when progress.csv exists, the last global_step in
metrics.csv is >= 0.95 * total_timesteps, and config.json carries the total_timesteps this
sweep asks for. Use --force to rerun anyway.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.ablations import (  # noqa: E402
    ABLATION_ENVS,
    ABLATIONS,
    BASELINE,
    ENV_CONFIGS,
    build_config,
)
from ppo.config import Config  # noqa: E402

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
FINISHED_FRACTION = 0.95


@dataclass
class Job:
    env_id: str
    name: str
    seed: int
    cfg: Config

    @property
    def run_dir(self) -> Path:
        return Path(self.cfg.out_dir) / self.env_id / self.name / f"seed_{self.seed}"

    @property
    def tag(self) -> str:
        return f"{self.env_id}/{self.name}/seed_{self.seed}"


def cfg_to_args(cfg: Config) -> List[str]:
    """Translate a Config into `ppo.train` flags (every field is a flag, bools as --x/--no-x)."""
    default = Config()
    args: List[str] = []
    for key, value in cfg.to_dict().items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            args.append(flag if value else "--no-" + key.replace("_", "-"))
        elif key == "hidden_sizes":
            if tuple(value) != tuple(default.hidden_sizes):  # only when it differs from the default
                args += [flag, *map(str, value)]
        else:
            args += [flag, str(value)]
    return args


def build_command(job: Job) -> List[str]:
    return [sys.executable, "-m", "ppo.train", *cfg_to_args(job.cfg)]


def last_global_step(csv_path: Path) -> Optional[int]:
    """Last global_step of a CSV (None if missing/empty/unparseable)."""
    try:
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.reader(f) if r]
        header, data = rows[0], rows[1:]
        if not data:
            return None
        return int(float(data[-1][header.index("global_step")]))
    except (OSError, IndexError, ValueError):
        return None


def is_finished(job: Job) -> bool:
    d = job.run_dir
    if not (d / "progress.csv").exists():
        return False
    try:
        with open(d / "config.json") as f:
            total = int(json.load(f)["total_timesteps"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    if total != job.cfg.total_timesteps:  # stale run made with a different budget
        return False
    last = last_global_step(d / "metrics.csv")
    return last is not None and last >= FINISHED_FRACTION * total


def make_jobs(pairs: Sequence[tuple], seeds: Sequence[int], overrides: Dict) -> List[Job]:
    jobs = []
    for env_id, name in pairs:
        for seed in seeds:
            cfg = build_config(env_id, name, seed, **overrides)
            jobs.append(Job(env_id, name, seed, cfg))
    return jobs


def select_pairs(args: argparse.Namespace) -> List[tuple]:
    """(env, config-name) pairs in launch order: baselines first so main results appear early."""
    if args.all:
        main = [(e, BASELINE) for e in ENV_CONFIGS]
        abl = [(e, n) for n in ABLATIONS for e in ABLATION_ENVS]
        return main + abl
    envs = args.envs or list(ENV_CONFIGS)
    configs = args.configs or [BASELINE]
    if configs == ["all"]:
        configs = [BASELINE, *ABLATIONS]
    return [(e, n) for n in configs for e in envs]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true",
                   help="main runs (baseline on all envs) + every ablation on HalfCheetah and Hopper")
    p.add_argument("--envs", nargs="+", choices=sorted(ENV_CONFIGS), help="default: all four envs")
    p.add_argument("--configs", nargs="+", metavar="NAME",
                   help=f"'baseline', ablation names or 'all' (default: baseline). Ablations: {', '.join(ABLATIONS)}")
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--jobs", "-j", type=int, default=2, help="parallel processes (4 cores are shared)")
    p.add_argument("--max-steps", type=int, default=None, help="override total_timesteps for every run")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--device", default=None, help="override device (default: from config, cpu)")
    p.add_argument("--log-dir", default=str(REPO_ROOT / "runs" / "sweep_logs"),
                   help="where each run's stdout/stderr goes")
    p.add_argument("--force", action="store_true", help="rerun even if the run looks finished")
    p.add_argument("--dry-run", action="store_true", help="print commands and skip status, run nothing")
    args = p.parse_args(argv)
    bad = [c for c in (args.configs or []) if c != "all" and c != BASELINE and c not in ABLATIONS]
    if bad:
        p.error(f"unknown config(s) {bad}; known: baseline, all, {', '.join(ABLATIONS)}")
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    return args


def run_jobs(jobs: List[Job], n_parallel: int, log_dir: Path) -> List[Job]:
    """Run jobs with at most n_parallel processes. Returns the failed jobs."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: List[tuple] = []  # (job, Popen, logfile, start)
    failed: List[Job] = []
    done = 0
    try:
        while pending or running:
            while pending and len(running) < n_parallel:
                job = pending.pop(0)
                log_path = log_dir / f"{job.env_id}__{job.name}__seed{job.seed}.log"
                logf = open(log_path, "w")
                proc = subprocess.Popen(build_command(job), cwd=REPO_ROOT, env=env,
                                        stdout=logf, stderr=subprocess.STDOUT)
                running.append((job, proc, logf, time.time()))
                print(f"[start] {job.tag} (log: {log_path})", flush=True)
            time.sleep(0.5)
            for item in list(running):
                job, proc, logf, t0 = item
                rc = proc.poll()
                if rc is None:
                    continue
                running.remove(item)
                logf.close()
                done += 1
                status = "ok" if rc == 0 else f"FAILED (exit {rc})"
                if rc != 0:
                    failed.append(job)
                print(f"[{done}/{len(jobs)}] {job.tag} {status} in {time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        print("interrupted: terminating running jobs", flush=True)
        for job, proc, logf, _ in running:
            proc.terminate()
        for job, proc, logf, _ in running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            logf.close()
        raise
    return failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    overrides: Dict = {"out_dir": args.out_dir}
    if args.max_steps is not None:
        overrides["total_timesteps"] = args.max_steps
    if args.device is not None:
        overrides["device"] = args.device
    jobs = make_jobs(select_pairs(args), args.seeds, overrides)

    if args.dry_run:
        for job in jobs:
            print(shlex.join(build_command(job)))
        print(f"# {len(jobs)} runs (dry run; completion status not checked)", file=sys.stderr)
        return 0

    todo = jobs if args.force else [j for j in jobs if not is_finished(j)]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already finished, "
          f"{len(todo)} to run with {args.jobs} parallel process(es)", flush=True)
    if not todo:
        return 0
    failed = run_jobs(todo, args.jobs, Path(args.log_dir))
    if failed:
        print(f"{len(failed)} run(s) failed:", file=sys.stderr)
        for j in failed:
            print(f"  {j.tag}", file=sys.stderr)
        return 1
    print("all runs finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
