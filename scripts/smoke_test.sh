#!/usr/bin/env bash
# Short end-to-end check: 1 CartPole run + 2 HalfCheetah runs (baseline and one ablation),
# then verify that every expected log file exists and is non-empty.
# Usage: bash scripts/smoke_test.sh   (PYTHON=/path/to/python to override the interpreter)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
else
    PY=python
fi

OUT="$ROOT/runs/smoke"          # runs/ is git-ignored; never touches results/
LOGS="$OUT/_logs"
rm -rf "$OUT"
mkdir -p "$OUT"

CARTPOLE_STEPS=20000            # ~39 iterations of 512 steps
CHEETAH_STEPS=6144              # 3 iterations of 2048 steps

echo "== CartPole-v1 baseline, seed 0, $CARTPOLE_STEPS steps"
"$PY" experiments/run_sweep.py --envs CartPole-v1 --configs baseline --seeds 0 \
    --max-steps "$CARTPOLE_STEPS" --out-dir "$OUT" --log-dir "$LOGS" --jobs 1

echo "== HalfCheetah-v5 baseline + no_gae, seed 0, $CHEETAH_STEPS steps"
"$PY" experiments/run_sweep.py --envs HalfCheetah-v5 --configs baseline no_gae --seeds 0 \
    --max-steps "$CHEETAH_STEPS" --out-dir "$OUT" --log-dir "$LOGS" --jobs 2

echo "== checking log files"
fail=0
check_file() {
    if [[ ! -s "$1" ]]; then
        echo "MISSING OR EMPTY: $1"
        fail=1
    fi
}
for run in CartPole-v1/baseline HalfCheetah-v5/baseline HalfCheetah-v5/no_gae; do
    dir="$OUT/$run/seed_0"
    for f in config.json progress.csv metrics.csv; do
        check_file "$dir/$f"
    done
    if [[ -s "$dir/progress.csv" && "$(wc -l < "$dir/progress.csv")" -lt 2 ]]; then
        echo "NO EPISODE ROWS: $dir/progress.csv"
        fail=1
    fi
    if [[ -s "$dir/metrics.csv" && "$(wc -l < "$dir/metrics.csv")" -lt 2 ]]; then
        echo "NO METRIC ROWS: $dir/metrics.csv"
        fail=1
    fi
    if [[ -s "$dir/metrics.csv" ]] && ! head -n 1 "$dir/metrics.csv" | grep -q "global_step,policy_loss,value_loss,entropy,approx_kl,clipfrac,explained_variance,learning_rate,sps"; then
        echo "UNEXPECTED metrics.csv HEADER: $dir/metrics.csv"
        fail=1
    fi
    if [[ -s "$dir/progress.csv" ]] && ! head -n 1 "$dir/progress.csv" | grep -q "^global_step,episodic_return,episodic_length"; then
        echo "UNEXPECTED progress.csv HEADER: $dir/progress.csv"
        fail=1
    fi
done

if [[ "$fail" -ne 0 ]]; then
    echo "SMOKE TEST FAILED (stdout/stderr of each run: $LOGS)"
    exit 1
fi
echo "SMOKE TEST PASSED"
