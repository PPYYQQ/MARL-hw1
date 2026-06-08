#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR/codebase"
python -m compileall agent_target_dqn agent_ppo tests
python tests/test_target_dqn_features.py
python tests/test_target_dqn_static.py
python tests/test_hyperparams_static.py
python tests/test_target_dqn_smoke.py

cd "$ROOT_DIR"
git diff --check
./scripts/package_submission.sh
zipinfo -1 dist/marl_hw1_codebase.zip | grep -E '(^log/|__pycache__|\.pyc$|ckpt/|\.pkl$|screenshot|REPORT|RUNBOOK|PROGRESS|AGENTS|icml)' && {
  echo "unexpected file found in submission package" >&2
  exit 1
} || true

echo "Offline checks completed"
