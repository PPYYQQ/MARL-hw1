#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
PACKAGE_NAME="${1:-marl_hw1_codebase.zip}"
PACKAGE_PATH="$DIST_DIR/$PACKAGE_NAME"

mkdir -p "$DIST_DIR"
rm -f "$PACKAGE_PATH"

cd "$ROOT_DIR/codebase"
zip -r "$PACKAGE_PATH" . \
  -x "log/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x ".pytest_cache/*" \
  -x "*/ckpt/*" \
  -x "*/models/*" \
  -x "*/saved_models/*" \
  -x "*.pkl" \
  -x "*.log"

printf 'Created %s\n' "$PACKAGE_PATH"
