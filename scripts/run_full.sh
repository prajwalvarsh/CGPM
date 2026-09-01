#!/usr/bin/env bash
# The full table: every policy on the test split, in one run, so the numbers
# are directly comparable. This is what fills Table 1 of the paper.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
RUN_NAME="${RUN_NAME:-main}"

if [ ! -f experiments/runs/gate/gate.pt ]; then
  echo "no trained gate found; running scripts/train_gate.sh first"
  bash scripts/train_gate.sh
fi

$PYTHON -m src.evaluate \
  --config configs/default.yaml \
  --policies never always verbalized gate oracle \
  --gate-dir experiments/runs/gate \
  --run-name "$RUN_NAME" \
  --split test \
  --allow-clarify \
  "$@"

echo
echo "paste experiments/runs/$RUN_NAME/results.md into experiments/log_template.md"
