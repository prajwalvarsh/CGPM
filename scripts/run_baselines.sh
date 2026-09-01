#!/usr/bin/env bash
# Baselines only. No trained gate needed, so this is the first real run
# after the smoke test. Expect it to show the gap the gate is competing for:
# Always Retrieve should win on memory-dependent turns and lose on general ones.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
RUN_NAME="${RUN_NAME:-baselines}"

$PYTHON -m src.evaluate \
  --config configs/default.yaml \
  --policies never always oracle \
  --run-name "$RUN_NAME" \
  --split test \
  "$@"

echo
echo "results in experiments/runs/$RUN_NAME/results.md"
