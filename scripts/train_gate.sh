#!/usr/bin/env bash
# Train and calibrate the confidence gate.
#
# The first pass uses --no-probe: surface and sketch features only, so it runs
# in seconds on CPU and tells you whether the sketch alone carries signal.
# The second pass adds the language model probe. Compare the two AUROCs before
# you spend GPU time on anything else, because that comparison is the first
# row of the ablation table in the paper.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
COST_MISSED="${COST_MISSED:-3.0}"

echo "=== pass 1: no language model probe (cheap) ==="
$PYTHON -m src.train_gate \
  --config configs/default.yaml \
  --run-name gate_no_probe \
  --no-probe \
  --cost-missed "$COST_MISSED"

echo
echo "=== pass 2: with the probe ==="
$PYTHON -m src.train_gate \
  --config configs/default.yaml \
  --run-name gate \
  --cost-missed "$COST_MISSED"

echo
echo "compare val_auroc in:"
echo "  experiments/runs/gate_no_probe/train_report.json"
echo "  experiments/runs/gate/train_report.json"
