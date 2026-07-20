#!/usr/bin/env bash
# Run the notebook-09 validation suite for every model/task setting, consecutively.
#
#   ./run_validation_all.sh                      # all 5 experiments, cluster profile
#   ./run_validation_all.sh cnn_cifar vit_mnist  # only these
#   NB09_MODE=local ./run_validation_all.sh      # cheap profile (laptop smoke test)
#
# Outputs: figs/09_validation/<EXP>/*.pdf, data/results/nb09_<EXP>.json,
#          notebooks/executed_09_<EXP>.ipynb, logs/nb09_<EXP>.log

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUPYTER="${JUPYTER:-$REPO/.venv/bin/jupyter}"
MODE="${NB09_MODE:-cluster}"
TIMEOUT="${NB_TIMEOUT:-100000}"
NB="09_validation_all_models.ipynb"

ALL=(mlp_even_odd mlp_digit cnn_cifar vit_mnist imagenet_cnn)
EXPS=("${@:-}"); [ -z "${1:-}" ] && EXPS=("${ALL[@]}")

[ -x "$JUPYTER" ] || { echo "ERROR: jupyter not found at $JUPYTER (set JUPYTER=...)"; exit 1; }
mkdir -p "$REPO/logs"
cd "$REPO/notebooks" || exit 1

echo "repo=$REPO  mode=$MODE  experiments: ${EXPS[*]}"
declare -a OK=() FAIL=()

for EXP in "${EXPS[@]}"; do
  LOG="$REPO/logs/nb09_${EXP}.log"
  echo "=== [$(date +%H:%M:%S)] $EXP -> $LOG"
  if NB09_EXP="$EXP" NB09_MODE="$MODE" "$JUPYTER" nbconvert \
        --to notebook --execute \
        --output "executed_09_${EXP}.ipynb" \
        --ExecutePreprocessor.timeout="$TIMEOUT" \
        "$NB" >"$LOG" 2>&1; then
    echo "    OK   -> data/results/nb09_${EXP}.json"
    OK+=("$EXP")
  else
    echo "    FAIL -> see $LOG"
    tail -n 15 "$LOG" | sed 's/^/      | /'
    FAIL+=("$EXP")
  fi
done

echo
echo "=== summary ==="
echo "  passed (${#OK[@]}): ${OK[*]:-none}"
echo "  failed (${#FAIL[@]}): ${FAIL[*]:-none}"
echo "  results: $REPO/data/results/nb09_*.json"
echo "  figures: $REPO/figs/09_validation/<EXP>/"
[ ${#FAIL[@]} -eq 0 ]
