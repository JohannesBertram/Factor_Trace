#!/usr/bin/env bash
# Run the notebook-09 validation suite for every model/task setting, consecutively.
#
#   ./run_validation_all.sh                      # all 5 experiments, cluster profile
#   ./run_validation_all.sh cnn_cifar vit_mnist  # only these
#   NB09_MODE=local ./run_validation_all.sh      # cheap profile (laptop smoke test)
#   PYTHON=/path/to/python ./run_validation_all.sh    # pick the interpreter explicitly
#
# Does NOT rely on the `jupyter` wrapper script (often missing on clusters):
# it calls `python -m nbconvert`. If ipykernel is unavailable it falls back to
# converting the notebook to a plain .py and running it — figures and JSON are
# produced either way.
#
# Outputs: figs/09_validation/<EXP>/*.pdf, data/results/nb09_<EXP>.json,
#          notebooks/executed_09_<EXP>.ipynb (nbconvert path), logs/nb09_<EXP>.log

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${NB09_MODE:-cluster}"
TIMEOUT="${NB_TIMEOUT:-100000}"
NB="09_validation_all_models.ipynb"
export MPLBACKEND="${MPLBACKEND:-Agg}"     # headless-safe plotting

# ── locate a usable python ───────────────────────────────────────────────────
pick_python() {
  local c
  for c in "${PYTHON:-}" "$REPO/.venv2/bin/python" "$REPO/venv2/bin/python" \
           "$REPO/.venv/bin/python" "$REPO/venv/bin/python" \
           "$(command -v python3 || true)" "$(command -v python || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}
PY="$(pick_python)" || { echo "ERROR: no python found. Set PYTHON=/path/to/python"; exit 1; }

if ! "$PY" -c "import nbconvert, nbformat" 2>/dev/null; then
  echo "ERROR: nbconvert/nbformat missing in: $PY"
  echo "  fix:  $PY -m pip install nbconvert nbformat ipykernel"
  echo "  (or point at the right env:  PYTHON=/path/to/venv2/bin/python $0 )"
  exit 1
fi

# ── choose execution strategy ────────────────────────────────────────────────
# Preferred: nbconvert --execute with a kernel pinned to THIS interpreter.
# Fallback: no ipykernel -> convert to script and run directly (no kernel needed).
KERNEL="bft-nb09"
if "$PY" -c "import ipykernel" 2>/dev/null; then
  "$PY" -m ipykernel install --user --name "$KERNEL" \
        --display-name "BFT nb09" >/dev/null 2>&1 \
    || { echo "WARN: could not register kernel; using default python3"; KERNEL="python3"; }
  STRATEGY="nbconvert"
else
  echo "NOTE: ipykernel not installed -> using script-conversion fallback."
  echo "      (for executed .ipynb outputs:  $PY -m pip install ipykernel )"
  STRATEGY="script"
fi

ALL=(mlp_even_odd mlp_digit cnn_cifar vit_mnist imagenet_cnn)
# avoid expanding "$@" when empty (errors under `set -u` on bash < 4.4)
if [ $# -eq 0 ]; then EXPS=("${ALL[@]}"); else EXPS=("$@"); fi

mkdir -p "$REPO/logs"
cd "$REPO/notebooks" || exit 1
echo "repo=$REPO"
echo "python=$PY"
echo "strategy=$STRATEGY  mode=$MODE  experiments: ${EXPS[*]}"

declare -a OK=() FAIL=()
for EXP in "${EXPS[@]}"; do
  LOG="$REPO/logs/nb09_${EXP}.log"
  echo "=== [$(date +%H:%M:%S)] $EXP -> $LOG"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "    DRY_RUN: would run [$STRATEGY] NB09_EXP=$EXP NB09_MODE=$MODE"
    OK+=("$EXP"); continue
  fi
  if [ "$STRATEGY" = "nbconvert" ]; then
    NB09_EXP="$EXP" NB09_MODE="$MODE" "$PY" -m nbconvert \
      --to notebook --execute \
      --output "executed_09_${EXP}.ipynb" \
      --ExecutePreprocessor.timeout="$TIMEOUT" \
      --ExecutePreprocessor.kernel_name="$KERNEL" \
      "$NB" >"$LOG" 2>&1
  else
    "$PY" -m nbconvert --to script --output "_run_09_${EXP}" "$NB" >"$LOG" 2>&1 \
      && NB09_EXP="$EXP" NB09_MODE="$MODE" "$PY" "_run_09_${EXP}.py" >>"$LOG" 2>&1
  fi
  if [ $? -eq 0 ]; then
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
