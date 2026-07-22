#!/usr/bin/env bash
# Complete the notebook-09 hyperparameter sweeps on a CPU node, without SLURM.
#
#   ./run_final_settings.sh                    # imagenet_cnn, then cnn_cifar
#   ./run_final_settings.sh imagenet_cnn       # only the one that is still undecided
#   NB10_JOBS=16 ./run_final_settings.sh       # more BLAS threads (wall time only)
#   PYTHON=/path/to/python ./run_final_settings.sh
#   DRY_RUN=1 ./run_final_settings.sh          # preflight checks only, run nothing
#
# Runs in the foreground and tees to logs/nb10_<EXP>.log. For an unattended run:
#   nohup ./run_final_settings.sh > logs/nb10_all.log 2>&1 &
#   tail -f logs/nb10_imagenet_cnn.log
#
# Notebook 10 RESUMES a killed nb09 run: it reads data/results/nb09_<EXP>.json, runs only the
# sweep configurations that are missing, and applies nb09's preregistered selection rule to the
# merged set. Re-running this script after a crash is safe and cheap — every configuration whose
# fingerprints already landed on disk is skipped, and only the ~1.3 h setup repeats.
#
# CPU is not a compromise here: nb09's cluster run recorded device=cpu for both conv
# experiments, so these rows come out of the same code path as the ones they are completing.
#
# Expected wall clock (8 threads):
#   imagenet_cnn  ~4.5 h   setup 1.3 h + rank x1.3 1.6 h + one contender refit 1.6 h
#   cnn_cifar     ~3.5 h   setup 1.2 h + two contender refits 2.1 h  (+0.7 h per model seed)
#
# Outputs: data/results/nb10_<EXP>.json          — merged sweep, final_hp, bootstrap CIs
#          data/results/nb10_<EXP>_fingerprints/ — one .npz per configuration, kept forever
#          figs/10_final_settings/<EXP>/         — the selection figure

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" || exit 1

PY="${PYTHON:-$REPO/.venv2/bin/python}"
NB="notebooks/10_final_settings.ipynb"
NB09="notebooks/09_validation_all_models.ipynb"

# Threads: everything here is BLAS-bound. Default to the node's cores, capped at 16 — past that
# the NMF updates stop scaling and just contend.
CORES="$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) | head -1 )"
THREADS="${NB10_JOBS:-$(( CORES > 16 ? 16 : CORES ))}"

export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export NB10_MODE="${NB10_MODE:-cluster}"
export NB10_JOBS="$THREADS"
export CUDA_VISIBLE_DEVICES=""            # CPU node: do not let torch hunt for a GPU
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

if [ $# -eq 0 ]; then EXPS=(imagenet_cnn cnn_cifar); else EXPS=("$@"); fi
mkdir -p logs

# ── preflight: every check below has cost someone a night at least once ──────
die() { echo "ERROR: $*" >&2; exit 1; }

[ -x "$PY" ] || die "no interpreter at $PY (set PYTHON=/path/to/python)"
"$PY" -c 'import numpy, torch, sklearn, nbformat' 2>/dev/null \
  || die "$PY cannot import numpy/torch/sklearn/nbformat"
[ -f "$NB" ]   || die "$NB missing — git pull on this machine first"
[ -f "$NB09" ] || die "$NB09 missing — nb10 executes its §0-§2 setup cells"

prior_json() {
  for p in "data/results/nb09_$1.json" "logs/results/nb09_$1.json"; do
    [ -f "$REPO/$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

echo "repo    = $REPO"
echo "python  = $PY"
echo "threads = $THREADS (of $CORES cores)   mode = $NB10_MODE   device = cpu"
echo

FAILED_PREFLIGHT=0
for EXP in "${EXPS[@]}"; do
  J="$(prior_json "$EXP")" || {
    echo "  $EXP: NO nb09 RESULTS. nb10 completes a run, it does not start one." >&2
    FAILED_PREFLIGHT=1; continue; }
  echo "  $EXP: prior $J"
  "$PY" - "$EXP" "$J" <<'EOF'
import json, sys
exp, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
hp = d.get('HP_sweep', {})
rows = [r for r in hp.get('configs', []) if 'error' not in r]
print(f"      sections={d.get('completed_sections')}")
print(f"      sweep: {len(rows)} config(s) scored, complete={hp.get('complete')}, "
      f"final_hp={'yes' if d.get('final_hp') else 'NO'}")
if d.get('mode') != 'cluster':
    print(f"      WARNING mode={d.get('mode')!r}: rows from a different compute profile are "
          "NOT comparable to what this run will produce")
EOF
  case "$EXP" in
    imagenet_cnn)
      [ -d "$REPO/data/val" ] || [ -f "$REPO/data/meta.bin" ] \
        || echo "      WARNING no ImageNet val data (data/val ImageFolder or a torchvision" \
                "root at data/) — build_experiment will raise ~1 min in" ;;
    cnn_cifar)
      [ -f "$REPO/data/models/cifar10_cnn_seed0/weights.pt" ] \
        || echo "      WARNING no CIFAR checkpoint at data/models/cifar10_cnn_seed0/"
      SEEDS="$(ls -d "$REPO"/data/models/cifar10_cnn_seed* 2>/dev/null | wc -l | tr -d ' ')"
      [ "${SEEDS:-0}" -ge 2 ] \
        || echo "      NOTE $SEEDS CIFAR checkpoint(s): no model-seed error bar." \
                "Run scripts/train_extra_seeds.sh cnn first if you want one." ;;
  esac
done
[ "$FAILED_PREFLIGHT" -eq 0 ] || die "preflight failed for at least one experiment"

if [ -n "${DRY_RUN:-}" ]; then
  echo; echo "DRY_RUN: preflight passed for ${EXPS[*]} — nothing executed."
  exit 0
fi

# ── run ──────────────────────────────────────────────────────────────────────
declare -a OK=() FAIL=()
for EXP in "${EXPS[@]}"; do
  LOG="logs/nb10_${EXP}.log"
  echo
  echo "=== [$(date '+%F %T')] $EXP -> $LOG"
  NB10_EXP="$EXP" "$PY" scripts/run_nb.py "$NB" 2>&1 | tee "$LOG"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "    [$(date '+%F %T')] OK -> data/results/nb10_${EXP}.json"
    OK+=("$EXP")
  else
    echo "    [$(date '+%F %T')] FAIL -> $LOG (partial results are still on disk;"
    echo "    re-run this script to resume — finished configurations are skipped)"
    FAIL+=("$EXP")
  fi
done

# ── summary: the answer, per experiment ──────────────────────────────────────
echo
echo "=== summary ==="
echo "  passed (${#OK[@]}): ${OK[*]:-none}"
echo "  failed (${#FAIL[@]}): ${FAIL[*]:-none}"
"$PY" - <<'EOF'
import glob, json
for f in sorted(glob.glob('data/results/nb10_*.json')):
    d = json.load(open(f))
    hp, fh = d.get('HP_sweep', {}), d.get('final_hp')
    print(f"\n  {d['experiment']}  (grid complete: {hp.get('complete')})")
    if not fh:
        print('    no final_hp — the sweep did not finish'); continue
    print(f"    selected : {fh['selected_config']}")
    print(f"    settings : {fh['bft_kwargs']}")
    sep = fh.get('separated_from_incumbent')
    print(f"    vs incumbent: " + {True: 'a real margin', False:
          'INSIDE THE NOISE — call the choice immaterial, not an improvement',
          None: 'not measured'}[sep])
    if fh.get('selected_below_stab_gate'):
        print('    WARNING selected config is below the 0.85 stability gate')
EOF
echo
echo "  Paste the settings into the matching notebook (nb03 / nb05) and update"
echo "  PUBLICATION_SETTINGS.md. Full record: data/results/nb10_<EXP>.json"
[ ${#FAIL[@]} -eq 0 ]
