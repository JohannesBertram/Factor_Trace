#!/usr/bin/env bash
# Assemble the anonymous code+data submission into ./submission/.
# Copies only the include-list, clears notebook outputs, scrubs identity leaks,
# and strips cruft. README.md, LICENSE and website/serve.sh are written separately
# (they are authored, not scrubbed). Run from the repo root:
#     bash scripts/make_submission.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/submission"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

echo ">> resetting $DEST"
rm -rf "$DEST"
mkdir -p "$DEST/src" "$DEST/scripts" "$DEST/notebooks" \
         "$DEST/figures/figdata" "$DEST/figures/appendix" \
         "$DEST/website"

# ---- src/ : all modules except side-projects / dead code -------------------
echo ">> src/"
for f in src/*.py; do
  case "$(basename "$f")" in
    flyvis_trace_utils.py|encoding_manifold_utils.py|old_code.py) continue ;;
  esac
  cp "$f" "$DEST/src/"
done

# ---- scripts/ : python pipeline only (no cluster .sh) ----------------------
echo ">> scripts/"
for b in render_figures.py train.py build_validation_bundles.py \
         build_pruning_bundle.py add_activation_baselines.py; do
  cp "scripts/$b" "$DEST/scripts/"
done

# ---- notebooks/ : the 15 paper notebooks -----------------------------------
echo ">> notebooks/"
for n in 01_MLP_8_4_0134 02_MLP_40_20_digits 03_CNN_CIFAR10 04_ViT 05_imagenet_cnn \
         09_validation_all_models 10_final_settings \
         13_pruning_mlp_even_odd 14_pruning_digit_cnn_imagenet \
         fig01_mlp_even_odd fig02_mlp_digits fig03_cnn_cifar fig04_vit fig05_imagenet fig09_validation; do
  cp "notebooks/$n.ipynb" "$DEST/notebooks/"
done

# ---- figures/figdata/ : bundle subset (drop 2 big circuits + non-paper) ----
echo ">> figures/figdata/"
cp figures/figdata/*.npz figures/figdata/*.json "$DEST/figures/figdata/"
rm -f "$DEST"/figures/figdata/nb05_circuits.* "$DEST"/figures/figdata/nb03_circuits.* \
      "$DEST"/figures/figdata/error_consistency.json \
      "$DEST"/figures/figdata/nb11_*_silhouette.* \
      "$DEST"/figures/figdata/nb12_error_consistency.*

# ---- figures/ : rendered paper PDFs (main + appendix, no preview) ----------
echo ">> figures/ (rendered)"
cp figures/BFT_overview.pdf \
   figures/fig2_mlp_circuits.pdf figures/fig4_fingerprints_main.pdf \
   figures/fig6_cnn_circuits.pdf figures/fig8_imagenet_circuits.pdf "$DEST/figures/"
for b in figA_mlp_details figB_digit_mlp_details figE_cnn_details figG_vit_circuits \
         figN_imagenet_details figfp_ood figfp_structure figP_validation; do
  cp "figures/appendix/$b.pdf" "$DEST/figures/appendix/"
done

# ---- root support ----------------------------------------------------------
echo ">> root support"
cp figstyle.py check_figure.py "$DEST/"
cp -R .figstyle "$DEST/.figstyle"
cp requirements.txt "$DEST/requirements.txt"
grep -qi '^Pillow' "$DEST/requirements.txt" || echo 'Pillow>=10.0.0' >> "$DEST/requirements.txt"

# ---- website/ : self-contained static site ---------------------------------
echo ">> website/"
rsync -a --exclude '__pycache__' --exclude '.DS_Store' docs/ "$DEST/website/"

# ---- clear notebook outputs + scrub identity leaks -------------------------
echo ">> clear outputs + scrub"
"$PY" - "$DEST" <<'PY'
import sys, os, glob, nbformat
dest = sys.argv[1]

for p in glob.glob(os.path.join(dest, 'notebooks', '*.ipynb')):
    nb = nbformat.read(p, as_version=4)
    for c in nb.cells:
        if c.get('cell_type') == 'code':
            c['outputs'] = []
            c['execution_count'] = None
            c.get('metadata', {}).pop('execution', None)
    nbformat.write(nb, p)

REPL = [
    ("/Users/johannesbertram/repos/Weight_Interpretability", "."),
    ("/home/jb3879/Factor_Trace", "."),
    ("/Users/johannesbertram", "/path/to"),
    ("/home/jb3879", "/path/to"),
    ("https://johannesbertram.github.io/Factor_Trace/", ""),
    ("johannesbertram.github.io/Factor_Trace", ""),
    ("johannesbertram", "anon"),
    ("jb3879", "anon"),
    ("Johannes Bertram", "Anonymous Authors"),
    ("Luciano Dyballa", "Anonymous Authors"),
    ("Luciano", "Anonymous"),
    ("dyballa@gmail.com", ""),
    ("johannes.bertram@student.uni-tuebingen.de", ""),
]
EXTS = {'.py', '.md', '.ipynb', '.txt', '.yaml', '.yml', '.html', '.js', '.css', '.sh'}
for root, _, files in os.walk(dest):
    for fn in files:
        if os.path.splitext(fn)[1].lower() not in EXTS:
            continue
        fp = os.path.join(root, fn)
        try:
            s = open(fp, encoding='utf-8').read()
        except Exception:
            continue
        o = s
        for a, b in REPL:
            s = s.replace(a, b)
        if s != o:
            open(fp, 'w', encoding='utf-8').write(s)
print("   cleared + scrubbed")
PY

# ---- strip cruft -----------------------------------------------------------
echo ">> strip cruft"
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$DEST" -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +
find "$DEST" -name '.DS_Store' -delete

echo ">> done. size:"
du -sh "$DEST"
