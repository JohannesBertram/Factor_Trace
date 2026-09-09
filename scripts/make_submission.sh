#!/usr/bin/env bash
# Assemble the anonymous code+data submission into ./submission/.
# After the publication refactor the repo IS the submission: copy the tracked
# tree, drop private/heavy parts, clear notebook outputs, scrub identity leaks.
# Run from the repo root:
#     bash scripts/make_submission.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/submission"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

echo ">> resetting $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"

# ---- the tracked tree, exactly as committed ---------------------------------
echo ">> git archive HEAD"
git archive HEAD | tar -x -C "$DEST"

# ---- drop what the submission does not carry --------------------------------
# logs/ holds run provenance with cluster paths; PUBLICATION.md is planning.
rm -rf "$DEST/logs"
rm -f "$DEST/PUBLICATION.md"
mv "$DEST/docs" "$DEST/website"

# ---- clear notebook outputs + scrub identity leaks --------------------------
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
EXTS = {'.py', '.md', '.ipynb', '.txt', '.yaml', '.yml', '.html', '.js', '.css', '.sh',
        '.json'}
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

# ---- strip cruft ------------------------------------------------------------
echo ">> strip cruft"
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$DEST" -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +
find "$DEST" -name '.DS_Store' -delete

echo ">> done. size:"
du -sh "$DEST"
