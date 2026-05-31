#!/usr/bin/env python3
"""Execute a Jupyter notebook without saving any output.

Usage:
    python scripts/run_nb.py notebooks/02_MLP_40_20_digits.ipynb
"""
import sys
import os
import nbformat
import matplotlib
matplotlib.use('Agg')

def run_notebook(path):
    nb = nbformat.read(path, as_version=4)
    # Set working directory to repo root so relative paths work
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    ns = {'__file__': os.path.abspath(path)}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code' or not cell.source.strip():
            continue
        try:
            exec(compile(cell.source, f'{path}:cell{i}', 'exec'), ns)
        except SystemExit:
            pass
        except Exception as e:
            print(f'\n[ERROR] Cell {i}:\n{cell.source[:200]}\n{type(e).__name__}: {e}')
            raise

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_nb.py <notebook.ipynb>')
        sys.exit(1)
    run_notebook(sys.argv[1])
