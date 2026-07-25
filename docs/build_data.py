#!/usr/bin/env python
"""Build the static circuit-explorer assets from the committed figdata bundles.

For each model's ``nb0N_circuits`` bundle this script:

  * reconstructs the Backward-Factor-Trace tree from the depth-first ``nodes``
    list (a node's children are the nodes whose ``path`` extends it by one factor),
  * renders every factor's weighted-average image and its top example stimuli into
    one sprite sheet per node (denormalized so colors are faithful), and
  * writes a compact ``manifest.json`` describing the tree, the layers, and the
    per-factor metadata (importance share, dominant-class profile, child link).

The web app (``docs/index.html`` + ``docs/assets``) is pure static files that read
these manifests and sprites — no Python, models, or datasets at serve time.

    python docs/build_data.py            # rebuild every model
    python docs/build_data.py nb03 nb05  # rebuild a subset

Output lands in ``docs/models/`` (a folder name deliberately avoided being ``data``,
which the repo-wide .gitignore would swallow).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import figdata  # noqa: E402

OUT_DIR = ROOT / "docs" / "models"

# One entry per model: bundle name, display title/subtitle, and the sprite tile
# size (native px each factor image is stored at; the browser scales up crisply).
MODELS = [
    dict(id="nb01", bundle="nb01_circuits", tile=56,
         title="MLP · MNIST even/odd",
         subtitle="3-layer fully-connected net, even-vs-odd digit"),
    dict(id="nb02", bundle="nb02_circuits", tile=56,
         title="MLP · MNIST digits",
         subtitle="3-layer fully-connected net, 10-way digit classifier"),
    dict(id="nb03", bundle="nb03_circuits", tile=64,
         title="CNN · CIFAR-10",
         subtitle="4-block small convolutional network"),
    dict(id="nb04", bundle="nb04_circuits", tile=56,
         title="ViT · MNIST even/odd",
         subtitle="Tiny vision transformer, 1 block / 2 heads"),
    dict(id="nb05", bundle="nb05_circuits", tile=64,
         title="SqueezeNet · ImageNet",
         subtitle="SqueezeNet-style CNN, 8-category ImageNet subset"),
]


# ── image conversion ──────────────────────────────────────────────────────────

def to_rgb(chw, mean, std, is_wavg):
    """One (C,H,W) array -> an RGB PIL image, ready to paste into a sprite.

    RGB models carry (mean,std): denormalize and clip so colors are faithful.
    Grayscale models are already in [0,1]; weighted averages are contrast-
    stretched so the (necessarily faint) average pattern is legible.
    """
    x = np.asarray(chw, np.float32)
    c = x.shape[0]
    if mean is not None:
        m = np.asarray(mean, np.float32).reshape(-1, 1, 1)
        s = np.asarray(std, np.float32).reshape(-1, 1, 1)
        x = np.clip(x * s + m, 0.0, 1.0)
    else:
        if is_wavg:
            lo, hi = float(x.min()), float(x.max())
            if hi > lo:
                x = (x - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
    if c == 1:
        arr = (x[0] * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(arr, "L").convert("RGB")
    arr = (np.transpose(x, (1, 2, 0)) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def resample_for(is_rgb):
    # Natural images upscale smoothly; MNIST-style stays crisp/blocky (honest px).
    return Image.LANCZOS if is_rgb else Image.NEAREST


# ── labeling ──────────────────────────────────────────────────────────────────

def pick_profile(node):
    """Return (profile_matrix (K,C), source_key) — the richest class breakdown."""
    if "digit_profile" in node:
        return np.asarray(node["digit_profile"], np.float32), "digit"
    if "class_profile" in node:
        return np.asarray(node["class_profile"], np.float32), "class"
    return None, None


def profile_labels(bundle, source):
    if source == "digit":
        digits = bundle.get("digits")
        if digits is not None:
            return [str(int(d)) for d in np.asarray(digits)]
        return [str(i) for i in range(10)]
    for key in ("class_names", "category_names"):
        if key in bundle and bundle[key]:
            return [str(x) for x in bundle[key]]
    order = bundle.get("digit_order")
    if order is not None:
        return [str(int(d)) for d in np.asarray(order)]
    classes = bundle.get("meta", {}).get("classes")
    if classes is not None:
        return [str(int(c)) for c in np.asarray(classes)]
    return None


def factor_label(prof_row, labels, fallback):
    if prof_row is None or labels is None or not len(labels):
        return fallback
    i = int(np.argmax(prof_row))
    share = float(prof_row[i]) / (float(prof_row.sum()) + 1e-12)
    return f"{labels[i]} · {share * 100:.0f}%"


# ── per-model build ───────────────────────────────────────────────────────────

def build_model(spec):
    bundle = figdata.load(spec["bundle"])
    nodes = bundle["nodes"]
    meta = bundle.get("meta", {})
    tile = spec["tile"]

    mean = bundle.get("image_mean")
    std = bundle.get("image_std")
    if mean is not None:
        mean = np.asarray(mean, np.float32)
        std = np.asarray(std, np.float32)
    is_rgb = nodes[0]["wavg"].shape[1] == 3
    resample = resample_for(is_rgb)

    labels = None
    _, src = pick_profile(nodes[0])
    if src:
        labels = profile_labels(bundle, src)

    # root-factor names supplied by the traced circuits (e.g. "even"/"odd")
    root_names = {}
    for c in bundle.get("circuits", []):
        if isinstance(c, dict) and "name" in c and "k" in c:
            root_names[int(c["k"])] = str(c["name"])

    # tree: map path -> node index; find the root (empty path)
    def path_key(p):
        return ",".join(str(int(x)) for x in np.asarray(p).ravel())

    path_to_idx = {path_key(n["path"]): i for i, n in enumerate(nodes)}
    root_idx = path_to_idx.get("", 0)

    model_dir = OUT_DIR / spec["id"]
    sprite_dir = model_dir / "sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)

    out_nodes = []
    n_tiles = 0
    for i, node in enumerate(nodes):
        wavg = np.asarray(node["wavg"])            # (K, C, H, W)
        exs = np.asarray(node["top_images"])       # (K, T, C, H, W)
        K, T = wavg.shape[0], exs.shape[1]
        cols = 1 + T
        prof, _ = pick_profile(node)
        path = [int(x) for x in np.asarray(node["path"]).ravel()]
        lam = np.asarray(node.get("lam_share", np.ones(K) / K), np.float32)

        sprite = Image.new("RGB", (cols * tile, K * tile), (16, 17, 22))
        factors = []
        for k in range(K):
            # column 0: weighted-average image; columns 1..T: example stimuli
            w = to_rgb(wavg[k], mean, std, is_wavg=True).resize((tile, tile), resample)
            sprite.paste(w, (0, k * tile))
            for t in range(T):
                e = to_rgb(exs[k, t], mean, std, is_wavg=False).resize((tile, tile), resample)
                sprite.paste(e, ((1 + t) * tile, k * tile))
            n_tiles += cols

            child = path_to_idx.get(path_key(path + [k]))
            prow = prof[k] if prof is not None else None
            fb = f"factor {k}"
            if i == root_idx and k in root_names:
                lbl = root_names[k]
            else:
                lbl = factor_label(prow, labels, fb)
            factors.append(dict(
                k=k,
                child=(int(child) if child is not None else None),
                lam=round(float(lam[k]), 4),
                label=lbl,
                profile=([round(float(v), 4) for v in prow] if prow is not None else None),
                n_ex=int(T),
            ))

        # photographs compress far better as JPEG; keep crisp line-art as PNG
        if is_rgb:
            sprite_name = f"n{i:03d}.jpg"
            sprite.save(sprite_dir / sprite_name, quality=88, optimize=True)
        else:
            sprite_name = f"n{i:03d}.png"
            sprite.save(sprite_dir / sprite_name, optimize=True)

        ws = [int(x) for x in np.asarray(node["weight_shape"]).ravel()]
        out_nodes.append(dict(
            id=i,
            path=path,
            depth=len(path),
            layer_idx=int(node["layer_idx"]),
            layer_name=str(node["layer_name"]),
            layer_type=str(node["layer_type"]),
            n_factors=int(K),
            sprite=f"sprites/{sprite_name}",
            tile=tile,
            cols=cols,
            weight_shape=ws,
            factors=factors,
        ))

    # ordered layer list (input side -> output side) for the network schematic
    layer_map = {}
    for n in out_nodes:
        li = n["layer_idx"]
        if li not in layer_map:
            layer_map[li] = dict(idx=li, name=n["layer_name"], type=n["layer_type"],
                                 size=int(n["weight_shape"][0]))
    layers = [layer_map[k] for k in sorted(layer_map)]
    in_dim = out_nodes and out_nodes[0]["weight_shape"]
    input_size = int(out_nodes[min(range(len(out_nodes)),
                     key=lambda j: out_nodes[j]["layer_idx"])]["weight_shape"][1]) \
        if in_dim and len(in_dim) > 1 else 0

    manifest = dict(
        id=spec["id"],
        title=spec["title"],
        subtitle=spec["subtitle"],
        image_kind=("rgb" if is_rgb else "gray"),
        tile=tile,
        root=int(root_idx),
        root_layer=int(meta.get("root_layer", layers[-1]["idx"] if layers else 0)),
        depth=int(max((n["depth"] for n in out_nodes), default=0)),
        profile_labels=labels,
        layers=layers,
        input=dict(name="input image", type="image", size=input_size),
        meta=dict(
            n_stimuli=int(meta.get("n_stimuli", 0)),
            test_acc=(round(float(meta["test_acc"]), 4) if "test_acc" in meta else None),
            n_layers=len(layers),
            n_nodes=len(out_nodes),
        ),
        nodes=out_nodes,
    )
    with open(model_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, separators=(",", ":"))

    size_kb = sum(p.stat().st_size for p in sprite_dir.iterdir() if p.is_file()) / 1024
    print(f"[{spec['id']}] {len(out_nodes)} nodes, {n_tiles} tiles, "
          f"{size_kb:.0f} kB sprites  ->  {model_dir.relative_to(ROOT)}")
    return dict(id=spec["id"], title=spec["title"], subtitle=spec["subtitle"],
                file=f"models/{spec['id']}/manifest.json",
                n_nodes=len(out_nodes), depth=manifest["depth"],
                image_kind=manifest["image_kind"])


def main(argv):
    wanted = set(argv)
    specs = [m for m in MODELS if not wanted or m["id"] in wanted]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = [build_model(s) for s in specs]
    if not wanted:  # only rewrite the top-level index on a full build
        with open(OUT_DIR / "index.json", "w") as f:
            json.dump(dict(models=index), f, indent=1)
        print(f"[index] {len(index)} models -> {(OUT_DIR / 'index.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
