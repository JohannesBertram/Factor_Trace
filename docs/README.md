# Circuit Explorer

An interactive, static web tool for browsing the circuits that **Backward Factor Trace (BFT)** decomposes out of each trained model. Pick a model, then click through its factor tree: every factor shows its weighted-average image and top example stimuli, the panel on the left tracks where you are in the network and which circuit led there, and clicking a factor descends into the sub-factors that compose it.

All five paper models are included: MLP (even/odd), MLP (digits), CNN (CIFAR-10), ViT (even/odd), and SqueezeNet (ImageNet subset).

## How it works

The explorer is a dependency-free static site — plain HTML/CSS/JS, no build step, no framework. It reads only files under `models/`:

```
docs/
├── index.html            # the app shell
├── assets/
│   ├── app.js            # tree navigation + sprite-crop rendering
│   └── style.css
├── models/
│   ├── index.json        # list of models
│   └── nb0N/
│       ├── manifest.json # tree, layers, per-factor metadata
│       └── sprites/      # one sprite sheet per node (PNG for line-art, JPEG for photos)
└── build_data.py         # regenerates everything above from the figdata bundles
```

Each node's factor images (weighted average + example stimuli) are packed into a single sprite sheet; the app crops individual tiles onto `<canvas>`, so a node loads in one request and the tree stays snappy even as it grows.

## Regenerating the data

The assets are generated from the committed `figures/figdata/nb0N_circuits` bundles — no models or datasets are needed. From the repo root:

```bash
python docs/build_data.py            # rebuild all five models
python docs/build_data.py nb03 nb05  # rebuild a subset
```

If you add larger traces later, re-export the `nb0N_circuits` bundle and re-run the script; the app picks up the new tree automatically. To add a brand-new model, append an entry to the `MODELS` list in `build_data.py`.

## Viewing locally

Because the app fetches JSON, open it through a web server rather than `file://`:

```bash
python -m http.server 8000 --directory docs
# then visit http://localhost:8000/
```

## Hosting on GitHub Pages

This folder is self-contained and ready to serve. On GitHub:

1. **Settings → Pages**
2. **Build and deployment → Source:** *Deploy from a branch*
3. **Branch:** `main`, **folder:** `/docs` → **Save**

After a minute the explorer is live at `https://<user>.github.io/<repo>/` (for this repo, `https://johannesbertram.github.io/Factor_Trace/`). All paths are relative, so it works under that project sub-path unchanged. The `.nojekyll` file disables Jekyll so nothing is post-processed.
