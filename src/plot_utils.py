import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def plot_nmf_scree(results):
    """
    Plot explained variance vs n_components.

    Parameters
    ----------
    results : dict {k: ev} mapping n_components → explained variance

    Returns
    -------
    fig
    """
    ks = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(6, 3))
    for k in ks:
        ax.scatter(k, results[k], color='gray', alpha=0.5, s=15, zorder=5)
    ax.plot(ks, [results[k] for k in ks], color='blue', lw=2)
    ax.set(xlabel='n_components', ylabel='explained variance',
           title='NMF scree — pick the elbow')
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_nmf_component(fi, neural_factors, img_factors, layer_sizes,
                       class_images, class_digits, first_layer_weights,
                       image_side=28):
    """
    Summary plot for one NMF component.

    Panels: (1) grid heatmap of neural loadings by layer,
            (2) line plot of neural loadings with layer dividers,
            (3) per-image coefficient sorted by digit class,
            (4) weighted-average input image,
            (5) deconvolution: first-layer weights weighted by neural factor.

    Parameters
    ----------
    fi                 : component index
    neural_factors     : (n_neurons, n_components)
    img_factors        : (n_class_samples, n_components)
    layer_sizes        : list of int — neurons per layer (e.g. [20, 10, 2])
    class_images       : (n_class_samples, H, W) or (n_class_samples, C, H, W) images
    class_digits       : (n_class_samples,) original digit labels for x-axis sorting
    first_layer_weights: (n_hidden1, n_inputs) weight matrix of the first linear layer
    image_side         : int — image height in pixels (28 for MNIST)
    """
    canvas = np.full((max(layer_sizes), len(layer_sizes) * 2), np.nan)
    fig, (ax_grid, ax_line, ax_coeff, ax_img, ax_deconv) = plt.subplots(1, 5, figsize=(11, 1))

    tot = 0
    for lii, lsz in enumerate(layer_sizes):
        canvas[:lsz, lii * 2] = neural_factors[tot:tot + lsz, fi]
        ax_line.axvline(tot + lsz - 0.5, ls='--', color='r')
        tot += lsz

    ax_grid.imshow(canvas, aspect=0.5)
    ax_grid.set(xticks=[], yticks=[])
    ax_grid.axis('off')

    ax_line.plot(neural_factors[:, fi], 'bo-', markersize=3)
    ax_line.set_title(f'component {fi}')

    sort_order = np.argsort(class_digits)
    ax_coeff.plot(img_factors[sort_order, fi])

    imgs_2d = class_images.reshape(len(class_images), image_side, -1)
    w_avg = np.sum([c * imgs_2d[i] for i, c in enumerate(img_factors[:, fi])], 0)
    w_avg /= img_factors[:, fi].sum()
    ax_img.imshow(w_avg, cmap='gist_ncar')
    ax_img.set(xticks=[], yticks=[])

    n_first = len(first_layer_weights)
    deconv = (neural_factors[:n_first, fi][:, None] * first_layer_weights).sum(0)
    absmax = np.abs(deconv).max()
    ax_deconv.imshow(deconv.reshape(image_side, -1), vmin=-absmax, vmax=absmax, cmap='seismic')
    ax_deconv.set(xticks=[], yticks=[])

    return fig


def plot_neuron_nmf_component(fi, img_factors, arbor_factors, input_shape,
                              images, targets, classes,
                              class_names=None, digit_targets=None):
    """
    Visualize one NMF component from a per-neuron synaptic arbor factorization.

    Parameters
    ----------
    fi             : component index
    img_factors    : (n_samples, K) — per-image NMF coefficients
    arbor_factors  : (n_inputs, K) — per-input-dimension NMF basis vectors
    input_shape    : int or (H, W) — for reshaping arbor_factors[:, fi] as imshow
    images         : (n_samples, H, W) — original input images (used for weighted avg)
    targets        : (n_samples,) task class labels
    classes        : list of class ids
    class_names    : optional dict {cl: name}
    digit_targets  : optional (n_samples,) original digit labels

    Panels
    ------
    1. arbor_factors[:, fi] reshaped  — the "receptive field pattern"
    2. weighted-average input image   — weighted by img_factors[:, fi]
    3. coefficient distribution per task class (jittered strip + mean)
    4. coefficient distribution per digit (if digit_targets given)
    """
    n_panels = 4 if digit_targets is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 2.5))

    # panel 1: arbor pattern
    af = arbor_factors[:, fi]
    if isinstance(input_shape, (tuple, list)):
        af_img = af.reshape(input_shape)
    else:
        af_img = af.reshape(int(input_shape), -1)
    absmax = np.abs(af_img).max() or 1
    axes[0].imshow(af_img, vmin=-absmax, vmax=absmax, cmap='seismic')
    axes[0].set(title=f'arbor pattern {fi}', xticks=[], yticks=[])

    # panel 2: weighted average image
    coefs = img_factors[:, fi]
    imgs_2d = images.reshape(len(images), -1)
    w_avg = (coefs[:, None] * imgs_2d).sum(0) / (coefs.sum() + 1e-12)
    img_h, img_w = images.shape[-2], images.shape[-1]
    axes[1].imshow(w_avg.reshape(img_h, img_w), cmap='gray')
    axes[1].set(title='weighted avg image', xticks=[], yticks=[])

    cmap = matplotlib.colormaps['tab10']

    def _strip(ax, group_ids, group_names, title):
        for xi, cl in enumerate(group_ids):
            mask = (targets if group_ids is classes else digit_targets) == cl
            y = coefs[mask]
            x = np.full(len(y), xi) + np.random.default_rng(0).uniform(-0.2, 0.2, len(y))
            label = group_names.get(cl, str(cl)) if group_names else str(cl)
            ax.scatter(x, y, s=6, alpha=0.4, color=cmap(xi / max(len(group_ids) - 1, 1)))
            ax.plot([xi - 0.3, xi + 0.3], [y.mean(), y.mean()], color='k', lw=1.5)
        ax.set_xticks(range(len(group_ids)))
        ax.set_xticklabels(
            [group_names.get(cl, str(cl)) if group_names else str(cl) for cl in group_ids],
            fontsize=8,
        )
        ax.set(title=title, ylabel='coefficient')

    # panel 3: task class distribution
    _strip(axes[2], classes, class_names or {}, 'by task class')

    # panel 4: digit distribution (optional)
    if digit_targets is not None:
        digits = sorted(np.unique(digit_targets).tolist())
        # temporarily swap targets/group_ids inside _strip via closure tweak
        _orig_targets = targets
        # use a local version that reads digit_targets
        def _strip_digit(ax):
            for xi, d in enumerate(digits):
                mask = digit_targets == d
                y = coefs[mask]
                x = np.full(len(y), xi) + np.random.default_rng(0).uniform(-0.2, 0.2, len(y))
                ax.scatter(x, y, s=6, alpha=0.4, color=cmap(xi / max(len(digits) - 1, 1)))
                ax.plot([xi - 0.3, xi + 0.3], [y.mean(), y.mean()], color='k', lw=1.5)
            ax.set_xticks(range(len(digits)))
            ax.set_xticklabels([str(d) for d in digits], fontsize=7)
            ax.set(title='by digit', ylabel='coefficient')
        _strip_digit(axes[3])

    fig.suptitle(f'component {fi}')
    fig.tight_layout()
    return fig


def plot_neuron_nmf_scatter(img_factors, targets, classes,
                             fi=0, fj=1,
                             class_names=None, digit_targets=None):
    """
    Scatter of NMF component fi vs fj coefficients.

    Left panel coloured by task class; right panel coloured by digit
    (only shown when digit_targets is provided).

    Parameters
    ----------
    img_factors   : (n_samples, K)
    targets       : (n_samples,) task class labels
    classes       : list of class ids
    fi, fj        : component indices to plot on x and y axes
    class_names   : optional dict {cl: name}
    digit_targets : optional (n_samples,) digit labels
    """
    n_panels = 2 if digit_targets is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5),
                             squeeze=False)
    cmap = matplotlib.colormaps['tab10']
    x = img_factors[:, fi]
    y = img_factors[:, fj]

    # left: task class
    ax = axes[0, 0]
    for xi, cl in enumerate(classes):
        mask = targets == cl
        label = class_names.get(cl, str(cl)) if class_names else str(cl)
        ax.scatter(x[mask], y[mask], s=8, alpha=0.5,
                   color=cmap(xi / max(len(classes) - 1, 1)), label=label)
    ax.set(xlabel=f'component {fi}', ylabel=f'component {fj}',
           title='task class')
    ax.legend(markerscale=2)

    # right: digit
    if digit_targets is not None:
        digits = sorted(np.unique(digit_targets).tolist())
        ax2 = axes[0, 1]
        for xi, d in enumerate(digits):
            mask = digit_targets == d
            ax2.scatter(x[mask], y[mask], s=8, alpha=0.5,
                        color=cmap(xi / max(len(digits) - 1, 1)), label=str(d))
        ax2.set(xlabel=f'component {fi}', ylabel=f'component {fj}',
                title='digit')
        ax2.legend(markerscale=2, ncol=2, fontsize=7)

    fig.tight_layout()
    return fig


def plot_scaffold_graph(loading, edge_matrices, layer_sizes, neg_edge_matrices=None):
    """
    Scaffold graph with precomputed edge matrices.

    Positive edges (from positive weight×input products) are drawn in PuRd.
    Negative edges (from negative weight×input products) are drawn in Blues.

    Parameters
    ----------
    loading           : (n_neurons_total,) non-negative node loadings
    edge_matrices     : list of (n_target, n_source) arrays, one per layer boundary
    layer_sizes       : list of int — neurons per layer
    neg_edge_matrices : optional list of (n_target, n_source) arrays for negative edges
    """
    n = sum(layer_sizes)
    node_pos = np.zeros((n, 2))
    A_pos = np.zeros((n, n))
    A_neg = np.zeros((n, n))
    height = max(layer_sizes)
    tot = 0
    tots = []

    # Normalize each layer's loadings independently so all nodes are coloured
    # regardless of scale differences between layers (e.g. raw activations vs
    # λ-weighted NMF coefficients which can be 100× larger).
    norm_load = np.zeros_like(loading, dtype=float)
    layer_start = 0
    for lsz in layer_sizes:
        seg = loading[layer_start:layer_start + lsz]
        norm_load[layer_start:layer_start + lsz] = seg / (seg.max() + 1e-12)
        layer_start += lsz

    for lii, lsz in enumerate(layer_sizes):
        gap = height / lsz
        node_pos[tot:tot + lsz, 1] = np.arange(height - gap / 2, 0, -gap)
        node_pos[tot:tot + lsz, 0] = lii
        if lii > 0 and lii - 1 < len(edge_matrices):
            E = edge_matrices[lii - 1]
            A_pos[tots[-1]:tot, tot:tot + lsz] = E.T
            if neg_edge_matrices is not None and lii - 1 < len(neg_edge_matrices):
                A_neg[tots[-1]:tot, tot:tot + lsz] = neg_edge_matrices[lii - 1].T
        tots.append(tot)
        tot += lsz

    # Normalize each layer boundary independently so L1→L2 and L2→L3 edges
    # are on comparable scales rather than dominated by the larger boundary.
    for bi in range(len(tots) - 1):
        rs, re = tots[bi], tots[bi + 1]
        cs, ce = tots[bi + 1], (tots[bi + 2] if bi + 2 < len(tots) else n)
        block_max = max(A_pos[rs:re, cs:ce].max(), A_neg[rs:re, cs:ce].max())
        if block_max > 0:
            scale = 3.0 / block_max
            A_pos[rs:re, cs:ce] *= scale
            A_neg[rs:re, cs:ce] *= scale

    node_cmap = matplotlib.colormaps['PuRd']
    node_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    node_colors = [node_cmap(node_norm(v)) for v in norm_load]
    nodelist = list(range(n))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Positive edges (PuRd)
    G_pos = nx.Graph(A_pos)
    pos_widths = nx.get_edge_attributes(G_pos, 'weight')
    if pos_widths:
        pw = list(pos_widths.values())
        ecmap = matplotlib.colormaps['PuRd']
        enorm = matplotlib.colors.Normalize(vmin=min(pw), vmax=max(pw))
        nx.draw_networkx_edges(G_pos, node_pos, ax=ax,
                               edgelist=list(pos_widths.keys()),
                               width=pw,
                               edge_color=[ecmap(enorm(w)) for w in pw],
                               alpha=0.8)

    # Negative edges (Blues)
    if neg_edge_matrices is not None:
        G_neg = nx.Graph(A_neg)
        neg_widths = nx.get_edge_attributes(G_neg, 'weight')
        if neg_widths:
            nw = list(neg_widths.values())
            ncmap = matplotlib.colormaps['Blues']
            nnorm = matplotlib.colors.Normalize(vmin=min(nw), vmax=max(nw))
            nx.draw_networkx_edges(G_neg, node_pos, ax=ax,
                                   edgelist=list(neg_widths.keys()),
                                   width=nw,
                                   edge_color=[ncmap(nnorm(w)) for w in nw],
                                   alpha=0.8)

    # Nodes and labels drawn last so they appear on top of edges
    nx.draw_networkx_nodes(G_pos, node_pos, ax=ax, nodelist=nodelist, node_size=150,
                           node_color=node_colors, linewidths=0.5, edgecolors='k', alpha=1)
    nx.draw_networkx_labels(G_pos, pos=node_pos, ax=ax,
                            labels={nd: nd for nd in nodelist},
                            font_color='white', font_size=7)
    return fig


def plot_factor_graph(fi, neural_factors, model_layers, layer_sizes, linear_layer_indices,
                      img_factors=None, top_pct=0.05):
    """
    Visualize one NMF component as a weighted network graph.

    Nodes represent neurons; edge thickness/colour reflects weight × neural-factor loading.
    Only positive edge weights are shown.

    Parameters
    ----------
    fi                    : component index
    neural_factors        : (n_neurons, n_components)
    model_layers          : nn.Sequential (model.layers)
    layer_sizes           : list of int — neurons per layer
    linear_layer_indices  : list of int — Sequential indices of Linear layers
    img_factors           : (n_samples, n_components) optional — when provided, the top
                            top_pct stimuli by img_factors[:, fi] are selected and their
                            full NMF reconstruction is averaged to give realistic node
                            activations (instead of the raw factor vector).
    top_pct               : float — fraction of stimuli to select (default 0.05 = 5%)
    """
    n = len(neural_factors)

    if img_factors is not None:
        n_top = max(1, int(np.ceil(img_factors.shape[0] * top_pct)))
        top_idx = np.argsort(img_factors[:, fi])[-n_top:]
        node_vals = (img_factors[top_idx, :] @ neural_factors.T).mean(axis=0)
    else:
        node_vals = neural_factors[:, fi]

    normalized_nf = node_vals / (node_vals.max() + 1e-12)

    pos = np.zeros((n, 2))
    A = np.zeros((n, n))
    height = max(layer_sizes)
    tot = 0
    tots = []

    for lii, lsz in enumerate(layer_sizes):
        gap = height / lsz
        pos[tot:tot + lsz, 1] = np.arange(height - gap / 2, 0, -gap)
        pos[tot:tot + lsz, 0] = lii
        if lii > 0:
            ws = model_layers[linear_layer_indices[lii]].weight.detach().numpy()
            edge_ws = (ws * normalized_nf[tots[lii - 1]:tot]).T
            edge_ws[edge_ws < 0] = 0
            A[tots[lii - 1]:tot, tot:tot + lsz] = edge_ws
        tots.append(tot)
        tot += lsz

    A /= A.max() / 3
    G = nx.Graph(A)
    widths = nx.get_edge_attributes(G, 'weight')

    edge_cmap = matplotlib.colormaps['PuRd']
    w_vals = list(widths.values())
    edge_norm = matplotlib.colors.Normalize(vmin=min(w_vals), vmax=max(w_vals))
    edge_colors = [edge_cmap(edge_norm(w)) for w in w_vals]

    node_cmap = matplotlib.colormaps['PuRd']
    node_norm = matplotlib.colors.Normalize(vmin=normalized_nf.min(), vmax=normalized_nf.max())
    node_colors = [node_cmap(node_norm(v)) for v in normalized_nf]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    nodelist = list(G.nodes())
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodelist, node_size=150,
                           node_color=node_colors, linewidths=0.5, edgecolors='k', alpha=1)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(widths.keys()),
                           width=w_vals, edge_color=edge_colors, alpha=1)
    nx.draw_networkx_labels(G, pos=pos, ax=ax,
                            labels={n: n for n in nodelist},
                            font_color='white', font_size=7)
    ax.set_title(f'component {fi}')
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Standardised visualisation utilities (Plots 1–7)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Private helpers ───────────────────────────────────────────────────────────

def _display_normalize(images):
    """Scale image array to [0, 1] float32, handling CHW and HW formats."""
    arr = np.asarray(images, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)


def _digit_colors():
    """Return {digit: rgba} — Blues for even digits, Oranges for odd digits."""
    blues   = matplotlib.colormaps['Blues'](np.linspace(0.4, 0.9, 5))
    oranges = matplotlib.colormaps['Oranges'](np.linspace(0.4, 0.9, 5))
    c = {}
    for i, d in enumerate([0, 2, 4, 6, 8]):
        c[d] = blues[i]
    for i, d in enumerate([1, 3, 5, 7, 9]):
        c[d] = oranges[i]
    return c


def _class_color_map(classes):
    """Return {class_id: rgba} using tab10."""
    cmap = matplotlib.colormaps['tab10']
    n = max(len(classes) - 1, 1)
    return {c: cmap(i / n) for i, c in enumerate(classes)}


def _show_image(ax, img):
    """Display a single image on ax, handling CHW→HWC conversion."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]
    img = np.clip(img, 0, 1)
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])


def _cname_fn(class_names):
    """Return a callable {class_id → str} from a list or dict."""
    if isinstance(class_names, dict):
        return lambda c: class_names.get(int(c), str(c))
    return lambda c: str(class_names[int(c)]) if int(c) < len(class_names) else str(c)


def _neural_heatmap_array(node, k):
    """Reshape neural_factors[:, k] to 2D (n_out, n_in_flat) using W shape."""
    nf = node['neural_factors'][:, k]
    W  = node['W']
    ltype = node.get('layer_type', 'fc')
    if ltype == 'conv':
        C_out = W.shape[0]
        return nf.reshape(C_out, nf.shape[0] // C_out)
    else:
        n_out = W.shape[0]
        return nf.reshape(n_out, nf.shape[0] // n_out)


# ── Plot 1 — Factor overview panels ──────────────────────────────────────────

def plot_factor_overview_panel(node, images, targets, class_names,
                                digit_targets=None, n_top=5):
    """
    Return one figure per factor k in `node`.

    Each figure has two rows:
      Row 0 – lambda bar chart (factor k highlighted in red).
      Row 1 – [neural heatmap | stim distribution | weighted-avg image | top-n stimuli]

    Parameters
    ----------
    node          : BFT tree node dict (img_factors, neural_factors, lambdas, W)
    images        : (N, *img_shape) raw input images (any value range)
    targets       : (N,) int task-class labels
    class_names   : list[str] or dict {int: str}
    digit_targets : (N,) int digit labels; enables per-digit colouring for even/odd tasks
    n_top         : number of top stimuli to display (default 5)

    Returns
    -------
    list[Figure]  — one figure per factor k
    """
    img_factors    = node['img_factors']     # (N, K)
    neural_factors = node['neural_factors']  # (n_flat, K)
    lambdas        = node['lambdas']         # (K,)
    K = img_factors.shape[1]
    layer_name = node.get('layer_name', f"L{node.get('layer_idx', '?')}")

    imgs = _display_normalize(images)
    cname = _cname_fn(class_names)

    # Colour group setup
    if digit_targets is not None:
        dcmap = _digit_colors()
        group_ids    = sorted(np.unique(digit_targets).tolist())
        group_labels = {d: f"{'E' if d % 2 == 0 else 'O'}{d}" for d in group_ids}
        group_colors = {d: dcmap.get(int(d), (0.5, 0.5, 0.5, 1.0)) for d in group_ids}
        def _group_vals(k_):
            return {d: img_factors[np.asarray(digit_targets) == d, k_] for d in group_ids}
    else:
        classes = sorted(np.unique(targets).tolist())
        ccmap   = _class_color_map(classes)
        group_ids    = classes
        group_labels = {c: cname(c) for c in classes}
        group_colors = ccmap
        def _group_vals(k_):
            return {c: img_factors[np.asarray(targets) == c, k_] for c in classes}

    figs = []
    for k in range(K):
        lam_frac = float(lambdas[k]) / (float(lambdas.sum()) + 1e-12)
        coefs = img_factors[:, k]

        # Layout: row 0 height=1 (lambda), row 1 height=4 (content)
        n_cols = n_top + 3  # heatmap | histogram | wavg | n_top stimuli
        fig = plt.figure(figsize=(2.6 * n_cols, 6.5))
        gs  = fig.add_gridspec(2, n_cols, height_ratios=[1, 2],
                               hspace=0.4, wspace=0.35)

        # ── Row 0: lambda bar ────────────────────────────────────────────────
        ax_lam = fig.add_subplot(gs[0, :])
        bar_colors = ['#e15759' if ki == k else '#aec7e8' for ki in range(K)]
        ax_lam.bar(range(K), lambdas, color=bar_colors, edgecolor='k', linewidth=0.5)
        ax_lam.set_xticks(range(K))
        ax_lam.set_xticklabels([f'k={ki}' for ki in range(K)], fontsize=8)
        ax_lam.set_ylabel('λ', fontsize=9)
        ax_lam.set_title(
            f'{layer_name}  ·  Factor k={k}  ({lam_frac:.1%} of total λ)',
            fontsize=10, fontweight='bold',
        )

        # ── Row 1, col 0: neural factor heatmap ─────────────────────────────
        ax_hm = fig.add_subplot(gs[1, 0])
        hm = _neural_heatmap_array(node, k)
        vmax = np.abs(hm).max() or 1.0
        ax_hm.imshow(hm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax,
                     interpolation='nearest')
        ax_hm.set_xlabel('in', fontsize=8)
        ax_hm.set_ylabel('out', fontsize=8)
        ax_hm.set_title('connection map', fontsize=9)
        ax_hm.tick_params(labelsize=6)

        # ── Row 1, col 1: stimulus weight distribution ───────────────────────
        ax_hist = fig.add_subplot(gs[1, 1])
        vals_by_group = _group_vals(k)
        for gid in group_ids:
            v = vals_by_group[gid]
            if len(v) == 0:
                continue
            ax_hist.hist(v, bins=25, alpha=0.55,
                         color=group_colors[gid],
                         label=group_labels[gid],
                         density=True)
        ax_hist.set_xlabel('img_factor weight', fontsize=8)
        ax_hist.set_ylabel('density', fontsize=8)
        ax_hist.set_title('stimulus weights', fontsize=9)
        ax_hist.legend(fontsize=6, ncol=2)
        ax_hist.tick_params(labelsize=7)

        # ── Row 1, col 2: weighted-average image ─────────────────────────────
        ax_wavg = fig.add_subplot(gs[1, 2])
        flat = imgs.reshape(len(imgs), -1)
        wavg = (coefs[:, None] * flat).sum(0) / (coefs.sum() + 1e-12)
        wavg_img = _display_normalize(wavg.reshape(imgs.shape[1:]))
        _show_image(ax_wavg, wavg_img)
        ax_wavg.set_anchor('N')
        ax_wavg.set_title('weighted avg', fontsize=9)

        # ── Row 1, cols 3+: top stimuli ──────────────────────────────────────
        top_idx = np.argsort(coefs)[::-1][:n_top]
        for ti, si in enumerate(top_idx):
            ax_t = fig.add_subplot(gs[1, 3 + ti])
            _show_image(ax_t, imgs[si])
            ax_t.set_anchor('N')
            lbl = cname(targets[si])
            if digit_targets is not None:
                lbl += f'\nd={int(digit_targets[si])}'
            ax_t.set_title(f'{lbl}\n{coefs[si]:.3f}', fontsize=7)

        figs.append(fig)

    return figs


# ── Plot 2 — Input-layer spatial factors ──────────────────────────────────────

def plot_input_layer_factors(node, images, arch='fc', image_shape=None):
    """
    Visualise the input-layer factors in pixel / spatial / attention space.

    Parameters
    ----------
    node         : BFT leaf node dict (neural_factors, W, img_factors)
    images       : (N, *img_shape) input images
    arch         : 'fc' | 'conv_rgb' | 'attn'
    image_shape  : tuple — spatial shape for FC pixel RFs (inferred if None)

    Returns
    -------
    list[Figure]  — one figure per factor k
    """
    neural_factors = node['neural_factors']  # (n_flat, K)
    img_factors    = node['img_factors']     # (N, K)
    K = neural_factors.shape[1]
    W = node['W']
    layer_name = node.get('layer_name', f"L{node.get('layer_idx', '?')}")
    imgs = _display_normalize(images)

    if image_shape is None:
        image_shape = images.shape[1:]

    figs = []

    if arch == 'fc':
        n_out, n_in = W.shape
        for k in range(K):
            nf = neural_factors[:, k].reshape(n_out, n_in)
            fig, axes = plt.subplots(1, n_out, figsize=(max(6, 2.2 * n_out), 2.8),
                                     squeeze=False)
            axes = axes[0]
            for ni in range(n_out):
                try:
                    rf = nf[ni].reshape(image_shape)
                except ValueError:
                    side = int(np.ceil(n_in ** 0.5))
                    rf = np.pad(nf[ni], (0, side * side - n_in)).reshape(side, side)
                if rf.ndim == 3 and rf.shape[0] in (1, 3):
                    rf = np.moveaxis(rf, 0, -1)
                    rf = np.clip((rf - rf.min()) / (rf.max() - rf.min() + 1e-12), 0, 1)
                    axes[ni].imshow(rf)
                else:
                    absmax = np.abs(rf).max() or 1.0
                    axes[ni].imshow(rf.squeeze(), cmap='seismic',
                                    vmin=-absmax, vmax=absmax)
                axes[ni].set_title(f'out {ni}', fontsize=8)
                axes[ni].axis('off')
            fig.suptitle(f'{layer_name} — Factor k={k} pixel RFs',
                         fontsize=10, fontweight='bold')
            fig.tight_layout()
            figs.append(fig)

    elif arch == 'conv_rgb':
        C_out, C_in, k_h, k_w = W.shape
        for k in range(K):
            nf = neural_factors[:, k].reshape(C_out, C_in, k_h, k_w)
            ncols = min(8, C_out)
            nrows = (C_out + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(2.2 * ncols, 2.5 * nrows),
                                     squeeze=False)
            axes_flat = axes.flatten()
            for oc in range(C_out):
                ax  = axes_flat[oc]
                kern = nf[oc]  # (C_in, k_h, k_w)
                if C_in == 3:
                    kern_rgb = np.moveaxis(kern, 0, -1)
                    lo, hi = kern_rgb.min(), kern_rgb.max()
                    ax.imshow(np.clip((kern_rgb - lo) / (hi - lo + 1e-12), 0, 1))
                else:
                    absmax = np.abs(kern).max() or 1.0
                    ax.imshow(kern.mean(0), cmap='seismic',
                              vmin=-absmax, vmax=absmax)
                ax.set_title(f'ch{oc}', fontsize=7)
                ax.axis('off')
            for ax in axes_flat[C_out:]:
                ax.axis('off')
            fig.suptitle(f'{layer_name} — Factor k={k} conv kernels',
                         fontsize=10, fontweight='bold')
            fig.tight_layout()
            figs.append(fig)

    elif arch == 'attn':
        attn_w = node.get('attn_weights', None)  # (N, T) if stored
        H_img  = images.shape[-2] if images.ndim >= 3 else 28
        W_img  = images.shape[-1] if images.ndim >= 2 else 28
        for k in range(K):
            coefs   = img_factors[:, k]
            n_show  = 5
            top_idx = np.argsort(coefs)[::-1][:n_show]
            fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5.5))
            for ti, si in enumerate(top_idx):
                _show_image(axes[0, ti], imgs[si])
                axes[0, ti].set_title(f'{coefs[si]:.3f}', fontsize=7)
                if attn_w is not None:
                    tokens = attn_w[si, 1:]
                    n_patches = len(tokens)
                    side = int(np.ceil(n_patches ** 0.5))
                    attn_map = np.pad(tokens, (0, side * side - n_patches)).reshape(side, side)
                    from PIL import Image as _PILImage
                    attn_r = np.array(
                        _PILImage.fromarray(
                            (attn_map / (attn_map.max() + 1e-12) * 255).astype(np.uint8)
                        ).resize((W_img, H_img), _PILImage.BILINEAR)
                    ) / 255.0
                    _show_image(axes[0, ti], imgs[si])
                    axes[1, ti].imshow(attn_r, cmap='hot')
                    axes[1, ti].set_title('attn', fontsize=7)
                else:
                    axes[1, ti].text(0.5, 0.5, 'no attn', ha='center', va='center',
                                     transform=axes[1, ti].transAxes, fontsize=8)
                axes[1, ti].axis('off')
            fig.suptitle(f'{layer_name} — Factor k={k} attention spatial patterns',
                         fontsize=10, fontweight='bold')
            fig.tight_layout()
            figs.append(fig)

    return figs


# ── Plot 4 — Per-factor image gallery ─────────────────────────────────────────

def plot_factor_gallery(node, images, targets, class_names, k, n=10):
    """
    2-row image gallery for factor k: top-n most active and bottom-n least active.

    Parameters
    ----------
    node        : BFT tree node dict
    images      : (N, *img_shape) input images
    targets     : (N,) int task-class labels
    class_names : list[str] or dict {int: str}
    k           : factor index
    n           : number of images per row (default 10)

    Returns
    -------
    Figure
    """
    coefs = node['img_factors'][:, k]
    layer_name = node.get('layer_name', f"L{node.get('layer_idx', '?')}")
    imgs  = _display_normalize(images)
    cname = _cname_fn(class_names)
    n     = min(n, len(coefs))

    top_idx = np.argsort(coefs)[::-1][:n]
    bot_idx = np.argsort(coefs)[:n]

    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.2))
    if n == 1:
        axes = axes[:, None]

    for ti, si in enumerate(top_idx):
        _show_image(axes[0, ti], imgs[si])
        axes[0, ti].set_title(f'{cname(targets[si])}\n{coefs[si]:.3f}', fontsize=7)

    for ti, si in enumerate(bot_idx):
        _show_image(axes[1, ti], imgs[si])
        axes[1, ti].set_title(f'{cname(targets[si])}\n{coefs[si]:.3f}', fontsize=7)

    axes[0, 0].set_ylabel('most active', fontsize=9)
    axes[1, 0].set_ylabel('least active', fontsize=9)
    fig.suptitle(f'{layer_name}  ·  Factor k={k}  —  image gallery',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Plot 6 — Pruning results ──────────────────────────────────────────────────

def plot_pruning_results(pruning_data, class_names, methods, fractions,
                          method_colors=None, method_labels=None,
                          pruning_stds=None):
    """
    One figure per class: left panel = target class accuracy, right = mean bystander.

    Parameters
    ----------
    pruning_data : dict {class_d: {method: {fraction: {class_id: accuracy}}}}
                   Include fraction=0 for the baseline (no-ablation) point.
    class_names  : list[str] or dict {int: str}
    methods      : list[str] — which methods to plot
    fractions    : list[float] — x-axis values (must include 0)
    method_colors : optional dict {method: color}
    method_labels : optional dict {method: display label}
    pruning_stds  : optional dict {class_d: {method: {fraction: {class_id: std}}}}

    Returns
    -------
    list[Figure]  — one figure per class
    """
    _default_colors = {
        'bft_top':    '#e15759',
        'bft_bottom': '#f28e2b',
        'magnitude':  '#76b7b2',
        'random':     '#333333',
        'algo_top':   '#e15759',
        'algo_bottom': '#f28e2b',
    }
    _default_labels = {
        'bft_top':    'BFT most important',
        'bft_bottom': 'BFT least important',
        'magnitude':  'Magnitude',
        'random':     'Random',
        'algo_top':   'BFT most important',
        'algo_bottom': 'BFT least important',
    }
    mc = {**_default_colors, **(method_colors or {})}
    ml = {**_default_labels, **(method_labels or {})}

    cname  = _cname_fn(class_names)
    classes = sorted(pruning_data.keys())
    fracs   = sorted(fractions)

    figs = []
    for d in classes:
        others = [c for c in classes if c != d]
        fig, (ax_tgt, ax_bys) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

        for method in methods:
            curve = pruning_data[d].get(method, {})
            stds  = (pruning_stds or {}).get(d, {}).get(method, {})
            color = mc.get(method, None)
            label = ml.get(method, method)
            lstyle = '--' if method in ('bft_bottom', 'algo_bottom') else '-'
            dashes = (4, 2) if method == 'random' else None

            xs_tgt, ys_tgt, ye_tgt = [], [], []
            xs_bys, ys_bys, ye_bys = [], [], []

            for f in fracs:
                acc = curve.get(f, {})
                if not isinstance(acc, dict):
                    continue
                if d in acc:
                    xs_tgt.append(f)
                    ys_tgt.append(acc[d])
                    std_d = stds.get(f, {}).get(d, 0.0)
                    ye_tgt.append(std_d)
                if others:
                    bys_vals = [acc[c] for c in others if c in acc]
                    if bys_vals:
                        bys_std_vals = [stds.get(f, {}).get(c, 0.0) for c in others if c in acc]
                        xs_bys.append(f)
                        ys_bys.append(float(np.mean(bys_vals)))
                        ye_bys.append(float(np.mean(bys_std_vals)))

            kw = dict(color=color, label=label, linestyle=lstyle,
                      marker='o', markersize=4, linewidth=1.5)
            if dashes:
                kw['dashes'] = dashes

            if xs_tgt:
                ax_tgt.plot(xs_tgt, ys_tgt, **kw)
                if any(e > 0 for e in ye_tgt):
                    ax_tgt.fill_between(xs_tgt,
                                        [y - e for y, e in zip(ys_tgt, ye_tgt)],
                                        [y + e for y, e in zip(ys_tgt, ye_tgt)],
                                        alpha=0.15, color=color)
            if xs_bys:
                kw2 = {**kw}
                kw2.pop('label', None)
                ax_bys.plot(xs_bys, ys_bys, **kw2)
                if any(e > 0 for e in ye_bys):
                    ax_bys.fill_between(xs_bys,
                                        [y - e for y, e in zip(ys_bys, ye_bys)],
                                        [y + e for y, e in zip(ys_bys, ye_bys)],
                                        alpha=0.15, color=color)

        for ax, ttl in [(ax_tgt, f'Target: {cname(d)}'),
                        (ax_bys, 'Mean bystander accuracy')]:
            ax.set_xlabel('pruned fraction', fontsize=9)
            ax.set_ylabel('accuracy', fontsize=9)
            ax.set_title(ttl, fontsize=10)
            ax.set_xlim(left=-0.01)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)

        ax_tgt.legend(fontsize=8)
        fig.suptitle(f'Pruning results — class "{cname(d)}"',
                     fontsize=11, fontweight='bold', y=1.01)
        fig.tight_layout()
        figs.append(fig)

    return figs


# ── Plot 7 — Embedding comparison (MDS / PCA) ─────────────────────────────────

def plot_embedding_comparison(fingerprints, activations_last, labels, class_names,
                               digit_targets=None, condition_labels=None, title=''):
    """
    3-panel figure: MDS(fingerprints) | PCA(fingerprints) | PCA(last-layer activations).

    Panel 0: MDS on pairwise cosine distance of fingerprints (current default).
    Panel 1: PCA on fingerprints — same data, different projection.
    Panel 2: PCA on raw last-layer activations — different representation.

    This design isolates the projection-method effect (0 vs 1) from the
    representation effect (1 vs 2), making comparisons fair.

    Parameters
    ----------
    fingerprints      : (N, d_fp) fingerprint matrix (from extract_fingerprint_matrix)
    activations_last  : (N, d_act) last-layer activation matrix
    labels            : (N,) int task-class labels
    class_names       : list[str] or dict {int: str}
    digit_targets     : (N,) int digit labels; enables per-digit colouring
    condition_labels  : (N,) str/int — condition per sample (e.g. 'ID', 'near-OOD');
                        if provided, uses different markers per condition
    title             : optional suptitle

    Returns
    -------
    Figure
    """
    from sklearn.preprocessing import normalize as _sk_normalize

    cname = _cname_fn(class_names)

    # MDS on cosine distance
    fp_norm = _sk_normalize(fingerprints, norm='l2')
    dist_mat = np.clip(1.0 - fp_norm @ fp_norm.T, 0, None)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
              n_init=4, max_iter=300)
    emb_mds = mds.fit_transform(dist_mat)

    # PCA on fingerprints
    pca_fp  = PCA(n_components=2, random_state=42)
    emb_pca_fp = pca_fp.fit_transform(fingerprints)
    var_fp  = pca_fp.explained_variance_ratio_[:2].sum()

    # PCA on last-layer activations
    pca_act = PCA(n_components=2, random_state=42)
    emb_pca_act = pca_act.fit_transform(activations_last)
    var_act = pca_act.explained_variance_ratio_[:2].sum()

    # Colour setup
    unique_labels = sorted(np.unique(labels).tolist())
    if digit_targets is not None:
        dcmap = _digit_colors()
        colors = np.array([dcmap.get(int(d), (0.5, 0.5, 0.5, 1.0))
                           for d in digit_targets])
    else:
        ccmap  = _class_color_map(unique_labels)
        colors = np.array([ccmap[int(l)] for l in labels])

    # Marker setup for conditions
    marker_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    if condition_labels is not None:
        cond_arr  = np.asarray(condition_labels)
        unique_c  = list(dict.fromkeys(condition_labels))
        cond2mark = {c: marker_list[i % len(marker_list)] for i, c in enumerate(unique_c)}
    else:
        cond_arr = None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panel_info = [
        (emb_mds,     'MDS (fingerprints)\ncosine distance'),
        (emb_pca_fp,  f'PCA (fingerprints)\n{100*var_fp:.1f}% variance'),
        (emb_pca_act, f'PCA (last-layer activations)\n{100*var_act:.1f}% variance'),
    ]

    for ax, (emb, ttl) in zip(axes, panel_info):
        if cond_arr is not None:
            for cond in unique_c:
                mask = (cond_arr == cond)
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=colors[mask], marker=cond2mark[cond],
                           s=18, alpha=0.7, label=str(cond), linewidths=0)
        else:
            ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=18, alpha=0.7, linewidths=0)
        ax.set(title=ttl, xlabel='dim 1', ylabel='dim 2')
        ax.tick_params(labelsize=7)

    # Class / digit legend on first axis
    if digit_targets is not None:
        dcmap_ = _digit_colors()
        for d in sorted(dcmap_.keys()):
            axes[0].scatter([], [], color=dcmap_[d],
                            label=f"{'E' if d % 2 == 0 else 'O'}{d}", s=14)
    else:
        ccmap_ = _class_color_map(unique_labels)
        for c in unique_labels:
            axes[0].scatter([], [], color=ccmap_[c], label=cname(c), s=14)
    axes[0].legend(fontsize=6, ncol=2, markerscale=2)

    # Condition legend on last axis
    if cond_arr is not None:
        axes[2].legend(fontsize=7, markerscale=1.5)

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig
