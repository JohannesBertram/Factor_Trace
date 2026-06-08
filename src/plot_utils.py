import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def plot_scaffold_graph(loading, edge_matrices, layer_sizes, neg_edge_matrices=None, *,
                        normalize='global', magnitude_transform=None,
                        max_nodes_per_layer=None, figsize=(6, 5), node_size=150,
                        node_cmap='PuRd', pos_edge_cmap='PuRd', neg_edge_cmap='Blues',
                        show_labels=True, show_neg=True, title=None, ax=None):
    """
    Scaffold graph with precomputed edge matrices.

    Positive edges are drawn in pos_edge_cmap; negative (inhibitory) edges in neg_edge_cmap.

    Parameters
    ----------
    loading              : (n_neurons_total,) non-negative node loadings
    edge_matrices        : list of (n_target, n_source) arrays, one per layer boundary
    layer_sizes          : list of int — neurons per layer
    neg_edge_matrices    : optional list of (n_target, n_source) arrays for negative edges
    normalize            : 'global' (default) — single max across all nodes/edges preserving
                           absolute scale; 'layer' — per-layer max-norm for within-layer contrast
    magnitude_transform  : None | 'sqrt' | 'cbrt' | 'log' — monotone squeeze applied after
                           normalization for better contrast with sparse distributions
    max_nodes_per_layer  : int | None — if set, only the top-K neurons per layer by loading
                           are shown; others are zeroed (layout preserved)
    figsize              : (width, height) in inches
    node_size            : marker size for nodes
    node_cmap            : matplotlib colormap name for node colors
    pos_edge_cmap        : matplotlib colormap name for positive edges
    neg_edge_cmap        : matplotlib colormap name for negative (inhibitory) edges
    show_labels          : whether to draw layer-relative neuron index labels
    show_neg             : whether to draw negative edges
    title                : optional axes title
    ax                   : existing Axes to draw into; if None a new figure is created
    """
    n = sum(layer_sizes)
    node_pos = np.zeros((n, 2))
    A_pos = np.zeros((n, n))
    A_neg = np.zeros((n, n))
    height = max(layer_sizes)
    tot = 0
    tots = []

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

    # Optionally hide low-loading neurons beyond top-K per layer
    loading = np.array(loading, dtype=float)
    if max_nodes_per_layer is not None:
        layer_start = 0
        for li, lsz in enumerate(layer_sizes):
            seg = loading[layer_start:layer_start + lsz]
            if lsz > max_nodes_per_layer:
                threshold = np.sort(seg)[-max_nodes_per_layer]
                mask_off = np.where(seg < threshold)[0] + layer_start
                loading[mask_off] = 0.0
                for idx in mask_off:
                    A_pos[idx, :] = 0.0
                    A_pos[:, idx] = 0.0
                    A_neg[idx, :] = 0.0
                    A_neg[:, idx] = 0.0
            layer_start += lsz

    # Normalize node loadings
    if normalize == 'global':
        norm_load = loading / (loading.max() + 1e-12)
    else:  # 'layer'
        norm_load = np.zeros_like(loading, dtype=float)
        layer_start = 0
        for lsz in layer_sizes:
            seg = loading[layer_start:layer_start + lsz]
            norm_load[layer_start:layer_start + lsz] = seg / (seg.max() + 1e-12)
            layer_start += lsz

    # Normalize edges
    if normalize == 'global':
        global_max = max(A_pos.max(), A_neg.max())
        if global_max > 0:
            A_pos *= 3.0 / global_max
            A_neg *= 3.0 / global_max
    else:  # 'layer' — per boundary
        for bi in range(len(tots) - 1):
            rs, re = tots[bi], tots[bi + 1]
            cs, ce = tots[bi + 1], (tots[bi + 2] if bi + 2 < len(tots) else n)
            block_max = max(A_pos[rs:re, cs:ce].max(), A_neg[rs:re, cs:ce].max())
            if block_max > 0:
                scale = 3.0 / block_max
                A_pos[rs:re, cs:ce] *= scale
                A_neg[rs:re, cs:ce] *= scale

    # Apply optional non-linear monotone squeeze
    def _transform(x):
        if magnitude_transform is None:
            return x
        if magnitude_transform == 'sqrt':
            return np.sqrt(np.clip(x, 0, None))
        if magnitude_transform == 'cbrt':
            return np.cbrt(x)
        if magnitude_transform == 'log':
            return np.log1p(np.clip(x, 0, None) * 9) / np.log1p(9)
        raise ValueError(f"Unknown magnitude_transform: {magnitude_transform!r}")

    norm_load = _transform(norm_load)
    A_pos = _transform(A_pos)
    A_neg = _transform(A_neg)

    ncmap_nodes = matplotlib.colormaps[node_cmap]
    node_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    node_colors = [ncmap_nodes(node_norm(v)) for v in norm_load]
    nodelist = list(range(n))

    # Layer-relative labels (e.g. neuron 4 in layer 2, not global index 12)
    labels = {}
    layer_start = 0
    for lsz in layer_sizes:
        for j in range(lsz):
            labels[layer_start + j] = j
        layer_start += lsz

    return_fig = ax is None
    if return_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Positive edges
    G_pos = nx.Graph(A_pos)
    pos_widths = nx.get_edge_attributes(G_pos, 'weight')
    if pos_widths:
        pw = list(pos_widths.values())
        ecmap = matplotlib.colormaps[pos_edge_cmap]
        enorm = matplotlib.colors.Normalize(vmin=0, vmax=max(pw))
        nx.draw_networkx_edges(G_pos, node_pos, ax=ax,
                               edgelist=list(pos_widths.keys()),
                               width=pw,
                               edge_color=[ecmap(enorm(w)) for w in pw],
                               alpha=0.8)

    # Negative (inhibitory) edges
    if show_neg and neg_edge_matrices is not None:
        G_neg = nx.Graph(A_neg)
        neg_widths = nx.get_edge_attributes(G_neg, 'weight')
        if neg_widths:
            nw = list(neg_widths.values())
            ncmap_edges = matplotlib.colormaps[neg_edge_cmap]
            nnorm = matplotlib.colors.Normalize(vmin=0, vmax=max(nw))
            nx.draw_networkx_edges(G_neg, node_pos, ax=ax,
                                   edgelist=list(neg_widths.keys()),
                                   width=nw,
                                   edge_color=[ncmap_edges(nnorm(w)) for w in nw],
                                   alpha=0.8)

    # Nodes and labels drawn last so they appear on top of edges
    nx.draw_networkx_nodes(G_pos, node_pos, ax=ax, nodelist=nodelist,
                           node_size=node_size,
                           node_color=node_colors, linewidths=0.5, edgecolors='k', alpha=1)
    if show_labels:
        nx.draw_networkx_labels(G_pos, pos=node_pos, ax=ax,
                                labels=labels, font_color='white', font_size=7)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)

    if return_fig:
        return fig
    return None



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
    """Return {class_id: rgba} using categorical tab10 (≤10) or tab20 (>10) colors."""
    palette = matplotlib.colormaps['tab20' if len(classes) > 10 else 'tab10'].colors
    return {c: palette[i % len(palette)] for i, c in enumerate(classes)}


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
    """Reshape connection_factors[:, k] to 2D (n_out, n_in_flat) using weight shape."""
    nf = node.connection_factors[:, k]
    W  = node.weight
    ltype = node.layer_type
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
    img_factors    = node.img_factors
    neural_factors = node.connection_factors
    lambdas        = node.lambdas
    K = img_factors.shape[1]
    layer_name = node.layer_name

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

        # Layout: row 0 (lambda | connection map | stimulus weights),
        #         row 1 (weighted avg | top-n stimuli)
        fig = plt.figure(figsize=(2.6 * (n_top + 3), 6.5))
        gs  = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.4)

        # Row 0: three panels with width ratios 1 : 2 : 2
        gs0 = gs[0].subgridspec(1, 3, width_ratios=[1, 2, 2], wspace=0.35)

        # ── Row 0, col 0: lambda bar ─────────────────────────────────────────
        ax_lam = fig.add_subplot(gs0[0])
        bar_colors = ['#e15759' if ki == k else '#aec7e8' for ki in range(K)]
        ax_lam.bar(range(K), lambdas, color=bar_colors, edgecolor='k', linewidth=0.5)
        ax_lam.set_xticks(range(K))
        ax_lam.set_xticklabels([f'k={ki}' for ki in range(K)], fontsize=8)
        ax_lam.set_ylabel('λ', fontsize=9)
        ax_lam.set_title(
            f'{layer_name}  ·  Factor k={k}  ({lam_frac:.1%} of total λ)',
            fontsize=10, fontweight='bold',
        )

        # ── Row 0, col 1: neural factor heatmap ─────────────────────────────
        ax_hm = fig.add_subplot(gs0[1])
        hm = _neural_heatmap_array(node, k)
        vmax = np.abs(hm).max() or 1.0
        ax_hm.imshow(hm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax,
                     interpolation='nearest')
        ax_hm.set_xlabel('in', fontsize=8)
        ax_hm.set_ylabel('out', fontsize=8)
        ax_hm.set_title('connection map', fontsize=9)
        ax_hm.tick_params(labelsize=6)

        # ── Row 0, col 2: mean activation per class (bar plot) ──────────────
        ax_hist = fig.add_subplot(gs0[2])
        vals_by_group = _group_vals(k)
        coefs_sum = coefs.sum() + 1e-12
        bar_means, bar_stds, bar_cols, bar_lbls = [], [], [], []
        for gid in group_ids:
            v = vals_by_group[gid] / coefs_sum
            bar_means.append(float(v.mean()) if len(v) > 0 else 0.0)
            bar_stds.append(float(v.std())  if len(v) > 1 else 0.0)
            bar_cols.append(group_colors[gid])
            bar_lbls.append(group_labels[gid])
        bar_means = np.array(bar_means)
        bar_stds  = np.array(bar_stds)
        max_mean  = float(bar_means.max()) if bar_means.max() > 0 else 1.0
        # clip upper error bars so they never exceed max_mean
        yerr_upper = np.minimum(bar_stds, max_mean - bar_means)
        x_pos = np.arange(len(group_ids))
        ax_hist.bar(x_pos, bar_means, color=bar_cols, edgecolor='k',
                    linewidth=0.4, alpha=0.8)
        ax_hist.errorbar(x_pos, bar_means, yerr=[np.zeros_like(bar_stds), yerr_upper],
                         fmt='none', ecolor='black', elinewidth=1.2, capsize=3)
        ax_hist.set_xticks(x_pos)
        ax_hist.set_xticklabels(bar_lbls, fontsize=6, rotation=45, ha='right')
        ax_hist.set_ylim(0, max_mean)
        ax_hist.set_ylabel('mean norm. weight', fontsize=8)
        ax_hist.set_title('stimulus weights', fontsize=9)
        ax_hist.tick_params(labelsize=7)

        # Row 1: weighted avg + top-n stimuli
        gs1 = gs[1].subgridspec(1, n_top + 1, wspace=0.35)

        # ── Row 1, col 0: weighted-average image ─────────────────────────────
        ax_wavg = fig.add_subplot(gs1[0])
        flat = imgs.reshape(len(imgs), -1)
        wavg = (coefs[:, None] * flat).sum(0) / (coefs.sum() + 1e-12)
        wavg_img = _display_normalize(wavg.reshape(imgs.shape[1:]))
        _show_image(ax_wavg, wavg_img)
        ax_wavg.set_anchor('N')
        ax_wavg.set_title('weighted avg', fontsize=9)

        # ── Row 1, cols 1+: top stimuli ──────────────────────────────────────
        top_idx = np.argsort(coefs)[::-1][:n_top]
        for ti, si in enumerate(top_idx):
            ax_t = fig.add_subplot(gs1[1 + ti])
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
    neural_factors = node.connection_factors
    img_factors    = node.img_factors
    K = neural_factors.shape[1]
    W = node.weight
    layer_name = node.layer_name
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


# ── Conv spatial activation maps ─────────────────────────────────────────────

def plot_spatial_activation_maps(model, images_np, node, layer_data, device,
                                  n_images=6, denorm_fn=None, title=None):
    """4-row grid showing top and bottom stimuli alongside their channel-weighted activation maps.

    Parameters
    ----------
    model      : nn.Module — the trained model
    images_np  : (N, C, H, W) float32 ndarray
    node       : BFTNode — must be a conv node (layer_type == 'conv')
    layer_data : list of layer dicts from collect_layer_dicts; node.layer_idx indexes into it
    device     : torch device
    n_images   : number of top/bottom stimuli to show (default 6)
    denorm_fn  : callable(img_chw) -> img_hwc float [0,1], or None for raw display
    title      : optional figure suptitle; defaults to "<layer_name>: spatial activation maps"

    Returns
    -------
    Figure  — 4-row grid: top images / top maps / bottom images / bottom maps
    """
    import torch

    scores  = node.img_factors[:, 0]
    top_idx = np.argsort(scores)[::-1][:n_images]
    bot_idx = np.argsort(scores)[:n_images]

    ld = layer_data[node.layer_idx]
    C_out, C_in, kH, kW = ld['weight'].shape
    con_f = node.connection_factors
    ch_importance = np.maximum(con_f[:, 0].reshape(C_out, C_in * kH * kW).sum(1), 0)
    if ch_importance.sum() > 0:
        ch_importance /= ch_importance.sum()

    def _get_maps(sel_idx):
        fmaps = {}
        tmod = dict(model.named_modules())[node.layer_name]
        hook = tmod.register_forward_hook(
            lambda m, i, o: fmaps.update({'out': o.detach().cpu()}))
        model.eval()
        with torch.no_grad():
            model(torch.from_numpy(images_np[sel_idx]).float().to(device))
        hook.remove()
        fmap = fmaps['out'].numpy()
        return np.maximum((fmap * ch_importance[None, :, None, None]).sum(1), 0)

    top_maps = _get_maps(top_idx)
    bot_maps = _get_maps(bot_idx)

    n = len(top_idx)
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 8))
    for col in range(n):
        top_img = denorm_fn(images_np[top_idx[col]]) if denorm_fn else images_np[top_idx[col]]
        bot_img = denorm_fn(images_np[bot_idx[col]]) if denorm_fn else images_np[bot_idx[col]]
        _show_image(axes[0, col], top_img)
        axes[1, col].imshow(top_maps[col], cmap='hot')
        axes[1, col].axis('off')
        _show_image(axes[2, col], bot_img)
        axes[3, col].imshow(bot_maps[col], cmap='hot')
        axes[3, col].axis('off')

    for row, lbl in enumerate(['Top image', 'Top map', 'Bot image', 'Bot map']):
        axes[row, 0].set_ylabel(lbl, fontsize=9)

    fig.suptitle(title or f'{node.layer_name}: spatial activation maps', fontsize=10)
    fig.tight_layout()
    return fig


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
    coefs = node.img_factors[:, k]
    layer_name = node.layer_name
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


# ── Plot 6b — Pruning by layer depth ─────────────────────────────────────────

def plot_pruning_by_layer_depth(
    layer_sweep, fractions, target_class, class_names, methods,
    method_colors=None, method_labels=None, fraction_alphas=None,
):
    """Layer-depth pruning plot from ablation_layer_sweep() output.

    X-axis: pruning depth (1 = last layer only, 2 = last 2 layers, …).
    Y-axis: accuracy.
    2 panels: target class (left), mean bystander (right).
    One curve per (method × fraction); fractions of the same method share color
    and are distinguished by alpha (smallest → lightest).

    Parameters
    ----------
    layer_sweep     : dict {depth: AblationResult} from ablation_layer_sweep()
    fractions       : list[float] — which fractions to show
    target_class    : int
    class_names     : list or dict {int: str}
    methods         : list[str]
    method_colors   : optional dict {method: color}
    method_labels   : optional dict {method: label}
    fraction_alphas : optional dict {fraction: alpha}

    Returns
    -------
    matplotlib.figure.Figure
    """
    _default_colors = {
        'bft_top':    '#e15759',
        'bft_bottom': '#f28e2b',
        'magnitude':  '#76b7b2',
        'random':     '#333333',
    }
    _default_labels = {
        'bft_top':    'BFT most important',
        'bft_bottom': 'BFT least important',
        'magnitude':  'Magnitude',
        'random':     'Random',
    }
    mc = {**_default_colors, **(method_colors or {})}
    ml = {**_default_labels, **(method_labels or {})}

    cname = _cname_fn(class_names)
    fracs = sorted(fractions)
    depths = sorted(layer_sweep.keys())

    # Default alphas: evenly spaced from 0.35 to 1.0
    if fraction_alphas is None:
        n = len(fracs)
        alphas_list = [0.35 + 0.65 * i / max(n - 1, 1) for i in range(n)]
        fraction_alphas = {f: a for f, a in zip(fracs, alphas_list)}

    # Determine all classes from the baseline of the first result
    first_result = layer_sweep[depths[0]]
    all_classes = sorted(first_result.baseline.keys())
    bystander_classes = [c for c in all_classes if c != target_class]

    fig, (ax_tgt, ax_bys) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # Baseline: unablated accuracy (constant across depths; use depth=1 result)
    baseline_tgt = first_result.baseline.get(target_class, None)
    baseline_bys = (float(np.mean([first_result.baseline[c] for c in bystander_classes]))
                    if bystander_classes else None)

    for method in methods:
        color = mc.get(method, '#888888')
        label_base = ml.get(method, method)
        lstyle = '--' if method in ('bft_bottom',) else '-'
        dashes = (4, 2) if method == 'random' else None

        for frac in fracs:
            alpha = fraction_alphas.get(frac, 1.0)
            xs, ys_tgt, ys_bys = [], [], []
            for depth in depths:
                result = layer_sweep[depth]
                method_results = result.results.get(method, {})
                acc = method_results.get(frac, {})
                if not isinstance(acc, dict) or target_class not in acc:
                    continue
                xs.append(depth)
                ys_tgt.append(acc[target_class])
                if bystander_classes:
                    bys_vals = [acc[c] for c in bystander_classes if c in acc]
                    ys_bys.append(float(np.mean(bys_vals)) if bys_vals else np.nan)

            if not xs:
                continue

            kw = dict(color=color, alpha=alpha, linestyle=lstyle,
                      marker='o', markersize=4, linewidth=1.5)
            if dashes:
                kw['dashes'] = dashes

            # Only label the darkest (last) fraction per method
            kw_tgt = dict(kw)
            if frac == fracs[-1]:
                kw_tgt['label'] = label_base
            ax_tgt.plot(xs, ys_tgt, **kw_tgt)
            if ys_bys:
                ax_bys.plot(xs, ys_bys, **kw)

    # Baseline dashed lines
    if baseline_tgt is not None:
        ax_tgt.axhline(baseline_tgt, color='black', linestyle=':', linewidth=1.0,
                       label='baseline (no pruning)', alpha=0.6)
    if baseline_bys is not None:
        ax_bys.axhline(baseline_bys, color='black', linestyle=':', linewidth=1.0,
                       alpha=0.6)

    # Fraction legend as subtitle
    frac_str = '  |  '.join(f'α={fraction_alphas[f]:.2f} → {f*100:.0f}%' for f in fracs)

    for ax, ttl in [(ax_tgt, f'Target: {cname(target_class)}'),
                    (ax_bys, 'Mean bystander accuracy')]:
        ax.set_xlabel('pruning depth (layers from output)', fontsize=9)
        ax.set_ylabel('accuracy', fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks(depths)
        ax.set_xlim(min(depths) - 0.3, max(depths) + 0.3)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    ax_tgt.legend(fontsize=8)
    fig.suptitle(
        f'Pruning by layer depth — target class "{cname(target_class)}"\n'
        f'Fraction levels: {frac_str}',
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    return fig


# ── Plot 7 — Embedding comparison (PCA / MDS) ─────────────────────────────────

def plot_embedding_comparison(fingerprints, activations_last, labels, class_names,
                               digit_targets=None, condition_labels=None,
                               far_ood_conditions=None, activations_all=None, title=''):
    """
    3- or 4-panel figure comparing BFT fingerprints and network activations.

    Panel order:
      PCA(fingerprints) | PCA(last-layer activations) | [PCA(all-layer activations)] | MDS(fingerprints)

    The optional 4th panel (PCA over all layers concatenated) appears only when
    `activations_all` is provided.  Panels 0–2 use the same method (PCA) on different
    representations; the final MDS panel is a nonlinear cross-check.

    Coloring is driven by class label throughout.  Circle markers are used for all
    in-distribution and near-OOD samples; distinct shapes are used only for the
    far-OOD conditions listed in `far_ood_conditions`.

    Parameters
    ----------
    fingerprints       : (N, d_fp) fingerprint matrix (from extract_fingerprint_matrix)
    activations_last   : (N, d_act) last-layer activation matrix
    labels             : (N,) int task-class labels
    class_names        : list[str] or dict {int: str}
    digit_targets      : (N,) int digit labels; enables per-digit colouring
    condition_labels   : (N,) str/int — condition per sample (e.g. 'ID', 'OOD-CIFAR100',
                         'gaussian_noise')
    far_ood_conditions : list/set of condition names that should be rendered with distinct
                         marker shapes instead of circles (all others get circles)
    activations_all    : (N, d_all) optional — all-layer activations concatenated; when
                         provided a third PCA panel is inserted before the MDS panel
    title              : optional suptitle

    Returns
    -------
    Figure
    """
    from sklearn.preprocessing import normalize as _sk_normalize

    cname = _cname_fn(class_names)

    # PCA on fingerprints (panel 0)
    pca_fp = PCA(n_components=2, random_state=42)
    emb_pca_fp = pca_fp.fit_transform(fingerprints)
    var_fp = pca_fp.explained_variance_ratio_[:2].sum()

    # PCA on last-layer activations (panel 1)
    pca_act = PCA(n_components=2, random_state=42)
    emb_pca_act = pca_act.fit_transform(activations_last)
    var_act = pca_act.explained_variance_ratio_[:2].sum()

    # PCA on all-layer activations (panel 2, optional)
    if activations_all is not None:
        pca_all = PCA(n_components=2, random_state=42)
        emb_pca_all = pca_all.fit_transform(activations_all)
        var_all = pca_all.explained_variance_ratio_[:2].sum()

    # MDS on cosine distance of fingerprints (last panel)
    fp_norm = _sk_normalize(fingerprints, norm='l2')
    dist_mat = np.clip(1.0 - fp_norm @ fp_norm.T, 0, None)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
              n_init=4, max_iter=300)
    emb_mds = mds.fit_transform(dist_mat)

    # Color setup — driven by class label
    unique_labels = sorted(np.unique(labels).tolist())
    if digit_targets is not None:
        dcmap_ = _digit_colors()
        point_colors = np.array([dcmap_.get(int(d), (0.5, 0.5, 0.5, 1.0))
                                  for d in digit_targets])
    else:
        ccmap_ = _class_color_map(unique_labels)
        point_colors = np.array([ccmap_[int(l)] for l in labels])

    # Marker setup — circles for ID/near-OOD, shapes for far-OOD only
    far_ood_set = set(far_ood_conditions) if far_ood_conditions is not None else set()
    ood_shapes = ['^', 'D', 'P', 'X', 's', 'v', '<', '>', 'p', 'h']
    if condition_labels is not None:
        cond_arr = np.asarray(condition_labels)
        seen = set(); far_ood_ordered = []
        for c in condition_labels:
            if c in far_ood_set and c not in seen:
                far_ood_ordered.append(c); seen.add(c)
        cond2mark = {c: ood_shapes[i % len(ood_shapes)]
                     for i, c in enumerate(far_ood_ordered)}
    else:
        cond_arr = None
        far_ood_ordered = []
        cond2mark = {}

    n_panels = 4 if activations_all is not None else 3
    fig_width = 20 if n_panels == 4 else 15
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.5))
    if activations_all is not None:
        panel_info = [
            (emb_pca_fp,  f'PCA — BFT fingerprints\n{100*var_fp:.1f}% var'),
            (emb_pca_act, f'PCA — last-layer activations\n{100*var_act:.1f}% var'),
            (emb_pca_all, f'PCA — all-layer activations\n{100*var_all:.1f}% var'),
            (emb_mds,     'MDS — BFT fingerprints\n(cosine distance)'),
        ]
    else:
        panel_info = [
            (emb_pca_fp,  f'PCA — BFT fingerprints\n{100*var_fp:.1f}% var'),
            (emb_pca_act, f'PCA — last-layer activations\n{100*var_act:.1f}% var'),
            (emb_mds,     'MDS — BFT fingerprints\n(cosine distance)'),
        ]

    for ax, (emb, ttl) in zip(axes, panel_info):
        if cond_arr is not None:
            # non-far-OOD samples: circles colored by class
            non_ood_mask = np.array([c not in far_ood_set for c in cond_arr])
            if non_ood_mask.any():
                ax.scatter(emb[non_ood_mask, 0], emb[non_ood_mask, 1],
                           c=point_colors[non_ood_mask], marker='o',
                           s=18, alpha=0.7, linewidths=0)
            # far-OOD samples: distinct shapes colored by class
            for cond in far_ood_ordered:
                mask = (cond_arr == cond)
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=point_colors[mask], marker=cond2mark[cond],
                           s=35, alpha=0.85, linewidths=0.4, edgecolors='k')
        else:
            ax.scatter(emb[:, 0], emb[:, 1], c=point_colors,
                       marker='o', s=18, alpha=0.7, linewidths=0)
        ax.set(title=ttl, xlabel='dim 1', ylabel='dim 2')
        ax.tick_params(labelsize=7)

    # Class/digit color legend on panel 0
    if digit_targets is not None:
        dcmap_leg = _digit_colors()
        for d in sorted(dcmap_leg.keys()):
            axes[0].scatter([], [], color=dcmap_leg[d],
                            label=f"{'E' if d % 2 == 0 else 'O'}{d}", s=14, marker='o')
    else:
        ccmap_leg = _class_color_map(unique_labels)
        for c in unique_labels:
            axes[0].scatter([], [], color=ccmap_leg[c], label=cname(c), s=14, marker='o')
    axes[0].legend(fontsize=6, ncol=2, markerscale=2)

    # Far-OOD shape legend on last panel (MDS)
    if far_ood_ordered:
        for cond in far_ood_ordered:
            mask = (cond_arr == cond)
            rep_color = tuple(point_colors[mask][0]) if mask.any() else (0.5, 0.5, 0.5, 1.0)
            axes[-1].scatter([], [], color=rep_color, marker=cond2mark[cond],
                             label=str(cond), s=30, edgecolors='k', linewidths=0.4)
        axes[-1].legend(fontsize=7, markerscale=1.5, title='far-OOD')

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ── Fingerprint similarity heatmap ───────────────────────────────────────────

def plot_similarity_heatmap(S, block_sizes, block_labels,
                             cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                             title='', figsize=(8, 7), ax=None):
    """Render a blocked cosine-similarity matrix with block boundary lines.

    Parameters
    ----------
    S            : (N, N) similarity matrix (e.g. from compute_stimulus_similarity)
    block_sizes  : list[int] — number of samples per named block; must sum to N.
                   Pass an empty list to skip boundary lines.
    block_labels : list[str] — one label per block, used for tick annotations
    cmap         : matplotlib colormap (default 'RdBu_r')
    vmin, vmax   : colour scale limits (default -1 to 1 for cosine similarity)
    title        : axes title
    figsize      : figure size; ignored when ax is provided
    ax           : existing Axes to draw into; if None a new figure is created

    Returns
    -------
    Figure (new figure) or None (when ax was provided)
    """
    return_fig = ax is None
    if return_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(S, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Cosine similarity')

    if block_sizes:
        bl_ends = list(np.cumsum(block_sizes))
        for b in bl_ends[:-1]:
            ax.axhline(b - 0.5, color='k', lw=1.5)
            ax.axvline(b - 0.5, color='k', lw=1.5)
        centres = np.array([0] + bl_ends[:-1]) + np.array(block_sizes) / 2
        ax.set_xticks(centres)
        ax.set_xticklabels(block_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(centres)
        ax.set_yticklabels(block_labels, fontsize=8)

    if title:
        ax.set_title(title, fontsize=10)

    if return_fig:
        fig.tight_layout()
        return fig
    return None


# ── BFT tree visualisation ────────────────────────────────────────────────────

class _NodeMeta(dict):
    """Metadata dict with attribute access — makes extract_tree_nodes() dicts
    compatible with functions expecting BFTNode attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'_NodeMeta' has no key {key!r}")

    def __setattr__(self, key, value):
        self[key] = value


def extract_tree_nodes(root_node):
    """Walk the BFT tree (BFS) and return a flat list of node metadata dicts.

    Parameters
    ----------
    root_node : BFTNode or BFTResult

    Returns
    -------
    list[dict]  BFS order (root first, leaves last).  Each dict has:
        node_id, parent_id, layer_idx, layer_name, layer_type, factor_idx,
        img_factors, connection_factors, weight, lambdas, stimulus_weights, path
    """
    from .types import BFTResult
    if isinstance(root_node, BFTResult):
        root_node = root_node.root
    nodes = []
    queue = [(root_node, None)]
    while queue:
        node, parent_id = queue.pop(0)
        nid = node.path
        nodes.append(_NodeMeta({
            'node_id':             nid,
            'parent_id':           parent_id,
            'layer_idx':           node.layer_idx,
            'layer_name':          node.layer_name,
            'layer_type':          node.layer_type,
            'factor_idx':          node.factor_idx,
            'img_factors':         node.img_factors,
            'connection_factors':  node.connection_factors,
            'weight':              node.weight,
            'lambdas':             node.lambdas,
            'stimulus_weights':    node.stimulus_weights,
            'path':                node.path,
        }))
        for child in node.children:
            queue.append((child, nid))
    return nodes


def compute_node_activations(tree_nodes, stimulus_indices):
    """Compute a scalar per-node activation for a set of stimuli.

    Uses img_factors[stimulus_indices, 0] (primary factor loading).

    Parameters
    ----------
    tree_nodes       : list[dict]  from extract_tree_nodes()
    stimulus_indices : array-like of int

    Returns
    -------
    dict  {node_id: float}
    """
    s_idx = np.asarray(stimulus_indices)
    return {
        e['node_id']: float(e['img_factors'][s_idx, 0].mean())
        for e in tree_nodes
    }


def _hierarchical_layout(tree_nodes):
    """Compute (x, y) positions for drawing a BFT tree.

    Root at top (y = max layer_idx); leaves at bottom (y = 0).
    """
    by_id       = {e['node_id']: e for e in tree_nodes}
    children_of = {e['node_id']: [] for e in tree_nodes}
    for e in tree_nodes:
        if e['parent_id'] is not None:
            children_of[e['parent_id']].append(e['node_id'])

    positions    = {}
    leaf_counter = [0]

    def _assign(nid):
        children = children_of[nid]
        if not children:
            x = float(leaf_counter[0])
            leaf_counter[0] += 1
        else:
            for c in children:
                _assign(c)
            x = float(np.mean([positions[c][0] for c in children]))
        y = float(by_id[nid]['layer_idx'])
        positions[nid] = (x, y)

    for e in tree_nodes:
        if e['parent_id'] is None:
            _assign(e['node_id'])
    return positions


def plot_factor_tree(
    tree_nodes,
    node_activations,
    ax=None,
    cmap='viridis',
    node_size=1400,
    font_size=7,
    title=None,
):
    """Draw the BFT factor tree coloured by per-node stimulus activations.

    Parameters
    ----------
    tree_nodes       : list[dict]  from extract_tree_nodes()
    node_activations : dict  {node_id: float}  from compute_node_activations()
    ax               : matplotlib Axes or None
    cmap, node_size, font_size, title : styling options

    Returns
    -------
    fig, ax
    """
    G = nx.DiGraph()
    for e in tree_nodes:
        G.add_node(e['node_id'])
        if e['parent_id'] is not None:
            G.add_edge(e['parent_id'], e['node_id'])

    pos  = _hierarchical_layout(tree_nodes)
    vals = np.array([node_activations[e['node_id']] for e in tree_nodes])
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)
    node_colors = [colormap(norm(node_activations[nid])) for nid in G.nodes()]

    labels = {}
    for e in tree_nodes:
        nid   = e['node_id']
        lname = e['layer_name']
        if 'factor_k' in e:
            labels[nid] = f"L{e['layer_idx']}\nf{e['factor_k']}"
        else:
            depth = len(e['path'])
            if depth == 0:
                labels[nid] = f"L{e['layer_idx']}\n{lname}"
            else:
                labels[nid] = f"L{e['layer_idx']}\nf{e['factor_idx']}"

    n_nodes = len(tree_nodes)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, n_nodes * 0.9), 4.0))
    else:
        fig = ax.get_figure()

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        labels=labels,
        font_size=font_size,
        font_color='white',
        edge_color='#555555',
        arrows=True,
        arrowsize=15,
    )

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Activation', shrink=0.6)
    ax.set_axis_off()
    if title:
        ax.set_title(title)

    return fig, ax


def extract_factor_tree_nodes(root_node):
    """Expand BFT tree into factor-level nodes — one node per (path, factor_k).

    Parameters
    ----------
    root_node : BFTNode or BFTResult

    Returns
    -------
    list[dict]  keys: node_id, parent_id, layer_idx, layer_name, layer_type,
                path, factor_k, img_factors, lambdas, stimulus_weights
    """
    from .types import BFTResult
    if isinstance(root_node, BFTResult):
        root_node = root_node.root
    path_nodes = {}
    queue = [root_node]
    while queue:
        node = queue.pop(0)
        path_nodes[node.path] = node
        queue.extend(node.children)

    factor_nodes = []
    for pt, node in path_nodes.items():
        K = node.img_factors.shape[1]
        parent_id = None if len(pt) == 0 else (pt[:-1], pt[-1])
        for k in range(K):
            factor_nodes.append(_NodeMeta({
                'node_id':          (pt, k),
                'parent_id':        parent_id,
                'layer_idx':        node.layer_idx,
                'layer_name':       node.layer_name,
                'layer_type':       node.layer_type,
                'path':             node.path,
                'factor_k':         k,
                'img_factors':      node.img_factors,
                'lambdas':          node.lambdas,
                'stimulus_weights': node.stimulus_weights,
            }))
    return factor_nodes


def compute_factor_activations(factor_nodes, stimulus_indices):
    """Compute per-node activation for a factor-level tree.

    Parameters
    ----------
    factor_nodes     : list[dict]  from extract_factor_tree_nodes()
    stimulus_indices : array-like of int

    Returns
    -------
    dict  {node_id: float}
    """
    s_idx = np.asarray(stimulus_indices)
    return {
        e['node_id']: float(e['img_factors'][s_idx, e['factor_k']].mean())
        for e in factor_nodes
    }


def top_stimuli_factor_activations(factor_nodes, source_path_node, factor_k, top_n):
    """Select top-N stimuli for a factor and compute factor-level tree activations.

    Parameters
    ----------
    factor_nodes     : list[dict]  from extract_factor_tree_nodes()
    source_path_node : BFTNode whose img_factors column is used for ranking
    factor_k         : int
    top_n            : int

    Returns
    -------
    acts    : dict {node_id: float}
    top_idx : np.ndarray
    """
    loadings = source_path_node.img_factors[:, factor_k]
    top_idx  = np.argsort(loadings)[::-1][:top_n]
    acts     = compute_factor_activations(factor_nodes, top_idx)
    return acts, top_idx
