import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


def plot_nmf_scree(results):
    """
    Plot explained variance vs n_components from nmf_component_sweep output.

    Shows mean ± std across seeds as a line with error bars, and individual
    seed values as small grey dots.

    Parameters
    ----------
    results : dict {k: ev} from nmf_component_sweep

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


def plot_factor_graph(fi, neural_factors, model_layers, layer_sizes, linear_layer_indices):
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
    """
    n = len(neural_factors)
    normalized_nf = neural_factors[:, fi] / neural_factors[:, fi].max()

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
