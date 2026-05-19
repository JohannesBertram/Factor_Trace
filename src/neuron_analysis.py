import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ── internal helpers ─────────────────────────────────────────────────────────

def _reshape(arr, input_shape):
    """Reshape arr using input_shape: int → (int, -1); tuple → exact shape."""
    if isinstance(input_shape, (tuple, list)):
        return arr.reshape(input_shape)
    return arr.reshape(int(input_shape), -1)


def _cl_name(cl, class_names):
    """Return display name for a class id."""
    if class_names and cl in class_names:
        return class_names[cl]
    return str(cl)


# ── core computation ──────────────────────────────────────────────────────────

def compute_synaptic_arbors(weights, inputs):
    """
    Element-wise weight × input product for a single neuron.

    Parameters
    ----------
    weights : (n_inputs,) array — neuron's incoming weight vector
    inputs  : (n_samples, n_inputs) array — input activations or pixel values

    Returns
    -------
    (n_samples, n_inputs) array of weight * input products
    """
    return inputs * weights.flatten()


def pca_decoding(arbors):
    """Fit PCA(2) and return (Y, pca) where Y is (n_samples, 2)."""
    pca = PCA(2, random_state=0)
    Y = pca.fit_transform(arbors)
    return Y, pca


# ── PCA scatter plots ─────────────────────────────────────────────────────────

def plot_pca_scatter(Y, targets, classes, title='', class_names=None):
    """Scatter plot of PCA projection coloured by class label."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    cmap = matplotlib.colormaps['tab10']
    norm = matplotlib.colors.Normalize(vmin=min(classes), vmax=max(classes))
    for cl in classes:
        mask = targets == cl
        ax.scatter(*Y[mask].T, color=cmap(norm(cl)), label=_cl_name(cl, class_names), s=10)
    ax.set(title=title)
    ax.legend(markerscale=2)
    return fig


def plot_pca_alpha(Y, targets, activations_neuron, classes, title='', class_names=None):
    """
    PCA scatter where each point's alpha encodes the neuron's activation.

    activations_neuron : (n_samples,) activation values for one neuron, all classes
    """
    cmap = matplotlib.colormaps['tab10']
    norm = matplotlib.colors.Normalize(vmin=min(classes), vmax=max(classes))
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for cl in classes:
        mask = targets == cl
        acts = activations_neuron[mask]
        assert acts.min() >= 0
        acts = acts / acts.max()
        ax.scatter(*Y[mask].T, color=cmap(norm(cl)), alpha=acts,
                   label=_cl_name(cl, class_names))
    ax.set(title=title)
    ax.legend(markerscale=2)
    return fig


# ── mean arbor plots ──────────────────────────────────────────────────────────

def plot_mean_arbors(arbors, targets, weights, classes, input_shape,
                     title='mean synaptic config.', class_names=None):
    """Show weights alongside per-class mean weight*input pattern."""
    n_cols = len(classes) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2))
    cmap = 'seismic'
    vlim = max(abs(weights))
    axes[0].imshow(_reshape(weights, input_shape), vmin=-vlim, vmax=vlim, cmap=cmap)
    axes[0].set(title='weights', xticks=[], yticks=[])
    for i, cl in enumerate(classes):
        mean_arbor = arbors[targets == cl].mean(0)
        vlim = max(abs(mean_arbor))
        axes[i + 1].imshow(_reshape(mean_arbor, input_shape), vmin=-vlim, vmax=vlim, cmap=cmap)
        axes[i + 1].set(title=_cl_name(cl, class_names), xticks=[], yticks=[])
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_pos_mean_arbors(arbors, targets, weights, classes, input_shape,
                         title='mean pos. synaptic config.', class_names=None):
    """Like plot_mean_arbors but zeroing out negative products."""
    n_cols = len(classes) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2))
    cmap = 'seismic'
    vlim = max(abs(weights))
    axes[0].imshow(_reshape(weights, input_shape), vmin=-vlim, vmax=vlim, cmap=cmap)
    axes[0].set(title='weights', xticks=[], yticks=[])
    for i, cl in enumerate(classes):
        mean_arbor = arbors[targets == cl].mean(0).copy()
        mean_arbor[mean_arbor < 0] = 0
        vlim = max(mean_arbor) if mean_arbor.max() > 0 else 1
        axes[i + 1].imshow(_reshape(mean_arbor, input_shape), vmin=0, vmax=vlim, cmap=cmap)
        axes[i + 1].set(title=_cl_name(cl, class_names), xticks=[], yticks=[])
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_activation_weighted_arbors(arbors, targets, activations_neuron, weights, classes,
                                    input_shape,
                                    title='mean pos. synaptic config. weighted by activation',
                                    class_names=None):
    """Mean positive arbor weighted by the neuron's scalar activation for each sample."""
    n_cols = len(classes) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2))
    cmap = 'seismic'
    vlim_w = max(abs(weights))
    axes[0].imshow(_reshape(weights, input_shape), vmin=-vlim_w, vmax=vlim_w, cmap=cmap)
    axes[0].set(title='weights', xticks=[], yticks=[])
    for i, cl in enumerate(classes):
        mask = targets == cl
        acts = activations_neuron[mask]
        mean_arbor = np.sum(arbors[mask] * acts[:, None], 0) / acts.sum()
        mean_arbor[mean_arbor < 0] = 0
        vlim = max(mean_arbor) if mean_arbor.max() > 0 else 1
        axes[i + 1].imshow(_reshape(mean_arbor, input_shape), vmin=0, vmax=vlim, cmap=cmap)
        axes[i + 1].set(title=_cl_name(cl, class_names), xticks=[], yticks=[])
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_activation_hist(activations_neuron, targets, classes,
                         bin_edges=None, class_names=None):
    """Histogram of a single neuron's activation distribution per class."""
    if bin_edges is None:
        bin_edges = np.arange(0, 1.05, 0.05)
    cmap = matplotlib.colormaps['tab10']
    norm = matplotlib.colors.Normalize(vmin=min(classes), vmax=max(classes))
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for cl in classes:
        acts = activations_neuron[targets == cl]
        ax.hist(acts, bin_edges, color=cmap(norm(cl)), alpha=0.6,
                label=_cl_name(cl, class_names))
    ax.legend()
    return fig


# ── full single-neuron report ─────────────────────────────────────────────────

def single_neuron_report(model_layers, layer_idx, neuron_idx,
                         all_activations, all_images,
                         all_targets, classes, input_shape,
                         class_names=None,
                         digit_targets=None, digit_classes=None,
                         digit_subset=None,
                         inputs=None,
                         global_neuron_idx=None):
    """
    Full single-neuron analysis: PCA decoding + mean-arbor visualizations + histogram.

    Parameters
    ----------
    model_layers      : nn.Sequential (model.layers)
    layer_idx         : int — index of the Linear layer in model_layers
    neuron_idx        : int — local neuron index within that layer
    all_activations   : (n_total, n_neurons) concatenated activation array
    all_images        : (n_total, H, W) images as numpy; ignored when `inputs` is given
    all_targets       : (n_total,) task class labels (e.g. 0=even, 1=odd)
    classes           : list of class ids matching all_targets
    input_shape       : int or (H, W) tuple — shape for imshow
                        • int  → reshape as (int, -1), e.g. 28 → (28, 28) for MNIST pixels
                        • tuple → used directly, e.g. (4, 5) for 20-dim layer-1 activations
    class_names       : optional dict {class_id: display_name}, e.g. {0: 'even', 1: 'odd'}
    digit_targets     : optional (n_total,) original digit labels (0-9)
    digit_classes     : optional list of all digit ids; auto-detected if None
    digit_subset      : optional list of specific digits to show in mean-arbor plots
                        (e.g. [0, 1]); requires digit_targets
    inputs            : optional (n_total, n_inputs) array to use instead of all_images
                        for arbor computation — use this for non-pixel layers
    global_neuron_idx : optional int — column in all_activations for activation lookup;
                        defaults to neuron_idx (correct for first-layer neurons whose
                        local and global indices coincide)
    """
    ws = model_layers[layer_idx].weight[neuron_idx].detach().numpy()

    # inputs for arbor: explicit override, or fall back to flattened images
    arb_inputs = inputs if inputs is not None else all_images.reshape(len(all_images), -1)
    arbors = compute_synaptic_arbors(ws, arb_inputs)

    # activation column: use global index (layer offset + local idx) if given
    g_idx = neuron_idx if global_neuron_idx is None else global_neuron_idx
    acts_neuron = all_activations[:, g_idx]

    layer_label = f'layer {layer_idx}, neuron {neuron_idx}'
    Y, _ = pca_decoding(arbors)

    figs = []

    # ── PCA scatter: task classes ──
    figs.append(plot_pca_scatter(Y, all_targets, classes,
                                 title=f'{layer_label} (task)', class_names=class_names))
    figs.append(plot_pca_alpha(Y, all_targets, acts_neuron, classes,
                               title=f'{layer_label} (task, α=activation)',
                               class_names=class_names))

    # ── PCA scatter: digit coloring ──
    if digit_targets is not None:
        if digit_classes is None:
            digit_classes = sorted(np.unique(digit_targets).tolist())
        figs.append(plot_pca_scatter(Y, digit_targets, digit_classes,
                                     title=f'{layer_label} (digit)'))
        figs.append(plot_pca_alpha(Y, digit_targets, acts_neuron, digit_classes,
                                   title=f'{layer_label} (digit, α=activation)'))

    # ── mean arbors: task classes ──
    figs.append(plot_mean_arbors(
        arbors, all_targets, ws, classes, input_shape,
        title='mean synaptic config. (task)', class_names=class_names))
    figs.append(plot_pos_mean_arbors(
        arbors, all_targets, ws, classes, input_shape,
        title='mean pos. synaptic config. (task)', class_names=class_names))
    figs.append(plot_activation_weighted_arbors(
        arbors, all_targets, acts_neuron, ws, classes, input_shape,
        title='mean pos. synaptic config. weighted by act. (task)', class_names=class_names))

    # ── mean arbors: digit subset ──
    if digit_targets is not None and digit_subset is not None:
        mask = np.isin(digit_targets, digit_subset)
        digit_names = {d: str(d) for d in digit_subset}
        figs.append(plot_mean_arbors(
            arbors[mask], digit_targets[mask], ws, digit_subset, input_shape,
            title=f'mean synaptic config. (digits {digit_subset})',
            class_names=digit_names))
        figs.append(plot_pos_mean_arbors(
            arbors[mask], digit_targets[mask], ws, digit_subset, input_shape,
            title=f'mean pos. synaptic config. (digits {digit_subset})',
            class_names=digit_names))
        figs.append(plot_activation_weighted_arbors(
            arbors[mask], digit_targets[mask], acts_neuron[mask], ws,
            digit_subset, input_shape,
            title=f'mean pos. synaptic config. weighted by act. (digits {digit_subset})',
            class_names=digit_names))

    # ── activation histogram ──
    figs.append(plot_activation_hist(acts_neuron, all_targets, classes,
                                     class_names=class_names))
    return figs
