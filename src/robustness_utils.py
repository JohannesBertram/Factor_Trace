"""NMF and BFT robustness analysis."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim


def _hungarian_align(A, B):
    """
    Align columns of B to columns of A by maximum cosine similarity (Hungarian).

    Parameters
    ----------
    A, B : (n_features, K) normalised factor matrices

    Returns
    -------
    sim_matrix : (K_A, K_B) cosine similarity grid
    perm       : (K_A,) int — perm[i] is the column of B matched to column i of A
    aligned_B  : (n_features, K_A) reordered B
    """
    # Trim to common feature length if seeds produced differently-sized arbors.
    min_feat = min(A.shape[0], B.shape[0])
    A, B = A[:min_feat], B[:min_feat]
    S = _cos_sim(A.T, B.T)        # (K_A, K_B)
    row_ind, col_ind = linear_sum_assignment(-S)
    aligned_B = B[:, col_ind]
    return S, col_ind, aligned_B


def _run_nmf_once(X32, k, seed, max_iter, batch_size):
    """Fit MiniBatchNMF and return L2-normalised H (n_features, k)."""
    from sklearn.decomposition import MiniBatchNMF
    nmf = MiniBatchNMF(n_components=k, random_state=seed, max_iter=max_iter,
                       batch_size=batch_size, init='random')
    nmf.fit(X32)
    H = nmf.components_.T      # (n_features, k)
    norms = np.linalg.norm(H, axis=0, keepdims=True)
    return H / (norms + 1e-12)


def compute_nmf_stability(X, k, n_seeds=10, max_iter=500, batch_size=1024):
    """
    Run NMF n_seeds times with different random seeds and compute pairwise
    cosine similarity after Hungarian alignment.

    Parameters
    ----------
    X        : (n_samples, n_features) non-negative matrix
    k        : number of NMF components
    n_seeds  : number of independent runs (default 10)
    max_iter : max iterations per run
    batch_size : MiniBatchNMF batch size

    Returns
    -------
    sim_matrix : (n_seeds, n_seeds) mean cosine similarity after alignment
    factors    : list[ndarray (n_features, k)] — one normalised H per seed
    """
    X32 = np.asarray(X, dtype=np.float32)
    factors = [_run_nmf_once(X32, k, seed, max_iter, batch_size)
               for seed in range(n_seeds)]

    sim_matrix = np.eye(n_seeds)
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            S, perm, _ = _hungarian_align(factors[i], factors[j])
            mean_sim = float(np.array([S[ki, perm[ki]] for ki in range(k)]).mean())
            sim_matrix[i, j] = mean_sim
            sim_matrix[j, i] = mean_sim

    return sim_matrix, factors


def align_factors(reference, others):
    """
    Align a list of factor matrices to a common reference via Hungarian matching.

    Useful for model-seed robustness (5b): after running BFT on N model seeds,
    align the img_factors columns across seeds before computing similarity.

    Parameters
    ----------
    reference : (n_features, K) normalised reference factor matrix
    others    : list of (n_features, K) factor matrices to align

    Returns
    -------
    aligned   : list of (n_features, K) aligned factor matrices
    sim_scores : list[float] — mean cosine similarity to reference per matrix
    """
    aligned, sim_scores = [], []
    for h in others:
        S, perm, h_aligned = _hungarian_align(reference, h)
        mean_sim = float(np.array([S[ki, perm[ki]] for ki in range(reference.shape[1])]).mean())
        aligned.append(h_aligned)
        sim_scores.append(mean_sim)
    return aligned, sim_scores


def compute_k_sensitivity(X, k_star, n_seeds=5, max_iter=500, batch_size=1024):
    """
    Compare NMF factors at K*-1, K*, K*+1 to measure sensitivity to K choice.

    For each factor k in K*, the best-matching factor at K*-1 and K*+1 is found
    via cosine similarity (no alignment needed: just row-wise max).

    Parameters
    ----------
    X       : (n_samples, n_features) non-negative matrix
    k_star  : int — chosen number of components
    n_seeds : number of seeds to average at each K level
    max_iter, batch_size : NMF hyperparameters

    Returns
    -------
    result : dict with keys:
        'k_star'   : np.ones(k_star)          — self-similarity (always 1)
        'k_minus1' : (k_star,) float or None  — best match at K*-1
        'k_plus1'  : (k_star,) float          — best match at K*+1
    H_star : (n_features, k_star) mean factor matrix at K*
    """
    from sklearn.metrics.pairwise import cosine_similarity as cs

    X32 = np.asarray(X, dtype=np.float32)

    def _mean_factors(k):
        hlist = [_run_nmf_once(X32, k, s, max_iter, batch_size) for s in range(n_seeds)]
        # align all to seed-0
        ref = hlist[0]
        aligned = [ref]
        for h in hlist[1:]:
            _, _, ha = _hungarian_align(ref, h)
            aligned.append(ha)
        return np.stack(aligned).mean(0)   # (n_features, k)

    H_star = _mean_factors(k_star)
    result  = {'k_star': np.ones(k_star)}

    if k_star > 1:
        H_minus = _mean_factors(k_star - 1)
        S_minus  = cs(H_star.T, H_minus.T)   # (k_star, k_star-1)
        result['k_minus1'] = S_minus.max(axis=1)
    else:
        result['k_minus1'] = None

    H_plus = _mean_factors(k_star + 1)
    S_plus  = cs(H_star.T, H_plus.T)          # (k_star, k_star+1)
    result['k_plus1'] = S_plus.max(axis=1)

    return result, H_star


def plot_nmf_stability_figure(sim_matrix, title='NMF initialisation stability'):
    """
    Heatmap of pairwise similarity + boxplot of off-diagonal values.

    Parameters
    ----------
    sim_matrix : (n_seeds, n_seeds) float from compute_nmf_stability
    title      : figure title

    Returns
    -------
    Figure
    """
    n = sim_matrix.shape[0]
    mask_off = ~np.eye(n, dtype=bool)
    off_diag = sim_matrix[mask_off]

    fig, (ax_hm, ax_box) = plt.subplots(1, 2, figsize=(10, 4.2))

    im = ax_hm.imshow(sim_matrix, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    ax_hm.set(xlabel='seed', ylabel='seed',
               title=f'{title}\npairwise cosine similarity')
    ax_hm.set_xticks(range(n))
    ax_hm.set_yticks(range(n))
    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    ax_box.boxplot(off_diag, vert=True, widths=0.45,
                   medianprops=dict(color='black', linewidth=1.5))
    ax_box.axhline(0.9, ls='--', color='#e15759', lw=1.5, label='threshold 0.90')
    ax_box.set_ylabel('cosine similarity', fontsize=9)
    ax_box.set_xticklabels(['off-diagonal'])
    ax_box.set_ylim(0, 1.05)
    ax_box.legend(fontsize=8)
    ax_box.set_title('Off-diagonal distribution', fontsize=9)
    ax_box.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def plot_k_sensitivity_figure(sensitivity_result, k_star, title='K sensitivity'):
    """
    Line plot of best-match cosine similarity for K*-1, K*, K*+1.

    Parameters
    ----------
    sensitivity_result : dict from compute_k_sensitivity
    k_star             : int — the chosen K
    title              : figure title

    Returns
    -------
    Figure
    """
    factors = list(range(k_star))
    fig, ax = plt.subplots(figsize=(max(5, k_star), 3.8))

    if sensitivity_result.get('k_minus1') is not None:
        ax.plot(factors, sensitivity_result['k_minus1'], 'o--',
                color='#76b7b2', label=f'K={k_star - 1}', linewidth=1.5, markersize=5)
    ax.plot(factors, sensitivity_result['k_star'], 's-',
            color='#e15759', label=f'K={k_star} (self)', linewidth=1.5, markersize=5)
    ax.plot(factors, sensitivity_result['k_plus1'], '^--',
            color='#f28e2b', label=f'K={k_star + 1}', linewidth=1.5, markersize=5)

    ax.axhline(0.9, ls=':', color='gray', lw=1, label='threshold 0.90')
    ax.set(xlabel='factor index (in K*)', ylabel='best cosine similarity',
           title=title, ylim=(0, 1.05))
    ax.set_xticks(factors)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


