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
