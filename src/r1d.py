import numpy as np
from scipy import sparse as sp


def approxRankOneSubmatrix(A, gamma, maxiters=25, monotonic=False, penalize_lownorm=False):

    assert gamma > 1, "gamma should be > 1"

    normsq = lambda arr, ax: np.square(arr).sum(ax)

    j0 = np.linalg.norm(A, axis=0).argmax()
    M = np.arange(A.shape[0], dtype='int')
    N = [j0]
    sigma = np.linalg.norm(A[:, j0])
    if sigma == 0:
        raise ValueError("A has no nonzero columns")

    if penalize_lownorm:
        eta = 1 / 20.
        g = gamma / (gamma - 1)
        rho_ = eta * (gamma - 1)**2 * sigma**2 / A.shape[0]
    else:
        rho_ = 0

    u = A[:, j0:j0+1] / sigma
    v = np.zeros(A.shape[1], dtype=A.dtype)

    tol = 1e-5

    for it in range(maxiters):

        v_ = A[M].T @ u[M]
        if monotonic:
            N = np.union1d(N, np.flatnonzero((gamma * np.square(v_).ravel() - normsq(A[M], 0) - rho_ * len(M)) > 0))
        else:
            N = np.flatnonzero((gamma * np.square(v_).ravel() - normsq(A[M], 0) - rho_ * len(M)) > 0)
        if N.size == 0:
            break
        new_v = np.zeros_like(v_)
        new_v[N] = v_[N] / np.linalg.norm(v_[N])

        u_ = A[:, N] @ new_v[N]
        if monotonic:
            M = np.setdiff1d(M, np.flatnonzero((gamma * np.square(u_).ravel() - normsq(A[:, N], 1) - rho_ * len(N)) <= 0))
        else:
            M = np.flatnonzero((gamma * np.square(u_).ravel() - normsq(A[:, N], 1) - rho_ * len(N)) > 0)
        if M.size == 0:
            break
        new_sigma = np.linalg.norm(u_[M])
        new_u = np.zeros_like(u_)
        new_u[M] = u_[M] / new_sigma

        if abs(new_sigma - sigma) < tol \
                and np.linalg.norm(new_u - u) < tol \
                and np.linalg.norm(new_v - v) < tol:
            break

        u = new_u
        v = new_v
        sigma = new_sigma

    return M, N, u.ravel(), sigma, v.ravel(), it + 1


def r1d(X, k, gamma=2, maxiters=50, penalize_lownorm=True, monotonic=True):
    """Greedy rank-1 deflation factorisation of a non-negative matrix.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Non-negative input matrix.
    k : int
        Maximum number of rank-1 components to extract.
    gamma : float
        Approximation quality parameter; must be > 1.

    Returns
    -------
    W : ndarray, shape (m, k_actual)
    H : ndarray, shape (n, k_actual)
    sigmas : list[float]
        k_actual may be less than k if early stopping occurs.
    """
    assert X.min() >= 0, "X must be non-negative"

    A = X.copy()
    W = np.zeros((X.shape[0], k))
    H = np.zeros((X.shape[1], k))
    sigmas = []
    for i in range(k):

        if A.max() == 0:
            break

        M, N, u, sigma, v, niters = approxRankOneSubmatrix(
            A, gamma, maxiters, penalize_lownorm=penalize_lownorm, monotonic=monotonic
        )

        if len(M) == 0 or len(N) == 0 or sigma == 0:
            break

        W[M, i] = sigma * u[M]
        H[N, i] = v[N]
        sigmas.append(sigma)

        A[np.ix_(M, N)] = 0

    return W, H, sigmas


def rec_err_curve(A, W, H, relative=True):
    """Relative Frobenius reconstruction error for k=1..K.

    Parameters
    ----------
    A : dense ndarray or scipy.sparse matrix, shape (m, n)
    W : ndarray, shape (m, K)
    H : ndarray, shape (n, K)
    """
    K = W.shape[1]
    errs = np.empty(K, dtype=float)

    A_norm2 = A.multiply(A).sum() if sp.issparse(A) else np.sum(A * A)

    GW = W.T @ W
    GH = H.T @ H

    xnorm2 = 0.0
    cross = 0.0

    for k in range(K):
        wk = W[:, k]
        hk = H[:, k]

        Ahk = A @ hk if sp.issparse(A) else A.dot(hk)
        cross += wk @ Ahk

        if k == 0:
            xnorm2 = GW[0, 0] * GH[0, 0]
        else:
            xnorm2 += 2.0 * np.sum(GW[:k, k] * GH[:k, k]) + GW[k, k] * GH[k, k]

        err2 = max(A_norm2 + xnorm2 - 2.0 * cross, 0.0)
        err = np.sqrt(err2)
        errs[k] = err / np.sqrt(A_norm2) if relative else err

    return errs


def run_r1d(X, n_components, gamma=2, maxiters=50,
            penalize_lownorm=True, monotonic=True, **_):
    """Adapter matching the run_nmf(X, n_components) -> (W, H, aux) signature.

    sklearn-specific kwargs (random_state, max_iter, init, l1_ratio) are
    silently absorbed via **_ so they can flow through the BFT call chain
    without causing errors.

    Returns
    -------
    W      : (m, k_actual)  — columns are sigma * u vectors
    H      : (n, k_actual)  — columns are unit-norm v vectors
    sigmas : list[float]    — one per extracted component
    """
    W, H, sigmas = r1d(X, n_components, gamma=gamma, maxiters=maxiters,
                       penalize_lownorm=penalize_lownorm, monotonic=monotonic)
    actual_k = len(sigmas)
    return W[:, :actual_k], H[:, :actual_k], sigmas
