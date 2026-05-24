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


# ── Sparse R1D variants ────────────────────────────────────────────────────────

def _approx_rank_one_submatrix_sparse(A_csr, A_csc, gamma, maxiters=25, tol=1e-5):
    """Sparse-friendly R1D inner routine with explicit submatrix slicing.

    Computes column/row norms by slicing A to the active support (M, N)
    before squaring, which is exact but requires forming sub-matrices.
    Prefer _approx_rank_one_submatrix_sparse2 for large matrices where
    forming A[M, :] is expensive.
    """
    assert gamma > 1, "gamma should be > 1"

    m, n = A_csr.shape

    col_norm2_full = np.asarray(A_csc.multiply(A_csc).sum(axis=0)).ravel()
    j0 = int(col_norm2_full.argmax())
    sigma = float(np.sqrt(col_norm2_full[j0]))
    if sigma == 0:
        raise ValueError("A has no nonzero columns")

    u = np.asarray(A_csc[:, j0].toarray()).ravel() / sigma
    v = np.zeros(n, dtype=float)

    M = np.arange(m, dtype=int)
    N = np.array([j0], dtype=int)

    niters = 0
    for niters in range(1, maxiters + 1):
        A_M = A_csr[M, :]
        v_ = np.asarray(A_M.T @ u[M]).ravel()

        col_norm2_M = np.asarray(A_M.multiply(A_M).sum(axis=0)).ravel()

        new_N = np.flatnonzero(gamma * (v_ ** 2) - col_norm2_M > 0)
        if new_N.size == 0:
            break

        new_v = np.zeros(n, dtype=float)
        nv = np.linalg.norm(v_[new_N])
        if nv == 0:
            break
        new_v[new_N] = v_[new_N] / nv

        A_N = A_csc[:, new_N]
        u_ = np.asarray(A_N @ new_v[new_N]).ravel()

        row_norm2_N = np.asarray(A_N.multiply(A_N).sum(axis=1)).ravel()

        new_M = np.flatnonzero(gamma * (u_ ** 2) - row_norm2_N > 0)
        if new_M.size == 0:
            break

        new_sigma = float(np.linalg.norm(u_[new_M]))
        if new_sigma == 0:
            break

        new_u = np.zeros(m, dtype=float)
        new_u[new_M] = u_[new_M] / new_sigma

        if (
            np.array_equal(new_M, M)
            and np.array_equal(new_N, N)
            and abs(new_sigma - sigma) < tol
            and np.linalg.norm(new_u - u) < tol
            and np.linalg.norm(new_v - v) < tol
        ):
            M, N, u, v, sigma = new_M, new_N, new_u, new_v, new_sigma
            break

        M, N, u, v, sigma = new_M, new_N, new_u, new_v, new_sigma

    return M, N, u, sigma, v, niters


def r1d_sparse(X, k, gamma=2.0, maxiters=25, copy=True, tol=1e-5):
    """Sparse-friendly R1D: inner routine uses explicit submatrix slicing.

    Accepts dense arrays (converted to CSR) or any scipy sparse matrix.
    Operates only on the active support (M, N) during each inner iteration
    by slicing A[M, :] and A[:, N] explicitly.

    Parameters
    ----------
    X       : ndarray or scipy.sparse matrix, non-negative
    k       : number of rank-1 components
    gamma   : approximation quality parameter (> 1)
    maxiters: inner-loop iteration cap
    copy    : if True, do not modify the input
    tol     : inner-loop convergence tolerance

    Returns
    -------
    W      : (m, k_actual)
    H      : (n, k_actual)
    sigmas : list[float]
    """
    assert gamma > 1, "gamma should be > 1"

    if sp.issparse(X):
        A_csr = X.tocsr(copy=copy)
    else:
        A_csr = sp.csr_matrix(X.copy() if copy else X)

    A_csc = A_csr.tocsc()

    m, n = A_csr.shape
    W = np.zeros((m, k), dtype=float)
    H = np.zeros((n, k), dtype=float)
    sigmas = []

    for i in range(k):
        if A_csr.nnz == 0:
            break

        M, N, u, sigma, v, niters = _approx_rank_one_submatrix_sparse(
            A_csr, A_csc, gamma=gamma, maxiters=maxiters, tol=tol
        )

        if len(M) == 0 or len(N) == 0 or sigma == 0:
            break

        W[M, i] = u[M]
        H[N, i] = sigma * v[N]
        sigmas.append(sigma)

        A_csr[np.ix_(M, N)] = 0
        A_csr.eliminate_zeros()
        A_csc = A_csr.tocsc()

    actual_k = len(sigmas)
    return W[:, :actual_k], H[:, :actual_k], sigmas


def _approx_rank_one_submatrix_sparse2(A_csr, A_csc, A2_csr, A2_csc, gamma,
                                        maxiters=25, tol=1e-5):
    """Sparse-friendly R1D inner routine using squared-matrix matvecs.

    Avoids forming A[M, :] or A[:, N] by keeping pre-computed elementwise-
    squared copies (A2_csr, A2_csc) and using binary mask vectors to restrict
    norms to the active support via a single sparse matvec.  Faster than
    _approx_rank_one_submatrix_sparse when the matrix is large and the support
    is small.
    """
    assert gamma > 1, "gamma should be > 1"

    m, n = A_csr.shape

    col_norm2_full = np.asarray(A2_csc.sum(axis=0)).ravel()
    j0 = int(col_norm2_full.argmax())
    sigma = float(np.sqrt(col_norm2_full[j0]))
    if sigma == 0:
        raise ValueError("A has no nonzero columns")

    u = np.asarray(A_csc[:, j0].toarray()).ravel() / sigma
    v = np.zeros(n, dtype=float)

    M = np.flatnonzero(u)
    N = np.array([j0], dtype=int)

    mask_M = np.zeros(m, dtype=float)
    mask_N = np.zeros(n, dtype=float)

    niters = 0
    for niters in range(1, maxiters + 1):
        mask_M.fill(0.0)
        mask_M[M] = 1.0

        v_ = np.asarray(A_csc.T @ u).ravel()
        col_norm2_M = np.asarray(A2_csc.T @ mask_M).ravel()

        new_N = np.flatnonzero(gamma * (v_ ** 2) - col_norm2_M > 0)
        if new_N.size == 0:
            break

        new_v = np.zeros(n, dtype=float)
        nv = np.linalg.norm(v_[new_N])
        if nv == 0:
            break
        new_v[new_N] = v_[new_N] / nv

        mask_N.fill(0.0)
        mask_N[new_N] = 1.0

        u_ = np.asarray(A_csr @ new_v).ravel()
        row_norm2_N = np.asarray(A2_csr @ mask_N).ravel()

        new_M = np.flatnonzero(gamma * (u_ ** 2) - row_norm2_N > 0)
        if new_M.size == 0:
            break

        new_sigma = float(np.linalg.norm(u_[new_M]))
        if new_sigma == 0:
            break

        new_u = np.zeros(m, dtype=float)
        new_u[new_M] = u_[new_M] / new_sigma

        if (
            np.array_equal(new_M, M)
            and np.array_equal(new_N, N)
            and abs(new_sigma - sigma) < tol
            and np.linalg.norm(new_u - u) < tol
            and np.linalg.norm(new_v - v) < tol
        ):
            M, N, u, v, sigma = new_M, new_N, new_u, new_v, new_sigma
            break

        M, N, u, v, sigma = new_M, new_N, new_u, new_v, new_sigma

    return M, N, u, sigma, v, niters


def r1d_sparse2(X, k, gamma=2.0, maxiters=25, copy=True, tol=1e-5):
    """Sparse-friendly R1D: inner routine uses squared-matrix matvecs.

    Like r1d_sparse but maintains elementwise-squared copies (A²_csr, A²_csc)
    so that restricted column/row norms are computed via a single matvec with
    a binary mask rather than by slicing the submatrix.  Preferred when the
    matrix is large and the support is a small fraction of the full size.

    Parameters
    ----------
    X       : ndarray or scipy.sparse matrix, non-negative
    k       : number of rank-1 components
    gamma   : approximation quality parameter (> 1)
    maxiters: inner-loop iteration cap
    copy    : if True, do not modify the input
    tol     : inner-loop convergence tolerance

    Returns
    -------
    W      : (m, k_actual)
    H      : (n, k_actual)
    sigmas : list[float]
    """
    assert gamma > 1, "gamma should be > 1"

    if sp.issparse(X):
        A_csr = X.tocsr(copy=copy)
    else:
        A_csr = sp.csr_matrix(X.copy() if copy else X)

    A_csc = A_csr.tocsc()
    A2_csr = A_csr.multiply(A_csr)
    A2_csc = A_csc.multiply(A_csc)

    m, n = A_csr.shape
    W = np.zeros((m, k), dtype=float)
    H = np.zeros((n, k), dtype=float)
    sigmas = []

    for i in range(k):
        if A_csr.nnz == 0:
            break

        M, N, u, sigma, v, niters = _approx_rank_one_submatrix_sparse2(
            A_csr, A_csc, A2_csr, A2_csc,
            gamma=gamma, maxiters=maxiters, tol=tol
        )

        if len(M) == 0 or len(N) == 0 or sigma == 0:
            break

        W[M, i] = u[M]
        H[N, i] = sigma * v[N]
        sigmas.append(sigma)

        A_csr[np.ix_(M, N)] = 0
        A_csr.eliminate_zeros()

        A_csc = A_csr.tocsc()
        A2_csr = A_csr.multiply(A_csr)
        A2_csc = A_csc.multiply(A_csc)

    actual_k = len(sigmas)
    return W[:, :actual_k], H[:, :actual_k], sigmas


def run_r1d_sparse(X, n_components, gamma=2.0, maxiters=25, tol=1e-5, **_):
    """Adapter for r1d_sparse matching the run_nmf(X, n_components) signature.

    sklearn-specific kwargs are silently absorbed via **_.

    Returns
    -------
    W      : (m, k_actual)
    H      : (n, k_actual)
    sigmas : list[float]
    """
    W, H, sigmas = r1d_sparse(X, n_components, gamma=gamma, maxiters=maxiters, tol=tol)
    return W, H, sigmas


def run_r1d_sparse2(X, n_components, gamma=2.0, maxiters=25, tol=1e-5, **_):
    """Adapter for r1d_sparse2 matching the run_nmf(X, n_components) signature.

    sklearn-specific kwargs are silently absorbed via **_.

    Returns
    -------
    W      : (m, k_actual)
    H      : (n, k_actual)
    sigmas : list[float]
    """
    W, H, sigmas = r1d_sparse2(X, n_components, gamma=gamma, maxiters=maxiters, tol=tol)
    return W, H, sigmas
