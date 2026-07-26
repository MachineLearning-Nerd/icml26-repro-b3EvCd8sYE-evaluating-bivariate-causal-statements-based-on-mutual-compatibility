"""Linear compatibility scores for bivariate causal statements.

Faithful implementation of Section 2 of

    Erik Jahn, Dominik Janzing.
    "Evaluating Bivariate Causal Statements Based on Mutual Compatibility."
    arXiv:2606.00278v1 [cs.AI], 29 May 2026.

Symbol conventions follow the paper exactly:

* ``Gamma`` -- strictly lower-triangular matrix of causal coefficients of a
  multivariate linear SEM ``X = Gamma X + N`` (equation (1)).
* ``A``     -- unit lower-triangular matrix of *bivariate* causal statements,
  ``A[j, i] = alpha_ij`` for ``i < j`` (Section 2.2).  Lemma 2.3 ties the two
  together via ``Gamma = I - A^{-1}``, equivalently ``A = (I - Gamma)^{-1}``.
* ``Sigma`` -- covariance matrix of the *observed* vector ``X``.

The only subtle quantity is the path-tracing term of Definition 2.6.  See
:func:`disjoint_path_pair_sums` for the dynamic program that computes it.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "gamma_from_A",
    "A_from_gamma",
    "sigma_x_from_model",
    "disjoint_path_pair_sums",
    "disjoint_path_pair_sums_bruteforce",
    "observed_path_term",
    "C_biv",
    "C_mult",
    "compatibility_score",
    "decompose_B_eps",
    "standardize",
]


# --------------------------------------------------------------------------
# Lemma 2.3: the bijection between statement matrices and SEM coefficients
# --------------------------------------------------------------------------

def gamma_from_A(A: np.ndarray) -> np.ndarray:
    """``Gamma = I - A^{-1}`` (Lemma 2.3)."""
    n = A.shape[0]
    return np.eye(n) - np.linalg.inv(A)


def A_from_gamma(Gamma: np.ndarray) -> np.ndarray:
    """``A = (I - Gamma)^{-1}``: the matrix of *total* bivariate causal effects."""
    n = Gamma.shape[0]
    return np.linalg.inv(np.eye(n) - Gamma)


def sigma_x_from_model(Gamma: np.ndarray, Sigma_N: np.ndarray) -> np.ndarray:
    """``Sigma_X = (I - Gamma)^{-1} Sigma_N (I - Gamma)^{-T}`` (Section 2.6)."""
    A = A_from_gamma(Gamma)
    return A @ Sigma_N @ A.T


# --------------------------------------------------------------------------
# Definition 2.6: sums over pairs of vertex-disjoint directed paths
# --------------------------------------------------------------------------

def disjoint_path_pair_sums(Gamma: np.ndarray) -> np.ndarray:
    """Return ``F`` with ``F[k, i, j] = sum over disjoint path pairs from k``.

    Precisely, for every ``k <= i < j``,

    .. math::

        F[k,i,j] = \\sum_{\\substack{P_1 : k \\leadsto i,\\ P_2 : k \\leadsto j\\\\
                                     P_1 \\cap P_2 = \\emptyset}}
                   \\Gamma_{P_1} \\Gamma_{P_2}

    where the paths are directed paths in the SEM's coefficient graph, they are
    vertex-disjoint apart from their shared start ``k``, and
    ``Gamma_P = prod_s Gamma[t_{s+1}, t_s]`` is the product of coefficients
    along the path (the empty path contributes ``1``).

    Because ``Gamma`` is strictly lower-triangular, every directed path visits
    strictly increasing indices.  This is what makes an ``O(n^4)`` dynamic
    program possible.  A DP state ``(u, v)`` records the current head of the
    path aimed at ``i`` and of the path aimed at ``j``; while neither path has
    arrived we always extend whichever head is *smaller*.  Under that rule every
    vertex already committed to either path is strictly below the head we are
    about to move -- the sole exception being the other head -- so
    vertex-disjointness reduces to the single local test
    ``new_head != other_head``.  Once one path has arrived at its target the
    other is extended alone, which is still safe because its head only grows and
    is already above every vertex of the finished path.

    Note the state is *not* symmetric in ``u`` and ``v``: the two paths have
    different destinations.
    """
    n = Gamma.shape[0]
    # total[b, a] = sum of Gamma_P over all directed paths a ~> b
    total = A_from_gamma(Gamma)

    F = np.zeros((n, n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # w[u, v]: sum of coefficient products over all vertex-disjoint
            # completions of a path pair with heads (u -> i, v -> j).
            w = np.zeros((n, n))
            w[i, j] = 1.0
            # Every transition strictly increases u + v, so a single sweep in
            # decreasing u + v order resolves the recursion.
            for s in range(2 * n - 3, -1, -1):
                for u in range(n):
                    v = s - u
                    if v < 0 or v >= n or u == v or (u == i and v == j):
                        continue
                    if u > i or v > j:
                        continue  # target already overshot; unreachable
                    acc = 0.0
                    if u == i:
                        extend_u = False          # path 1 has arrived
                    elif v == j:
                        extend_u = True           # path 2 has arrived
                    else:
                        extend_u = u < v          # extend the smaller head
                    if extend_u:
                        for up in range(u + 1, n):
                            g = Gamma[up, u]
                            if g == 0.0 or up == v:
                                continue
                            acc += g * w[up, v]
                    else:
                        for vp in range(v + 1, n):
                            g = Gamma[vp, v]
                            if g == 0.0 or vp == u:
                                continue
                            acc += g * w[u, vp]
                    w[u, v] = acc
            # Assemble F[k, i, j] for every start k <= i.
            for k in range(i + 1):
                if k == i:
                    # P1 is the empty path at i; P2 ranges over all i ~> j.
                    F[k, i, j] = total[j, i]
                    continue
                acc = 0.0
                for a in range(k + 1, n):
                    ga = Gamma[a, k]
                    if ga == 0.0:
                        continue
                    for b in range(k + 1, n):
                        if b == a:
                            continue
                        gb = Gamma[b, k]
                        if gb == 0.0:
                            continue
                        acc += ga * gb * w[a, b]
                F[k, i, j] = acc
    return F


def _all_increasing_paths(Gamma: np.ndarray, src: int, dst: int):
    """Enumerate every directed path ``src ~> dst`` as a vertex tuple."""
    n = Gamma.shape[0]
    if src == dst:
        yield (src,)
        return
    stack = [(src, (src,))]
    while stack:
        head, path = stack.pop()
        for nxt in range(head + 1, n):
            if Gamma[nxt, head] == 0.0:
                continue
            if nxt == dst:
                yield path + (nxt,)
            else:
                stack.append((nxt, path + (nxt,)))


def _path_weight(Gamma: np.ndarray, path) -> float:
    w = 1.0
    for s in range(len(path) - 1):
        w *= Gamma[path[s + 1], path[s]]
    return w


def disjoint_path_pair_sums_bruteforce(Gamma: np.ndarray) -> np.ndarray:
    """Reference implementation of :func:`disjoint_path_pair_sums`.

    Enumerates path pairs explicitly.  Exponential in ``n`` -- used only as an
    independent checker of the dynamic program on small models.
    """
    n = Gamma.shape[0]
    F = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(i + 1):
                acc = 0.0
                for p1 in _all_increasing_paths(Gamma, k, i):
                    s1 = set(p1)
                    w1 = _path_weight(Gamma, p1)
                    if w1 == 0.0:
                        continue
                    for p2 in _all_increasing_paths(Gamma, k, j):
                        # disjoint apart from the shared start k
                        if len(s1 & set(p2)) != 1:
                            continue
                        acc += w1 * _path_weight(Gamma, p2)
                F[k, i, j] = acc
    return F


def observed_path_term(Sigma: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """``T[i, j] = sum_{k<=i} Sigma[k,k] * F[k,i,j]`` -- the bracket of Def. 2.6."""
    F = disjoint_path_pair_sums(Gamma)
    diag = np.diag(Sigma)
    n = Sigma.shape[0]
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            T[i, j] = float(np.dot(diag[: i + 1], F[: i + 1, i, j]))
    return T


# --------------------------------------------------------------------------
# Definitions 2.5 / 2.6 / 2.7
# --------------------------------------------------------------------------

def C_biv(Sigma: np.ndarray, A: np.ndarray, i: int, j: int) -> float:
    """Definition 2.5: ``(Sigma_ij - alpha_ij Sigma_ii)^2`` with ``alpha_ij = A[j,i]``."""
    return float((Sigma[i, j] - A[j, i] * Sigma[i, i]) ** 2)


def C_mult(Sigma: np.ndarray, Gamma: np.ndarray, i: int, j: int,
           T: np.ndarray | None = None) -> float:
    """Definition 2.6: squared part of ``Sigma_ij`` unexplained by observed paths."""
    if T is None:
        T = observed_path_term(Sigma, Gamma)
    return float((Sigma[i, j] - T[i, j]) ** 2)


def compatibility_score(Sigma: np.ndarray, A: np.ndarray) -> float:
    """Definition 2.7: ``sum_{i<j} C_biv_ij - C_mult_ij``.

    ``Sigma`` and ``A`` are expected to already be on the standardized scale
    (see :func:`standardize`); the sign of the score is scale-invariant either
    way, as the paper notes in Section 2.5.
    """
    Gamma = gamma_from_A(A)
    T = observed_path_term(Sigma, Gamma)
    n = Sigma.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += C_biv(Sigma, A, i, j) - float((Sigma[i, j] - T[i, j]) ** 2)
    return total


def decompose_B_eps(Sigma: np.ndarray, A: np.ndarray):
    """Return ``(B, eps)`` from the proof of Theorem 2.9.

    ``B[i,j]`` is the covariance contribution of *observed* back-door paths and
    ``eps[i,j]`` the contribution of unobserved ones, so that (equation (4))

        ``Sigma[i,j] = Sigma[i,i] * A[j,i] + B[i,j] + eps[i,j]``.

    The proof then reads ``C_biv_ij = (B + eps)^2`` and ``C_mult_ij = eps^2``.
    """
    Gamma = gamma_from_A(A)
    F = disjoint_path_pair_sums(Gamma)
    diag = np.diag(Sigma)
    n = Sigma.shape[0]
    B = np.zeros((n, n))
    eps = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            B[i, j] = float(np.dot(diag[:i], F[:i, i, j]))  # k < i only
            T_ij = diag[i] * F[i, i, j] + B[i, j]
            eps[i, j] = float(Sigma[i, j] - T_ij)
    return B, eps


def compat_and_decomposition(Sigma: np.ndarray, A: np.ndarray):
    """Return ``(comp, B, eps)`` sharing a single path-tracing pass.

    ``comp`` is computed from Definition 2.7 directly (``C_biv - C_mult``), not
    from the proof identity ``sum(B^2 + 2 B eps)`` -- keeping the two available
    independently is what lets the identity itself be tested rather than assumed.
    """
    Gamma = gamma_from_A(A)
    F = disjoint_path_pair_sums(Gamma)
    diag = np.diag(Sigma)
    n = Sigma.shape[0]
    B = np.zeros((n, n))
    eps = np.zeros((n, n))
    comp = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            B[i, j] = float(np.dot(diag[:i], F[:i, i, j]))
            T_ij = diag[i] * F[i, i, j] + B[i, j]
            eps[i, j] = float(Sigma[i, j] - T_ij)
            comp += (Sigma[i, j] - A[j, i] * Sigma[i, i]) ** 2 - (Sigma[i, j] - T_ij) ** 2
    return float(comp), B, eps


def standardize(Sigma: np.ndarray, A: np.ndarray):
    """Rescale to unit variances (Section 2.5).

    Returns ``(corr, A_std)`` where ``corr`` is the correlation matrix and
    ``A_std`` holds the correspondingly rescaled causal statements,
    ``A_std[j,i] = A[j,i] * sd_i / sd_j``.
    """
    sd = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(sd, sd)
    A_std = A * np.outer(1.0 / sd, sd)
    np.fill_diagonal(A_std, 1.0)
    return corr, A_std
