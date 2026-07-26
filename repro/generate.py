"""Synthetic model generation, following Appendix D.1 of arXiv:2606.00278.

Two procedures are specified in the paper and both are implemented verbatim:
:func:`sample_linear_model` (used for Figures 2, 3, 8, 9) and
:func:`sample_graphical_model` (used for Figure 5).
"""

from __future__ import annotations

import numpy as np

from .graphical import (
    StatementGraph,
    has_confounding_path,
    transitive_closure,
)
from .linear import A_from_gamma

__all__ = [
    "sample_linear_model",
    "perturb_statements",
    "sample_graphical_model",
    "inject_graph_errors",
    "marginal_admg",
]


# --------------------------------------------------------------------------
# D.1, linear procedure (steps 1-6)
# --------------------------------------------------------------------------

def sample_linear_model(n: int, m: int, p: float, rng: np.random.Generator):
    """Sample a ground-truth linear Gaussian SEM and its true statements.

    Returns ``(Sigma_obs, A_true, Gamma_obs)`` where ``Sigma_obs`` is the
    correlation matrix of the ``n`` observed variables, ``Gamma_obs`` the
    marginalised causal coefficient matrix and ``A_true = (I - Gamma_obs)^{-1}``
    the matrix of correct bivariate causal statements.

    Steps, quoting Appendix D.1:

    1. draw causal coefficients for ``n + m`` variables from a standard normal
       and error variances from an exponential distribution with variance 1;
    2. zero each coefficient independently with probability ``1 - p``;
    3. compute the covariance matrix assuming independent errors;
    4. rescale error variances and coefficients so every variable has unit
       variance;
    5. marginalise to ``n`` uniformly chosen observed variables (Lemma 2.1);
    6. read off ``A = (I - Gamma)^{-1}``.
    """
    N = n + m
    Gamma = np.tril(rng.normal(size=(N, N)), -1)
    Gamma *= (rng.random((N, N)) < p)
    Gamma = np.tril(Gamma, -1)
    # exponential with variance 1 => scale 1 (mean 1, variance 1)
    err_var = rng.exponential(scale=1.0, size=N)

    A_full = A_from_gamma(Gamma)
    Sigma = A_full @ np.diag(err_var) @ A_full.T

    # Step 4: rescale so that every variable has unit variance.  Writing
    # S = diag(sd), the standardised model has Gamma' = S^{-1} Gamma S and
    # error variances err_var / sd^2.
    sd = np.sqrt(np.diag(Sigma))
    Gamma = Gamma * np.outer(1.0 / sd, sd)
    Sigma = Sigma / np.outer(sd, sd)

    # Step 5: marginalise to n observed variables.
    hidden = rng.choice(N, size=m, replace=False)
    obs = np.array(sorted(set(range(N)) - set(hidden.tolist())))
    Z = np.array(sorted(hidden.tolist()))
    if m > 0:
        G_YY = Gamma[np.ix_(obs, obs)]
        G_YZ = Gamma[np.ix_(obs, Z)]
        G_ZZ = Gamma[np.ix_(Z, Z)]
        G_ZY = Gamma[np.ix_(Z, obs)]
        Gamma_obs = G_YY + G_YZ @ np.linalg.solve(np.eye(m) - G_ZZ, G_ZY)
    else:
        Gamma_obs = Gamma.copy()
    Gamma_obs = np.tril(Gamma_obs, -1)
    Sigma_obs = Sigma[np.ix_(obs, obs)]

    A_true = A_from_gamma(Gamma_obs)
    return Sigma_obs, A_true, Gamma_obs


def perturb_statements(A_true: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """D.1 step 7: add centred Gaussian noise of variance ``sigma`` to each
    bivariate statement.  ``sigma`` is a *variance*, as written in the paper, so
    the added noise has standard deviation ``sqrt(sigma)``."""
    n = A_true.shape[0]
    A = A_true.copy()
    noise = rng.normal(scale=np.sqrt(sigma), size=(n, n))
    idx = np.tril_indices(n, -1)
    A[idx] = A_true[idx] + noise[idx]
    np.fill_diagonal(A, 1.0)
    return np.tril(A)


# --------------------------------------------------------------------------
# D.1, graphical procedure
# --------------------------------------------------------------------------

def marginal_admg(D: np.ndarray, Bd: np.ndarray, keep: np.ndarray) -> StatementGraph:
    """Definition 3.2: marginalise a mixed graph onto the vertices ``keep``."""
    N = D.shape[0]
    full = StatementGraph(N, D, Bd)
    hidden = sorted(set(range(N)) - set(keep.tolist()))
    k = len(keep)

    # 1. directed edge v -> w if a directed path exists with all intermediate
    #    vertices hidden.
    reach = D.copy()
    for h in hidden:  # allow routing only through hidden vertices
        reach = reach | (np.outer(reach[:, h], reach[h, :]) & ~np.eye(N, dtype=bool))
    Dm = reach[np.ix_(keep, keep)].copy()
    np.fill_diagonal(Dm, False)

    # 2. bidirected edge v <-> w if a confounding path exists with all
    #    intermediate vertices hidden.  We test on the subgraph induced by
    #    {v, w} together with the hidden vertices, which is exactly the
    #    "all intermediates hidden" restriction.
    Bm = np.zeros((k, k), bool)
    for a in range(k):
        for b in range(a + 1, k):
            v, w = int(keep[a]), int(keep[b])
            sub = sorted(set(hidden) | {v, w})
            idx = np.array(sub)
            sg = StatementGraph(len(sub), D[np.ix_(idx, idx)], Bd[np.ix_(idx, idx)])
            if has_confounding_path(sg, sub.index(v), sub.index(w)):
                Bm[a, b] = Bm[b, a] = True
    return StatementGraph(k, Dm, Bm)


def sample_graphical_model(n: int, m: int, p: float, rng: np.random.Generator):
    """Sample ground truth and the correct graphical bivariate statements.

    Steps 1-4 of the graphical procedure in Appendix D.1.  The returned
    statement graph is the union of the pairwise marginal ADMGs, and is by
    construction graphically compatible (Lemma 3.5).
    """
    N = n + m
    perm = rng.permutation(N)
    D = np.zeros((N, N), bool)
    for a in range(N):
        for b in range(a + 1, N):
            if rng.random() < p:
                D[perm[a], perm[b]] = True
    Bd = np.zeros((N, N), bool)

    hidden = rng.choice(N, size=m, replace=False)
    keep = np.array(sorted(set(range(N)) - set(hidden.tolist())))
    marg = marginal_admg(D, Bd, keep)

    # Step 4: marginalise further onto each pair and take the union.  Under a
    # pairwise marginalisation every other vertex is hidden, so the directed
    # part of the union is exactly the transitive closure of the observed
    # graph, and a bidirected edge appears wherever a confounding path exists.
    Bs = np.zeros((n, n), bool)
    Ds = transitive_closure(marg.D)
    for a in range(n):
        for b in range(a + 1, n):
            if has_confounding_path(marg, a, b):
                Bs[a, b] = Bs[b, a] = True
    return StatementGraph(n, Ds, Bs)


def inject_graph_errors(g: StatementGraph, k: int,
                        rng: np.random.Generator) -> StatementGraph:
    """D.1 step 5: toggle ``k`` uniformly chosen edge slots.

    A slot is either one of the ``n(n-1)`` ordered pairs (a directed edge) or
    one of the ``C(n,2)`` unordered pairs (a bidirected edge); the chosen slot
    is deleted if present and inserted if absent.  Slots are drawn without
    replacement so that exactly ``k`` statements are altered.
    """
    n = g.n
    slots = [("d", u, v) for u in range(n) for v in range(n) if u != v]
    slots += [("b", u, v) for u in range(n) for v in range(u + 1, n)]
    pick = rng.choice(len(slots), size=k, replace=False)
    D, Bd = g.D.copy(), g.Bd.copy()
    for idx in pick:
        kind, u, v = slots[int(idx)]
        if kind == "d":
            D[u, v] = not D[u, v]
        else:
            Bd[u, v] = Bd[v, u] = not Bd[u, v]
    return StatementGraph(n, D, Bd)
