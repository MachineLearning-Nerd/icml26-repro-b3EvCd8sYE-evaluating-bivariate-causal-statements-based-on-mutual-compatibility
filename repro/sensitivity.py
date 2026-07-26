"""Reconstruction of the two ingredients Theorem 2.10's bound is built from.

Theorem 2.10 is a *sufficiency* statement quantified over every model and every
``(eps, delta)``, so measuring one algorithm's sample requirement can corroborate
it but never establish it.  What can be established is the derivation, and it
factors into two independently measurable pieces:

  (S)  **Sensitivity.**  ``comp(Sigma, A)`` is an explicit quadratic polynomial
       in the entries of ``Sigma``, so its exact gradient is available in closed
       form.  Write ``L(Sigma, A)`` for the ``l1`` norm of that gradient, i.e.
       the largest first-order change in ``comp`` caused by an entrywise
       perturbation of ``Sigma`` of size 1.  To get ``|comp(Sigmahat) -
       comp(Sigma)| <= eps`` it suffices to estimate every entry of ``Sigma`` to
       accuracy ``t = eps / L``.

  (C)  **Concentration.**  For a centered Gaussian, ``Sigmahat_ab`` is an average
       of ``N`` iid sub-exponential variables with variance
       ``(Sigma_aa Sigma_bb + Sigma_ab^2)/N <= 2V^2/N``.  A union bound over the
       ``n(n+1)/2`` entries gives ``max_ab |Sigmahat_ab - Sigma_ab| <= t`` with
       probability ``1 - delta`` once ``N >= C (V/t)^2 log(n^2/delta)``.

Composing them, ``N >= C V^2 L^2 / eps^2 * log(n^2/delta)``.  This has the
theorem's shape exactly when ``L <= K n^2 (1+a+b)^2 V``, since then

    V^2 L^2 = K^2 n^4 (1+a+b)^4 V^4.

So the theorem's four exponents are *predictions* about two measurable
quantities, and neither measurement uses the theorem's formula:

    exponent of L in n           = 2      (=> exponent 4 of N* in n)
    exponent of L in (1+a+b)     = 2      (=> exponent 4 in (1+a+b))
    exponent of L in V           = 1      (=> exponent 4 in V, with the V^2 above)
    exponent of t in 1/sqrt(N)   = 1      (=> exponent 2 in 1/eps)
    exponent of t in sqrt(log(n^2/delta)) = 1  (=> exponent 1 in log(n/delta))

``L`` is deterministic -- no Monte-Carlo noise at all -- so its exponents come
out sharply, which is exactly where the end-to-end ``N*`` measurement is weakest.
"""

from __future__ import annotations

import numpy as np

from .linear import disjoint_path_pair_sums, gamma_from_A


def comp_gradient(Sigma: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Exact gradient of ``comp`` w.r.t. the free parameters of a symmetric Sigma.

    Returns an upper-triangular array ``g`` with ``g[a, b] = d comp / d theta_ab``
    where ``theta_ab`` is the shared value of ``Sigma_ab`` and ``Sigma_ba``.

    ``comp = sum_{i<j} u_ij^2 - w_ij^2`` with ``u_ij = Sigma_ij - A_ji Sigma_ii``
    and ``w_ij = Sigma_ij - sum_{k<=i} Sigma_kk F[k,i,j]``, both *linear* in
    Sigma, so the derivative is elementary and needs no finite differences.
    """
    n = Sigma.shape[0]
    F = disjoint_path_pair_sums(gamma_from_A(A))
    diag = np.diag(Sigma)
    g = np.zeros((n, n))
    u = np.zeros((n, n))
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u[i, j] = Sigma[i, j] - A[j, i] * Sigma[i, i]
            w[i, j] = Sigma[i, j] - float(np.dot(diag[: i + 1], F[: i + 1, i, j]))
    for i in range(n):
        for j in range(i + 1, n):
            g[i, j] += 2.0 * u[i, j] - 2.0 * w[i, j]      # via Sigma_ij itself
            g[i, i] += -2.0 * u[i, j] * A[j, i]           # via Sigma_ii in u
            for k in range(i + 1):                        # via Sigma_kk in w
                g[k, k] += 2.0 * w[i, j] * F[k, i, j]
    return g


def sensitivity(Sigma: np.ndarray, A: np.ndarray) -> float:
    """``L = ||grad comp||_1`` -- worst first-order change per unit sup-perturbation."""
    return float(np.abs(comp_gradient(Sigma, A)).sum())


def curvature(Sigma: np.ndarray, A: np.ndarray) -> float:
    """``Q``: the second-order coefficient, so that for every perturbation D

        ``|comp(Sigma+D) - comp(Sigma)| <= L ||D||_inf + Q ||D||_inf^2``.

    ``comp`` is exactly quadratic in Sigma, so this two-term bound is exact --
    there is no higher-order remainder to neglect.  ``Q`` is the ``l1`` norm of
    the quadratic form ``sum_{i<j} (du_ij)^2 - (dw_ij)^2``, bounded by summing
    the absolute coefficients.
    """
    n = Sigma.shape[0]
    F = disjoint_path_pair_sums(gamma_from_A(A))
    Q = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            du = 1.0 + abs(A[j, i])                       # |dSigma_ij| + |A_ji||dSigma_ii|
            dw = 1.0 + float(np.abs(F[: i + 1, i, j]).sum())
            Q += du * du + dw * dw
    return float(Q)


def covariance_quantiles(n: int, Sigma: np.ndarray, N: int, reps: int,
                         probs, rng, block: int = 4000) -> np.ndarray:
    """Upper quantiles of ``max_{a<=b} |Sigmahat_ab - Sigma_ab|`` over ``reps`` draws.

    ``probs`` are *exceedance* probabilities (i.e. ``delta``), so the returned
    value at ``delta`` is the smallest ``t`` with an empirical
    ``P(max dev > t) <= delta``.  Computed in blocks to bound memory.
    """
    L = np.linalg.cholesky(Sigma)
    iu = np.triu_indices(n)
    devs = np.empty(reps)
    done = 0
    while done < reps:
        r = min(block, reps - done)
        z = rng.standard_normal((r, N, n))
        x = z @ L.T
        S = np.einsum("rna,rnb->rab", x, x) / N
        devs[done:done + r] = np.abs(S - Sigma)[:, iu[0], iu[1]].max(axis=1)
        done += r
    return np.quantile(devs, 1.0 - np.asarray(probs, dtype=float))
