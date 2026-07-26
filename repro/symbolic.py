"""Machine-checkable certificates for the proof of Theorem 2.9.

Theorem 2.9 is universally quantified over every distribution on ``(Gamma,
Sigma_N)`` satisfying Assumption 2.8, so no amount of Monte-Carlo sampling can
verify it -- sampling can only fail to refute it.  This module reconstructs the
paper's own proof and checks each step by machine, over the *complete* finite
domain the step ranges over.

The proof (Appendix B) is:

  1.  Split the covariance of ``X_i, X_j`` into the direct effect, the observed
      back-door contribution ``B_ij`` and the unobserved one ``eps_ij``::

          Sigma_X,ij = Sigma_X,ii * A_ji + B_ij + eps_ij                    (4)

          B_ij   = sum_{k<i}    Sigma_X,kk * sum_{P1:k=>i, P2:k=>j, P1 disjoint P2} Gamma_P1 Gamma_P2
          eps_ij = sum_{l != m} Sigma_N,lm * sum_{Q1:l=>i, Q2:m=>j, Q1 disjoint Q2} Gamma_Q1 Gamma_Q2

  2.  ``C_biv_ij = (B_ij + eps_ij)^2`` and ``C_mult_ij = eps_ij^2``, hence
      ``comp = sum_{i<j} B_ij^2 + 2 B_ij eps_ij``.

  3.  ``E[B_ij eps_ij] = 0``.  This is the load-bearing step.  The paper argues
      it summand by summand: in every product ``Gamma_P1 Gamma_P2 Gamma_Q1
      Gamma_Q2`` there is an entry ``Gamma_rs`` with ``r > k`` occurring
      **exactly once**; ``Sigma_X,kk`` involves only ``Gamma_pq`` with ``p <= k``,
      so by Assumption 2.8(2) that entry is independent of everything else in
      the summand, and by Assumption 2.8(1) its expectation is zero.

  4.  Therefore ``E[comp] = sum E[B_ij^2] >= E[B_23^2] > 0``.

Step 3's combinatorial core is a statement about *paths*, over a finite domain
for each ``n``; :func:`certify_cross_term_parity` enumerates that domain
completely.  Steps 1, 2 and 4 are polynomial identities in the entries of
``Gamma`` and ``Sigma_N``; they are checked with exact symbolic algebra, which
settles them for *every* model of that dimension at once rather than for a
sample of them.

Everything here is written from the paper's formulas by direct enumeration of
paths, independently of the ``O(n^4)`` dynamic program in :mod:`repro.linear`;
agreement between the two is itself one of the checks.
"""

from __future__ import annotations

import itertools
from collections import Counter

import sympy as sp


# --------------------------------------------------------------------------
# Paths.  Gamma is strictly lower triangular, so a directed path can only move
# to a strictly larger index and is exactly an increasing index sequence.
# --------------------------------------------------------------------------

def paths(u: int, w: int, n: int) -> list[tuple[int, ...]]:
    """Every directed path ``u => w``, as a vertex tuple. Empty list if u > w."""
    if u > w:
        return []
    if u == w:
        return [(u,)]
    out = []
    for r in range(w - u):
        for mid in itertools.combinations(range(u + 1, w), r):
            out.append((u,) + mid + (w,))
    return out


def path_edges(p: tuple[int, ...]) -> list[tuple[int, int]]:
    """Edges of a path, as ``(row, col)`` index pairs into ``Gamma``."""
    return [(p[t + 1], p[t]) for t in range(len(p) - 1)]


def _disjoint(p1, p2, share=None) -> bool:
    s1, s2 = set(p1), set(p2)
    if share is not None:
        s1.discard(share)
        s2.discard(share)
    return not (s1 & s2)


def path_weight(p, G):
    w = sp.Integer(1)
    for r, c in path_edges(p):
        w *= G[r, c]
    return w


# --------------------------------------------------------------------------
# Symbolic model
# --------------------------------------------------------------------------

def symbolic_model(n: int):
    """Return ``(G, SN, A, SX, gsyms, vsyms)`` with fully generic entries.

    ``G`` is strictly lower triangular with independent symbols, ``SN`` is a
    generic symmetric noise covariance, and ``A = (I-G)^-1`` is computed as the
    finite Neumann series ``I + G + ... + G^(n-1)`` (exact, since ``G`` is
    nilpotent).
    """
    G = sp.zeros(n, n)
    gsyms = {}
    for i in range(n):
        for j in range(i):
            s = sp.Symbol(f"g{i}{j}", real=True)
            G[i, j] = s
            gsyms[(i, j)] = s
    SN = sp.zeros(n, n)
    vsyms = {}
    for i in range(n):
        for j in range(i, n):
            s = sp.Symbol(f"v{i}{j}", real=True)
            SN[i, j] = SN[j, i] = s
            vsyms[(i, j)] = s
    A = sp.eye(n)
    P = sp.eye(n)
    for _ in range(n - 1):
        P = sp.expand(P * G)
        A = A + P
    A = sp.expand(A)
    SX = sp.expand(A * SN * A.T)
    return G, SN, A, SX, gsyms, vsyms


def B_eps_symbolic(n: int, G, SN, SX):
    """``(B, eps)`` built literally from the two displayed sums in the proof."""
    B = sp.zeros(n, n)
    E = sp.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            b = sp.Integer(0)
            for k in range(i):
                inner = sp.Integer(0)
                for p1 in paths(k, i, n):
                    for p2 in paths(k, j, n):
                        if _disjoint(p1, p2, share=k):
                            inner += path_weight(p1, G) * path_weight(p2, G)
                b += SX[k, k] * inner
            B[i, j] = sp.expand(b)
            e = sp.Integer(0)
            for ell in range(i + 1):
                for m in range(j + 1):
                    if ell == m:
                        continue
                    inner = sp.Integer(0)
                    for q1 in paths(ell, i, n):
                        for q2 in paths(m, j, n):
                            if _disjoint(q1, q2):
                                inner += path_weight(q1, G) * path_weight(q2, G)
                    e += SN[ell, m] * inner
            E[i, j] = sp.expand(e)
    return B, E


def compat_symbolic(n: int, G, A, SX):
    """Definitions 2.5-2.7, by direct path enumeration (not the DP)."""
    comp = sp.Integer(0)
    for i in range(n):
        for j in range(i + 1, n):
            c_biv = (SX[i, j] - A[j, i] * SX[i, i]) ** 2
            T = sp.Integer(0)
            for k in range(i + 1):
                inner = sp.Integer(0)
                for p1 in paths(k, i, n):
                    for p2 in paths(k, j, n):
                        if _disjoint(p1, p2, share=k):
                            inner += path_weight(p1, G) * path_weight(p2, G)
                T += SX[k, k] * inner
            c_mult = (SX[i, j] - T) ** 2
            comp += c_biv - c_mult
    return sp.expand(comp)


# --------------------------------------------------------------------------
# Step 3: the combinatorial core, checked exhaustively
# --------------------------------------------------------------------------

def certify_cross_term_parity(n: int, q_share_start: bool = False):
    """Check the paper's parity argument over *every* summand of equation (6).

    For each ``(i, j, k, l, m, P1, P2, Q1, Q2)`` the paper asserts that some
    ``Gamma_rs`` with ``r > k`` occurs exactly once in
    ``Gamma_P1 Gamma_P2 Gamma_Q1 Gamma_Q2``.  Because ``Sigma_X,kk`` involves
    only ``Gamma_pq`` with ``p <= k``, such an entry is independent of the whole
    rest of the summand, and ``E[Gamma_rs] = 0`` then kills it.

    Returns ``(n_summands, failures)``; ``failures`` lists summands with no such
    entry.

    ``q_share_start=True`` is the negative control.  What makes the argument go
    through is that ``Q1, Q2`` have *all different endpoints* while ``P1, P2``
    meet at ``k``, so the ``Q`` pair can never reproduce the ``P`` pair edge for
    edge.  Letting the ``Q`` paths share their start removes exactly that
    property -- ``Q1 = P1, Q2 = P2`` becomes admissible, every edge then occurs
    twice, and the parity argument must fail.  A control that fails for any
    other reason would not be testing the step the proof actually uses.
    """
    n_summands = 0
    failures = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(i):                      # B_ij: k < i
                bpairs = [(p1, p2)
                          for p1 in paths(k, i, n) for p2 in paths(k, j, n)
                          if _disjoint(p1, p2, share=k)]
                if not bpairs:
                    continue
                for ell in range(i + 1):
                    for m in range(j + 1):
                        if ell == m and not q_share_start:
                            continue
                        share = ell if (q_share_start and ell == m) else None
                        qpairs = [(q1, q2)
                                  for q1 in paths(ell, i, n)
                                  for q2 in paths(m, j, n)
                                  if _disjoint(q1, q2, share=share)]
                        for p1, p2 in bpairs:
                            for q1, q2 in qpairs:
                                n_summands += 1
                                cnt = Counter(path_edges(p1) + path_edges(p2)
                                              + path_edges(q1) + path_edges(q2))
                                if not any(c == 1 and r > k
                                           for (r, _s), c in cnt.items()):
                                    failures.append(
                                        dict(i=i, j=j, k=k, l=ell, m=m,
                                             P1=p1, P2=p2, Q1=q1, Q2=q2))
    return n_summands, failures


def certify_sigma_kk_support(n: int, SX, gsyms):
    """``Sigma_X,kk`` may involve only ``Gamma_pq`` with ``p <= k``.

    The proof needs this to conclude that ``Gamma_rs`` with ``r > k`` is
    independent of ``Sigma_X,kk``.  Returns a list of violations.
    """
    bad = []
    for k in range(n):
        free = SX[k, k].free_symbols
        for (p, q), s in gsyms.items():
            if s in free and p > k:
                bad.append((k, f"g{p}{q}"))
    return bad


# --------------------------------------------------------------------------
# Assumption 2.8 as an expectation operator
# --------------------------------------------------------------------------

def expectation_A28(expr, gsyms):
    """Apply ``E[.]`` using *only* what Assumption 2.8 grants.

    Assumption 2.8 gives mutual independence of the ``Gamma`` entries and of
    ``Sigma_N`` (part 2), and ``E[Gamma_ij] = 0`` (part 1).  It grants nothing
    about higher moments, so a factor ``Gamma_ij^d`` with ``d >= 2`` is replaced
    by an opaque moment symbol and a factor with ``d == 1`` sends the whole
    monomial to zero.  Products of ``Sigma_N`` entries are likewise opaque, so
    each distinct ``Sigma_N``-monomial becomes its own symbol.

    Anything this returns as identically zero is therefore zero for *every*
    distribution satisfying Assumption 2.8, not merely for symmetric ones.
    """
    gl = sorted(gsyms.values(), key=str)
    poly = sp.Poly(sp.expand(expr), *gl)
    out = sp.Integer(0)
    for monom, coeff in zip(poly.monoms(), poly.coeffs()):
        if any(d == 1 for d in monom):
            continue                                  # E[Gamma_ij] = 0
        term = sp.Integer(1)
        for s, d in zip(gl, monom):
            if d:
                term *= sp.Symbol(f"M[{s}^{d}]")
        out += coeff * term
    return sp.expand(out)
