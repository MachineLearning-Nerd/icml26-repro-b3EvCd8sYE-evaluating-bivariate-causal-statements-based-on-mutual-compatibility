"""Claim 1 -- Lemma 2.3 (existence of a unique compatible SEM).

Exact statement (arXiv:2606.00278, Lemma 2.3)
---------------------------------------------
    "Let A be a unit lower-triangular matrix of linear bivariate causal
     statements for a vector X of observed variables and define Gamma = I -
     A^{-1}.  Then, the multivariate structural equation model X = Gamma X + N,
     where the distribution of N is defined as the distribution of (I - Gamma)X,
     is the unique linear SEM whose pairwise marginal submodels are
     X_j = A_ji X_i + Ntilde_ij."

Quantifier: *for all* unit lower-triangular A of every size n.  A finite
numerical spot-check therefore cannot establish it -- which is why this verifier
reconstructs the derivation **symbolically**, over matrices whose entries are
free symbols, so that each identity is proved as a rational-function identity
in n(n-1)/2 indeterminates rather than at particular numbers.

The proof has two halves and both are checked:

(E) *Existence.*  For an SEM with strictly lower-triangular coefficient matrix
    Gamma, marginalising to the pair (i, j) via Lemma 2.1 gives
    ``Gamma_YY + Gamma_YZ (I - Gamma_ZZ)^{-1} Gamma_ZY = [[0,0],[r,0]]`` with
    ``r = ((I - Gamma)^{-1})_ji``.  Hence the pairwise submodels of
    ``Gamma = I - A^{-1}`` are exactly ``X_j = A_ji X_i + Ntilde_ij``.

(U) *Uniqueness.*  Requiring all pairwise submodels to match A is, by (E),
    equivalent to ``(I - Gamma)^{-1} = A``; and ``Gamma -> (I - Gamma)^{-1}`` is
    a bijection from strictly lower-triangular matrices onto unit
    lower-triangular matrices.  Both directions of that bijection are verified
    symbolically, which is what makes the compatible SEM *unique*.
"""

from __future__ import annotations

import sympy as sp

from ..config import CFG
from ..harness import Verdict, banner, write_json


def _symbolic_gamma(n: int):
    G = sp.zeros(n, n)
    syms = []
    for i in range(n):
        for j in range(i):
            s = sp.Symbol(f"g_{i}_{j}")
            G[i, j] = s
            syms.append(s)
    return G, syms


def _symbolic_unit_lower(n: int):
    A = sp.eye(n)
    syms = []
    for i in range(n):
        for j in range(i):
            s = sp.Symbol(f"a_{i}_{j}")
            A[i, j] = s
            syms.append(s)
    return A, syms


def _pairwise_marginal(G, n, i, j):
    """Lemma 2.1 marginalisation of Gamma onto the ordered pair (i, j)."""
    Y = [i, j]
    Z = [k for k in range(n) if k not in Y]
    G_YY = G[Y, Y]
    if Z:
        G_YZ = G[Y, Z]
        G_ZZ = G[Z, Z]
        G_ZY = G[Z, Y]
        M = G_YY + G_YZ * (sp.eye(len(Z)) - G_ZZ).inv() * G_ZY
    else:
        M = G_YY
    return sp.simplify(M)


def run() -> dict:
    banner("CLAIM 1 -- Lemma 2.3: an acyclic list of linear bivariate statements "
           "induces exactly one multivariate SEM")
    v = Verdict("claim1", "Lemma 2.3 (existence of a unique compatible SEM)")
    records = []

    for n in CFG["c1_symbolic_n"]:
        G, gsyms = _symbolic_gamma(n)
        A_of_G = sp.simplify((sp.eye(n) - G).inv())

        # (E) every pairwise marginal equals ((I - Gamma)^{-1})_ji
        ok_marg = True
        worst = None
        for i in range(n):
            for j in range(i + 1, n):
                M = _pairwise_marginal(G, n, i, j)
                # marginal must be [[0, 0], [r, 0]] with r = A_ji
                good = (sp.simplify(M[0, 0]) == 0 and sp.simplify(M[0, 1]) == 0
                        and sp.simplify(M[1, 1]) == 0
                        and sp.simplify(M[1, 0] - A_of_G[j, i]) == 0)
                if not good:
                    ok_marg = False
                    worst = (i, j)
        v.check(f"n={n}: pairwise marginal of Gamma equals ((I-Gamma)^-1)_ji "
                f"for all {n*(n-1)//2} pairs (symbolic)", ok_marg,
                "" if ok_marg else f"first failure at pair {worst}")

        # (U) the map Gamma -> (I - Gamma)^{-1} is a bijection onto unit
        #     lower-triangular matrices; both round-trips are identities.
        A, asyms = _symbolic_unit_lower(n)
        G_of_A = sp.simplify(sp.eye(n) - A.inv())
        rt1 = sp.simplify((sp.eye(n) - G_of_A).inv() - A)
        rt2 = sp.simplify(sp.eye(n) - (sp.eye(n) - G).inv().inv() - G)
        strictly_lower = all(sp.simplify(G_of_A[i, j]) == 0
                             for i in range(n) for j in range(i, n))
        unit_lower = (all(sp.simplify(A_of_G[i, i] - 1) == 0 for i in range(n))
                      and all(sp.simplify(A_of_G[i, j]) == 0
                              for i in range(n) for j in range(i + 1, n)))
        v.check(f"n={n}: Gamma = I - A^-1 is strictly lower-triangular for every "
                f"unit lower-triangular A (symbolic)", strictly_lower)
        v.check(f"n={n}: A = (I-Gamma)^-1 is unit lower-triangular for every "
                f"strictly lower-triangular Gamma (symbolic)", unit_lower)
        v.check(f"n={n}: round-trips (I - (I - A^-1))^-1 = A and "
                f"I - ((I-Gamma)^-1)^-1 = Gamma are identities (symbolic)",
                rt1 == sp.zeros(n, n) and rt2 == sp.zeros(n, n))

        records.append(dict(n=n, n_free_symbols=len(gsyms),
                            n_pairs=n * (n - 1) // 2,
                            marginal_identity=bool(ok_marg),
                            bijection=bool(strictly_lower and unit_lower
                                           and rt1 == sp.zeros(n, n))))

    # --------------------------------------------------------------------
    # Negative control.  The verifier must reject a *wrong* construction.
    # We substitute the plausible-looking but incorrect Gamma = A - I and
    # confirm the marginal identity fails; a control that passed here would
    # prove the check has no discriminating power.
    # --------------------------------------------------------------------
    n = 3
    A, _ = _symbolic_unit_lower(n)
    G_wrong = A - sp.eye(n)
    A_wrong = sp.simplify((sp.eye(n) - G_wrong).inv())
    mismatch = [(i, j) for i in range(n) for j in range(i + 1, n)
                if sp.simplify(A_wrong[j, i] - A[j, i]) != 0]
    v.check("negative control: the incorrect construction Gamma = A - I fails "
            "the same marginal identity", len(mismatch) > 0,
            f"mismatching pairs: {mismatch}")

    # A second control: a Gamma that is *not* strictly lower-triangular breaks
    # the [[0,0],[r,0]] structure that the derivation relies on.
    G_bad = sp.Matrix([[0, sp.Symbol("u"), 0], [sp.Symbol("t"), 0, 0], [0, 0, 0]])
    M = _pairwise_marginal(G_bad, 3, 0, 1)
    v.check("negative control: a non-triangular Gamma violates the "
            "[[0,0],[r,0]] marginal structure", sp.simplify(M[0, 1]) != 0,
            f"upper entry = {sp.simplify(M[0,1])}")

    write_json("claim1", "symbolic_results.json",
               dict(records=records,
                    negative_control_mismatching_pairs=[list(p) for p in mismatch]))
    return v.finish("VERIFIED" if v.all_passed() else "BLOCKED")
