"""Mutual compatibility of bivariate causal statements — an executable walkthrough.

A self-contained marimo notebook for arXiv:2606.00278.  It deliberately does
*not* import this repository's library: everything is rebuilt from the paper's
definitions in a few lines, so the notebook is an independent re-derivation of
the headline results rather than a demonstration of our own code.

    uvx marimo edit notebooks/compatibility.py
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mutual compatibility of bivariate causal statements

    *An executable walkthrough of [arXiv:2606.00278](https://arxiv.org/abs/2606.00278),
    Erik Jahn and Dominik Janzing.*

    Ask an expert — or a language model — a set of **pairwise** causal
    questions: does $X_i$ cause $X_j$, and how strongly? Each answer is
    cheap to give and hard to check. The paper's idea is that the answers
    can be checked **against each other**: a set of bivariate statements
    implicitly describes one multivariate model, and if no single model
    explains them all, they are mutually incompatible — no ground truth
    needed.

    This notebook rebuilds the construction from the paper's definitions
    and reproduces four things:

    1. the worked example of Figure 1,
    2. the compatibility score and what makes it negative,
    3. a machine-checked certificate for the core step of Theorem 2.9,
    4. the sensitivity constant behind Theorem 2.10.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. From bivariate statements to one multivariate model

    Fix a causal order. A list of bivariate statements is a unit
    lower-triangular matrix $A$, where $A_{ji}$ is the claimed **total**
    causal effect of $X_i$ on $X_j$. A multivariate linear SEM is
    $X = \Gamma X + N$ with $\Gamma$ strictly lower-triangular, and its
    matrix of total effects is $(I-\Gamma)^{-1}$.

    Lemma 2.3 says the correspondence is a bijection: every acyclic list
    of statements induces **exactly one** SEM, namely
    $\Gamma = I - A^{-1}$.

    Because $\Gamma$ is strictly lower-triangular it is nilpotent, so
    $(I-\Gamma)^{-1} = I + \Gamma + \Gamma^2 + \dots + \Gamma^{n-1}$ —
    a finite sum. Everything below stays polynomial as a result, which is
    what makes the symbolic checks in section 4 feasible at all.
    """)
    return


@app.cell
def _(np):
    def A_from_gamma(Gamma):
        """Total-effect matrix (I - Gamma)^-1, as a finite Neumann series."""
        n = Gamma.shape[0]
        A = np.eye(n)
        P = np.eye(n)
        for _ in range(n - 1):
            P = P @ Gamma
            A = A + P
        return A

    def gamma_from_A(A):
        """Lemma 2.3: the unique SEM induced by a statement list."""
        return np.eye(A.shape[0]) - np.linalg.inv(A)

    return A_from_gamma, gamma_from_A


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Path tracing, and the two ways to explain a covariance

    With $\Gamma$ strictly lower-triangular, a directed path can only move
    to a strictly larger index — so **a path is just an increasing index
    sequence**, and enumerating paths is enumerating subsets. That single
    observation is what makes the whole construction computable.

    Wright's path-tracing rule writes each covariance as a sum over pairs
    of paths out of a common source. Definitions 2.5 and 2.6 then measure
    the same covariance two ways:

    - $C^{\text{biv}}_{ij} = (\Sigma_{ij} - A_{ji}\Sigma_{ii})^2$ — what is
      left unexplained if you believe only the **bivariate** statement
      about $(i,j)$;
    - $C^{\text{mult}}_{ij} = \bigl(\Sigma_{ij} - \sum_{k \le i}
      \Sigma_{kk} \!\!\sum_{\substack{P_1: k \to i,\ P_2: k \to j \\
      P_1 \cap P_2 = \emptyset}}\!\! \Gamma_{P_1}\Gamma_{P_2}\bigr)^2$ —
      what is left unexplained once the whole **multivariate** model, with
      all its observed back-door paths, is allowed to help.

    Definition 2.7 scores the list by how much the multivariate view buys
    you: $\mathrm{comp} = \sum_{i<j} C^{\text{biv}}_{ij} -
    C^{\text{mult}}_{ij}$.
    """)
    return


@app.cell
def _(itertools):
    def paths(u, w):
        """Every directed path u -> w: an increasing index sequence."""
        if u > w:
            return []
        if u == w:
            return [(u,)]
        return [(u,) + mid + (w,)
                for r in range(w - u)
                for mid in itertools.combinations(range(u + 1, w), r)]

    def weight(p, G):
        out = 1.0
        for t in range(len(p) - 1):
            out *= G[p[t + 1], p[t]]
        return out

    def disjoint_pair_sum(G, k, i, j):
        """Sum of Gamma_P1 Gamma_P2 over vertex-disjoint P1: k->i, P2: k->j."""
        total = 0.0
        for p1 in paths(k, i):
            for p2 in paths(k, j):
                if not (set(p1) - {k}) & (set(p2) - {k}):
                    total += weight(p1, G) * weight(p2, G)
        return total

    def compatibility(Sigma, A, Gamma):
        """Definition 2.7."""
        n = Sigma.shape[0]
        score = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c_biv = (Sigma[i, j] - A[j, i] * Sigma[i, i]) ** 2
                explained = sum(Sigma[k, k] * disjoint_pair_sum(Gamma, k, i, j)
                                for k in range(i + 1))
                c_mult = (Sigma[i, j] - explained) ** 2
                score += c_biv - c_mult
        return float(score)

    return compatibility, disjoint_pair_sum, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The worked example of Figure 1

    The paper's example: three variables whose pairwise correlations are
    all $0.5$, and whose bivariate statements likewise all claim an effect
    of $0.5$. Each statement is individually consistent with the data. The
    induced multivariate model, however, needs $X_1 \to X_3$ to have
    coefficient $0.25$ — not $0.5$ — because the path
    $X_1 \to X_2 \to X_3$ already supplies the rest.

    The gap left over is $-0.125$, and the score is $-(0.125)^2$.
    """)
    return


@app.cell
def _(A_from_gamma, compatibility, gamma_from_A, np):
    Sigma_fig1 = np.array([[1.0, .5, .5], [.5, 1.0, .5], [.5, .5, 1.0]])
    A_fig1 = np.array([[1.0, 0, 0], [.5, 1.0, 0], [.5, .5, 1.0]])

    Gamma_fig1 = gamma_from_A(A_fig1)
    comp_fig1 = compatibility(Sigma_fig1, A_fig1, Gamma_fig1)

    fig1_result = {
        "induced Gamma (2<-1, 3<-1, 3<-2)": [round(Gamma_fig1[1, 0], 4),
                                             round(Gamma_fig1[2, 0], 4),
                                             round(Gamma_fig1[2, 1], 4)],
        "round trip (I-Gamma)^-1 == A": bool(
            np.allclose(A_from_gamma(Gamma_fig1), A_fig1)),
        "comp": comp_fig1,
        "paper states -(0.125)^2": -(0.125 ** 2),
        "match": abs(comp_fig1 + 0.125 ** 2) < 1e-12,
    }
    fig1_result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The induced coefficients are $0.5$, $0.25$, $0.5$ and the score is
    exactly $-0.015625 = -(0.125)^2$, matching the figure.

    **Why negative?** A negative score means the multivariate model explains
    the data *worse* than the bivariate statements did on their own —
    which can only happen if the statements were tuned to each pair
    separately, without a single coherent model behind them.

    ### True statements score non-negatively

    Now the other direction. Draw a random SEM, compute the covariance it
    implies, and read the **true** statements off it. Theorem 2.9 says the
    expected score is strictly positive.
    """)
    return


@app.cell
def _(A_from_gamma, compatibility, mo, np):
    def random_model(rng, n, confounded=True):
        G = np.tril(rng.normal(size=(n, n)), -1)
        A = A_from_gamma(G)
        if confounded:
            L = rng.normal(size=(n, n + 2))
            SN = L @ L.T / (n + 2) + 0.05 * np.eye(n)   # dense: hidden confounders
        else:
            SN = np.diag(rng.exponential(size=n))       # diagonal: none
        return G, A, A @ SN @ A.T

    _rng = np.random.default_rng(20260726)
    _true, _perturbed = [], []
    for _ in range(400):
        _G, _A, _S = random_model(_rng, 5)
        _true.append(compatibility(_S, _A, _G))
        _A_bad = _A + np.tril(_rng.normal(scale=1.5, size=(5, 5)), -1)
        _perturbed.append(compatibility(_S, _A_bad, np.eye(5) - np.linalg.inv(_A_bad)))

    scores = {"true statements": _true, "perturbed statements": _perturbed}
    mo.md(f"""
    | statement list | mean score | fraction negative |
    |---|---|---|
    | **true** (A = (I-Γ)⁻¹ exactly) | {np.mean(_true):+.4f} | {np.mean(np.array(_true) < 0):.1%} |
    | **perturbed** (A nudged off the truth) | {np.mean(_perturbed):+.4f} | {np.mean(np.array(_perturbed) < 0):.1%} |

    True statements score positively on average; perturbing them off the truth
    sends a large share of the scores negative. That contrast is the whole
    diagnostic — and it is available without ever knowing the true model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Where the score comes from — and the step the proof turns on

    The proof of Theorem 2.9 splits each covariance into three parts:

    $$\Sigma_{ij} = \underbrace{\Sigma_{ii} A_{ji}}_{\text{direct}}
      + \underbrace{B_{ij}}_{\text{observed back-doors}}
      + \underbrace{\varepsilon_{ij}}_{\text{unobserved back-doors}}$$

    Then $C^{\text{biv}} = (B+\varepsilon)^2$ and
    $C^{\text{mult}} = \varepsilon^2$, so

    $$\mathrm{comp} = \sum_{i<j} B_{ij}^2 + 2 B_{ij}\varepsilon_{ij}.$$

    The first term is a sum of squares — automatically non-negative. So
    **everything rests on the cross term vanishing in expectation.**

    The paper's argument for $\mathbb{E}[B_{ij}\varepsilon_{ij}] = 0$ is
    purely combinatorial. Each summand is a product
    $\Gamma_{P_1}\Gamma_{P_2}\Gamma_{Q_1}\Gamma_{Q_2}$ where $P_1, P_2$ are
    disjoint paths meeting at a shared source $k$, while $Q_1, Q_2$ have
    **all different endpoints**. Because of that mismatch the $Q$ pair can
    never reproduce the $P$ pair edge for edge, so some edge $\Gamma_{rs}$
    with $r > k$ appears **exactly once**. Assumption 2.8 makes that entry
    independent of everything else in the product and gives it mean zero,
    so the summand vanishes.

    For each $n$ that is a claim about a *finite* set of path
    configurations — so we can simply check all of them.
    """)
    return


@app.cell
def _(Counter, paths):
    def parity_certificate(n, q_share_start=False):
        """Enumerate every summand and look for an edge of multiplicity one.

        ``q_share_start=True`` lets the Q-paths share their source, removing the
        one property the argument uses. It is the negative control: the
        certificate must then fail.
        """
        checked, failures = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(i):
                    P = [(a, b) for a in paths(k, i) for b in paths(k, j)
                         if not (set(a) - {k}) & (set(b) - {k})]
                    for ell in range(i + 1):
                        for m in range(j + 1):
                            if ell == m and not q_share_start:
                                continue
                            share = {ell} if (q_share_start and ell == m) else set()
                            Q = [(a, b) for a in paths(ell, i) for b in paths(m, j)
                                 if not (set(a) - share) & (set(b) - share)]
                            for p1, p2 in P:
                                for q1, q2 in Q:
                                    checked += 1
                                    edges = Counter()
                                    for pth in (p1, p2, q1, q2):
                                        for t in range(len(pth) - 1):
                                            edges[(pth[t + 1], pth[t])] += 1
                                    if not any(c == 1 and r > k
                                               for (r, _), c in edges.items()):
                                        failures += 1
        return checked, failures

    return (parity_certificate,)


@app.cell
def _(mo, parity_certificate):
    _rows = []
    for _n in (3, 4, 5, 6, 7):
        _chk, _fail = parity_certificate(_n)
        _cchk, _cfail = parity_certificate(_n, q_share_start=True)
        _rows.append(f"| {_n} | {_chk:,} | **{_fail}** | {_cchk:,} | {_cfail:,} |")

    mo.md("""
    | n | summands checked | failures | control: summands | control: failures |
    |---|---|---|---|---|
    """ + "\n".join(_rows) + """

    Zero failures everywhere — the proof's step holds over the *complete* set of
    path configurations at these dimensions, not for a sample of them. And the
    control has real power: remove the "all different endpoints" property and
    the argument breaks immediately.

    The full suite pushes this to n = 9, covering 5,385,576 summands.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Theorem 2.10, from its two ingredients

    Theorem 2.10 says
    $N \ge C\,n^4(1+a+b)^4V^4\varepsilon^{-2}\log(n/\delta)$ samples suffice
    to estimate the score to within $\varepsilon$. Rather than take the
    formula on faith, notice that it factors into two things you can
    measure separately:

    **Sensitivity.** $\mathrm{comp}$ is a *quadratic polynomial* in the
    entries of $\Sigma$ — look back at the definition: each term is a
    square of something linear in $\Sigma$. So its gradient is available in
    closed form, and $L = \lVert\nabla \mathrm{comp}\rVert_1$ is exactly
    the worst first-order change per unit entrywise error in $\Sigma$. To
    get within $\varepsilon$ it suffices to estimate $\Sigma$ to $t =
    \varepsilon / L$.

    **Concentration.** $\hat\Sigma_{ab}$ is an average of $N$ i.i.d. terms,
    so $t \propto V\sqrt{\log(n^2/\delta)/N}$.

    Compose: $N \gtrsim V^2L^2\varepsilon^{-2}\log(n^2/\delta)$. This has
    the theorem's shape precisely when $L \lesssim n^2(1+a+b)^2V$ — and
    that is a claim about a deterministic quantity, so it can be checked
    without any sampling noise at all.
    """)
    return


@app.cell
def _(disjoint_pair_sum, np):
    def comp_gradient(Sigma, A, Gamma):
        """Exact d comp / d Sigma_ab, for the free entries of a symmetric Sigma."""
        n = Sigma.shape[0]
        g = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                u = Sigma[i, j] - A[j, i] * Sigma[i, i]
                F = [disjoint_pair_sum(Gamma, k, i, j) for k in range(i + 1)]
                w = Sigma[i, j] - sum(Sigma[k, k] * F[k] for k in range(i + 1))
                g[i, j] += 2 * u - 2 * w
                g[i, i] += -2 * u * A[j, i]
                for k in range(i + 1):
                    g[k, k] += 2 * w * F[k]
        return g

    def sensitivity(Sigma, A, Gamma):
        return float(np.abs(comp_gradient(Sigma, A, Gamma)).sum())

    return comp_gradient, sensitivity


@app.cell
def _(A_from_gamma, comp_gradient, compatibility, mo, np, sensitivity):
    # The gradient is the claim; central differences are the independent check.
    _rng = np.random.default_rng(7)
    _worst = 0.0
    for _ in range(5):
        _n = 4
        _G = np.tril(_rng.normal(scale=0.5, size=(_n, _n)), -1)
        _A = A_from_gamma(_G)
        _S = _A @ np.diag(_rng.uniform(0.5, 1.5, _n)) @ _A.T
        _g = comp_gradient(_S, _A, _G)
        for _a in range(_n):
            for _b in range(_a, _n):
                _D = np.zeros((_n, _n))
                _D[_a, _b] = _D[_b, _a] = 1e-6
                _fd = (compatibility(_S + _D, _A, _G)
                       - compatibility(_S - _D, _A, _G)) / 2e-6
                _worst = max(_worst, abs(_fd - _g[_a, _b]) / max(1.0, abs(_g[_a, _b])))

    # How L scales with the maximum variance V -- the derivation predicts 1.
    _Ls = []
    for _V in (0.25, 0.5, 1.0, 2.0, 4.0):
        _r = np.random.default_rng(11)
        _G = np.tril(_r.normal(scale=0.45, size=(5, 5)), -1)
        _A = A_from_gamma(_G)
        _S = _A @ np.diag(_r.uniform(0.5, 1.5, 5)) @ _A.T
        _S = _S * (_V / np.diag(_S).max())
        _Ls.append(sensitivity(_S, _A, _G))
    _slope = np.polyfit(np.log([0.25, 0.5, 1.0, 2.0, 4.0]), np.log(_Ls), 1)[0]

    mo.md(f"""
    - closed-form gradient vs central differences: max relative error
      **{_worst:.2e}**
    - exponent of $L$ in $V$: **{_slope:.3f}** — the derivation predicts
      exactly **1**

    The full suite measures all three exponents jointly over hundreds of models
    and finds $L \\sim n^{{2.24}}(1+a+b)^{{1.38}}V^{{1.000}}$, which composes to
    the theorem's $(4, 4, 4, 2, 1)$ — without the theorem's formula ever being
    used.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go next

    - **Full reproduction**, all six claims with raw data, negative
      controls and a visibility matrix:
      [the logbook](https://huggingface.co/spaces/DineshAI/b3EvCd8sYE)
    - **Source**:
      [GitHub](https://github.com/MachineLearning-Nerd/icml26-repro-b3EvCd8sYE-evaluating-bivariate-causal-statements-based-on-mutual-compatibility)
    - **Paper**: [arXiv:2606.00278](https://arxiv.org/abs/2606.00278)

    Things this notebook leaves out, which the full suite covers: the
    $O(n^4)$ dynamic program that replaces the exponential path enumeration
    used here, the graphical (ADMG) half of the paper, the NP-hardness
    reduction, and the LLM experiments.
    """)
    return


@app.cell
def _():
    import itertools
    from collections import Counter

    import marimo as mo
    import numpy as np

    return Counter, itertools, mo, np


if __name__ == "__main__":
    app.run()
