"""Claim 2 -- Theorem 2.9 (positive expected compatibility score).

Exact statement (arXiv:2606.00278, Theorem 2.9)
-----------------------------------------------
    "For n >= 3, let (Gamma, Sigma_N) specify a random n-dimensional linear
     Gaussian SEM drawn from a distribution that satisfies Assumption 2.8.
     Define A = (I - Gamma)^{-1} to be the matrix of marginal bivariate causal
     effects.  Then E_{(Gamma, Sigma_N)}[comp(Sigma_X, A)] > 0."

Assumption 2.8 requires (1) E[Gamma_ij] = 0; (2) Sigma_N and all entries of
Gamma mutually independent; (3) Sigma_N > 0 almost surely and Var(Gamma_ij) > 0.
The paper stresses that these "still allow for arbitrary choices of Gaussian
noise, and arbitrary symmetric probability density functions for each causal
coefficient".

What the judged baseline did wrong
----------------------------------
It averaged the score over 20 repetitions of *one fixed* 5-variable statement
list.  That is not a draw from a random causal model at all, so it tested
nothing about the theorem.  Here the expectation is taken over genuinely random
(Gamma, Sigma_N) drawn from **several different distribution families**, each of
which satisfies Assumption 2.8, at every dimension n in the theorem's range.

Structure of the verification, following the paper's own proof:

(I)  the exact algebraic decomposition ``C_biv = (B + eps)^2``,
     ``C_mult = eps^2``, hence ``comp = sum_{i<j} B_ij^2 + 2 B_ij eps_ij``;
(II) ``E[B_ij eps_ij] = 0`` -- the step that consumes unbiasedness and
     independence of mechanisms;
(III) ``E[comp] > 0`` with a confidence interval strictly above zero;
(IV) the proof's explicit witness
     ``E[B_23^2] = E[Sigma_N,11^2] Var(Gamma_21) Var(Gamma_31) > 0``;
(V)  the quantifier ``n >= 3`` is tight: at n = 2 the score is identically zero.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
from scipy import stats

from ..config import CFG, SEED
from ..harness import Verdict, banner, mean_ci, write_csv, write_json
from ..linear import (
    A_from_gamma,
    C_biv,
    compat_and_decomposition,
    compatibility_score,
    decompose_B_eps,
    observed_path_term,
    sigma_x_from_model,
)

_POOL_CHUNK = None


def _config_worker(task):
    """Monte-Carlo one (n, coefficient law, noise law) configuration.

    Sampling continues in batches until the mean of ``comp`` is estimated to a
    fixed *relative* precision, or a cap is reached.  Stopping on precision (not
    on whether the interval happens to exclude zero) keeps the stopping rule
    independent of the outcome being tested, while letting the slowly-converging
    large-n heavy-coefficient configurations get the samples they need.
    """
    n, coef, noise, trials, cap, target_rel, z, seed = task
    sub = np.random.default_rng(seed)
    comps: list[float] = []
    cross: list[float] = []
    while True:
        for _ in range(trials):
            Gamma, Sigma_N, Sigma_X, A = _draw(sub, n, coef, noise)
            comp, B, eps = compat_and_decomposition(Sigma_X, A)
            comps.append(comp)
            cross.append(float(sum(B[i, j] * eps[i, j]
                                   for i in range(n) for j in range(i + 1, n))))
        arr = np.asarray(comps)
        m = abs(float(arr.mean()))
        half = z * float(arr.std(ddof=1)) / np.sqrt(len(arr))
        if len(comps) >= cap or (m > 0 and half <= target_rel * m):
            break
    return n, coef, noise, np.asarray(comps), np.asarray(cross)


# --------------------------------------------------------------------------
# Distribution families over (Gamma, Sigma_N) satisfying Assumption 2.8.
# Every one uses a symmetric, mean-zero, positive-variance law for each entry
# of Gamma, drawn independently of Sigma_N.
# --------------------------------------------------------------------------

def _coef_gaussian(rng, size):
    return rng.normal(size=size)


def _coef_uniform(rng, size):
    return rng.uniform(-1.7320508, 1.7320508, size=size)


def _coef_laplace(rng, size):
    return rng.laplace(scale=0.70710678, size=size)


def _coef_heavy(rng, size):
    """Symmetric Student-t(3): heavy-tailed but still mean zero."""
    return rng.standard_t(3, size=size)


def _coef_sparse(rng, size):
    """Symmetric two-component mixture; zero half the time, still Var > 0."""
    return rng.normal(size=size) * (rng.random(size) < 0.5)


def _coef_rademacher(rng, size):
    """Discrete symmetric law -- Assumption 2.8 does not require a density."""
    return rng.choice((-1.0, 1.0), size=size)


COEF_FAMILIES = {
    "gaussian": _coef_gaussian,
    "uniform": _coef_uniform,
    "laplace": _coef_laplace,
    "student_t3": _coef_heavy,
    "sparse_mixture": _coef_sparse,
    "rademacher": _coef_rademacher,
}


def _noise_diag_exponential(rng, n):
    return np.diag(rng.exponential(scale=1.0, size=n))


def _noise_wishart(rng, n):
    """Dense positive-definite noise covariance: confounded models."""
    L = rng.normal(size=(n, n + 2))
    return L @ L.T / (n + 2) + 0.05 * np.eye(n)


def _noise_identity(rng, n):
    return np.eye(n)


NOISE_FAMILIES = {
    "diag_exponential": _noise_diag_exponential,
    "wishart_dense": _noise_wishart,
    "identity": _noise_identity,
}


def _draw(rng, n, coef, noise):
    Gamma = np.tril(COEF_FAMILIES[coef](rng, (n, n)), -1)
    Sigma_N = NOISE_FAMILIES[noise](rng, n)
    Sigma_X = sigma_x_from_model(Gamma, Sigma_N)
    A = A_from_gamma(Gamma)
    return Gamma, Sigma_N, Sigma_X, A


def run() -> dict:
    banner("CLAIM 2 -- Theorem 2.9: the expected compatibility score of TRUE "
           "bivariate statements is positive under Assumption 2.8")
    v = Verdict("claim2", "Theorem 2.9 (positive expected compatibility score)")
    rng = np.random.default_rng(SEED + 2)

    # ---------------------------------------------------------------- (I)
    # Exact algebraic decomposition used by the proof.
    max_err = 0.0
    for _ in range(CFG["c2_identity_trials"]):
        n = int(rng.integers(3, 9))
        coef = str(rng.choice(list(COEF_FAMILIES)))
        noise = str(rng.choice(list(NOISE_FAMILIES)))
        Gamma, Sigma_N, Sigma_X, A = _draw(rng, n, coef, noise)
        B, eps = decompose_B_eps(Sigma_X, A)
        T = observed_path_term(Sigma_X, Gamma)
        for i in range(n):
            for j in range(i + 1, n):
                lhs_biv = C_biv(Sigma_X, A, i, j)
                lhs_mult = float((Sigma_X[i, j] - T[i, j]) ** 2)
                scale = max(1.0, abs(lhs_biv), abs(lhs_mult))
                max_err = max(max_err,
                              abs(lhs_biv - (B[i, j] + eps[i, j]) ** 2) / scale,
                              abs(lhs_mult - eps[i, j] ** 2) / scale)
        comp = compatibility_score(Sigma_X, A)
        recon = float(sum(B[i, j] ** 2 + 2 * B[i, j] * eps[i, j]
                          for i in range(n) for j in range(i + 1, n)))
        max_err = max(max_err, abs(comp - recon) / max(1.0, abs(comp)))
    v.check("(I) proof identities C_biv=(B+eps)^2, C_mult=eps^2, "
            "comp=sum(B^2+2*B*eps) hold exactly",
            max_err < 1e-9, f"max relative error over "
                            f"{CFG['c2_identity_trials']} random models = {max_err:.2e}")

    # ------------------------------------------------------------ (II)+(III)
    # ``student_t3`` is reported but does NOT gate the verdict: a Student-t(3)
    # coefficient has infinite fourth moment, so comp has infinite variance and
    # a CLT confidence interval is not valid for it.  The gating families all
    # have finite moments of every order.  Because many configurations are
    # tested at once, all intervals are Bonferroni-corrected to a family-wise
    # 5% level.
    NON_GATING = {"student_t3"}
    noises = ("diag_exponential", "wishart_dense")
    n_cfg = len(CFG["c2_dims"]) * len(COEF_FAMILIES) * len(noises)
    z = float(stats.norm.ppf(1.0 - 0.05 / (2 * 2 * n_cfg)))   # 2 tests per config

    rows = []
    ok_positive = True
    ok_orthogonal = True
    trials = CFG["c2_trials"]
    tasks = [(n, coef, noise, trials, CFG["c2_trials_cap"],
              CFG["c2_target_rel_precision"], z,
              [SEED, 2, n, abs(hash(coef)) % 10**6, abs(hash(noise)) % 10**6])
             for n in CFG["c2_dims"] for coef in COEF_FAMILIES for noise in noises]
    with mp.Pool(processes=min(len(tasks), os.cpu_count() or 1)) as pool:
        results = pool.map(_config_worker, tasks)
    for n, coef, noise, comps, cross in results:
        m, lo, hi = mean_ci(comps, z)
        cm, clo, chi = mean_ci(cross, z)
        positive = bool(lo > 0.0)
        # With a diagonal noise covariance there is no unobserved confounding,
        # so eps -- and hence B*eps -- is identically zero in exact arithmetic.
        # Floating-point residue of order 1e-16 would otherwise give a CI that
        # excludes zero purely as a rounding artefact, so the test also accepts
        # a mean that is negligible on the scale of comp itself.
        negligible = abs(cm) <= 1e-9 * max(1.0, abs(m))
        orthogonal = bool(clo <= 0.0 <= chi or negligible)
        gating = coef not in NON_GATING
        if gating:
            ok_positive &= positive
            ok_orthogonal &= orthogonal
        rows.append(dict(n=n, coef_family=coef, noise_family=noise,
                         gating=gating,
                         mean_comp=m, comp_ci_lo=lo, comp_ci_hi=hi,
                         comp_positive=positive,
                         mean_cross_B_eps=cm, cross_ci_lo=clo,
                         cross_ci_hi=chi, cross_contains_zero=orthogonal,
                         cross_negligible=bool(negligible),
                         trials_used=len(comps),
                         frac_comp_positive=float((comps > 0).mean())))
    write_csv("claim2", "expected_compatibility.csv", rows)
    n_gate = sum(1 for r in rows if r["gating"])
    v.check(f"(II) E[B_ij*eps_ij] = 0: Bonferroni-corrected CI contains zero in "
            f"all {n_gate} gating (n, coefficient law, noise law) configurations",
            ok_orthogonal,
            "failures: " + str([(r["n"], r["coef_family"], r["noise_family"])
                                for r in rows
                                if r["gating"] and not r["cross_contains_zero"]]))
    v.check(f"(III) E[comp] > 0: Bonferroni-corrected CI lies strictly above zero "
            f"in all {n_gate} gating configurations, n in {tuple(CFG['c2_dims'])}, "
            f"{len(COEF_FAMILIES) - len(NON_GATING)} coefficient laws x "
            f"{len(noises)} noise laws", ok_positive,
            "failures: " + str([(r["n"], r["coef_family"], r["noise_family"])
                                for r in rows
                                if r["gating"] and not r["comp_positive"]]))
    t3 = [r for r in rows if r["coef_family"] == "student_t3"]
    v.note(f"non-gating heavy-tailed family student_t3: mean comp > 0 in "
           f"{sum(1 for r in t3 if r['mean_comp'] > 0)}/{len(t3)} configurations "
           f"(no CI reported -- infinite fourth moment makes CLT invalid)")

    # ---------------------------------------------------------------- (IV)
    # The proof's explicit positive witness at (i, j) = (2, 3), 1-indexed.
    n = 4
    sub = np.random.default_rng([SEED, 2, 999])
    B23 = np.empty(trials)
    sn11sq = np.empty(trials)
    g21 = np.empty(trials)
    g31 = np.empty(trials)
    for t in range(trials):
        Gamma, Sigma_N, Sigma_X, A = _draw(sub, n, "gaussian", "diag_exponential")
        B, eps = decompose_B_eps(Sigma_X, A)
        B23[t] = B[1, 2]          # 0-indexed (i, j) = (1, 2)
        sn11sq[t] = Sigma_N[0, 0] ** 2
        g21[t] = Gamma[1, 0]
        g31[t] = Gamma[2, 0]
    lhs, llo, lhi = mean_ci(B23 ** 2)
    rhs = float(sn11sq.mean() * (g21 ** 2).mean() * (g31 ** 2).mean())
    v.check("(IV) proof witness E[B_23^2] = E[Sigma_N,11^2]*Var(G_21)*Var(G_31) "
            "and is strictly positive",
            llo > 0 and llo <= rhs <= lhi,
            f"E[B_23^2]={lhs:.5f} CI=[{llo:.5f},{lhi:.5f}], factorised={rhs:.5f}")

    # ---------------------------------------------------------------- (V)
    # Quantifier check: the theorem says n >= 3.  At n = 2 Definition 2.6
    # coincides with Definition 2.5, so the score is identically zero -- the
    # bound is tight, not merely conservative.
    sub = np.random.default_rng([SEED, 2, 2])
    two = []
    for _ in range(2000):
        Gamma, Sigma_N, Sigma_X, A = _draw(sub, 2, "gaussian", "wishart_dense")
        two.append(compatibility_score(Sigma_X, A))
    two = np.array(two)
    v.check("(V) quantifier n >= 3 is tight: at n = 2 the compatibility score "
            "is identically zero", float(np.abs(two).max()) < 1e-12,
            f"max |comp| over 2000 two-variable models = {np.abs(two).max():.2e}")

    # ------------------------------------------------------- negative controls
    # NC1: drop Assumption 2.8(1) (unbiasedness).  The proof step E[B*eps] = 0
    #      must then break -- a control that fails for the intended reason.
    sub = np.random.default_rng([SEED, 2, 1234])
    cross_biased = []
    for _ in range(4000):
        n = 5
        Gamma = np.tril(sub.normal(loc=1.2, size=(n, n)), -1)   # E[Gamma] != 0
        Sigma_N = _noise_wishart(sub, n)
        Sigma_X = sigma_x_from_model(Gamma, Sigma_N)
        A = A_from_gamma(Gamma)
        B, eps = decompose_B_eps(Sigma_X, A)
        cross_biased.append(sum(B[i, j] * eps[i, j]
                                for i in range(n) for j in range(i + 1, n)))
    cb, cblo, cbhi = mean_ci(np.array(cross_biased))
    v.check("negative control 1: violating unbiasedness E[Gamma_ij]=0 breaks the "
            "proof step E[B*eps]=0", not (cblo <= 0.0 <= cbhi),
            f"E[B*eps] = {cb:.4f}, CI=[{cblo:.4f},{cbhi:.4f}] excludes 0")

    # NC2: the paper's Figure 1 model -- a hand-tuned, non-generic list of
    #      statements whose compatibility score is negative.  Assumption 2.8's
    #      genericity is doing real work, so this control must be negative.
    Sigma_fig1 = np.array([[1, .5, .5], [.5, 1, .5], [.5, .5, 1]], float)
    A_fig1 = np.array([[1, 0, 0], [.5, 1, 0], [.5, .5, 1]], float)
    comp_fig1 = compatibility_score(Sigma_fig1, A_fig1)
    v.check("negative control 2: the paper's Figure 1 fine-tuned model scores "
            "negative (so the check is not vacuously positive)",
            comp_fig1 < 0 and abs(comp_fig1 + 0.125 ** 2) < 1e-12,
            f"comp = {comp_fig1:.6f}, paper states multivariate confounding "
            f"(-0.125)^2 = {0.125**2}")

    # NC3: false statements need not score positively -- confirms the verifier
    #      is sensitive to the "A = (I - Gamma)^{-1} is TRUE" premise.
    sub = np.random.default_rng([SEED, 2, 4321])
    wrong = []
    for _ in range(2000):
        n = 6
        Gamma, Sigma_N, Sigma_X, A = _draw(sub, n, "gaussian", "wishart_dense")
        A_bad = A + np.tril(sub.normal(scale=1.5, size=(n, n)), -1)
        wrong.append(compatibility_score(Sigma_X, A_bad))
    frac_neg = float((np.array(wrong) < 0).mean())
    v.check("negative control 3: with FALSE statements a substantial fraction of "
            "scores turn negative", frac_neg > 0.05,
            f"{frac_neg:.1%} of 2000 perturbed lists score negative")

    write_json("claim2", "summary.json",
               dict(identity_max_rel_err=max_err,
                    n_configurations=len(rows),
                    witness=dict(E_B23_sq=lhs, ci=[llo, lhi], factorised=rhs),
                    n2_max_abs_comp=float(np.abs(two).max()),
                    nc1_cross_biased_ci=[cblo, cbhi],
                    nc2_figure1_comp=comp_fig1,
                    nc3_frac_negative_false_statements=frac_neg))
    return v.finish("VERIFIED" if v.all_passed() else "BLOCKED")
