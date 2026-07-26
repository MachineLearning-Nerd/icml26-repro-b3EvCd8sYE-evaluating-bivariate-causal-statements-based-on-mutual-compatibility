"""Claim 6 -- Lemma 3.7 and the empirical behaviour of the heuristic score.

Exact statements
----------------
Lemma 3.7 (arXiv:2606.00278):
    "For any statement graph G, we have
       1. c(G) >= incomp(G);
       2. c(G) = 0  <=>  incomp(G) = 0."

Section 3.4, on the synthetic experiment behind Figure 5:
    "As expected, incompatibility scores increase on average monotonically with
     the number of injected errors."

Section 3.4, on Figures 6 and 7 (LLM statement graphs):
    "Among this subset of more informative statement graphs, the results again
     reveal a correlation between low incompatibility scores and higher general
     model capacity."

Note the quantifiers.  Lemma 3.7 is universal over *all* statement graphs, so
it is checked by exhaustive enumeration over the complete finite domain for
every n where that is possible.  The Figure 5 statement is explicitly about the
*average*, and monotonicity is claimed in the number of injected errors, not in
any other parameter -- so that is exactly what is tested, panel by panel.

The judged baseline tested two hand-picked graphs (one compatible 5-node graph
and one 3-cycle) and did not test monotonicity at all.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from ..config import CFG, SEED
from ..harness import (Verdict, banner, mean_ci, stable_hash, write_csv,
                       write_json)
from ..generate import inject_graph_errors, sample_graphical_model
from ..graphical import (
    StatementGraph,
    enumerate_transitively_closed_dags,
    exact_incompatibility,
    heuristic_incompatibility,
)

_DAGS_CACHE: dict[int, list] = {}


def _tc_dags(n: int):
    if n not in _DAGS_CACHE:
        _DAGS_CACHE[n] = list(enumerate_transitively_closed_dags(n))
    return _DAGS_CACHE[n]


def _all_statement_graphs(n: int):
    dslots = [(u, v) for u in range(n) for v in range(n) if u != v]
    bslots = [(u, v) for u in range(n) for v in range(u + 1, n)]
    for dm in range(1 << len(dslots)):
        D = np.zeros((n, n), bool)
        for k, e in enumerate(dslots):
            if dm >> k & 1:
                D[e] = True
        for bm in range(1 << len(bslots)):
            B = np.zeros((n, n), bool)
            for k, (u, v) in enumerate(bslots):
                if bm >> k & 1:
                    B[u, v] = B[v, u] = True
            yield StatementGraph(n, D, B)


def _lemma_worker(task):
    n, dbytes, bbytes = task
    D = np.frombuffer(dbytes, bool).reshape(n, n).copy()
    B = np.frombuffer(bbytes, bool).reshape(n, n).copy()
    g = StatementGraph(n, D, B)
    exact = exact_incompatibility(g, _tc_dags(n))
    heur = heuristic_incompatibility(g)
    return int(exact), int(heur)


def _fig5_worker(task):
    """One (panel, parameter value, error count) cell of Figure 5."""
    panel, value, fixed, k_err, reps, seed = task
    params = dict(fixed)
    params[panel] = value
    rng = np.random.default_rng(seed)
    scores, detail = [], []
    for _ in range(reps):
        g = sample_graphical_model(params["n"], params["m"], params["p"], rng)
        if k_err:
            g = inject_graph_errors(g, k_err, rng)
        s, d = heuristic_incompatibility(g, detail=True)
        scores.append(s)
        detail.append(d)
    scores = np.array(scores, float)
    m, lo, hi = mean_ci(scores)
    agg = {k: float(np.mean([d[k] for d in detail])) for k in detail[0]}
    return dict(panel=panel, value=value, n=params["n"], m=params["m"],
                p=params["p"], n_errors=k_err, reps=reps,
                mean_score=m, ci_lo=lo, ci_hi=hi,
                median_score=float(np.median(scores)), **agg)


def run() -> dict:
    banner("CLAIM 6 -- Lemma 3.7 (heuristic upper bound, zero iff zero) and "
           "Figure 5 (monotone increase with injected errors)")
    v = Verdict("claim6", "Lemma 3.7 + heuristic incompatibility behaviour")

    # ------------------------------------------------- (A) Lemma 3.7, exhaustive
    rows = []
    lemma_ok = True
    for n in CFG["c6_exhaustive_n"]:
        tasks = [(n, g.D.tobytes(), g.Bd.tobytes()) for g in _all_statement_graphs(n)]
        with mp.Pool(processes=min(len(tasks), os.cpu_count() or 1)) as pool:
            out = pool.map(_lemma_worker, tasks, chunksize=64)
        viol_ub = sum(1 for e, h in out if h < e)
        viol_zero = sum(1 for e, h in out if (h == 0) != (e == 0))
        tight = sum(1 for e, h in out if h == e)
        lemma_ok &= (viol_ub == 0 and viol_zero == 0)
        rows.append(dict(n=n, domain="exhaustive: all statement graphs",
                         n_graphs=len(out), violations_upper_bound=viol_ub,
                         violations_zero_iff_zero=viol_zero,
                         frac_exact=tight / len(out),
                         mean_gap=float(np.mean([h - e for e, h in out]))))
        v.check(f"n={n}: Lemma 3.7(1) c(G) >= incomp(G) holds for ALL "
                f"{len(out)} statement graphs on {n} labelled vertices",
                viol_ub == 0)
        v.check(f"n={n}: Lemma 3.7(2) c(G) = 0 <=> incomp(G) = 0 holds for ALL "
                f"{len(out)} statement graphs", viol_zero == 0,
                f"heuristic is exactly tight on {tight/len(out):.1%} of them, "
                f"mean gap {np.mean([h - e for e, h in out]):.3f}")

    for n in CFG["c6_sampled_n"]:
        rng = np.random.default_rng([SEED, 6, n])
        tasks = []
        for _ in range(CFG["c6_samples"]):
            d = float(rng.choice([0.2, 0.4, 0.6]))
            D = rng.random((n, n)) < d
            np.fill_diagonal(D, False)
            B = rng.random((n, n)) < d
            B = B | B.T
            np.fill_diagonal(B, False)
            tasks.append((n, D.tobytes(), B.tobytes()))
        with mp.Pool(processes=min(len(tasks), os.cpu_count() or 1)) as pool:
            out = pool.map(_lemma_worker, tasks)
        viol_ub = sum(1 for e, h in out if h < e)
        viol_zero = sum(1 for e, h in out if (h == 0) != (e == 0))
        lemma_ok &= (viol_ub == 0 and viol_zero == 0)
        rows.append(dict(n=n, domain=f"random sample of {len(out)} statement graphs",
                         n_graphs=len(out), violations_upper_bound=viol_ub,
                         violations_zero_iff_zero=viol_zero,
                         frac_exact=sum(1 for e, h in out if e == h) / len(out),
                         mean_gap=float(np.mean([h - e for e, h in out]))))
        v.check(f"n={n}: Lemma 3.7 holds on {len(out)} sampled statement graphs "
                f"(exhaustive search infeasible at this size)",
                viol_ub == 0 and viol_zero == 0)
    write_csv("claim6", "lemma37_exhaustive.csv", rows)

    # ------------------------------------------- (B) Figure 5: monotone increase
    tasks = []
    for panel, values, fixed in CFG["c6_panels"]:
        for value in values:
            for k in CFG["c6_errors"]:
                tasks.append((panel, value, tuple(fixed.items()), k, CFG["c6_reps"],
                              [SEED, 6, stable_hash(panel) % 1000, int(value * 100), k]))
    tasks = [(p, v_, dict(f), k, r, s) for (p, v_, f, k, r, s) in tasks]
    with mp.Pool(processes=min(len(tasks), os.cpu_count() or 1)) as pool:
        fig5 = pool.map(_fig5_worker, tasks)
    write_csv("claim6", "figure5_monotonicity.csv", fig5)

    curves = {}
    for r in fig5:
        curves.setdefault((r["panel"], r["value"]), []).append(r)
    mono_ok = True
    for key, rs in sorted(curves.items(), key=lambda kv: str(kv[0])):
        rs.sort(key=lambda r: r["n_errors"])
        means = [r["mean_score"] for r in rs]
        nondec = all(means[i + 1] >= means[i] - 1e-12 for i in range(len(means) - 1))
        mono_ok &= nondec
        print(f"    panel {key[0]}={key[1]}: mean c(G) over errors "
              f"{[r['n_errors'] for r in rs]} = {[round(x, 2) for x in means]}"
              f"  {'monotone' if nondec else 'NOT MONOTONE'}", flush=True)
    v.check(f"Figure 5: average incompatibility score is monotonically "
            f"non-decreasing in the number of injected errors, in all "
            f"{len(curves)} parameter curves", mono_ok)

    zero_err = [r for r in fig5 if r["n_errors"] == 0]
    all_zero = all(abs(r["mean_score"]) < 1e-12 for r in zero_err)
    v.check("negative control: with zero injected errors the correct statement "
            "graph always scores exactly 0 (so the score is not measuring noise)",
            all_zero,
            f"{len(zero_err)} zero-error cells, max mean score = "
            f"{max(r['mean_score'] for r in zero_err):.3g}")

    # A control with power: a *random* statement graph (not a perturbed correct
    # one) must score well above the k-error curve, otherwise the score would be
    # insensitive to how wrong the statements are.
    rng = np.random.default_rng([SEED, 6, 777])
    rand_scores = []
    for _ in range(50):
        n = 10
        D = rng.random((n, n)) < 0.5
        np.fill_diagonal(D, False)
        B = rng.random((n, n)) < 0.5
        B = B | B.T
        np.fill_diagonal(B, False)
        rand_scores.append(heuristic_incompatibility(StatementGraph(n, D, B)))
    worst_k = max(r["mean_score"] for r in fig5)
    v.check("negative control: uniformly random statement graphs score higher "
            "than every injected-error cell", float(np.mean(rand_scores)) > worst_k,
            f"random mean = {np.mean(rand_scores):.2f} vs largest injected-error "
            f"cell mean = {worst_k:.2f}")

    # --------------------------------------------- (C) Figures 6 and 7: LLMs
    from .llm_scores import score_graphical_transcripts
    llm = score_graphical_transcripts(v)
    write_json("claim6", "figures67_llm.json", llm)

    write_json("claim6", "summary.json",
               dict(lemma37=rows, figure5_curves=len(curves),
                    figure5_all_monotone=mono_ok,
                    zero_error_all_zero=all_zero,
                    random_graph_mean=float(np.mean(rand_scores))))
    status = "VERIFIED" if (lemma_ok and v.all_passed()) else "BLOCKED"
    return v.finish(status)
