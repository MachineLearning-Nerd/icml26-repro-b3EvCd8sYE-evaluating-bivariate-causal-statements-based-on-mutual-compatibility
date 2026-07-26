"""Claim 4 -- Figure 2 (synthetic) and Figure 4 (LLM) compatibility scores.

Exact statements
----------------
Section 2.7, on Figure 2:
    "Across all tested combinations of model parameters, the fraction of
     statements with positive compatibility scores strictly decreases as the
     error increases, which implies that our compatibility score successfully
     distinguishes between correct and incorrect statements on average.  Our
     results also provide empirical evidence for the validity of Assumption 2.4,
     as most of the true sets of bivariate causal statements have positive
     compatibility scores."

Section 2.7, on Figure 4:
    "The results show that our compatibility score can demonstrate systematic
     differences across models, with higher-capacity models tending to achieve
     higher scores.  Even though the random baseline scores positively in our
     experiment, many LLMs still receive negative scores, in which case our
     approach provides evidence for the incorrectness of the generated causal
     statements."

An important correction to the anchored wording
-----------------------------------------------
The claim as anchored for this reproduction reads "compatibility scores decrease
monotonically as injected error increases".  That is *not* what the paper
asserts.  The paper's monotonicity statement is about the **fraction of
statement lists whose score is positive**, not about the magnitude of the score.
The distinction matters a great deal: the magnitude of the score is a sum of
squared quantities that generally grows in absolute value as the statements get
worse, so the mean score is *not* expected to decrease monotonically.

Both readings are therefore measured and reported separately:

  (A) the paper's actual claim -- fraction positive, strictly decreasing;
  (B) the anchored paraphrase -- mean score, monotonically decreasing.

Reporting (B) as a failure of the paper would be a misreading; reporting (A) as
if it were (B) would be a misreading in the other direction.  The verdict for
this claim is driven by (A), with (B) recorded as a documented discrepancy
between the anchored wording and the source.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from ..config import CFG, SEED
from ..harness import Verdict, banner, wilson, write_csv, write_json
from ..generate import perturb_statements, sample_linear_model
from ..linear import compatibility_score, standardize


def _cell(task):
    """One (panel, parameter value, sigma) point of Figure 2."""
    panel, value, fixed, sigma, n_models, n_noise, seed = task
    params = dict(fixed)
    params[panel] = value
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_models):
        Sigma, A_true, _ = sample_linear_model(int(params["n"]), int(params["m"]),
                                               float(params["p"]), rng)
        for _ in range(n_noise):
            A = perturb_statements(A_true, sigma, rng)
            # Section 2.5: scores are computed after standardising to unit
            # variance.  sample_linear_model already returns a correlation
            # matrix, so this is a no-op guard rather than a transformation.
            S, As = standardize(Sigma, A)
            scores.append(compatibility_score(S, As))
    scores = np.array(scores, float)
    k = int((scores > 0).sum())
    lo, hi = wilson(k, len(scores))
    return dict(panel=panel, value=value, n=params["n"], m=params["m"],
                p=params["p"], sigma=sigma, n_scores=len(scores),
                frac_positive=k / len(scores), frac_ci_lo=lo, frac_ci_hi=hi,
                mean_score=float(scores.mean()),
                median_score=float(np.median(scores)))


def run() -> dict:
    banner("CLAIM 4 -- Figure 2: the fraction of positive compatibility scores "
           "strictly decreases as statement error grows")
    v = Verdict("claim4", "Figure 2 (synthetic) and Figure 4 (LLM) behaviour")

    tasks = []
    for panel, values, fixed in CFG["c4_panels"]:
        for value in values:
            for sigma in CFG["c4_sigmas"]:
                tasks.append((panel, value, tuple(fixed.items()), sigma,
                              CFG["c4_models"], CFG["c4_noise"],
                              [SEED, 4, abs(hash(panel)) % 1000,
                               int(value * 100), int(sigma * 1000)]))
    tasks = [(p, val, dict(f), s, nm, nn, sd) for (p, val, f, s, nm, nn, sd) in tasks]
    with mp.Pool(processes=min(len(tasks), os.cpu_count() or 1)) as pool:
        rows = pool.map(_cell, tasks)
    write_csv("claim4", "figure2_synthetic.csv", rows)

    curves = {}
    for r in rows:
        curves.setdefault((r["panel"], r["value"]), []).append(r)

    # -------------------------------------------------- (A) the paper's claim
    strict_ok = True
    for key, rs in sorted(curves.items(), key=lambda kv: str(kv[0])):
        rs.sort(key=lambda r: r["sigma"])
        fr = [r["frac_positive"] for r in rs]
        # "strictly decreases as the error increases": require a strict decrease
        # from the zero-error baseline to the largest error, and no increase
        # beyond sampling noise anywhere along the curve.
        ends = fr[-1] < fr[0]
        no_rise = all(rs[i + 1]["frac_ci_lo"] <= rs[i]["frac_ci_hi"]
                      for i in range(len(rs) - 1))
        ok = ends and no_rise
        strict_ok &= ok
        print(f"    panel {key[0]}={key[1]}: frac positive over sigma "
              f"{[r['sigma'] for r in rs]} = {[round(x, 3) for x in fr]} "
              f"{'ok' if ok else 'VIOLATION'}", flush=True)
    v.check(f"(A) paper's statement: the fraction of positive compatibility "
            f"scores strictly decreases with injected error, in all "
            f"{len(curves)} parameter curves "
            f"({CFG['c4_models']} model draws x {CFG['c4_noise']} noise draws "
            f"= {CFG['c4_models']*CFG['c4_noise']} lists per point)", strict_ok)

    # "most of the true sets of bivariate causal statements have positive scores"
    base = [r for r in rows if r["sigma"] == 0.0]
    frac0 = float(np.mean([r["frac_positive"] for r in base]))
    v.check("(A2) at zero error most true statement lists score positively "
            "(evidence for Assumption 2.4)",
            all(r["frac_positive"] > 0.5 for r in base),
            f"mean fraction positive at sigma=0 across {len(base)} settings "
            f"= {frac0:.3f}, min = {min(r['frac_positive'] for r in base):.3f}")

    # -------------------------------- (B) the anchored paraphrase, for the record
    mean_mono = 0
    for key, rs in curves.items():
        rs.sort(key=lambda r: r["sigma"])
        ms = [r["mean_score"] for r in rs]
        if all(ms[i + 1] <= ms[i] + 1e-12 for i in range(len(ms) - 1)):
            mean_mono += 1
    v.note(f"(B) the anchored paraphrase -- 'compatibility scores decrease "
           f"monotonically' -- holds for the MEAN score in only "
           f"{mean_mono}/{len(curves)} curves. The paper does not claim this; "
           f"its Figure 2 plots the fraction of positive scores. Recorded as a "
           f"discrepancy between the anchored wording and the source, not as "
           f"evidence against the paper.")

    # ------------------------------------------------------- negative control
    # If the statements carry no information at all, the fraction positive must
    # sit far below the zero-error baseline and stop responding to sigma.
    rng = np.random.default_rng([SEED, 4, 31337])
    rand_pos = []
    for _ in range(600):
        Sigma, A_true, _ = sample_linear_model(10, 3, 0.5, rng)
        A = perturb_statements(A_true * 0.0 + np.eye(10), 1.0, rng)
        S, As = standardize(Sigma, A)
        rand_pos.append(compatibility_score(S, As) > 0)
    frac_rand = float(np.mean(rand_pos))
    v.check("negative control: statement lists unrelated to the true model score "
            "positively far less often than true ones",
            frac_rand < frac0 - 0.1,
            f"unrelated lists {frac_rand:.3f} vs true lists {frac0:.3f}")

    write_json("claim4", "figure2_summary.json",
               dict(n_curves=len(curves), strict_decrease_all=strict_ok,
                    frac_positive_at_zero_error=frac0,
                    mean_score_monotone_curves=mean_mono,
                    negative_control_frac_positive=frac_rand,
                    lists_per_point=CFG["c4_models"] * CFG["c4_noise"]))

    # ------------------------------------------------------------- LLM (Fig 4)
    from .llm_scores import score_linear_transcripts
    llm = score_linear_transcripts(v)
    write_json("claim4", "figure4_llm.json", llm)

    status = "VERIFIED" if v.all_passed() else "BLOCKED"
    return v.finish(status)
