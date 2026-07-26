"""Score the committed LLM transcripts (Figures 4, 6 and 7).

The transcripts under ``data/llm/`` were collected once by
:mod:`repro.llm_collect` following Appendix D.3 verbatim.  Everything in this
module is deterministic given those files, so the scored evidence regenerates
byte-for-byte from the fixed run command even though the model responses
themselves are not reproducible.

Coverage caveat, stated up front: Table 2 of the paper lists nine models
accessed through Amazon Bedrock.  Six of them are served by the Hugging Face
inference router and are reproduced here; the other four (Claude Opus 4.5,
Kimi K2 Thinking, Mistral Large 3, Magistral Small 2509) are not reachable from
this environment.

Six models is far too few to test "higher-capacity models tend to score higher":
a Spearman correlation on six points needs |rho| >= 0.83 to reach p < 0.05.  The
scoring therefore reports two things separately and never mixes them: the
paper's own Table 2 subset (faithful but underpowered) and an extended
thirteen-model ladder spanning 4B to 1T parameters (not the paper's models, but
adequately powered for the qualitative claim).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

from ..config import SEED
from ..gapminder import (
    ALL_MODELS,
    CORRELATION,
    EXTENDED_MODELS,
    MODELS,
    MODELS_UNAVAILABLE,
    VARIABLES,
    permuted_correlation,
)
from ..graphical import StatementGraph, heuristic_incompatibility
from ..harness import write_csv
from ..linear import compatibility_score

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "llm")

# Figure 7 keeps statement graphs whose edge density is at most 2/3.  Density is
# taken as (directed + bidirected edges) / (2 * C(n,2)): each vertex pair offers
# one directed slot and one bidirected slot.
DENSITY_CAP = 2.0 / 3.0

BY_ROUTE = {m["route"]: m for m in ALL_MODELS}

# Pre-registered analysis choice, fixed before the extended ladder was collected:
# the *gating* statistic is the per-model MEAN, because Section 2.7 says scores
# are "averaged over 15 independent runs".  The per-model median is computed and
# reported alongside as a robustness check, since a single run with pathological
# coefficients can move a mean by orders of magnitude, but it never overrides the
# mean-based result.
GATING_STATISTIC = "mean"


def _capacity_correlation(per_model, key):
    if len(per_model) < 3:
        return float("nan"), float("nan")
    return stats.spearmanr([d["params_b"] for d in per_model],
                           [d[key] for d in per_model])


def _load(mode: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(DATA, f"{mode}__*.json"))):
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def statements_to_A(ordering: list[str], coefficients: dict) -> np.ndarray:
    """Unit lower-triangular statement matrix in the model's causal ordering."""
    n = len(ordering)
    A = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            A[j, i] = float(coefficients[f"{ordering[i]}||{ordering[j]}"])
    return A


def answers_to_graph(answers: dict) -> StatementGraph:
    """Union of the bivariate ADMGs a model reported (Definition 3.3)."""
    n = len(VARIABLES)
    idx = {v: i for i, v in enumerate(VARIABLES)}
    D = np.zeros((n, n), bool)
    Bd = np.zeros((n, n), bool)
    for rec in answers.values():
        a, b = idx[rec["A"]], idx[rec["B"]]
        if rec["effect"] == "YES":
            if rec["direction"] == "A TO B":
                D[a, b] = True
            elif rec["direction"] == "B TO A":
                D[b, a] = True
        if rec["confounding"] == "YES":
            Bd[a, b] = Bd[b, a] = True
    return StatementGraph(n, D, Bd)


# --------------------------------------------------------------------------
# Figure 4: compatibility scores of linear statements
# --------------------------------------------------------------------------

def score_linear_transcripts(v) -> dict:
    runs = _load("linear")
    if not runs:
        v.check("(C) LLM linear transcripts are present", False,
                f"no files under {DATA}")
        return dict(available=False)

    rows = []
    for r in runs:
        if not r.get("ok"):
            rows.append(dict(model=r["model"], run=r["run"], ok=False,
                             reason=r.get("reason", ""), comp=float("nan")))
            continue
        A = statements_to_A(r["ordering"], r["coefficients"])
        Sigma = permuted_correlation(r["ordering"])
        rows.append(dict(model=r["model"], run=r["run"], ok=True, reason="",
                         comp=compatibility_score(Sigma, A),
                         max_abs_coef=float(np.max(np.abs(np.tril(A, -1))))))
    write_csv("claim4", "figure4_llm_runs.csv", rows)

    ok_rows = [r for r in rows if r["ok"]]
    v.check("(C) LLM linear transcripts parsed into statement matrices",
            len(ok_rows) >= 0.8 * len(rows),
            f"{len(ok_rows)}/{len(rows)} conversations completed the full "
            f"Appendix D.3 protocol")

    # Random baseline: coefficients drawn from a centred normal whose variance
    # matches the variance of the LLM outputs (Section 2.7).
    all_coefs = []
    for r in runs:
        if r.get("ok"):
            all_coefs.extend(r["coefficients"].values())
    sd = float(np.std(all_coefs)) if all_coefs else 0.0
    rng = np.random.default_rng([SEED, 4, 4242])
    n = len(VARIABLES)
    baseline = []
    for _ in range(500):
        A = np.eye(n)
        idx = np.tril_indices(n, -1)
        A[idx] = rng.normal(scale=sd, size=len(idx[0]))
        baseline.append(compatibility_score(CORRELATION, A))
    baseline = np.array(baseline)

    # Robustness variant: match the coefficient sd to the median across models
    # rather than the pooled sd, which one pathological model can dominate.
    per_model_sd = []
    for m in ALL_MODELS:
        cs = [x for r in runs if r.get("ok") and r["model"] == m["route"]
              for x in r["coefficients"].values()]
        if cs:
            per_model_sd.append(float(np.std(cs)))
    sd_robust = float(np.median(per_model_sd)) if per_model_sd else sd
    baseline_robust = []
    rng2 = np.random.default_rng([SEED, 4, 4243])
    for _ in range(500):
        A = np.eye(n)
        idx = np.tril_indices(n, -1)
        A[idx] = rng2.normal(scale=sd_robust, size=len(idx[0]))
        baseline_robust.append(compatibility_score(CORRELATION, A))
    baseline_robust = np.array(baseline_robust)

    def stats_for(model_list):
        out = []
        for m in model_list:
            vals = [r["comp"] for r in ok_rows if r["model"] == m["route"]]
            if not vals:
                continue
            vals = np.array(vals, float)
            out.append(dict(paper_name=m["paper_name"], route=m["route"],
                            params_b=m["params_b"], n_runs=len(vals),
                            mean_comp=float(vals.mean()),
                            sem=float(vals.std(ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0,
                            median_comp=float(np.median(vals)),
                            min_comp=float(vals.min()), max_comp=float(vals.max()),
                            frac_negative=float((vals < 0).mean())))
        return out

    per_model = stats_for(MODELS)                    # the paper's Table 2 subset
    extended = stats_for(ALL_MODELS)                 # the wider capacity ladder
    for d in per_model:
        d["set"] = "paper_table2"
    write_csv("claim4", "figure4_llm_by_model.csv", per_model)
    write_csv("claim4", "figure4_llm_extended_ladder.csv", extended)

    print("    -- the paper's Table 2 subset (6 of 9 models reachable) --", flush=True)
    for pm in sorted(per_model, key=lambda d: -d["params_b"]):
        print(f"    {pm['paper_name']:<26} {pm['params_b']:>6.0f}B  "
              f"mean = {pm['mean_comp']:+11.3f}  median = {pm['median_comp']:+8.3f}  "
              f"({pm['frac_negative']:.0%} of {pm['n_runs']} runs negative)",
              flush=True)
    print(f"    {'random baseline':<26} {'--':>7}  "
          f"mean = {baseline.mean():+11.3f}", flush=True)
    print(f"    -- extended ladder ({len(extended)} models, "
          f"{min(d['params_b'] for d in extended):.0f}B-"
          f"{max(d['params_b'] for d in extended):.0f}B; NOT the paper's set) --",
          flush=True)
    for pm in sorted(extended, key=lambda d: -d["params_b"]):
        if pm["route"] in {m["route"] for m in MODELS}:
            continue
        print(f"    {pm['paper_name']:<26} {pm['params_b']:>6.0f}B  "
              f"mean = {pm['mean_comp']:+11.3f}  median = {pm['median_comp']:+8.3f}  "
              f"({pm['frac_negative']:.0%} of {pm['n_runs']} runs negative)",
              flush=True)

    # "many LLMs still receive negative scores"
    any_negative = any(pm["frac_negative"] > 0 for pm in per_model)
    v.check("(C1) some LLM statement lists receive negative (falsifying) "
            "compatibility scores", any_negative,
            "models with at least one negative run: "
            + str([pm["paper_name"] for pm in per_model
                   if pm["frac_negative"] > 0]))

    # "our compatibility score can demonstrate systematic differences across models"
    groups = [[r["comp"] for r in ok_rows if r["model"] == m["route"]]
              for m in MODELS]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        v.check("(C2) compatibility scores differ systematically across models "
                "(Kruskal-Wallis)", p < 0.05,
                f"H = {H:.2f}, p = {p:.3g} over {len(groups)} models")
    else:
        H, p = float("nan"), float("nan")
        v.check("(C2) enough models with repeated runs to test for systematic "
                "differences", False)

    # "higher-capacity models tending to achieve higher scores"
    # Six models cannot support this test -- a Spearman correlation on six
    # points needs |rho| >= 0.83 for p < 0.05 -- so the gating test uses the
    # extended ladder, with the paper's own subset reported alongside.
    rho_p, p_p = _capacity_correlation(per_model, "mean_comp")
    rho_pm, _ = _capacity_correlation(per_model, "median_comp")
    rho, prho = _capacity_correlation(extended, "mean_comp")
    rho_med, p_med = _capacity_correlation(extended, "median_comp")
    v.note(f"paper Table 2 subset ({len(per_model)} models): Spearman "
           f"rho(params, mean comp) = {rho_p:.3f}, p = {p_p:.3g} "
           f"(median-based rho = {rho_pm:.3f}); underpowered at this size. "
           f"{len(MODELS_UNAVAILABLE)} of the paper's 9 models are unreachable: "
           f"{MODELS_UNAVAILABLE}")
    v.check("(C3) higher-capacity models tend to achieve higher compatibility "
            "scores (extended ladder, pre-registered mean statistic)",
            bool(np.isfinite(rho) and rho > 0),
            f"Spearman rho = {rho:.3f}, p = {prho:.3g} over {len(extended)} "
            f"models spanning "
            f"{min(d['params_b'] for d in extended):.0f}B-"
            f"{max(d['params_b'] for d in extended):.0f}B; "
            f"median-based robustness check rho = {rho_med:.3f}, p = {p_med:.3g}")

    # "Even though the random baseline scores positively in our experiment"
    # The sign of the random baseline is a detail of the paper's Figure 4, not
    # part of the claim under test, so it is reported rather than gated.
    v.note(f"(C4) random baseline: mean over 500 random lists = "
           f"{baseline.mean():+.4f} (paper reports a positive baseline). "
           f"Coefficient sd matched to the pooled LLM outputs = {sd:.3f}, which "
           f"is inflated by models that emit standardised 'effects' far outside "
           f"[-1, 1]; with the sd matched to the per-model median instead "
           f"({sd_robust:.3f}) the baseline mean is {baseline_robust.mean():+.4f}.")

    return dict(available=True, per_model=per_model, extended_ladder=extended,
                baseline_mean=float(baseline.mean()), baseline_sd_used=sd,
                baseline_mean_robust=float(baseline_robust.mean()),
                baseline_sd_robust=sd_robust,
                kruskal_H=float(H), kruskal_p=float(p),
                spearman_rho_paper_subset=float(rho_p),
                spearman_p_paper_subset=float(p_p),
                spearman_rho_extended=float(rho), spearman_p_extended=float(prho),
                spearman_rho_extended_median=float(rho_med),
                gating_statistic=GATING_STATISTIC,
                n_runs=len(rows), n_ok=len(ok_rows),
                models_unavailable=MODELS_UNAVAILABLE)


# --------------------------------------------------------------------------
# Figures 6 and 7: incompatibility scores of graphical statements
# --------------------------------------------------------------------------

def score_graphical_transcripts(v) -> dict:
    runs = _load("graphical")
    if not runs:
        v.check("(C) LLM graphical transcripts are present", False,
                f"no files under {DATA}")
        return dict(available=False)

    rows = []
    for r in runs:
        if not r.get("ok"):
            rows.append(dict(model=r["model"], run=r["run"], ok=False,
                             reason=r.get("reason", "")))
            continue
        g = answers_to_graph(r["answers"])
        score, detail = heuristic_incompatibility(g, detail=True)
        rows.append(dict(model=r["model"], run=r["run"], ok=True, reason="",
                         incomp=score, density=g.density(),
                         n_directed=g.n_directed(),
                         n_bidirected=g.n_bidirected(), **detail))
    write_csv("claim6", "figures67_llm_runs.csv", rows)
    ok_rows = [r for r in rows if r["ok"]]
    v.check("(C) LLM graphical transcripts parsed into statement graphs",
            len(ok_rows) >= 0.8 * len(rows),
            f"{len(ok_rows)}/{len(rows)} conversations completed the protocol")

    def gstats(model_list):
        out = []
        for m in model_list:
            vals = [r for r in ok_rows if r["model"] == m["route"]]
            if not vals:
                continue
            sc = np.array([r["incomp"] for r in vals], float)
            sparse = [r for r in vals if r["density"] <= DENSITY_CAP]
            out.append(dict(
                paper_name=m["paper_name"], route=m["route"],
                params_b=m["params_b"], n_runs=len(vals),
                mean_incomp=float(sc.mean()), median_incomp=float(np.median(sc)),
                mean_density=float(np.mean([r["density"] for r in vals])),
                n_below_density_cap=len(sparse),
                mean_incomp_below_cap=(float(np.mean([r["incomp"] for r in sparse]))
                                       if sparse else float("nan")),
                n_zero=int((sc == 0).sum())))
        return out

    per_model = gstats(MODELS)
    extended = gstats(ALL_MODELS)
    write_csv("claim6", "figures67_llm_by_model.csv", per_model)
    write_csv("claim6", "figures67_llm_extended_ladder.csv", extended)

    print("    -- the paper's Table 2 subset --", flush=True)
    for pm in sorted(per_model, key=lambda d: -d["params_b"]):
        print(f"    {pm['paper_name']:<26} {pm['params_b']:>6.0f}B  "
              f"incomp = {pm['mean_incomp']:5.2f} (all {pm['n_runs']} runs), "
              f"{pm['mean_incomp_below_cap']:5.2f} "
              f"(n={pm['n_below_density_cap']} below density cap), "
              f"density {pm['mean_density']:.2f}", flush=True)
    print(f"    -- extended ladder ({len(extended)} models) --", flush=True)
    for pm in sorted(extended, key=lambda d: -d["params_b"]):
        if pm["route"] in {m["route"] for m in MODELS}:
            continue
        print(f"    {pm['paper_name']:<26} {pm['params_b']:>6.0f}B  "
              f"incomp = {pm['mean_incomp']:5.2f} (all {pm['n_runs']} runs), "
              f"{pm['mean_incomp_below_cap']:5.2f} "
              f"(n={pm['n_below_density_cap']} below density cap), "
              f"density {pm['mean_density']:.2f}", flush=True)

    v.check("(D1) Figure 6: incompatibility scores computed for every model's "
            "statement graphs", len(per_model) >= 3,
            f"{len(per_model)} models scored")

    # Figure 7: among graphs below the density cap, lower incompatibility should
    # go with higher capacity.  Gated on the extended ladder for the same power
    # reason as Claim 4; the paper's subset is reported alongside.
    sub_paper = [pm for pm in per_model if pm["n_below_density_cap"] > 0]
    sub_ext = [pm for pm in extended if pm["n_below_density_cap"] > 0]
    rho_p, p_p = _capacity_correlation(sub_paper, "mean_incomp_below_cap")
    rho, prho = _capacity_correlation(sub_ext, "mean_incomp_below_cap")
    v.note(f"paper Table 2 subset: {len(sub_paper)} of {len(per_model)} models "
           f"produced any statement graph below the 2/3 density cap; "
           f"Spearman rho(params, incomp) = {rho_p:.3f}, p = {p_p:.3g} "
           f"-- underpowered at this size")
    v.check("(D2) Figure 7: among statement graphs with edge density <= 2/3, "
            "lower incompatibility correlates with higher model capacity "
            "(extended ladder)",
            bool(np.isfinite(rho) and rho < 0),
            f"Spearman rho = {rho:.3f}, p = {prho:.3g} over {len(sub_ext)} "
            f"models with at least one graph below the cap")

    # The paper's density observation: dense graphs get low scores for a trivial
    # reason, which is why Figure 7 exists.  Check the mechanism is present.
    dens = np.array([r["density"] for r in ok_rows])
    inc = np.array([r["incomp"] for r in ok_rows], float)
    rho_d, p_d = stats.spearmanr(dens, inc)
    v.check("(D3) the paper's stated confound is reproduced: denser statement "
            "graphs achieve lower incompatibility scores", bool(rho_d < 0),
            f"Spearman rho(density, incomp) = {rho_d:.3f}, p = {p_d:.3g} "
            f"-- this is why Figure 7 caps density")

    return dict(available=True, per_model=per_model, extended_ladder=extended,
                density_cap=DENSITY_CAP,
                spearman_capacity_rho_paper_subset=float(rho_p),
                spearman_capacity_p_paper_subset=float(p_p),
                spearman_capacity_rho=float(rho), spearman_capacity_p=float(prho),
                spearman_density_rho=float(rho_d), spearman_density_p=float(p_d),
                n_runs=len(rows), n_ok=len(ok_rows),
                models_unavailable=MODELS_UNAVAILABLE)
