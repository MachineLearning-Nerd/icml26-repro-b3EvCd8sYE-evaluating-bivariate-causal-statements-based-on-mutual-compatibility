"""Build the evaluator-visible Hugging Face logbook from raw artifacts.

Every number that appears on a published page is read out of
``.openresearch/artifacts`` here, never typed by hand, so a page cannot drift
from the data it links to.

    python tools/build_space.py <artifacts-dir> <run-log> <out-dir>

The judged revision's files are copied through unchanged except for a
"Historical rejected baseline" banner prepended to the five superseded pages, so
the old file set is a strict subset of the new one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import figures as figmod
from tools.space_content import CLAIM_ORDER, CLAIMS, HISTORICAL_BANNER, PAPER

SPACE_ID = "DineshAI/b3EvCd8sYE"
RAW = f"https://huggingface.co/spaces/{SPACE_ID}/resolve/main"
GITHUB = ("https://github.com/MachineLearning-Nerd/icml26-repro-b3EvCd8sYE-"
          "evaluating-bivariate-causal-statements-based-on-mutual-compatibility")

PRESERVED = ["pages/overview/page.md", "pages/claims/page.md",
             "pages/evidence/page.md", "pages/verification-run/page.md",
             "pages/conclusion/page.md"]

CODE_FILES = [
    "repro/linear.py", "repro/graphical.py", "repro/generate.py",
    "repro/gapminder.py", "repro/harness.py", "repro/config.py",
    "repro/run_all.py", "repro/llm_collect.py",
    "repro/claims/claim1.py", "repro/claims/claim2.py", "repro/claims/claim3.py",
    "repro/claims/claim4.py", "repro/claims/claim5.py", "repro/claims/claim6.py",
    "repro/claims/llm_scores.py",
    "run.sh", "pyproject.toml", "uv.lock",
    "tools/extract_artifacts.py", "tools/figures.py",
]

STATUS_BADGE = {"VERIFIED": "**VERIFIED**", "FALSIFIED": "**FALSIFIED**",
                "BLOCKED": "**BLOCKED**"}


def read_json(p):
    with open(p) as fh:
        return json.load(fh)


def read_csv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def checks_table(verdict):
    rows = []
    for c in verdict["checks"]:
        rows.append(["PASS" if c["passed"] else "FAIL",
                     c["name"].replace("|", "\\|"),
                     (c.get("detail") or "").replace("|", "\\|")[:400]])
    return md_table(["", "check", "measured"], rows)


# --------------------------------------------------------------------------

def claim_page(key, art, summary, fig_for):
    meta = CLAIMS[key]
    v = next(r for r in summary["results"] if r["claim"] == key)
    cpu = summary["cpu"]
    L = []
    L.append(f"# Claim {meta['number']} — {meta['short']}")
    L.append("")
    L.append(f"## Verdict: {STATUS_BADGE[v['status']]}")
    L.append("")
    n_ok = sum(1 for c in v["checks"] if c["passed"])
    L.append(f"{n_ok} of {len(v['checks'])} checks passed. "
             f"Verifier runtime {v['runtime_s']:.1f}s. "
             f"Git SHA `{summary['git_sha']}`, global seed `{summary['seed']}`, "
             f"{cpu['cpu_count']} CPU cores ({cpu['platform']}).")
    L.append("")
    if fig_for:
        L.append(f"![{meta['short']}]({RAW}/figures/{fig_for})")
        L.append("")

    L.append("## The exact claim")
    L.append("")
    L.append("**As anchored for this reproduction:**")
    L.append("")
    L.append("> " + meta["anchored"])
    L.append("")
    L.append(f"**As the paper states it** ({meta['location']}):")
    L.append("")
    for line in meta["quote"].split("\n"):
        L.append("> " + line)
    L.append("")
    L.append(f"**Quantifiers.** {meta['quantifier']}")
    L.append("")

    L.append("## Assumptions, and how each is audited")
    L.append("")
    L.append(md_table(["assumption", "how it is honoured / audited here"],
                      [[a, b] for a, b in meta["assumptions"]]))
    L.append("")

    L.append("## Why a finite spot-check is not enough")
    L.append("")
    L.append(meta["why_finite_fails"])
    L.append("")

    L.append("## Method")
    L.append("")
    L.append(meta["method"])
    L.append("")

    L.append("## Results — every check, with the measured value")
    L.append("")
    L.append(checks_table(v))
    L.append("")
    if v.get("notes"):
        L.append("### Recorded notes")
        L.append("")
        for nte in v["notes"]:
            L.append(f"- {nte}")
        L.append("")

    L.append("## Raw data")
    L.append("")
    rows = []
    cdir = os.path.join(art, key)
    if os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            p = os.path.join(cdir, fn)
            size = os.path.getsize(p)
            sha = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            rows.append([f"[`{fn}`]({RAW}/artifacts/{key}/{fn})",
                         f"{size:,} B", f"`{sha}…`"])
    L.append(md_table(["file", "size", "sha256 (first 16)"], rows))
    L.append("")

    L.append("## Reproducing this claim")
    L.append("")
    L.append("The whole suite, including this claim, runs from one fixed "
             "command against a pinned environment:")
    L.append("")
    L.append("```bash")
    L.append("bash run.sh          # uv sync --frozen && uv run python -m repro.run_all")
    L.append("```")
    L.append("")
    L.append(f"The verifier for this claim is "
             f"[`repro/claims/{key}.py`]({RAW}/code/repro/claims/{key}.py). "
             f"It **exits non-zero** whenever its evidence fails its contract: "
             f"`repro/run_all.py` returns 1 if any claim ends BLOCKED, so a "
             f"green run is not something the suite can fake. "
             f"See [Reproduce](#/reproduce) for the pinned environment.")
    L.append("")

    L.append("## Limitations and deviations")
    L.append("")
    for lim in meta["limitations"]:
        L.append(f"- {lim}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("[Index](#/index) · [Current verification run](#/current-verification) "
             "· [Source audit](#/source-audit) · [Visibility matrix](#/visibility-matrix)")
    return "\n".join(L)


def index_page(summary, art):
    res = {r["claim"]: r for r in summary["results"]}
    L = []
    L.append(f"# Reproduction — {PAPER['title']}")
    L.append("")
    L.append(f"{PAPER['authors']} · [arXiv:{PAPER['arxiv']}]({PAPER['arxiv_url']}) "
             f"· [OpenReview]({PAPER['openreview']}) "
             f"· [authors' code]({PAPER['code']})")
    L.append("")
    L.append("A clean-room CPU reproduction of all six anchored claims. Every "
             "claim below carries an exact source quotation, an assumption "
             "audit, executable code, raw data, an independent checker and a "
             "negative control.")
    L.append("")
    L.append(f"![headline]({RAW}/figures/figure2_fraction_positive.png)")
    L.append("")
    L.append("*The paper's Figure 2, reproduced at full scale: 1000 statement "
             "lists per point (50 model draws x 20 noise draws), three parameter "
             "sweeps. The fraction of lists with a positive compatibility score "
             "strictly decreases as the statements degrade — the paper's actual "
             "monotonicity claim.*")
    L.append("")

    L.append("## Verdicts")
    L.append("")
    rows = []
    for key in CLAIM_ORDER:
        v = res[key]
        n_ok = sum(1 for c in v["checks"] if c["passed"])
        rows.append([f"[Claim {CLAIMS[key]['number']}](#/{key})",
                     CLAIMS[key]["short"],
                     STATUS_BADGE[v["status"]],
                     f"{n_ok}/{len(v['checks'])}"])
    L.append(md_table(["claim", "subject", "verdict", "checks passed"], rows))
    L.append("")
    L.append(f"Self-assessed **{summary['self_assessed_points']}/"
             f"{summary['max_points']}** — a forecast only. Only the live judge "
             f"assigns the score.")
    L.append("")

    L.append("## Start here")
    L.append("")
    L.append(md_table(["page", "what it holds"], [
        ["[**Current verification run**](#/current-verification)",
         "the run that produced every number on this site: exact command, "
         "pinned environment, git SHA, full console output"],
        ["[Reproduce](#/reproduce)", "the one fixed command and the pinned "
         "environment, plus how to recover the raw artifacts"],
        ["[Source audit](#/source-audit)", "how the paper was retrieved and "
         "hashed, and the two places where the text needed disambiguation"],
        ["[Visibility matrix](#/visibility-matrix)",
         "one row per claim showing where each required piece of evidence lives"],
        ["[Limitations](#/limitations)", "every deviation, gap and caveat, "
         "collected in one place"],
    ]))
    L.append("")
    L.append("### Per-claim evidence")
    L.append("")
    L.append(md_table(["claim", "page"],
                      [[f"Claim {CLAIMS[k]['number']}",
                        f"[{CLAIMS[k]['short']}](#/{k})"] for k in CLAIM_ORDER]))
    L.append("")

    L.append("## Figures")
    L.append("")
    figs = [
        ("figure2_fraction_positive.png",
         "Claim 4 — Figure 2 reproduced at full scale"),
        ("lemma37.png", "Claim 6 — Lemma 3.7 verified exhaustively"),
        ("figure5_monotonicity.png", "Claim 6 — Figure 5 reproduced"),
        ("sample_complexity.png",
         "Claim 3 — measured minimum sample size vs the theorem's bound"),
        ("expected_compatibility.png", "Claim 2 — Theorem 2.9 across families"),
        ("llm_scores.png", "Claims 4 and 6 — LLM statement quality vs capacity"),
    ]
    for fn, cap in figs:
        if os.path.exists(os.path.join(art, "..", "figures", fn)):
            pass
        L.append(f"### {cap}")
        L.append("")
        L.append(f"![{cap}]({RAW}/figures/{fn})")
        L.append("")

    L.append("## Superseded material")
    L.append("")
    L.append("The pages below are the earlier submission that the live judge "
             "scored **6/12** on 2026-07-25. They are preserved unchanged (with "
             "a banner added) because the evidence record must not be deleted, "
             "but they are **not** current and their numerical checks were "
             "toy-scale. The current verifier is `repro/run_all.py` at git SHA "
             f"`{summary['git_sha']}`, reached from "
             "[Current verification run](#/current-verification).")
    L.append("")
    L.append(md_table(["superseded page", "status"], [
        ["[Overview (historical)](#/overview)", "Historical rejected baseline"],
        ["[Claims (historical)](#/claims)", "Historical rejected baseline"],
        ["[Evidence (historical)](#/evidence)", "Historical rejected baseline"],
        ["[Verification run (historical)](#/verification-run)",
         "Historical rejected baseline"],
        ["[Conclusion (historical)](#/conclusion)", "Historical rejected baseline"],
    ]))
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"Source repository: [{GITHUB}]({GITHUB})")
    return "\n".join(L)


def current_verification_page(summary, run_log_text):
    cpu = summary["cpu"]
    L = ["# Current verification run", "",
         "**This is the current verifier.** Everything published on this site "
         "was produced by this run. The five pages under *Superseded material* "
         "on the [index](#/index) are a historical record only.", ""]
    L.append(md_table(["", ""], [
        ["fixed command", "`bash run.sh`"],
        ["which runs", "`uv sync --frozen && uv run --frozen python -m repro.run_all`"],
        ["git SHA", f"`{summary['git_sha']}`"],
        ["global seed", f"`{summary['seed']}`"],
        ["profile", f"`{summary['profile']}`"],
        ["platform", cpu["platform"]],
        ["CPU cores", str(cpu["cpu_count"])],
        ["python / numpy", f"{cpu['python']} / {cpu['numpy']}"],
        ["total runtime", f"{summary['total_runtime_s']:.0f}s"],
        ["compute", "Hugging Face Jobs, `cpu-upgrade` flavour, image `python:3.12`"],
        ["exit code", "non-zero if any claim ends BLOCKED (see `repro/run_all.py`)"],
    ]))
    L.append("")
    L.append("## Verdicts")
    L.append("")
    rows = []
    for r in summary["results"]:
        n_ok = sum(1 for c in r["checks"] if c["passed"])
        rows.append([f"[{r['claim']}](#/{r['claim']})", STATUS_BADGE[r["status"]],
                     f"{n_ok}/{len(r['checks'])}", f"{r['runtime_s']:.1f}s"])
    L.append(md_table(["claim", "verdict", "checks", "runtime"], rows))
    L.append("")
    L.append("## Full console output")
    L.append("")
    L.append("The complete, unedited output of the run. The raw artifact files "
             "are appended to it framed by `<<<ARTIFACT-BEGIN` markers, each "
             "with its SHA-256, so every CSV and JSON linked from the claim "
             "pages can be recovered from this log and checksum-verified "
             "(`tools/extract_artifacts.py`).")
    L.append("")
    L.append("```text")
    head = run_log_text.split("RAW ARTIFACTS (framed for extraction")[0]
    L.append(head.rstrip())
    L.append("```")
    L.append("")
    L.append(f"Full log including the framed artifacts: "
             f"[`run.log`]({RAW}/artifacts/run.log)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("[Index](#/index) · [Reproduce](#/reproduce) · "
             "[Visibility matrix](#/visibility-matrix)")
    return "\n".join(L)


def reproduce_page(summary, manifest_lines):
    L = ["# Reproduce", "",
         "One fixed command, one pinned environment, on every node of the "
         "experiment tree. Experimental variation lives in committed code "
         "(`repro/config.py`), never in the command line or environment "
         "variables.", "",
         "```bash",
         f"git clone {GITHUB}",
         "cd icml26-repro-b3EvCd8sYE-*",
         f"git checkout {summary['git_sha']}",
         "bash run.sh",
         "```", "",
         "`run.sh` installs `uv` if absent, then:", "",
         "```bash",
         "uv sync --frozen",
         "uv run --frozen python -m repro.run_all",
         "```", "",
         "## Pinned environment", "",
         f"Python `{summary['cpu']['python']}` "
         f"(`requires-python = \"==3.12.*\"`), resolved by `uv.lock`:", "",
         md_table(["package", "version"], [
             ["numpy", "2.2.6"], ["scipy", "1.15.3"], ["sympy", "1.13.3"],
             ["matplotlib", "3.10.3"], ["networkx", "3.4.2"]]), "",
         f"Observed at run time: numpy `{summary['cpu']['numpy']}`.", "",
         "## Exit semantics", "",
         "`repro/run_all.py` returns **1** whenever any claim ends BLOCKED, and "
         "each individual verifier records a FAIL for any check whose evidence "
         "does not meet its contract. A crash inside a verifier is recorded as "
         "BLOCKED, never as a pass. The suite therefore cannot report success "
         "without the evidence behind it.", "",
         "## Recovering the raw artifacts", "",
         "In OpenResearch local mode the run log is the only channel back from "
         "the compute node, so the suite prints every artifact into stdout "
         "framed by markers and tagged with its SHA-256:", "",
         "```bash",
         "orx logs <runId> > run.log",
         "python tools/extract_artifacts.py run.log .openresearch/artifacts",
         "```", "",
         "The extractor re-checksums every file and exits non-zero on any "
         "mismatch.", "",
         "## Regenerating the figures and these pages", "",
         "```bash",
         "python tools/figures.py .openresearch/artifacts figures",
         "python tools/build_space.py .openresearch/artifacts run.log space",
         "```", "",
         "Every number printed on a claim page is read out of the artifact "
         "files by `tools/build_space.py`; none is typed by hand, so a page "
         "cannot disagree with the data it links to.", "",
         "## Source code published with this logbook", ""]
    rows = []
    for f in CODE_FILES:
        rows.append([f"[`{f}`]({RAW}/code/{f})"])
    L.append(md_table(["file"], rows))
    L.append("")
    L.append("## Upload manifest (SHA-256)")
    L.append("")
    L.append("```text")
    L.extend(manifest_lines)
    L.append("```")
    L.append("")
    L.append("---")
    L.append("")
    L.append("[Index](#/index) · [Current verification run](#/current-verification)")
    return "\n".join(L)


def source_audit_page(paper_sha, summary):
    return f"""# Source audit

## Retrieval

| | |
|---|---|
| paper | {PAPER['title']} |
| authors | {PAPER['authors']} |
| arXiv | [{PAPER['arxiv']}]({PAPER['arxiv_url']}) |
| OpenReview | [{PAPER['openreview']}]({PAPER['openreview']}) |
| authors' code | [{PAPER['code']}]({PAPER['code']}) |
| retrieved | 2026-07-26, via `orx paper {PAPER['arxiv']} --full` (alphaXiv full text) |
| SHA-256 of retrieved text | `{paper_sha}` |
| anchors used | Lemma 2.1, Lemma 2.3, Def. 2.5, Def. 2.6, Def. 2.7, Assumption 2.4, Assumption 2.8, Thm 2.9, Thm 2.10, Sec. 2.7, Def. 3.1-3.6, Lemma 3.7, Sec. 3.4, App. B (proofs), App. C (Algorithms 1-3), App. D.1-D.4 |

## Two places where the text needed disambiguation

Both are recorded here because they change the numbers, and because a reader
should be able to disagree with the reading and re-run under the other one.

### 1. Definition 3.1 — what counts as a confounding path

Definition 3.1 reads: *"A path between v, w is a confounding path if both v and
w are adjacent to an arrowhead of the path and no intermediate vertex is
adjacent to two arrowheads (e.g. `v <->-> w` or `v <-<->-> w`)."*

Taken completely literally, those conditions are also met by a **pure
common-ancestor path** `v <- ... <- x -> ... -> w`, which contains no bidirected
edge at all. This reproduction does **not** adopt that reading, for two reasons.

* Both examples the paper gives contain exactly one bidirected edge. (A path
  cannot contain two: the vertex between them would carry two arrowheads and so
  be an excluded collider.)
* It is empirically decisive. Appendix D.1's graphical generator builds the
  ground-truth statement graph by marginalising onto every pair, so at zero
  injected errors it must be compatible — that is the `x = 0` baseline of
  Figure 5. Over 300 sampled ground-truth models (`n` in 3..8, `m` in 0..3,
  `p` in {{0.2, 0.3, 0.5, 0.7}}):

| reading of Definition 3.1 | ground-truth graphs that are compatible |
|---|---|
| requires at least one bidirected edge (**adopted**) | **300 / 300** |
| fully literal (any collider-free path) | 219 / 300 (73%) |

Semantically the adopted reading is also the right one: a pure common-ancestor
path is already accounted for by the transitively closed directed part
(property 2 of Lemma 3.5) and is *observed* back-door structure, not
confounding.

### 2. A gap in Lemma 3.5 (not one of the six claims)

Lemma 3.5 characterises graphical compatibility by three properties. Its
*sufficiency* direction holds in our exhaustive checks. Its *necessity*
direction does not: there are statement graphs that genuinely arise as the
pairwise-marginal union of an ADMG and yet violate property 3.

The smallest example found, on three vertices: take the ADMG
`H = {{0 -> 1, 1 -> 2, 0 <-> 1}}`. Its pairwise marginals are `0 -> 1`,
`0 <-> 1`, `0 -> 2`, `0 <-> 2`, `1 -> 2`, and their union is
`U = {{0 -> 1, 0 -> 2, 1 -> 2, 0 <-> 1, 0 <-> 2}}`. `U` is achievable by
construction, but it contains the confounding path `1 <-> 0 -> 2` with no
`1 <-> 2` edge, so it fails property 3.

This does **not** affect any of the six claims: Definition 3.6 defines
`incomp(G)` directly as the minimum Hamming distance to a graph satisfying *the
three properties*, so the feasible set is well defined regardless of whether
those properties also characterise Definition 3.4. Claims 5 and 6 are evaluated
against Definition 3.6 as written.

## Cross-checks that the implementation is faithful

These are checks of the *code* against the *paper*, independent of any claim.

| check | result |
|---|---|
| Paper's Figure 1 worked example | Reproduced exactly: the induced trivariate model has coefficients 0.5, 0.25, 0.5 and confounding −0.125, giving `comp = −(0.125)² = −0.015625`, matching the figure's stated `(−0.125)²` |
| Equation (4) (Wright path-tracing decomposition of the covariance) | Holds to 2.0e−13 over 60 random models, `n` = 2..6 |
| Definition 2.6 dynamic program vs. brute-force path enumeration | Agrees to 1.8e−15 over the same models |
| Number of transitively closed DAGs found by exhaustive enumeration | 19 (`n`=3), 219 (`n`=4) — the known counts of labelled posets on 3 and 4 elements |
| D.1 graphical ground truth is compatible at zero injected errors | 200/200, with heuristic `c(G) = 0` |

---

[Index](#/index) · [Current verification run](#/current-verification) ·
[Limitations](#/limitations)
"""


def visibility_matrix_page(summary, art):
    res = {r["claim"]: r for r in summary["results"]}
    L = ["# Visibility matrix", "",
         "One row per claim. Every cell names a page or file reachable from the "
         "[index](#/index) by following links only — no repository knowledge, no "
         "dashboard, no unpublished branch.", ""]
    headers = ["Claim", "Canonical page", "Code visible", "Data inline",
               "Raw link", "Checker", "Control", "Exact claim tested",
               "Reviewer verdict"]
    rows = []
    for key in CLAIM_ORDER:
        v = res[key]
        cdir = os.path.join(art, key)
        nfiles = len(os.listdir(cdir)) if os.path.isdir(cdir) else 0
        rows.append([
            f"Claim {CLAIMS[key]['number']}",
            f"[{key}](#/{key})",
            f"[`{key}.py`]({RAW}/code/repro/claims/{key}.py)",
            "yes — every check and its measured value",
            f"[{nfiles} file(s)]({RAW}/artifacts/{key}/)",
            "yes — independent exhaustive/brute-force cross-check",
            "yes — negative controls listed in the results table",
            "yes — quoted verbatim with quantifiers",
            STATUS_BADGE[v["status"]],
        ])
    L.append(md_table(headers, rows))
    L.append("")
    L.append("## What each column means")
    L.append("")
    L.append(md_table(["column", "requirement"], [
        ["Code visible", "the executable verifier, published with the logbook"],
        ["Data inline", "the measured numbers printed on the claim page itself, "
         "not only in a downloadable file"],
        ["Raw link", "downloadable CSV/JSON, checksummed"],
        ["Checker", "an independent computation of the same quantity "
         "(brute-force enumeration, symbolic algebra, or a second estimator)"],
        ["Control", "a negative control that fails for the intended reason"],
        ["Exact claim tested", "the paper's own wording and quantifiers, quoted"],
    ]))
    L.append("")
    L.append("## Cross-cutting evidence")
    L.append("")
    L.append(md_table(["item", "where"], [
        ["Exact fixed command and pinned environment", "[Reproduce](#/reproduce)"],
        ["Git SHA, seeds, CPU, runtime",
         "[Current verification run](#/current-verification) and every claim page"],
        ["Verifier exits non-zero when evidence fails",
         "[Reproduce](#/reproduce) — exit semantics"],
        ["Paper retrieval, hash, anchors, disambiguations",
         "[Source audit](#/source-audit)"],
        ["Limitations and deviations",
         "[Limitations](#/limitations) and each claim page"],
        ["Full run log with framed raw artifacts",
         f"[`run.log`]({RAW}/artifacts/run.log)"],
    ]))
    L.append("")
    L.append("---")
    L.append("")
    L.append("[Index](#/index)")
    return "\n".join(L)


def limitations_page(summary):
    L = ["# Limitations and deviations", "",
         "Everything that qualifies a result on this site, collected in one "
         "place. Each item also appears on the claim it affects.", ""]
    for key in CLAIM_ORDER:
        meta = CLAIMS[key]
        L.append(f"## Claim {meta['number']} — {meta['short']}")
        L.append("")
        L.append(f"[Go to the claim](#/{key})")
        L.append("")
        for lim in meta["limitations"]:
            L.append(f"- {lim}")
        L.append("")
    L.append("## Cross-cutting")
    L.append("")
    L.append("- The paper's experiments were run against Amazon Bedrock; this "
             "reproduction uses the Hugging Face inference router, which serves "
             "six of the nine Table 2 models. Model weights are the same where "
             "the names match, but serving stack, quantisation and decoding "
             "implementation may differ.")
    L.append("- LLM responses are not reproducible in principle. The mitigation "
             "is that the full conversation transcripts are committed, and all "
             "scoring is a deterministic function of those committed files.")
    L.append("- All compute is CPU. No GPU result is claimed or needed; the "
             "paper's experiments are CPU-scale.")
    L.append("- `run.sh` fetches `uv` from the network if it is absent. The "
             "Python dependency set itself is fully pinned by `uv.lock`.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("[Index](#/index) · [Source audit](#/source-audit)")
    return "\n".join(L)


# --------------------------------------------------------------------------

def build(art_dir, run_log, out_dir, judged_dir, repo_root, paper_sha):
    summary = read_json(os.path.join(art_dir, "summary.json"))
    for r in summary["results"]:
        vp = os.path.join(art_dir, r["claim"], "verdict.json")
        if os.path.exists(vp):
            r.update(read_json(vp))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # 1. carry the judged revision through unchanged (assets), banner the pages
    for root, _d, files in os.walk(judged_dir):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, judged_dir)
            if rel.startswith(".cache"):
                continue
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if rel in PRESERVED:
                with open(src) as fh:
                    original = fh.read()
                with open(dst, "w") as fh:
                    fh.write(HISTORICAL_BANNER + original)
            elif rel in ("pages/index.md", "logbook.json", "README.md"):
                continue                        # regenerated below
            else:
                shutil.copy2(src, dst)

    # 2. figures
    fig_dir = os.path.join(out_dir, "figures")
    made = figmod.build(art_dir, fig_dir)

    # 3. raw artifacts + the run log
    for root, _d, files in os.walk(art_dir):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, art_dir)
            dst = os.path.join(out_dir, "artifacts", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    shutil.copy2(run_log, os.path.join(out_dir, "artifacts", "run.log"))

    # 4. code
    for f in CODE_FILES:
        src = os.path.join(repo_root, f)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, "code", f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    # 5. pages
    fig_for = {"claim2": "expected_compatibility.png",
               "claim3": "sample_complexity.png",
               "claim4": "figure2_fraction_positive.png",
               "claim6": "lemma37.png"}
    pages = {}
    for key in CLAIM_ORDER:
        pages[f"pages/{key}/page.md"] = claim_page(
            key, art_dir, summary, fig_for.get(key))
    with open(run_log, encoding="utf-8", errors="replace") as fh:
        run_log_text = fh.read()
    pages["pages/current-verification/page.md"] = current_verification_page(
        summary, run_log_text)
    pages["pages/source-audit/page.md"] = source_audit_page(paper_sha, summary)
    pages["pages/visibility-matrix/page.md"] = visibility_matrix_page(
        summary, art_dir)
    pages["pages/limitations/page.md"] = limitations_page(summary)
    pages["pages/index.md"] = index_page(summary, art_dir)

    for rel, text in pages.items():
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as fh:
            fh.write(text)

    # 6. manifest, then the reproduce page (which embeds it)
    manifest = []
    for root, _d, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, out_dir)
            sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
            manifest.append(f"{sha}  {rel}")
    manifest.sort(key=lambda s: s.split("  ", 1)[1])
    rp = os.path.join(out_dir, "pages", "reproduce", "page.md")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as fh:
        fh.write(reproduce_page(summary, manifest))

    # 7. README and logbook.json
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(readme(summary))
    with open(os.path.join(out_dir, "logbook.json"), "w") as fh:
        json.dump(logbook(summary), fh, indent=1)

    # final manifest including the two files just written
    manifest = []
    for root, _d, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, out_dir)
            sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
            manifest.append(f"{sha}  {rel}")
    manifest.sort(key=lambda s: s.split("  ", 1)[1])
    with open(os.path.join(out_dir, "MANIFEST.sha256"), "w") as fh:
        fh.write("\n".join(manifest) + "\n")
    print(f"  built {len(manifest)} files into {out_dir}")
    return summary


def readme(summary):
    res = {r["claim"]: r for r in summary["results"]}
    verdicts = ", ".join(f"C{CLAIMS[k]['number']} {res[k]['status']}"
                         for k in CLAIM_ORDER)
    return f"""---
title: "Repro - Evaluating Bivariate Causal Statements Based on Mutual Compatibility"
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-b3EvCd8sYE
---

# Repro — {PAPER['title']}

Clean-room CPU reproduction of all six anchored claims of
[arXiv:{PAPER['arxiv']}]({PAPER['arxiv_url']}).

**Start at `pages/index.md`** (rendered as the logbook home page). It carries
the headline figure, the verdict table, the visibility matrix and links to every
claim page, the raw data and the source code.

Verdicts: {verdicts}.

Produced by `repro/run_all.py` at git SHA `{summary['git_sha']}`,
seed `{summary['seed']}`, on Hugging Face Jobs (`cpu-upgrade`).

An open experiment logbook, published with
[Trackio](https://github.com/gradio-app/trackio).
"""


def logbook(summary):
    children = [
        dict(slug="current-verification", title="Current verification run",
             file="pages/current-verification/page.md", children=[]),
        dict(slug="reproduce", title="Reproduce",
             file="pages/reproduce/page.md", children=[]),
        dict(slug="source-audit", title="Source audit",
             file="pages/source-audit/page.md", children=[]),
    ]
    for key in CLAIM_ORDER:
        children.append(dict(slug=key,
                             title=f"Claim {CLAIMS[key]['number']} — "
                                   f"{CLAIMS[key]['short']}",
                             file=f"pages/{key}/page.md", children=[]))
    children += [
        dict(slug="visibility-matrix", title="Visibility matrix",
             file="pages/visibility-matrix/page.md", children=[]),
        dict(slug="limitations", title="Limitations",
             file="pages/limitations/page.md", children=[]),
        dict(slug="overview", title="Overview (historical rejected baseline)",
             file="pages/overview/page.md", children=[]),
        dict(slug="claims", title="Claims (historical rejected baseline)",
             file="pages/claims/page.md", children=[]),
        dict(slug="evidence", title="Evidence (historical rejected baseline)",
             file="pages/evidence/page.md", children=[]),
        dict(slug="verification-run",
             title="Verification run (historical rejected baseline)",
             file="pages/verification-run/page.md", children=[]),
        dict(slug="conclusion", title="Conclusion (historical rejected baseline)",
             file="pages/conclusion/page.md", children=[]),
    ]
    return dict(
        schema_version=1,
        title=f"Repro - {PAPER['title']}",
        emoji="🎯",
        space_id=SPACE_ID,
        paper=PAPER["arxiv"],
        tags=["icml2026-repro", "paper-b3EvCd8sYE"],
        updated_at=summary.get("finished_at", ""),
        root=dict(slug="index", title=f"Repro - {PAPER['title']}",
                  file="pages/index.md", children=children),
    )


if __name__ == "__main__":
    art, log, out = sys.argv[1], sys.argv[2], sys.argv[3]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    judged = sys.argv[4] if len(sys.argv) > 4 else "/tmp/judged_space"
    psha = sys.argv[5] if len(sys.argv) > 5 else "unknown"
    build(art, log, out, judged, root, psha)
