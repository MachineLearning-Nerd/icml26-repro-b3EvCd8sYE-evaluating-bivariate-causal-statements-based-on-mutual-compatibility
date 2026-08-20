# Reproducing *Evaluating Bivariate Causal Statements Based on Mutual Compatibility*

## Collection classification and audit boundary

This repository is a **legacy/source workspace** for *Evaluating Bivariate Causal Statements Based on Mutual Compatibility*
(arXiv `2606.00278`, OpenReview `b3EvCd8sYE`). It is preserved
separately from the standardized canonical record at
[`icml26-bivariate-causal-compatibility`](https://github.com/MachineLearning-Nerd/icml26-bivariate-causal-compatibility).

The claim results and scores recorded below are historical results of this
workspace. They are not new paper-level verifications performed while
organizing the collection. The collection audit did not run the scientific
implementation; the canonical record documents its own scoped status and
limitations.

### How the historical claim evidence is produced

The claim table and experiment log below are the authoritative mapping from
each paper claim to its producer, command, control, and evidence artifact. In
this workspace, the claim-specific scripts, notebooks, and checker routes write their raw bundles and verdict pages under `.openresearch/artifacts/`; the claim table below names each statistical or symbolic producer.

The former `orx/*` branches are historical workstreams, not additional final
publication claims. Their purposes and tips are preserved in
[`BRANCH_AUDIT.md`](BRANCH_AUDIT.md). Citation and author acknowledgment
details are in [`CITATION.cff`](CITATION.cff) and
[`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

A clean-room, CPU-only reproduction of all six anchored claims of
[arXiv:2606.00278](https://arxiv.org/abs/2606.00278) (Erik Jahn, Dominik
Janzing; OpenReview [b3EvCd8sYE](https://openreview.net/forum?id=b3EvCd8sYE)).

**Full evidence, per claim:** <https://huggingface.co/spaces/DineshAI/b3EvCd8sYE>
· **Report:** [`reports/report.md`](reports/report.md)
· **Notebook:** [`notebooks/compatibility.py`](notebooks/compatibility.py)

## Reproduce

One command, one pinned environment, on every node of the experiment tree.
Experimental variation lives in committed code (`repro/config.py`) — never in
the command line and never in environment variables.

```bash
git clone https://github.com/MachineLearning-Nerd/icml26-repro-b3EvCd8sYE-evaluating-bivariate-causal-statements-based-on-mutual-compatibility
cd icml26-repro-b3EvCd8sYE-evaluating-bivariate-causal-statements-based-on-mutual-compatibility
bash run.sh
```

`run.sh` installs [uv](https://docs.astral.sh/uv/) if it is absent, then runs:

```bash
uv sync --frozen
uv run --frozen python -m repro.run_all
```

Python is pinned to `3.12.*` and every dependency to an exact version by
`uv.lock` (numpy 2.2.6, scipy 1.15.3, sympy 1.13.3, matplotlib 3.10.3,
networkx 3.4.2). A smoke profile exercises the whole suite in about 40 seconds:

```bash
REPRO_PROFILE=smoke uv run --frozen python -m repro.run_all
```

**Exit semantics.** `repro/run_all.py` returns **1** if any claim ends BLOCKED,
and a crash inside a verifier is recorded as BLOCKED rather than skipped. Each
individual check records a FAIL when its evidence does not meet its stated
contract. The suite cannot report success without the evidence behind it.

**Recovering the raw data.** In OpenResearch local mode the run log is the only
channel back from the compute node, so the suite prints every artifact into
stdout framed by markers and tagged with its SHA-256:

```bash
orx logs <runId> > run.log
python tools/extract_artifacts.py run.log .openresearch/artifacts   # re-checksums; non-zero on mismatch
python tools/figures.py .openresearch/artifacts figures
python tools/build_space.py .openresearch/artifacts run.log space /tmp/judged_space <paper-sha>
```

## The six claims

| # | Claim | Anchor | How it is settled |
|---|---|---|---|
| 1 | An acyclic list of linear bivariate statements induces exactly one multivariate SEM | Lemma 2.3 | Symbolic identities in generic Γ, n = 2..6 |
| 2 | True statements have strictly positive expected compatibility | Theorem 2.9 | Exhaustive path-configuration certificate (n = 3..9) + exact polynomial identities (n = 3..6) + Monte Carlo |
| 3 | Polynomial sample complexity | Theorem 2.10 | Derivation reconstructed from exact sensitivity × measured concentration; δ direction as a large-deviation rate |
| 4 | Compatibility degrades monotonically with statement error; LLM statements are scored | Figures 2 and 4 | Full-scale reproduction of Figure 2; 13-model LLM ladder |
| 5 | Computing the exact incompatibility score is NP-hard | Section 3.3 | Exhaustive over all 25 (n=3) and all 543 (n=4) acyclic digraphs |
| 6 | The greedy heuristic upper-bounds the exact score and is zero exactly when it is | Lemma 3.7, Figures 5–7 | Exhaustive over all 512 (n=3) and all 262 144 (n=4) statement graphs |

Every claim page on the logbook carries the paper's wording with its
quantifiers, an assumption audit, the executable verifier, raw CSV/JSON, an
independent checker, and a negative control that fails for a stated reason.

## Experiment log

Each row is a node of the experiment tree. The run command is identical on
every node by construction; nodes differ only in committed code.

| Node | Branch | Commit | Run command (verbatim) | Compute | Runtime | Result |
|---|---|---|---|---|---|---|
| Baseline — faithful core library + honest claim contracts | `orx/baseline-faithful-core-library-honest-claim-cont` | `3778b35` | `bash run.sh` | HF Jobs `cpu-upgrade`, 8 vCPU | 47m52s | 6/12 self-assessed; claims 2, 3, 4 BLOCKED |
| Round 1 — statistical fixes + extended LLM capacity ladder | `orx/round-1-statistical-fixes-extended-llm-capacity` | `7cd0823` | `bash run.sh` | HF Jobs `cpu-upgrade`, 64 vCPU | 1h41m | 8/12; claim 4 fixed; 2 and 3 each with one failing check |
| Round 2 — symbolic certificates + reconstructed derivation | `orx/round-2-symbolic-certificates-for-thm-2-9-recons` | `38aeb66` | `bash run.sh` | HF Jobs `cpu-upgrade`, 64 vCPU | *(see report)* | *(see report)* |

The launcher is `orx exp run <expId> --backend hf --flavor cpu-upgrade --image
python:3.12 --timeout 8h`. All compute is CPU; no GPU result is claimed or
needed.

## Layout

```
repro/            the verification suite (the thing that runs)
  linear.py       Definitions 2.5-2.7 and the O(n^4) disjoint-path-pair DP
  graphical.py    statement graphs, exact incompatibility, Appendix C's greedies
  symbolic.py     machine-checkable certificates for the proof of Theorem 2.9
  sensitivity.py  the closed-form gradient of comp and the concentration probe
  generate.py     Appendix D.1's generative procedure
  gapminder.py    Table 1's correlation matrix and the model ladder
  llm_collect.py  Appendix D.3's prompts, verbatim (network; not part of a run)
  claims/         one verifier per claim
  run_all.py      the entrypoint; exits non-zero if any claim is BLOCKED
tools/            figures, artifact extraction, the logbook builder
notebooks/        a self-contained marimo walkthrough
data/llm/         committed LLM transcripts (scoring is deterministic given these)
reports/          the written report and its figures
```

## Two places where the paper needed disambiguation

Both are recorded in full on the logbook's *Source audit* page, because both
change the numbers and a reader should be able to disagree and re-run.

1. **Definition 3.1** — read fully literally, a pure common-ancestor path
   counts as a confounding path, which makes Lemma 3.5 false. Requiring at
   least one bidirected edge makes Appendix D.1's ground-truth graphs
   compatible 300/300 at zero injected errors; the literal reading manages
   219/300.
2. **Lemma 3.5's necessity direction has a genuine gap.** A three-vertex
   counterexample is given. It does not affect any of the six claims, because
   Definition 3.6 is stated directly in terms of the three properties.

## Provenance

Reproduction code written from the paper text; the authors' own code is at
<https://github.com/ejahn17/compatibility-scores>. LLM transcripts were
collected through the Hugging Face inference router (the paper used Amazon
Bedrock) and are committed in full, so all scoring is reproducible even though
the generation is not.
