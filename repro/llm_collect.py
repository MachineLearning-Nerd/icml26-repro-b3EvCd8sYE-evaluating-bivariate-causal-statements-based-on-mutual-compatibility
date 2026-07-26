"""Collect LLM causal statements, following Appendix D.3 verbatim.

This module talks to the network, so it is **not** part of the reproduction
run.  It is executed once, out of band, and its output -- the complete
conversation transcripts -- is committed under ``data/llm/``.  The verifiers
then score those committed transcripts deterministically.  Splitting it this
way means the scored evidence is auditable and byte-reproducible even though
the model responses themselves are not.

Usage:  python -m repro.llm_collect [linear|graphical] [--runs N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

from .gapminder import (
    GEN_PARAMS,
    MODELS,
    VARIABLES,
    correlation_string,
    variables_string,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "llm")
ENDPOINT = "https://router.huggingface.co/v1/chat/completions"

SYSTEM_LINEAR = (
    "You are a causality expert, tasked to estimate standardized TOTAL causal "
    "effects between country development indicators. Return your answer in HTML "
    "format:\n"
    "<answer>CAUSAL COEFFICIENT: <number></answer>\n"
    "For example: <answer>CAUSAL COEFFICIENT: 0.35</answer> or "
    "<answer>CAUSAL COEFFICIENT: -0.62</answer>\n"
    "The causal coefficient quantifies the expected change of the effect "
    "variable in standard deviations, given an intervention that changes the "
    "cause variable by 1 standard deviation. It includes the effect of all "
    "direct causal pathways from the cause to the effect variable. Do not "
    "assume away confounding; use realistic domain knowledge.\n"
    "No other text."
)

SYSTEM_GRAPHICAL = (
    "You are a causality expert analyzing relationships between country-level "
    "development indicators. For each pair of variables, you will assess:\n"
    "1. Whether there is a total causal effect between them, and in which "
    "direction\n"
    "2. Whether there is confounding (correlation not explained by the causal "
    "effect between the pair)\n\n"
    "Guidelines:\n"
    "- Answer YES for causal effect if you expect that an intervention on the "
    "cause variable would significantly change the effect variable\n"
    "- Be conservative about confounding - only answer YES for confounding if "
    "MOST of the correlation cannot be explained by the causal effect\n"
    "- Use realistic domain knowledge about socioeconomic factors\n\n"
    "Return your answer in the following HTML format:\n"
    "<causal effect>\n<exists>YES or NO</exists>\n"
    "<direction>A TO B or B TO A or NONE</direction>\n</causal effect>\n"
    "<confounding>\n<exists>YES or NO</exists>\n</confounding>\n"
    "Where A TO B means the first variable causes the second, and B TO A means "
    "the second causes the first."
)

INTRO_LINEAR = (
    "I have observational data on 7 country-level development indicators.\n"
    "Correlation matrix:\n{corr}\n"
    "Variable descriptions:\n{vars}\n\n"
    "Before we begin estimating causal coefficients, please determine a "
    "plausible causal ordering of these 7 variables (from root causes to "
    "downstream effects). Return the ordering as a comma-separated list of "
    "variable names inside <ordering> tags. Use the exact variable names shown "
    "above.\nFor example: <ordering>var a, var b, var c, ...</ordering>"
)

INTRO_GRAPHICAL = (
    "I have observational data on 7 country-level development indicators.\n"
    "Correlation matrix:\n{corr}\n"
    "Variable descriptions:\n{vars}\n\n"
    "I will now ask you about each pair of variables. For each pair:\n"
    "1. Assess if there is a causal effect\n"
    "2. Assess if there is strong confounding"
)

RETRY_ORDERING = ("Please provide the causal ordering as a comma-separated list "
                  "of the exact variable names inside <ordering> tags. The "
                  "variable names are: {vars}")
RETRY_COEF = ("Please provide your answer in the correct format: "
              "<answer>CAUSAL COEFFICIENT: <number></answer>")
RETRY_GRAPH = ("Please provide your answer in the correct format: "
               "<causal effect>\n<exists>YES or NO</exists>\n"
               "<direction>A TO B or B TO A or NONE</direction>\n"
               "</causal effect>\n<confounding>\n<exists>YES or NO</exists>\n"
               "</confounding>")

MAX_FORMAT_RETRIES = 5      # Appendix D.3: "up to 5 times in a row"


def _token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:                                   # optional, collection-only dep
            from huggingface_hub import get_token
            tok = get_token()
        except ImportError:
            tok = None
    if not tok:
        sys.exit("no Hugging Face token available (set HF_TOKEN)")
    return tok


def _chat(model: str, messages: list[dict], token: str) -> str:
    body = json.dumps(dict(model=model, messages=messages, **GEN_PARAMS)).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as fh:
                d = json.load(fh)
            msg = d["choices"][0]["message"]
            text = msg.get("content") or ""
            if not text:
                # some reasoning models put the visible answer after a
                # reasoning field; fall back to it rather than losing the turn
                text = msg.get("reasoning_content") or msg.get("reasoning") or ""
            return text
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(4 * (attempt + 1))
                continue
            raise
        except Exception as e:                       # transient network problem
            last = repr(e)
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"chat failed after retries: {last}")


def parse_ordering(text: str) -> list[str] | None:
    m = re.search(r"<ordering>(.*?)</ordering>", text, re.S | re.I)
    if not m:
        return None
    names = [x.strip().lower() for x in m.group(1).split(",")]
    canon = {v.lower(): v for v in VARIABLES}
    out = [canon[x] for x in names if x in canon]
    return out if sorted(out) == sorted(VARIABLES) else None


def parse_coefficient(text: str) -> float | None:
    m = re.search(r"CAUSAL\s+COEFFICIENT:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                  text, re.I)
    return float(m.group(1)) if m else None


def parse_graph_answer(text: str):
    ce = re.search(r"<causal\s*effect\s*>(.*?)</causal\s*effect\s*>", text, re.S | re.I)
    cf = re.search(r"<confounding>(.*?)</confounding>", text, re.S | re.I)
    if not ce or not cf:
        return None
    ex = re.search(r"<exists>\s*(YES|NO)\s*</exists>", ce.group(1), re.I)
    di = re.search(r"<direction>\s*(A TO B|B TO A|NONE)\s*</direction>",
                   ce.group(1), re.I)
    cx = re.search(r"<exists>\s*(YES|NO)\s*</exists>", cf.group(1), re.I)
    if not ex or not cx or (ex.group(1).upper() == "YES" and not di):
        return None
    return dict(effect=ex.group(1).upper(),
                direction=(di.group(1).upper() if di else "NONE"),
                confounding=cx.group(1).upper())


def _ask(model, messages, token, parser, retry_msg, transcript):
    """Send one question, re-prompting up to five times on a format error."""
    for attempt in range(MAX_FORMAT_RETRIES + 1):
        reply = _chat(model, messages, token)
        transcript.append(dict(role="assistant", content=reply))
        parsed = parser(reply)
        if parsed is not None:
            messages.append(dict(role="assistant", content=reply))
            return parsed
        if attempt == MAX_FORMAT_RETRIES:
            return None
        messages.append(dict(role="assistant", content=reply))
        messages.append(dict(role="user", content=retry_msg))
        transcript.append(dict(role="user", content=retry_msg))
    return None


def collect_linear(model_route: str, run_idx: int, token: str) -> dict:
    intro = INTRO_LINEAR.format(corr=correlation_string(), vars=variables_string())
    messages = [dict(role="system", content=SYSTEM_LINEAR),
                dict(role="user", content=intro)]
    transcript = list(messages)
    order = _ask(model_route, messages, token, parse_ordering,
                 RETRY_ORDERING.format(vars=", ".join(VARIABLES)), transcript)
    if order is None:
        return dict(model=model_route, run=run_idx, ok=False,
                    reason="no valid causal ordering", transcript=transcript)

    bridge = ("I will now ask you to estimate the total linear causal "
              "coefficient for several pairs of variables, following the "
              "ordering you provided. For each question, please provide your "
              "answer in the format: <answer>CAUSAL COEFFICIENT: <number></answer>")
    messages.append(dict(role="user", content=bridge))
    transcript.append(dict(role="user", content=bridge))

    coefficients = {}
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            q = (f"Estimate the total linear causal coefficient for the causal "
                 f"effect of {order[a]} on {order[b]}.")
            messages.append(dict(role="user", content=q))
            transcript.append(dict(role="user", content=q))
            val = _ask(model_route, messages, token, parse_coefficient,
                       RETRY_COEF, transcript)
            if val is None:
                return dict(model=model_route, run=run_idx, ok=False,
                            reason=f"no coefficient for ({order[a]},{order[b]})",
                            ordering=order, transcript=transcript)
            coefficients[f"{order[a]}||{order[b]}"] = val
    return dict(model=model_route, run=run_idx, ok=True, ordering=order,
                coefficients=coefficients, transcript=transcript)


def collect_graphical(model_route: str, run_idx: int, token: str) -> dict:
    intro = INTRO_GRAPHICAL.format(corr=correlation_string(),
                                   vars=variables_string())
    messages = [dict(role="system", content=SYSTEM_GRAPHICAL),
                dict(role="user", content=intro)]
    transcript = list(messages)

    pairs = [(a, b) for a in range(len(VARIABLES))
             for b in range(a + 1, len(VARIABLES))]
    # Appendix D.3: "we present each pair in a random order, since our graphical
    # incompatibility score also tests the extent to which the causal statements
    # are acyclic".  The order is randomised per run and recorded.
    rnd = random.Random(hash((model_route, run_idx)) & 0xFFFFFFFF)
    pairs = [(a, b) if rnd.random() < 0.5 else (b, a) for a, b in pairs]
    rnd.shuffle(pairs)

    answers = {}
    for (a, b) in pairs:
        va, vb = VARIABLES[a], VARIABLES[b]
        q = (f'Consider the pair: "{va.capitalize()}" (A) and '
             f'"{vb.capitalize()}" (B).\n'
             f"1. Is there a causal effect between them? If yes, in which "
             f"direction?\n"
             f"2. Is there confounding (significant correlation not explained by "
             f"the causal effect between these two)?\n"
             f"Provide your assessment in the required format.")
        messages.append(dict(role="user", content=q))
        transcript.append(dict(role="user", content=q))
        parsed = _ask(model_route, messages, token, parse_graph_answer,
                      RETRY_GRAPH, transcript)
        if parsed is None:
            return dict(model=model_route, run=run_idx, ok=False,
                        reason=f"no valid answer for ({va},{vb})",
                        transcript=transcript)
        answers[f"{va}||{vb}"] = dict(A=va, B=vb, **parsed)
    return dict(model=model_route, run=run_idx, ok=True,
                pair_order=[[VARIABLES[a], VARIABLES[b]] for a, b in pairs],
                answers=answers, transcript=transcript)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("linear", "graphical"))
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of routes")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent conversations (each is internally serial)")
    args = ap.parse_args()
    # Section 2.7: 15 runs per model for the linear scores; Figure 6: 10
    # repetitions for the graphical ones.
    runs = args.runs if args.runs else (15 if args.mode == "linear" else 10)
    token = _token()
    os.makedirs(DATA, exist_ok=True)
    routes = ([m["route"] for m in MODELS] if not args.models
              else args.models.split(","))

    jobs = []
    for route in routes:
        for run_idx in range(runs):
            slug = route.replace("/", "__")
            path = os.path.join(DATA, f"{args.mode}__{slug}__run{run_idx:02d}.json")
            if os.path.exists(path):
                continue
            jobs.append((route, run_idx, path))
    print(f"{len(jobs)} conversations to collect "
          f"({args.mode}, {len(routes)} models x {runs} runs)", flush=True)

    fn = collect_linear if args.mode == "linear" else collect_graphical

    def one(job):
        route, run_idx, path = job
        t0 = time.time()
        try:
            rec = fn(route, run_idx, token)
        except Exception as e:
            rec = dict(model=route, run=run_idx, ok=False,
                       reason=f"exception: {e!r}")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        return path, rec

    # Conversations are independent; run them concurrently.  Each conversation
    # is strictly sequential internally, so this does not change the protocol.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for fut in as_completed(futs):
            path, rec = fut.result()
            done += 1
            print(f"[{done}/{len(jobs)}] "
                  f"{'ok  ' if rec.get('ok') else 'FAIL'} {os.path.basename(path)} "
                  f"({rec['elapsed_s']}s)"
                  + ("" if rec.get("ok") else f" -- {rec.get('reason')}"),
                  flush=True)


if __name__ == "__main__":
    main()
