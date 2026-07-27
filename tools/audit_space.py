"""Evaluator-blind audit of the built logbook.  Exits non-zero on any failure.

    python tools/audit_space.py <space-dir>

This is the pre-publication red team, written as executable checks rather than
as a reading pass, because the property being tested is mechanical: *can a
reviewer who starts at the canonical entrypoint and follows links only reach
every required piece of evidence?*  A human skim cannot answer that reliably
across ninety files; a graph traversal can.

The audit deliberately knows nothing about the verification suite.  It reads
only what was built for publication, exactly as an evaluator would see it.
"""

from __future__ import annotations

import json
import os
import re
import sys

# What the evaluator must be able to find, per claim, starting from the index.
REQUIRED_SECTIONS = [
    ("## The exact claim", "the paper's own wording"),
    ("**Quantifiers.**", "the quantifiers spelled out"),
    ("## Assumptions, and how each is audited", "an assumption audit"),
    ("## Why a finite spot-check is not enough", "why the obvious test fails"),
    ("## Method", "what was actually done"),
    ("## Results — every check, with the measured value", "raw numbers inline"),
    ("## Raw data", "downloadable raw data"),
    ("## Reproducing this claim", "exact command and pinned environment"),
    ("## Limitations and deviations", "limitations"),
]

CLAIMS = [f"claim{i}" for i in range(1, 7)]


class Audit:
    def __init__(self):
        self.failures: list[str] = []
        self.passes = 0

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        if ok:
            self.passes += 1
        else:
            self.failures.append(f"{what}{(' -- ' + detail) if detail else ''}")
        print(f"  [{'ok  ' if ok else 'FAIL'}] {what}"
              + (f"  {detail}" if detail and not ok else ""))
        return ok


def main(root: str) -> int:
    a = Audit()
    logbook = json.load(open(os.path.join(root, "logbook.json")))

    print("\n-- logbook metadata --")
    a.check(logbook.get("space_id") == "DineshAI/b3EvCd8sYE",
            "logbook.json names the correct Space",
            f"got {logbook.get('space_id')!r}")
    a.check("Compatibility" in logbook.get("title", ""),
            "logbook.json title matches this paper",
            f"got {logbook.get('title')!r}")
    a.check(logbook.get("root", {}).get("file") == "pages/index.md",
            "the canonical entrypoint is pages/index.md")

    # every page named in the tree exists, and every page on disk is in the tree
    tree, stack = {}, [logbook["root"]]
    while stack:
        node = stack.pop()
        tree[node["slug"]] = node["file"]
        stack.extend(node.get("children", []))
    on_disk = set()
    for base, _d, files in os.walk(os.path.join(root, "pages")):
        for fn in files:
            if fn.endswith(".md"):
                on_disk.add(os.path.relpath(os.path.join(base, fn), root))
    a.check(all(os.path.exists(os.path.join(root, f)) for f in tree.values()),
            f"all {len(tree)} pages in the navigation tree exist on disk")
    a.check(on_disk == set(tree.values()),
            "no page on disk is missing from the navigation tree",
            f"orphans: {sorted(on_disk - set(tree.values()))}")

    print("\n-- reachability from the canonical entrypoint --")
    # follow #/slug links only, starting at index
    text = {slug: open(os.path.join(root, f)).read() for slug, f in tree.items()}
    seen, frontier = {"index"}, ["index"]
    while frontier:
        cur = frontier.pop()
        for slug in re.findall(r"\(#/([A-Za-z0-9\-_]+)\)", text.get(cur, "")):
            if slug in tree and slug not in seen:
                seen.add(slug)
                frontier.append(slug)
    unreachable = sorted(set(tree) - seen)
    a.check(not unreachable,
            f"all {len(tree)} pages are reachable from pages/index.md by "
            f"following links only", f"unreachable: {unreachable}")
    dangling = sorted({s for t in text.values()
                       for s in re.findall(r"\(#/([A-Za-z0-9\-_]+)\)", t)}
                      - set(tree))
    a.check(not dangling, "no page link points at a missing slug",
            f"dangling: {dangling}")

    print("\n-- current verification is first, history is labelled --")
    idx = text["index"]
    cv = idx.find("current-verification")
    hist = idx.find("Historical rejected baseline")
    a.check(cv != -1 and (hist == -1 or cv < hist),
            "the current verification run appears before the superseded pages")
    for slug in ("overview", "claims", "evidence", "verification-run",
                 "conclusion"):
        a.check(text.get(slug, "").startswith("> ## Historical rejected baseline"),
                f"superseded page '{slug}' carries the exact banner")
    a.check(tree.get("verification-run") is not None
            and "historical" in [n["title"].lower() for n in
                                 [x for x in [logbook["root"]] + list(
                                     _walk(logbook["root"]))
                                  if x["slug"] == "verification-run"]][0],
            "the old 'Verification run' page is labelled historical in the nav")

    print("\n-- per-claim evidence --")
    for key in CLAIMS:
        page = text.get(key, "")
        a.check(bool(page), f"{key}: page exists")
        for marker, why in REQUIRED_SECTIONS:
            a.check(marker in page, f"{key}: {why}")
        a.check("/code/" in page, f"{key}: links to executable code")
        a.check("/artifacts/" in page, f"{key}: links to downloadable raw data")
        a.check("negative control" in page.lower(),
                f"{key}: reports a negative control")
        a.check("exits non-zero" in page,
                f"{key}: states that the verifier exits non-zero on failure")
        a.check(re.search(r"Git SHA `[0-9a-f]{7,40}`", page) is not None,
                f"{key}: records the git SHA")
        a.check("seed `" in page, f"{key}: records the seed")
        a.check("CPU cores" in page, f"{key}: records the CPU")
        a.check("runtime" in page.lower(), f"{key}: records the runtime")

    print("\n-- visibility matrix --")
    vm = text.get("visibility-matrix", "")
    for col in ["Claim", "Canonical page", "Code visible", "Data inline",
                "Raw link", "Checker", "Control", "Exact claim tested",
                "Reviewer verdict"]:
        a.check(col in vm, f"visibility matrix has the '{col}' column")
    for key in CLAIMS:
        row = [ln for ln in vm.splitlines()
               if ln.startswith("| Claim") and f"#/{key})" in ln]
        a.check(bool(row), f"visibility matrix has a row for {key}")
        if row:
            cells = [c.strip() for c in row[0].strip("|").split("|")]
            a.check(len(cells) == 9 and all(cells),
                    f"visibility matrix row for {key} has all 9 cells filled",
                    f"got {len(cells)} cells: {cells}")

    print("\n-- every referenced file actually exists --")
    missing = set()
    for slug, t in text.items():
        for url in re.findall(r"resolve/main/([A-Za-z0-9._/\-]+)", t):
            if not os.path.exists(os.path.join(root, url)):
                missing.add(f"{slug} -> {url}")
    a.check(not missing, "every raw/figure/code URL resolves to a built file",
            f"missing: {sorted(missing)[:6]}")

    print("\n-- verdicts are consistent with the run --")
    summ = json.load(open(os.path.join(root, "artifacts", "summary.json")))
    res = {r["claim"]: r["status"] for r in summ["results"]}
    for key in CLAIMS:
        a.check(f"## Verdict: **{res[key]}**" in text[key],
                f"{key}: page verdict matches summary.json ({res[key]})")
    a.check(all(v == "VERIFIED" for v in res.values())
            or "BLOCKED" in idx or "FALSIFIED" in idx,
            "any non-VERIFIED claim is visible on the index page")

    print(f"\n{a.passes} checks passed, {len(a.failures)} failed")
    for f in a.failures:
        print(f"  FAILED: {f}")
    return 1 if a.failures else 0


def _walk(node):
    for c in node.get("children", []):
        yield c
        yield from _walk(c)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
