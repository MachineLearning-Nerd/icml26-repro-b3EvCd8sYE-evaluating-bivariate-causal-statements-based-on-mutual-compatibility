"""Cumulative regression suite: every claim, every run, one fixed command.

    bash run.sh

The suite exits non-zero unless every claim reaches a terminal verdict that its
own evidence supports.  A claim whose evidence fails its contract is reported
BLOCKED, never silently downgraded to a pass.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

from .config import CFG, PROFILE, SEED
from .harness import ARTIFACTS, banner, cpu_info, git_sha, write_json
from .claims import claim1, claim2, claim3, claim4, claim5, claim6

CLAIMS = [
    ("claim1", claim1, "Lemma 2.3: unique induced multivariate SEM"),
    ("claim2", claim2, "Theorem 2.9: positive expected compatibility score"),
    ("claim3", claim3, "Theorem 2.10: polynomial sample complexity"),
    ("claim4", claim4, "Figures 2 & 4: synthetic and LLM compatibility scores"),
    ("claim5", claim5, "NP-hardness of the exact incompatibility score"),
    ("claim6", claim6, "Lemma 3.7 & Figures 5-7: heuristic incompatibility"),
]


def main() -> int:
    t0 = time.time()
    sha = git_sha()
    info = cpu_info()
    banner("REPRODUCTION OF arXiv:2606.00278 -- "
           "Evaluating Bivariate Causal Statements Based on Mutual Compatibility")
    print(f"  git sha      : {sha}")
    print(f"  profile      : {PROFILE}")
    print(f"  global seed  : {SEED}")
    print(f"  platform     : {info['platform']}")
    print(f"  cpu count    : {info['cpu_count']}")
    print(f"  python/numpy : {info['python']} / {info['numpy']}")
    print(f"  artifacts    : {ARTIFACTS}", flush=True)

    results = []
    for name, mod, title in CLAIMS:
        try:
            rec = mod.run()
        except Exception as exc:                       # a crash is not a pass
            import traceback
            traceback.print_exc()
            rec = dict(claim=name, title=title, status="BLOCKED",
                       checks=[dict(name="verifier completed", passed=False,
                                    detail=repr(exc))],
                       notes=[f"verifier raised {exc!r}"], runtime_s=0.0)
            write_json(name, "verdict.json", rec)
        results.append(rec)

    banner("VERDICT SUMMARY")
    points = {"VERIFIED": 2, "FALSIFIED": 2, "BLOCKED": 0}
    total = 0
    for rec in results:
        n_ok = sum(1 for c in rec["checks"] if c["passed"])
        total += points[rec["status"]]
        print(f"  {rec['claim']}: {rec['status']:<9} "
              f"({n_ok}/{len(rec['checks'])} checks passed, "
              f"{rec['runtime_s']:.1f}s)  {rec['title']}")
    print(f"\n  self-assessed: {total}/{2*len(results)} "
          f"(a forecast only -- the live judge assigns the score)")

    summary = dict(git_sha=sha, profile=PROFILE, seed=SEED, cpu=info,
                   config={k: str(v) for k, v in CFG.items()},
                   total_runtime_s=round(time.time() - t0, 1),
                   results=results,
                   self_assessed_points=total,
                   max_points=2 * len(results))
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(os.path.join(ARTIFACTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  wrote {os.path.join(ARTIFACTS, 'summary.json')}")
    print(f"  total runtime {summary['total_runtime_s']}s", flush=True)

    # Non-zero exit whenever any claim failed to reach a supported verdict.
    blocked = [r["claim"] for r in results if r["status"] == "BLOCKED"]
    if blocked:
        print(f"\n  FAILING: {len(blocked)} claim(s) BLOCKED: {blocked}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
