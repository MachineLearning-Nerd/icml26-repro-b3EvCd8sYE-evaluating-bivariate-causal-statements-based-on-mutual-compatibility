"""Shared verifier plumbing: artifact paths, result records, CSV/JSON writing."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, ".openresearch", "artifacts")


def artifact_dir(claim: str) -> str:
    d = os.path.join(ARTIFACTS, claim)
    os.makedirs(d, exist_ok=True)
    return d


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def cpu_info() -> dict:
    return dict(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        python=platform.python_version(),
        numpy=np.__version__,
    )


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def write_json(claim: str, name: str, obj) -> str:
    path = os.path.join(artifact_dir(claim), name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return path


def write_csv(claim: str, name: str, rows: list[dict], columns=None) -> str:
    import csv
    path = os.path.join(artifact_dir(claim), name)
    if not rows:
        open(path, "w").close()
        return path
    columns = columns or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(x, z: float = 1.959963985) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a normal-approximation CI on the mean.

    ``z`` may be raised to obtain a Bonferroni-corrected interval when many
    configurations are tested at once."""
    x = np.asarray(x, float)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, m - z * se, m + z * se


class Verdict:
    """One claim's outcome.  ``status`` is VERIFIED, FALSIFIED or BLOCKED."""

    def __init__(self, claim: str, title: str):
        self.claim = claim
        self.title = title
        self.checks: list[dict] = []
        self.status = "BLOCKED"
        self.notes: list[str] = []
        self.t0 = time.time()

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(dict(name=name, passed=bool(passed), detail=detail))
        mark = "ok " if passed else "FAIL"
        print(f"    [{mark}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
        return bool(passed)

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"    note: {text}", flush=True)

    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    def finish(self, status: str) -> dict:
        assert status in ("VERIFIED", "FALSIFIED", "BLOCKED")
        self.status = status
        rec = dict(claim=self.claim, title=self.title, status=status,
                   checks=self.checks, notes=self.notes,
                   runtime_s=round(time.time() - self.t0, 2))
        write_json(self.claim, "verdict.json", rec)
        print(f"\n  ==> {self.claim}: {status}  ({rec['runtime_s']}s)", flush=True)
        return rec
