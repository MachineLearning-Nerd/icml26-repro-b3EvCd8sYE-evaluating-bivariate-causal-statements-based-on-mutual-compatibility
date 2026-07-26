"""Recover the raw artifact files from a run log.

`orx logs <runId>` is the only evidence channel in OpenResearch local mode, so
`repro.run_all` prints every artifact framed by ARTIFACT-BEGIN/END markers.
This reverses that, verifying each file's SHA-256 as it goes.

    orx logs <runId> > run.log
    python tools/extract_artifacts.py run.log .openresearch/artifacts
"""
from __future__ import annotations

import hashlib
import os
import sys

BEGIN = "<<<ARTIFACT-BEGIN"
END = "<<<ARTIFACT-END"


def main(log_path: str, out_dir: str) -> int:
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines(keepends=True)

    i, written, bad = 0, 0, 0
    while i < len(lines):
        if not lines[i].startswith(BEGIN):
            i += 1
            continue
        header = lines[i].split()
        rel, sha = header[1], header[2].split("=", 1)[1]
        body: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].startswith(END):
            body.append(lines[i])
            i += 1
        i += 1
        data = "".join(body)
        if data.endswith("\n") and not data.endswith("\n\n"):
            candidates = [data, data[:-1]]
        else:
            candidates = [data]
        chosen = None
        for cand in candidates:
            if hashlib.sha256(cand.encode()).hexdigest() == sha:
                chosen = cand
                break
        path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if chosen is None:
            bad += 1
            print(f"  SHA MISMATCH {rel}", file=sys.stderr)
            chosen = candidates[0]
        with open(path, "w") as out:
            out.write(chosen)
        written += 1
    print(f"extracted {written} artifact files to {out_dir} "
          f"({bad} with a checksum mismatch)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
                  else ".openresearch/artifacts"))
