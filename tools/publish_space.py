"""Publish the built logbook to the existing Hugging Face Space.

    python tools/publish_space.py <space-dir> [--dry-run]

Three properties are enforced *in code*, before anything is uploaded, because
each of them is easy to violate by hand and expensive to undo:

1.  **Text only.**  Only files whose extension is on :data:`TEXT_EXT` are
    uploaded, and each is verified to decode as UTF-8 first.  The Space's
    existing binary assets (the Trackio logos) are left exactly where they are.

2.  **Nothing is ever deleted.**  The commit contains additions only, so the
    judged revision's file set is necessarily a subset of the published one.
    That is checked explicitly against the live repository as well, rather than
    inferred from the fact that no deletion was requested.

3.  **One Space.**  The target is hard-coded.  No flag can redirect it.

The script prints a SHA-256 manifest of everything it uploads and refuses to
proceed if any gate fails.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

SPACE_ID = "DineshAI/b3EvCd8sYE"        # not configurable, by design
JUDGED_REV = "af512d65233bb770d870f9ce6edadd1942c4d171"

TEXT_EXT = {".md", ".json", ".csv", ".svg", ".py", ".toml", ".lock", ".sh",
            ".txt", ".html", ".css", ".js", ".log", ".sha256", ".gitattributes",
            ".yaml", ".yml"}


def text_files(root: str) -> list[str]:
    out = []
    for base, _dirs, files in os.walk(root):
        for fn in files:
            path = os.path.join(base, fn)
            rel = os.path.relpath(path, root)
            ext = os.path.splitext(fn)[1] or os.path.basename(fn)
            if ext not in TEXT_EXT:
                continue
            with open(path, "rb") as fh:
                raw = fh.read()
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                print(f"  SKIP (not valid UTF-8): {rel}")
                continue
            out.append(rel)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("space_dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub import CommitOperationAdd

    api = HfApi()
    root = args.space_dir

    # -- gate 1: everything we intend to send is text ----------------------
    uploads = text_files(root)
    skipped = []
    for base, _dirs, files in os.walk(root):
        for fn in files:
            rel = os.path.relpath(os.path.join(base, fn), root)
            if rel not in uploads:
                skipped.append(rel)
    print(f"gate 1  text-only: {len(uploads)} file(s) to upload, "
          f"{len(skipped)} non-text file(s) left untouched")
    for rel in sorted(skipped):
        print(f"           not uploaded (binary): {rel}")

    # -- gate 2: the judged revision's file set survives -------------------
    live = set(api.list_repo_files(SPACE_ID, repo_type="space"))
    judged = set(api.list_repo_files(SPACE_ID, repo_type="space",
                                     revision=JUDGED_REV))
    after = live | set(uploads)
    missing = sorted(f for f in judged if f not in after)
    print(f"gate 2  subset: judged revision {JUDGED_REV[:12]} has "
          f"{len(judged)} file(s); after this commit the Space will have "
          f"{len(after)}; missing from the new set: {missing}")
    if missing:
        print("REFUSING: the judged revision's file set would not be preserved.")
        return 1
    if not judged:
        print("REFUSING: could not read the judged revision's file list.")
        return 1

    # -- gate 3: the target is the one Space -------------------------------
    print(f"gate 3  target: {SPACE_ID} (hard-coded)")

    # -- manifest ----------------------------------------------------------
    manifest = []
    ops = []
    for rel in uploads:
        path = os.path.join(root, rel)
        with open(path, "rb") as fh:
            raw = fh.read()
        manifest.append(f"{hashlib.sha256(raw).hexdigest()}  {rel}")
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=path))
    print(f"\nSHA-256 manifest ({len(manifest)} files):")
    for line in manifest:
        print(f"  {line}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0

    info = api.create_commit(
        repo_id=SPACE_ID, repo_type="space", operations=ops,
        commit_message="Publish the round-3 verification: all six claims with "
                       "raw data, checkers, controls and a visibility matrix",
        commit_description="Supersedes the revision judged 6/12. The five "
                           "superseded pages are preserved verbatim under a "
                           "'Historical rejected baseline' banner; no file is "
                           "deleted.")
    print(f"\npublished: {info.oid if hasattr(info, 'oid') else info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
