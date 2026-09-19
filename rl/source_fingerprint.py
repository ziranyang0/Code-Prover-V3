"""Fingerprint local source changes, including scoped untracked files."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

SOURCE_SCOPES = (
    "agents",
    "requirements-rl.txt",
    "rl",
    "scripts/1_algo",
    "scripts/2_perf",
    "scripts/3_platform",
    "tools",
    "train.py",
    "train_async.py",
    "verifier",
)


def git_source_fingerprint(root: Path) -> str:
    root = root.resolve()
    digest = hashlib.sha256()
    tracked_patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    digest.update(b"tracked-patch\0")
    digest.update(tracked_patch)

    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *SOURCE_SCOPES,
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.split(b"\0")
    for encoded in sorted(path for path in untracked if path):
        relative = encoded.decode("utf-8")
        path = root / relative
        if not path.is_file():
            continue
        digest.update(b"untracked\0")
        digest.update(encoded)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: source_fingerprint.py GIT_ROOT")
    print(git_source_fingerprint(Path(sys.argv[1])))
