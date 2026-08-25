"""Replay a finished module into git as a sequence of section-sized commits.

Each file is cut at its own banner comments (a ``# ---`` rule, a title line,
another ``# ---`` rule), so every intermediate state is a syntactically valid
prefix of the module rather than a half-written statement.
"""
import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANNER = re.compile(r"^\s*#\s*-{5,}\s*$")


def sections(text):
    """Yield (title, start_line) for each ``banner / title / banner`` header."""
    lines = text.split("\n")
    out = []
    for i in range(len(lines) - 2):
        if (BANNER.match(lines[i])
                and not BANNER.match(lines[i + 1])
                and BANNER.match(lines[i + 2])
                and lines[i + 1].lstrip().startswith("#")):
            out.append((lines[i + 1].lstrip("# ").strip(), i))
    return out


def git(*args):
    subprocess.run(["git", *args], check=True, cwd=ROOT)


def _write(path, text):
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)


def replay(path, prefix, intro=None):
    """Commit ``path`` one section at a time, then commit the finished file."""
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    full = io.open(path, encoding="utf-8").read()
    lines = full.split("\n")
    secs = sections(full)

    if intro and secs:
        _write(path, "\n".join(lines[:secs[0][1]]).rstrip() + "\n")
        git("add", rel)
        git("commit", "-q", "-m", intro)

    for idx, (title, _start) in enumerate(secs):
        end = secs[idx + 1][1] if idx + 1 < len(secs) else len(lines)
        _write(path, "\n".join(lines[:end]).rstrip() + "\n")
        git("add", rel)
        git("commit", "-q", "-m", f"{prefix}: {title.lower()}")

    _write(path, full)
    subprocess.run(["git", "add", rel], cwd=ROOT)
    subprocess.run(["git", "commit", "-q", "-m", f"{prefix}: finalise module"],
                   cwd=ROOT)


if __name__ == "__main__":
    replay(os.path.join(ROOT, sys.argv[1]), sys.argv[2],
           sys.argv[3] if len(sys.argv) > 3 else None)
