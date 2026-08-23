# Every runnable example in engine/README.md, run and compared to the output
# the README claims. A ```python block runs only if `<!-- run -->` precedes it
# and a ```text block with the expected stdout follows. `<!-- run:gpu -->` and
# `<!-- run:igraph -->` need an optional package and are reported as not run
# when it is missing. Each block runs in a subprocess with cwd=/tmp, so an
# example that only works inside the repository fails here.

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import Report, finish                         # noqa: E402

README = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "README.md"))

_FENCE = re.compile(r"^```(\w*)\s*$")
_MARKERS = {"<!-- run -->": "", "<!-- run:gpu -->": "gpu",
            "<!-- run:igraph -->": "igraph"}


def blocks(text):
    """[(language, code, line_number, marked)] for every fenced block."""
    out = []
    lines = text.splitlines()
    i = 0
    marked = None
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in _MARKERS:
            marked = _MARKERS[stripped]
            i += 1
            continue
        match = _FENCE.match(lines[i])
        if not match:
            if stripped:
                marked = None
            i += 1
            continue
        language = match.group(1) or "text"
        start = i + 1
        j = start
        while j < len(lines) and not _FENCE.match(lines[j]):
            j += 1
        out.append((language, "\n".join(lines[start:j]), start + 1, marked))
        marked = None
        i = j + 1
    return out


def examples(text):
    """[(code, expected, line, requires)] for blocks followed by a text block."""
    found = blocks(text)
    out = []
    for index, (language, code, line, marked) in enumerate(found):
        if language != "python" or marked is None:
            continue
        following = found[index + 1] if index + 1 < len(found) else None
        if following is None or following[0] != "text":
            continue
        out.append((code, following[1], line, marked))
    return out


def run(code):
    """(stdout, stderr, returncode) from a fresh interpreter, run in /tmp."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(code)
        path = handle.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, cwd=tempfile.gettempdir(), timeout=300)
        return proc.stdout, proc.stderr, proc.returncode
    finally:
        os.unlink(path)


def diff(got, want):
    """The first line that differs, or None."""
    got_lines = [line.rstrip() for line in got.strip("\n").splitlines()]
    want_lines = [line.rstrip() for line in want.strip("\n").splitlines()]
    for index in range(max(len(got_lines), len(want_lines))):
        a = got_lines[index] if index < len(got_lines) else "<missing>"
        b = want_lines[index] if index < len(want_lines) else "<extra>"
        if a != b:
            return f"line {index + 1}: got {a!r}, README says {b!r}"
    return None


def main():
    r = Report("test_readme", minimum=12)
    if not os.path.exists(README):
        r.check("README.md exists", False, README)
        finish(r)
    text = open(README, encoding="utf-8").read()
    found = blocks(text)
    runnable = examples(text)

    r.section(f"{README}")
    r.check("the README has fenced blocks", len(found) > 4, f"{len(found)} blocks")
    r.check("...of which several are marked runnable", len(runnable) >= 6,
            f"{len(runnable)} runnable examples")

    # A copyable snippet must at least compile.
    for language, code, line, _marked in found:
        if language != "python":
            continue
        try:
            compile(code, f"README.md:{line}", "exec")
            ok, detail = True, f"{len(code.splitlines())} lines"
        except SyntaxError as exc:
            ok, detail = False, f"{exc.msg} at line {exc.lineno}"
        r.check(f"block at README.md:{line} parses", ok, detail)

    from scigraphs_engine import filters, gpu_available
    has_gpu, why = gpu_available()
    have = {"": True, "gpu": has_gpu, "igraph": filters.IGRAPH_AVAILABLE}
    if not has_gpu:
        r.note(f"no GPU here ({why})")
    if not filters.IGRAPH_AVAILABLE:
        r.note("igraph is not installed here")

    for code, expected, line, requires in runnable:
        if not have[requires]:
            r.note(f"example at README.md:{line} NOT RUN -- needs {requires}")
            continue
        stdout, stderr, returncode = run(code)
        if returncode != 0:
            r.check(f"example at README.md:{line} runs", False,
                    (stderr.strip().splitlines() or ["exit "
                                                     + str(returncode)])[-1])
            continue
        r.check(f"example at README.md:{line} runs", True,
                f"{len(stdout.splitlines())} lines of output")
        mismatch = diff(stdout, expected)
        r.check(f"example at README.md:{line} prints what the README says",
                mismatch is None, mismatch or "identical")

    finish(r)


if __name__ == "__main__":
    main()
