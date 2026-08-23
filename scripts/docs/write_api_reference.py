"""Regenerate docs/reference/api/*.qmd from the source of SciGraphs/api/.

    python3 scripts/docs/write_api_reference.py

Runs as a Quarto pre-render hook, so `quarto render docs` republishes the
reference against the current API. Reads with `ast`, never by importing: every
module under SciGraphs/api imports `bpy`, and `--check` proves this generator
does not by refusing `bpy` on sys.meta_path first."""

import argparse
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
API_DIR = ROOT / "SciGraphs" / "api"
OUT_DIR = ROOT / "docs" / "reference" / "api"

# The order the narrative page introduces them in: build a graph, look at it
# fast, render it, export it, thin it. Alphabetical would open on `context`,
# the one module unusable without a graph already.
MODULES = ("graphs", "preview", "render", "context", "thin")

# A banner comment: a rule, a title, a rule. Every module in the package groups
# its functions this way, so the headings are the author's own sections.
BANNER_RULE = re.compile(r"^#\s*-{10,}\s*$")
BANNER_TITLE = re.compile(r"^#\s+(?P<title>\S.*?)\s*$")

# Past this, a first paragraph is an explanation rather than a summary.
PARAGRAPH_LIMIT = 320

HEADER = """---
title: "{title}"
---

::: {{.lead}}
{lead}
:::

Auto-generated from `SciGraphs/api/{module}.py`. Regenerate it with
`python3 scripts/docs/write_api_reference.py`, which also runs on every
`quarto render docs`. Edits made here are overwritten.
"""


def _no_dashes(text):
    """Replace dashes: a comma for a parenthetical, `to` for a numeric range."""
    text = re.sub(r"(?<=\d)\s*[--]\s*(?=\d)", " to ", text)
    text = re.sub(r"\s*[--]\s*", ", ", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",(\s*[.;:])", r"\1", text)
    return text


def _one_line(text):
    """Collapse a wrapped paragraph onto one line."""
    return " ".join(text.split())


def _summary(node):
    """The docstring's first paragraph, or its first line past PARAGRAPH_LIMIT."""
    raw = ast.get_docstring(node, clean=True)
    if not raw:
        return ""
    paragraph = raw.split("\n\n", 1)[0].strip()
    text = _one_line(paragraph)
    if len(text) > PARAGRAPH_LIMIT:
        text = _one_line(paragraph.split("\n", 1)[0])
    return _no_dashes(text)


def _signature(source, node):
    """The `def` line as written, minus the trailing colon. Sliced from the
    source, not round-tripped through `ast.unparse`, so defaults keep their
    spelling and long signatures keep their line breaks."""
    start = source.index("def ", _offset(source, node.lineno, node.col_offset))
    end = _end_of_header(source, start)
    text = source[start:end]
    # A trailing comma is an artifact of the wrapping, not part of the interface.
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).rstrip().rstrip(",")


def _end_of_header(source, start):
    """Offset of the colon closing a `def` header, tracking bracket depth and
    quoting so a default containing either does not end the scan early."""
    depth = 0
    quote = ""
    index = start
    while index < len(source):
        char = source[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ":" and depth == 0:
            return index
        index += 1
    return len(source)


def _offset(source, lineno, col):
    """Byte offset of a (1-based line, 0-based column) position."""
    lines = source.splitlines(keepends=True)
    return sum(len(line) for line in lines[:lineno - 1]) + col


def _banners(source):
    """Map a line number to the section title in force at that line."""
    lines = source.splitlines()
    marks = []
    for i in range(len(lines) - 2):
        if not BANNER_RULE.match(lines[i]) or not BANNER_RULE.match(lines[i + 2]):
            continue
        match = BANNER_TITLE.match(lines[i + 1])
        if match:
            marks.append((i + 3, _no_dashes(match.group("title"))))
    return marks


def _section_for(marks, lineno):
    title = None
    for start, name in marks:
        if lineno >= start:
            title = name
    return title


def _module_lead(tree):
    """The module docstring's opening paragraph, as the page's lead."""
    doc = ast.get_docstring(tree, clean=True) or ""
    if not doc:
        return ""
    return _no_dashes(_one_line(doc.split("\n\n", 1)[0]))


def render_module(name, source):
    """The full .qmd text for one API module."""
    tree = ast.parse(source, filename=f"{name}.py")
    marks = _banners(source)

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]

    out = [HEADER.format(
        title=f"sg.{name}",
        lead=_module_lead(tree) or f"The `{name}` module of the scripting API.",
        module=name,
    )]

    current = object()
    for node in functions:
        section = _section_for(marks, node.lineno)
        if section != current:
            current = section
            if section:
                out.append(f"\n## {section}\n")
        out.append(f"\n### `{node.name}`\n")
        out.append(f"\n```python\n{_signature(source, node)}\n```\n")
        summary = _summary(node)
        if summary:
            out.append(f"\n{summary}\n")

    return "".join(out)


def _check_no_bpy():
    """Fail loudly if generating the pages needs Blender: the finder raises
    rather than returning None, so no later finder can satisfy the import."""
    blocked = ("bpy", "gpu", "bmesh", "mathutils", "SciGraphs", "scigraphs_engine")

    class Refuse:
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == n or fullname.startswith(n + ".") for n in blocked):
                raise ModuleNotFoundError(
                    f"{fullname} must not be imported by the docs build",
                    name=fullname)
            return None

    sys.meta_path.insert(0, Refuse())
    print("[api-reference] --check: %s are blocked for this run"
          % ", ".join(blocked))


def _shown(path):
    """A repo-relative path when there is one, else the path itself."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT_DIR),
                        help="directory to write the .qmd pages into")
    parser.add_argument("--check", action="store_true",
                        help="block bpy and SciGraphs on sys.meta_path first")
    args = parser.parse_args(argv)

    if args.check:
        _check_no_bpy()

    if not API_DIR.is_dir():
        # A docs-only checkout is legitimate; do not fail the render.
        print(f"[api-reference] no {API_DIR}, leaving the pages as they are",
              file=sys.stderr)
        return 0

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name in MODULES:
        path = API_DIR / f"{name}.py"
        if not path.is_file():
            print(f"[api-reference] {path.name} is gone, skipping",
                  file=sys.stderr)
            continue
        text = render_module(name, path.read_text(encoding="utf-8"))
        target = out_dir / f"{name}.qmd"
        target.write_text(text, encoding="utf-8")
        written.append(target)
        print("[api-reference] %s (%d functions)"
              % (_shown(target), text.count("\n### ")))

    # Or the sidebar keeps pointing at a reference for a deleted module.
    keep = {p.name for p in written}
    for stale in out_dir.glob("*.qmd"):
        if stale.name not in keep:
            stale.unlink()
            print(f"[api-reference] removed stale {stale.name}")

    if not written:
        print("[api-reference] wrote nothing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
