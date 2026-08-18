# Rewrite the absolute `data-notebook="..."` paths Quarto leaves in the rendered
# HTML, for which there is no Quarto option. The attribute is inert once
# `notebook-links: false` is set, but a repo-relative path still says which
# notebook produced the cells below. Post-render, so it sees the final HTML.

import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent
REPO = DOCS.parent
SITE = DOCS / "_site"

# The whole attribute, so a rewrite cannot land inside prose containing a path.
ATTR = re.compile(r'(data-notebook=")([^"]+)(")')


def _relative(path):
    """Repo-relative form of an absolute build path, or None to leave it alone.
    The absolute test is load-bearing: without it a second run resolves the
    already-relative value against docs/ and rewrites `docs/notebooks/...` to
    `docs/docs/notebooks/...`, and rendering over an existing _site is that run.
    """
    candidate = pathlib.Path(path)
    if not candidate.is_absolute():
        return None
    try:
        return pathlib.PurePosixPath(candidate.resolve().relative_to(REPO)).as_posix()
    except ValueError:
        return None


def main():
    if not SITE.is_dir():
        sys.exit(f"{SITE} does not exist; run this after quarto render")

    rewritten = 0
    pages = 0
    for page in SITE.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        if "data-notebook=" not in text:
            continue

        hits = []

        def replace(match):
            relative = _relative(match.group(2))
            if relative is None or relative == match.group(2):
                return match.group(0)
            hits.append(relative)
            return match.group(1) + relative + match.group(3)

        updated = ATTR.sub(replace, text)
        if hits:
            page.write_text(updated, encoding="utf-8")
            rewritten += len(hits)
            pages += 1

    # Count the survivors too: an absolute path left here ships with the site.
    leaked = sum(
        page.read_text(encoding="utf-8", errors="ignore").count(str(REPO))
        for page in SITE.rglob("*.html"))
    print(f"[scrub-paths] {rewritten} notebook paths made relative "
          f"across {pages} pages, {leaked} absolute paths remaining")
    if leaked:
        sys.exit("[scrub-paths] build paths still present in the rendered site")


if __name__ == "__main__":
    main()
