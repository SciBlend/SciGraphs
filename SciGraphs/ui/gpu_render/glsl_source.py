# GLSL kept in files under glsl/, not in Python string literals.
#
# Blender's `gpu` module takes shader source as a string, never a path, so the
# text has to reach Python one way or another. Reading it from a file rather
# than concatenating adjacent literals buys the one thing that style cost:
# line numbers. A shader pasted together as `"void main()" "{" "  ..."` arrives
# at the compiler on a single line, and every error it reports is "line 1" with
# no source echoed. With real newlines the driver names the line and prints it.
#
# Files are read once and cached. The add-on ships them because
# build_extension.sh copies the whole tree, deleting only *.md.

import os

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glsl")

_CACHE = {}


def glsl(name):
    """Return the contents of ``glsl/<name>.glsl``.

    Raises FileNotFoundError rather than returning a placeholder: a shader that
    silently compiles to nothing draws an empty viewport, which is a slower way
    to learn the file is missing.
    """
    if name not in _CACHE:
        with open(os.path.join(_DIR, name + ".glsl"), encoding="utf-8") as fh:
            _CACHE[name] = fh.read()
    return _CACHE[name]
