# Licensing

Copyright (c) 2024-2026 José Marín.

| What | License | Text |
|---|---|---|
| The Blender add-on, everything under `SciGraphs/` | GPL-3.0-or-later | [`LICENSE`](LICENSE) |
| `scigraphs-core`, built from `core/` | MIT | [`core/LICENSE`](core/LICENSE) |
| `scigraphs-engine`, built from `engine/` | MIT | [`engine/LICENSE`](engine/LICENSE) |


## Why the two wheels are MIT

They are meant to be used without Blender. Someone computing betweenness on a
street network, or fitting a layout inside a program that renders with something
else, should not take on the GPL to do it. Both install from PyPI with numpy as
their only requirement and neither imports `bpy`.
GPL-3.0 permits incorporating MIT
code, so the add-on can bundle both wheels and stay GPL.


## Bundled dependencies

The add-on ships around sixty third-party wheels, listed in
`blender_manifest.toml`. They stay under their own licenses, all of which are
GPL-compatible: mostly BSD-3-Clause, MIT and Apache-2.0, with MPL-2.0 for
`certifi` and `tqdm`.

Two optional extras of `scigraphs-core` pull copyleft packages into the
environment they are installed in: `igraph` brings `python-igraph` (GPL), and
`sql` brings `mysql-connector-python` (GPL-2.0 with FOSS exception) alongside
`psycopg` and `pymssql` (LGPL). `cluster` brings `pysurprise` (GPL-3.0-or-later).
None of them is a hard dependency, and a default `pip install scigraphs-core`
pulls only numpy. Installing those extras is a choice about the environment you
assemble, and the obligations follow that choice rather than the wheel.
