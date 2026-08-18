# scigraphs-core

Graph algorithms, layouts, colormaps, OSM/geo helpers, and pipeline schema —
the SciGraphs bits that never touch Blender.

```bash
pip install ./core
```

Needs only numpy. Optional extras (`networkx`, `geo`, `sql`, …) are in
`pyproject.toml`. Import as `scigraphs_core`.

The add-on uses this package; this package never imports the add-on. Mesh
building, scene I/O, and rendering stay in the add-on / `scigraphs-engine`.

MIT. See `LICENSES.md` at the repo root.
