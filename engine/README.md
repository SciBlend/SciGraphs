# scigraphs-engine

Graph visualization as a plain Python package. Coordinates, an edge list and
whatever attributes they carry go in; geometry or pixels come out, with 26
filter channels, structural sparsification, edge-style tessellation and
hierarchical bundling in between.

```python
from scigraphs_engine import Graph

g = Graph.from_arrays(coords, edges)
g = g.filter(core__gte=0.5).simplify(backbone="disparity", alpha=0.05)
g.style(edges="bundled").render(size=(1920, 1080)).save("out.png")
```

Nothing in this package imports `bpy`, `gpu`, `bmesh` or `mathutils`. The
compute half needs only `numpy`, and `Graph.geometry()` runs with no graphics
library installed at all, returning `MeshSpec`s that name their shader as a
string, so any renderer can draw them. The GPU half is an optional extra.

## Install

```bash
pip install scigraphs-engine                  # numpy only
pip install "scigraphs-engine[igraph]"        # coreness, PageRank, betweenness, bridges
pip install "scigraphs-engine[scipy]"         # components, hop distance, crowding
pip install "scigraphs-engine[wgpu]"          # render() to pixels
pip install "scigraphs-engine[all]"           # everything
```

Requires Python 3.10 or newer. A missing extra makes individual channels
unavailable rather than breaking anything: `available_channels()` returns all 26
with a reason for each one that cannot run.

## Documentation

https://github.com/SciBlend/SciGraphs/tree/main/docs

```bash
python engine/scigraphs_engine/tests/run.py          # -v for full output
```

## License

MIT, see `LICENSE`. The SciGraphs Blender add-on is GPLv3 and is one caller of
this package, not its owner. It reaches the package through a shim that prefers
an installed `scigraphs_engine` over any copy shipped alongside it.
