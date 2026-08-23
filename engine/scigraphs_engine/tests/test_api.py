# The public surface: what it promises and what it refuses. Every semantic
# claim in the README has a check here, down to a typo raising in style().

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graphs                                              # noqa: E402
from harness import Report, finish                         # noqa: E402

import scigraphs_engine as sg                              # noqa: E402
from scigraphs_engine import (Camera, ChannelUnavailable, Graph,
                              UnknownChannel, channel, channel_names,
                              channel_specs)               # noqa: E402
from scigraphs_engine import filters                       # noqa: E402
from scigraphs_engine.source import ArraySource            # noqa: E402


def main():
    r = Report("test_api", minimum=90)

    coords, edges, node_attrs, edge_attrs = graphs.clustered()
    source = ArraySource(coords, edges, point_attrs=node_attrs,
                         edge_attrs=edge_attrs)
    g = Graph.from_source(source)

    r.section("1. the names a user types")
    for name in ("Graph", "Camera", "channel", "channel_specs",
                 "gpu_available"):
        r.check(f"scigraphs_engine.{name} exists", hasattr(sg, name))
    r.check("importing the package does not import wgpu",
            "wgpu" not in sys.modules,
            "wgpu absent from sys.modules after `import scigraphs_engine`")
    r.check("the package still exposes its modules",
            hasattr(sg, "filters") and hasattr(sg, "mesh"))

    r.section("2. all 26 channels, by public name")
    r.check("26 channels", len(channel_names()) == 26,
            f"{len(channel_names())}")
    ids = {spec.id for spec in channel_specs()}
    engine_ids = set(filters.NODE_CHANNELS) | set(filters.EDGE_CHANNELS)
    r.check("every engine channel id is covered exactly once",
            ids == engine_ids,
            f"missing {sorted(engine_ids - ids)}, extra {sorted(ids - engine_ids)}")
    r.check("EDGE_MINDEG is published as rich_club",
            channel("rich_club").name == "rich_club")
    r.check("BETWEEN is published as betweenness",
            channel("betweenness").name == "betweenness")
    r.check("the engine ids still resolve (a saved recipe must load)",
            channel("EDGE_MINDEG").name == "rich_club"
            and channel("BETWEEN").name == "betweenness")
    r.check("node/edge split matches the engine",
            len(channel_names("edge")) == len(filters.EDGE_CHANNELS),
            f"{len(channel_names('edge'))} edge channels")
    r.raises("an invented channel raises", UnknownChannel, channel, "eigenvector")

    r.section("3. measuring")
    computed = 0
    for name in channel_names():
        result = g.measure(name) if name not in ("attr", "edge_attr") else \
            g.measure(name, "community" if name == "attr" else "weight")
        r.check(f"measure({name!r}) returns a result", result is not None,
                "available" if result.available else result.reason)
        computed += int(result.available)
    # Predicted from the install: a bare count passes even if a channel stopped.
    expected_missing = {"hops"}                 # this fixture has no seed set
    if not filters.IGRAPH_AVAILABLE:
        expected_missing |= {"articulation", "betweenness", "bridge"}
        if not filters.SCIPY_AVAILABLE:
            expected_missing |= {"clustering", "pagerank"}
    if not filters.SCIPY_AVAILABLE:
        expected_missing |= {"crowding", "support", "overlap"}
    actual_missing = {n for n, v in g.available_channels().items() if v}
    r.check("exactly the channels this install cannot compute are unavailable",
            actual_missing == expected_missing,
            f"{computed} of 26 available; unexpected "
            f"{sorted(actual_missing ^ expected_missing)}")
    core = g.measure("core")
    r.check("normalized into [0, 1]",
            float(core.values.min()) >= 0.0 and float(core.values.max()) <= 1.0,
            f"{core.values.min():.3f}..{core.values.max():.3f}")
    r.check("native range is the graph's own units",
            core.native[1] > 1.5, f"coreness up to {core.native[1]:g}")
    r.check("from_native round-trips through to_native",
            abs(core.to_native(core.from_native(3.0)) - 3.0) < 1e-5,
            f"{core.to_native(core.from_native(3.0)):.4f}")
    r.check("Settings() never reaches the caller: the four channels that read "
            "coarsen_attr do not raise",
            all(g.measure(n) is not None
                for n in ("participation", "module_z", "thin", "span")))

    r.section("4. filtering is a conjunction of masks over the whole graph")
    a = g.filter(core__gte=0.5)
    b = a.filter(degree__gte=0.2)
    r.check("chaining narrows monotonically",
            int(b.node_mask.sum()) <= int(a.node_mask.sum()) <= g.num_nodes,
            f"{g.num_nodes} -> {int(a.node_mask.sum())} -> "
            f"{int(b.node_mask.sum())}")
    r.check("the original Graph is untouched",
            int(g.node_mask.sum()) == g.num_nodes, "still all nodes")
    r.check("degree is measured on the whole graph, not on the survivors",
            np.array_equal(b.measure("degree").values,
                           g.measure("degree").values),
            "identical arrays")
    r.check("the arrays are shared, not copied",
            a.source.coords() is g.source.coords()
            and b.source.coords() is g.source.coords(),
            "same object identity through two derivations")
    r.check("the channel cache is shared across derivations",
            a._base is g._base and b._base is g._base,
            f"{len(g._base._cache)} cached channels")
    n_cached = len(g._base._cache)
    g.filter(core__gte=0.9).node_mask
    r.check("a new threshold on a computed channel recomputes nothing",
            len(g._base._cache) == n_cached, f"{n_cached} entries, unchanged")

    r.check("nodes are indices into the array you passed in",
            b.nodes.max() < g.num_nodes and b.nodes.dtype == np.int32,
            f"max id {int(b.nodes.max())} of {g.num_nodes}")
    r.check("a node clause removes the edges that would dangle",
            bool(a.node_mask[g._base.edges[a.edge_mask][:, 0]].all()),
            f"{int(a.edge_mask.sum())} edges survive")

    r.section("5. filter syntax")
    r.check("kwargs and explicit conditions agree",
            np.array_equal(g.filter(core__gte=0.5).node_mask,
                           g.filter(channel("core") >= 0.5).node_mask))
    r.check("between()", int(g.filter(core__between=(0.2, 0.8))
                             .node_mask.sum()) < g.num_nodes)
    r.check("outside() is between() inverted",
            np.array_equal(g.filter(core__outside=(0.2, 0.8)).node_mask,
                           ~g.filter(core__between=(0.2, 0.8)).node_mask))
    r.check("a bare attribute name resolves to the attribute channel",
            int(g.filter(weight__gte=0.5).edge_mask.sum()) < g.num_edges,
            "weight is an EDGE attribute, not a channel")
    r.raises("a bare name that is neither raises", UnknownChannel,
             g.filter, nonsense__gte=0.5)
    r.raises("an unknown comparison raises", ValueError,
             g.filter, core__approximately=0.5)
    r.raises("> is refused rather than silently widened", TypeError,
             lambda: channel("core") > 0.5)
    r.raises("< is refused too", TypeError, lambda: channel("core") < 0.5)
    r.raises("a bare non-boolean value raises", ValueError,
             g.filter, core=0.5)
    # span, not bridge: bridge needs igraph, this runs numpy-only.
    r.check("a bare boolean works on a flag channel",
            0 < int(g.filter(span=True).edge_mask.sum()) < g.num_edges,
            f"span=True keeps {int(g.filter(span=True).edge_mask.sum())} "
            f"between-community edges of {g.num_edges}")
    r.check("...and its negation is the complement",
            int(g.filter(span=False).edge_mask.sum())
            + int(g.filter(span=True).edge_mask.sum()) == g.num_edges)
    r.check("the full range is exactly no clause",
            np.array_equal(g.filter(core__gte=0.0).node_mask, g.node_mask))

    r.section("6. an unavailable channel is visible, not silent")
    bare = Graph.from_arrays(coords, edges)          # no weight attribute
    result = bare.measure("disparity")
    r.check("declines rather than raising", not result.available)
    r.check("and says how to fix it", "weight=" in (result.reason or ""),
            result.reason)
    r.raises("filter() raises by default", ChannelUnavailable,
             lambda: bare.filter(disparity__gte=0.5).node_mask)
    skipped = bare.filter(disparity__gte=0.5, on_missing="skip")
    r.check("on_missing='skip' reproduces the add-on's behavior",
            int(skipped.edge_mask.sum()) == bare.num_edges,
            "nothing removed")
    r.check("...and the report says the clause was skipped",
            any("skipped" in w for w in skipped.report().warnings),
            skipped.report().warnings[0])
    emptied = bare.filter(disparity__gte=0.5, on_missing="empty")
    r.check("on_missing='empty' keeps nothing",
            int(emptied.edge_mask.sum()) == 0)
    listing = bare.available_channels()
    r.check("available_channels covers all 26", len(listing) == 26)
    r.check("...and gives a reason for each unavailable one",
            all(v for v in listing.values() if v is not None),
            f"{sum(v is not None for v in listing.values())} unavailable")

    r.section("7. simplify")
    disp = g.simplify(backbone="disparity", alpha=0.1)
    stats = disp.report().backbone
    r.check("the backbone reports its evidence",
            set(stats) == {"kept", "total", "weight_frac"}, str(stats))
    r.check("it kept a minority of edges carrying most of the weight",
            stats["kept"] < stats["total"] and stats["weight_frac"] > 0.5,
            f"{stats['kept']}/{stats['total']} edges, "
            f"{100 * stats['weight_frac']:.0f}% of weight")
    r.check("a disparity backbone is not a weight threshold",
            not np.array_equal(
                disp.edge_mask,
                g.filter(weight__gte=1.0 - stats["kept"] / stats["total"]
                         ).edge_mask),
            "different edge sets")
    r.raises("an invented backbone raises", ValueError, g.simplify, "pagerank")
    r.check("backbone=None clears one",
            int(disp.simplify(None).edge_mask.sum()) == g.num_edges)
    warned = Graph.from_arrays(coords, edges).simplify(backbone="disparity")
    r.check("running unweighted is warned about, not hidden",
            any("uniform weights" in w for w in warned.report().warnings),
            warned.report().warnings[0] if warned.report().warnings else "-")

    r.section("8. style")
    r.raises("an unknown style key raises", KeyError, g.style, node_raidus=1.0)
    r.raises("an unknown edge style raises", ValueError, g.style, edges="spiral")
    r.raises("an unknown node style raises", ValueError, g.style, nodes="cube")
    r.check("an edge-style parameter is accepted",
            g.style(edges="arc", curvature=0.9)._style["curvature"] == 0.9)
    r.check("style does not change what survives",
            np.array_equal(g.style(edges="arc").node_mask, g.node_mask))
    styles = {}
    for name in ("straight", "arc", "curved", "orthogonal", "bundled",
                 "tapered", "hierarchical"):
        geometry = g.style(edges=name, segments=5).geometry()
        styles[name] = geometry.segments
        r.check(f"style {name!r} tessellates", geometry.segments > 0
                and not geometry.validate(), f"{geometry.segments} segments")
    r.check("straight is one segment per edge, curved styles are more",
            styles["straight"] < styles["arc"],
            f"{styles['straight']} vs {styles['arc']}")

    r.section("9. compact() is the other semantics, explicitly")
    filtered = g.filter(core__gte=0.5)
    small = filtered.compact()
    r.check("compact keeps only the survivors",
            small.num_nodes == int(filtered.node_mask.sum()),
            f"{small.num_nodes} nodes")
    r.check("its edges are renumbered into the new space",
            small.num_edges == int(filtered.edge_mask.sum())
            and int(small._base.edges.max()) < small.num_nodes)
    r.check("degree now means degree among the survivors",
            not np.array_equal(
                small.measure("degree").values,
                filtered.measure("degree").values[filtered.node_mask]),
            "different arrays, which is the documented difference")
    r.check("attributes come across",
            "community" in small.source.scalar_names('POINT')
            and "weight" in small.source.scalar_names('EDGE'))

    r.section("10. camera")
    cam = Camera.fit(g)
    r.check("fit is deferred until it knows the frame", not cam.resolved)
    wide = cam.resolve((1920, 1080))
    tall = cam.resolve((1080, 1920))
    r.check("resolving gives concrete matrices", wide.resolved)
    r.check("...and the aspect ratio matters, which is why fit is deferred",
            not np.allclose(wide.proj, tall.proj),
            "16:9 and 9:16 projections differ")
    px = cam.project(coords, (800, 600))
    inside = ((px[:, 0] >= 0) & (px[:, 0] < 800)
              & (px[:, 1] >= 0) & (px[:, 1] < 600))
    r.check("every node projects inside the frame it was fitted to",
            bool(inside.all()), f"{int(inside.sum())} of {inside.size}")
    r.check("depth is in [0, 1], not OpenGL's [-1, 1]",
            float(px[:, 2].min()) >= 0.0 and float(px[:, 2].max()) <= 1.0,
            f"{px[:, 2].min():.3f}..{px[:, 2].max():.3f}")
    r.check("a filtered graph is framed on what survives",
            not np.allclose(
                Camera.fit(g.filter(pos_x__lte=0.3)).resolve((800, 600)).view,
                wide.view if False else
                Camera.fit(g).resolve((800, 600)).view),
            "different eye positions")

    r.section("11. geometry without a GPU")
    geometry = g.style(edges="arc").geometry()
    r.check("specs validate", geometry.validate() == [], "no problems")
    r.check("specs name shaders as strings",
            all(isinstance(s.shader.base, str) for s in geometry.specs),
            str([s.shader.base for s in geometry.specs]))
    r.check("the report survives into the geometry",
            geometry.report.nodes_out == geometry.nodes.size)
    r.check("geometry() needs no graphics library",
            "wgpu" not in sys.modules, "wgpu still not imported")

    r.section("12. an empty graph is a picture of nothing, not a crash")
    empty = Graph.from_arrays(np.zeros((0, 3), np.float32))
    r.check("no nodes, no edges", empty.num_nodes == 0 and empty.num_edges == 0)
    eg = empty.geometry()
    r.check("geometry is empty rather than raising", len(eg.specs) == 0)
    r.check("its report is honest", eg.report.nodes_out == 0)
    lonely = Graph.from_arrays(np.zeros((3, 3), np.float32))
    r.check("nodes with no edges still draw",
            len(lonely.geometry().specs) == 1,
            "one sphere spec, no ribbon")
    r.check("...and the edge channels decline rather than raise",
            lonely.measure("length").available is False,
            lonely.measure("length").reason)

    finish(r)


if __name__ == "__main__":
    main()
