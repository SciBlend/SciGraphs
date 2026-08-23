# The package with nothing but numpy, and with each extra removed in turn.
# Extras go away by refusing their imports, not by patching flags, so this hits
# the same ImportError path a real install without them takes. Run directly it
# forks one subprocess per configuration; `import igraph` cannot be undone.

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGURATIONS = ("none", "igraph", "scipy", "all")


class _Block:
    """A meta_path finder that refuses named top-level packages."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ImportError(f"blocked by test_degraded: {name}")
        return None


def child(which):
    """One configuration, in this process. Called by the subprocess."""
    blocked = {"none": ("igraph", "scipy", "networkx"),
               "igraph": ("igraph",),
               "scipy": ("scipy",),
               "all": ()}[which]
    if blocked:
        sys.meta_path.insert(0, _Block(blocked))

    sys.path.insert(0, HERE)
    import numpy as np
    import graphs
    from harness import Report, finish

    from scigraphs_engine import (Camera, ChannelUnavailable, Graph,
                                  channel_names)
    from scigraphs_engine import filters
    from scigraphs_engine.source import ArraySource

    r = Report(f"test_degraded[{which}]", minimum=20)
    r.section(f"blocked: {list(blocked) or 'nothing'}")
    # Blocked implies unavailable, not the converse: it may not be installed.
    r.check("blocking igraph makes it unavailable",
            not ("igraph" in blocked and filters.IGRAPH_AVAILABLE),
            f"IGRAPH_AVAILABLE={filters.IGRAPH_AVAILABLE}")
    r.check("blocking scipy makes it unavailable",
            not ("scipy" in blocked and filters.SCIPY_AVAILABLE),
            f"SCIPY_AVAILABLE={filters.SCIPY_AVAILABLE}")
    r.note(f"effective: igraph={filters.IGRAPH_AVAILABLE}, "
           f"scipy={filters.SCIPY_AVAILABLE}")

    coords, edges, node_attrs, edge_attrs = graphs.clustered(per_community=30)
    source = ArraySource(coords, edges, point_attrs=node_attrs,
                         edge_attrs=edge_attrs)
    g = Graph.from_source(source)

    listing = g.available_channels()
    r.check("all 26 channels answered", len(listing) == 26)
    unavailable = {k: v for k, v in listing.items() if v}
    r.check("every unavailable channel gave a reason",
            all(unavailable.values()), f"{len(unavailable)} unavailable")
    r.check("no reason is an empty string or a bare 'None'",
            all(len(v) > 12 and "None" != v for v in unavailable.values()),
            "; ".join(sorted(unavailable)) or "-")
    for name in sorted(unavailable):
        r.note(f"{name}: {unavailable[name]}")

    # The four the README singles out: they decline, never raise.
    for name in ("participation", "module_z", "thin", "span"):
        r.check(f"{name} declines or answers, never raises",
                g.measure(name) is not None,
                "available" if listing[name] is None else listing[name])

    # Coreness has a numpy peeler, so a failure here breaks the README claim.
    core = g.measure("core")
    r.check("core is available with no extras at all", core.available,
            core.reason or f"native 0..{core.native[1]:g}")
    r.check("component is available with no extras at all",
            g.measure("component").available)

    narrowed = g.filter(core__gte=0.3).simplify(backbone="topk", k=2) \
        .style(edges="arc", color_by="core")
    geometry = narrowed.geometry()
    r.check("the pipeline still produces geometry", len(geometry.specs) == 2,
            f"{[s.shader.base for s in geometry.specs]}")
    r.check("...which validates", geometry.validate() == [])
    r.check("...and a camera can frame it",
            Camera.fit(narrowed).resolve((640, 360)).resolved)

    if not filters.IGRAPH_AVAILABLE:
        raised = False
        try:
            g.filter(betweenness__gte=0.5).node_mask
        except ChannelUnavailable as exc:
            raised = "igraph" in exc.reason
        r.check("filtering on an igraph-only channel raises and names igraph",
                raised, "ChannelUnavailable mentioning the extra")
        skipped = g.filter(betweenness__gte=0.5, on_missing="skip")
        r.check("on_missing='skip' leaves everything and says so",
                int(skipped.node_mask.sum()) == g.num_nodes
                and any("skipped" in w for w in skipped.report().warnings))
    else:
        r.check("betweenness is available here", g.measure("betweenness").available)
        r.check("...and filtering on it narrows",
                int(g.filter(betweenness__gte=0.5).node_mask.sum()) <= g.num_nodes)

    hierarchy = g.hierarchy()
    r.check("a community tree is built without any GPLv3 detector",
            len(hierarchy) >= 1, f"{len(hierarchy)} level(s)")
    bundled = g.style(edges="hierarchical").geometry()
    r.check("hierarchical bundling produces curves", bundled.segments > 0,
            f"{bundled.segments} segments")

    # Level 1 comes from the fixture's `community` attribute, and a three-
    # community graph never reaches level 2, so detect() would never run here.
    from scigraphs_engine import communities
    bare = Graph.from_arrays(coords, edges)       # no community attribute
    r.check("the fixture's own communities are not visible to `bare`",
            bare.attributes["groups"] == "", f"groups={bare.attributes['groups']!r}")
    labels = communities.detect(edges, coords.shape[0], None)
    r.check(f"the shipped detector ({communities.name()}) partitions the graph",
            labels is not None and 2 <= np.unique(labels).size <= 12,
            f"{0 if labels is None else np.unique(labels).size} communities "
            f"from {coords.shape[0]} nodes")
    r.check("...deterministically",
            np.array_equal(labels,
                           communities.detect(edges, coords.shape[0], None)),
            "two calls, identical labels")
    # Half, not "almost all": Louvain recovers the planted communities nearly
    # exactly, but the numpy agglomeration splits each into two or three.
    dominant = max(int((labels[:30] == v).sum()) for v in np.unique(labels[:30]))
    r.check("...and it groups most of a planted community together",
            dominant >= 15, f"largest label covers {dominant} of 30")
    tree = bare.hierarchy()
    r.check("hierarchy() falls back to it when no attribute names the groups",
            len(tree) >= 1, f"{len(tree)} level(s)")
    r.check("...and bundling then works on a graph with no attributes at all",
            bare.style(edges="hierarchical").geometry().segments > 0)
    finish(r)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in CONFIGURATIONS:
        child(sys.argv[1])
        return 0
    print(f"test_degraded -- {len(CONFIGURATIONS)} dependency configurations")
    passed = re.compile(r"^\S+: (\d+) checks passed$", re.M)
    failed = re.compile(r"^\S+: (\d+) of (\d+) FAILED$", re.M)
    toofew = re.compile(r"^\S+: ONLY (\d+) CHECKS RAN", re.M)
    total = failures = 0
    for which in CONFIGURATIONS:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), which],
                              capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        print(out)
        match = passed.search(out)
        if match:
            total += int(match.group(1))
        else:
            match = failed.search(out) or toofew.search(out)
            total += int(match.group(len(match.groups()))) if match else 0
        if proc.returncode != 0:
            failures += 1
    if failures:
        print(f"\ntest_degraded: {failures} of {len(CONFIGURATIONS)} "
              f"configurations FAILED over {total} checks")
        return 1
    print(f"\ntest_degraded: {total} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
