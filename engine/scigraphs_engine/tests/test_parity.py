# Does the API express what the add-on's pipeline does?
# `SciGraphs/ui/gpu_render/batches.py::build_bundle` is the only end-to-end
# composition of these modules, so it is the spec. `reference()` transcribes it,
# and every array is compared elementwise: the failure this catches is a
# matching node count over permuted coordinates. Four configurations, one per
# build_bundle branch. Not checked, because the API does not claim it: spatial
# blocks, the adaptive cut, the GPU filter predicate, the ID pass, the volume
# field, the taper settings, the animated path.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graphs                                              # noqa: E402
from harness import Report, finish                         # noqa: E402

from scigraphs_engine import Graph, channel                # noqa: E402
from scigraphs_engine import (channels as _ch, edge_styles, filters,  # noqa: E402
                              mesh, simplify)
from scigraphs_engine.settings import Clause, Settings     # noqa: E402
from scigraphs_engine.source import ArraySource            # noqa: E402
from scigraphs_engine import palette                       # noqa: E402


def reference(source, *, clauses=(), backbone=None, style_params=None,
              base_radius, edge_radius_scale=0.45, weight_attr="",
              group_attr="", color_values=None, size_values=None,
              size_max_mult=4.0, edge_width_attr="", edge_width_max_mult=4.0,
              node_color=(0.3, 0.7, 1.0, 1.0),
              edge_color=(0.5, 0.5, 0.55, 0.35)):
    """build_bundle, transcribed. Calls the modules directly, never the API."""
    settings = Settings({"coarsen_attr": group_attr})
    coords = source.coords()
    edges = source.edges()
    node_mask = source.node_mask()
    num_verts = int(source.num_vertices)
    num_edges = 0 if edges is None else int(edges.shape[0])

    # --- filters.masks -----------------------------------------------------
    visible = np.ones(num_verts, dtype=bool)
    stack_edges = np.ones(num_edges, dtype=bool)
    for slot in clauses:
        values, _lo, _hi, domain = filters.channel_values(
            source, settings, slot.channel, slot.attr_name,
            group_attr=group_attr, weight_attr=weight_attr)
        if values is None:
            continue
        keep = filters.slot_mask(values, slot)
        if domain == 'EDGE':
            if keep.size == num_edges:
                stack_edges &= keep
        elif keep.size == num_verts:
            visible &= keep
    if num_edges and not visible.all():
        stack_edges &= visible[edges[:, 0]] & visible[edges[:, 1]]

    # --- the drawn node set ------------------------------------------------
    node_visible = visible if node_mask is None else (visible & node_mask)
    point_idx = np.nonzero(node_visible)[0]

    if color_values is not None:
        colors = palette.apply(color_values, "viridis")
    else:
        colors = palette.solid(node_color, num_verts)
    if size_values is not None:
        radii = (base_radius * (1.0 + size_values[point_idx]
                                * (size_max_mult - 1.0))).astype(np.float32)
    else:
        radii = np.full(point_idx.size, base_radius, dtype=np.float32)

    bb_mask = bb_stats = None
    weights = source.edge_scalar(weight_attr) if weight_attr else None
    if backbone is not None and num_edges:
        bb_mask, bb_stats = simplify.backbone_mask(
            edges, weights, backbone["mode"], backbone.get("k", 3),
            backbone.get("alpha", 0.05), backbone.get("sample", 0.25))

    # --- logical edges + the styled geometry -------------------------------
    styled = None
    logical = np.empty((0, 2), dtype=np.int32)
    if num_edges:
        recovered = edge_styles.recover_logical_edges(edges, node_mask)
        logical, mesh_ids = recovered
        if bb_mask is not None:
            m = bb_mask[mesh_ids]
            logical, mesh_ids = logical[m], mesh_ids[m]
        lkeep = node_visible[logical[:, 0]] & node_visible[logical[:, 1]]
        lkeep &= stack_edges[mesh_ids]
        logical, mesh_ids = logical[lkeep], mesh_ids[lkeep]

        edge_widths = None
        if edge_width_attr:
            raw = source.edge_scalar(edge_width_attr)
            if raw is not None:
                unit = _ch.scale_to_unit(raw, 'LOG')
                edge_widths = (1.0 + unit[mesh_ids]
                               * (edge_width_max_mult - 1.0)).astype(np.float32)
        if logical.shape[0]:
            styled = edge_styles.tessellate(
                coords, logical.astype(np.int64),
                edge_styles.default_style_params(**(style_params or {})),
                edge_widths=edge_widths, hierarchy=None, node_colors=colors,
                edge_color=np.asarray(edge_color, dtype=np.float32))

    sphere = mesh.sphere_spec(coords[point_idx], colors[point_idx], radii) \
        if point_idx.size else None
    ribbon = None
    if styled is not None:
        edge_radius = base_radius * edge_radius_scale
        ribbon = mesh.ribbon_spec(
            styled["seg_a"], styled["seg_b"], edge_color,
            (styled["scale_a"] * edge_radius).astype(np.float32),
            (styled["scale_b"] * edge_radius).astype(np.float32),
            styled.get("col_a"), styled.get("col_b"))
    return {"point_idx": point_idx, "colors": colors, "radii": radii,
            "node_coords": coords[point_idx],
            "logical": logical, "styled": styled, "sphere": sphere,
            "ribbon": ribbon, "backbone": bb_stats,
            "node_visible": node_visible, "stack_edges": stack_edges}


def compare(report, label, want, geometry, check_specs=True):
    """Compare every array elementwise, not by counts."""
    report.equal(f"{label}: drawn node ids", geometry.nodes, want["point_idx"])
    report.equal(f"{label}: node coordinates", geometry.node_coords,
                 want["node_coords"])
    report.equal(f"{label}: node radii", geometry.node_radii, want["radii"])
    report.equal(f"{label}: node colors", geometry.node_colors,
                 want["colors"][want["point_idx"]])
    report.equal(f"{label}: logical edges", geometry.edges, want["logical"])

    if want["styled"] is None:
        report.check(f"{label}: no segments either side", geometry.segments == 0,
                     f"api {geometry.segments}")
    else:
        report.equal(f"{label}: segment count",
                     np.array(geometry.segments),
                     np.array(want["styled"]["seg_a"].shape[0]))

    if not check_specs:
        return
    by_shader = {s.shader.base: s for s in geometry.specs}
    for name in ("sphere", "ribbon"):
        want_spec = want[name]
        got_spec = by_shader.get(name)
        if want_spec is None:
            report.check(f"{label}: no {name} spec either side",
                         got_spec is None, f"api {got_spec is not None}")
            continue
        if got_spec is None:
            report.check(f"{label}: {name} spec present", False,
                         f"api produced {sorted(by_shader)}")
            continue
        report.check(f"{label}: {name} attribute names",
                     set(got_spec.attrs) == set(want_spec.attrs),
                     f"{sorted(got_spec.attrs)}")
        for attr in sorted(want_spec.attrs):
            if attr in got_spec.attrs:
                report.equal(f"{label}: {name}.{attr}", got_spec.attrs[attr],
                             want_spec.attrs[attr])
        report.equal(f"{label}: {name} indices",
                     got_spec.indices if got_spec.indices is not None
                     else np.zeros(0),
                     want_spec.indices if want_spec.indices is not None
                     else np.zeros(0))


def main():
    r = Report("test_parity", minimum=60)

    r.section("1. plain graph, straight edges, no attributes")
    coords, edges = graphs.plain()
    source = ArraySource(coords, edges)
    g = Graph.from_source(source)
    radius = g.node_radius()
    want = reference(source, base_radius=radius)
    compare(r, "plain", want, g.geometry())
    r.check("plain: radius is 1.2% of the layout diagonal",
            abs(radius - float(np.linalg.norm(
                coords.max(axis=0) - coords.min(axis=0))) * 0.012) < 1e-6,
            f"{radius:.5g}")

    r.section("2. weighted + clustered: disparity backbone, arc edges, "
              "color by coreness, size by degree, width by weight")
    coords, edges, node_attrs, edge_attrs = graphs.clustered()
    source = ArraySource(coords, edges, point_attrs=node_attrs,
                         edge_attrs=edge_attrs)
    g = (Graph.from_source(source)
         .simplify(backbone="disparity", alpha=0.2)
         .style(edges="arc", curvature=0.45, segments=6, color_by="core",
                size_by="degree", edge_width_by="weight"))
    r.check("2: weight attribute guessed", g.attributes["weight"] == "weight",
            repr(g.attributes["weight"]))
    r.check("2: group attribute guessed", g.attributes["groups"] == "community",
            repr(g.attributes["groups"]))
    core = filters.channel_values(source, Settings({"coarsen_attr": "community"}),
                                  'CORE', "", group_attr="community",
                                  weight_attr="weight")[0]
    degree = filters.channel_values(source, Settings({"coarsen_attr": "community"}),
                                    'DEGREE', "", group_attr="community",
                                    weight_attr="weight")[0]
    want = reference(
        source, backbone={"mode": 'DISPARITY', "alpha": 0.2},
        style_params={"style_type": 'ARC', "curvature": 0.45, "segments": 6},
        base_radius=g.node_radius(), weight_attr="weight",
        group_attr="community", color_values=core, size_values=degree,
        edge_width_attr="weight")
    geometry = g.geometry()
    compare(r, "2", want, geometry)
    r.check("2: backbone stats identical",
            geometry.report.backbone == want["backbone"],
            f"{geometry.report.backbone}")
    r.check("2: the backbone actually removed edges",
            0 < want["backbone"]["kept"] < want["backbone"]["total"],
            f"{want['backbone']['kept']} of {want['backbone']['total']}")
    r.check("2: arcs produced more segments than edges",
            geometry.segments == geometry.edges.shape[0] * 6,
            f"{geometry.segments} segments, {geometry.edges.shape[0]} edges")

    r.section("3. baked polyline mesh: recover_logical_edges")
    bcoords, medges, node_mask, logical_truth = graphs.baked()
    source = ArraySource(bcoords, medges, node_mask=node_mask)
    g = Graph.from_source(source).style(edges="straight")
    want = reference(source, base_radius=g.node_radius())
    geometry = g.geometry()
    compare(r, "baked", want, geometry)
    r.check("baked: curve points are never drawn as nodes",
            geometry.nodes.max() < logical_truth.max() + 1
            and bool(node_mask[geometry.nodes].all()),
            f"{geometry.nodes.size} of {bcoords.shape[0]} vertices")
    r.equal("baked: the recovered edges are the original ones",
            np.sort(np.sort(geometry.edges, axis=1), axis=0),
            np.sort(np.sort(logical_truth, axis=1), axis=0))
    r.check("baked: mesh has twice as many segments as logical edges",
            medges.shape[0] == logical_truth.shape[0] * 2,
            f"{medges.shape[0]} vs {logical_truth.shape[0]}")

    r.section("4. two-domain conjunction: a node clause and an edge clause")
    coords, edges, node_attrs, edge_attrs = graphs.clustered()
    source = ArraySource(coords, edges, point_attrs=node_attrs,
                         edge_attrs=edge_attrs)
    g = (Graph.from_source(source)
         .filter(core__gte=0.35)
         .filter(channel("edge_attr", "weight") >= 0.15))
    clauses = (Clause(channel='CORE', range_min=0.35),
               Clause(channel='EDGE_ATTR', attr_name="weight", range_min=0.15))
    want = reference(source, clauses=clauses, base_radius=g.node_radius(),
                     weight_attr="weight", group_attr="community")
    geometry = g.geometry()
    compare(r, "conj", want, geometry)
    r.equal("conj: node mask", g.node_mask, want["node_visible"])
    r.equal("conj: edge mask", g.edge_mask, want["stack_edges"])
    r.check("conj: both clauses removed something",
            int(g.node_mask.sum()) < coords.shape[0]
            and int(g.edge_mask.sum()) < edges.shape[0],
            f"{int(g.node_mask.sum())}/{coords.shape[0]} nodes, "
            f"{int(g.edge_mask.sum())}/{edges.shape[0]} edges")
    # Only visible with both domains live: the dangling-edge pass runs after
    # the node clauses are ANDed together, not once per clause.
    dangling = geometry.edges
    r.check("conj: no drawn edge ends on a filtered-out node",
            bool(g.node_mask[dangling[:, 0]].all()
                 and g.node_mask[dangling[:, 1]].all()),
            f"{dangling.shape[0]} edges checked")

    r.section("5. the clause list is the engine's own serializable form")
    recipe = g.recipe()
    r.check("recipe names the engine's channel ids, not the API's",
            [c["channel"] for c in recipe["clauses"]] == ['CORE', 'EDGE_ATTR'],
            str([c["channel"] for c in recipe["clauses"]]))
    replayed = Graph.from_source(source).replay(recipe)
    r.equal("replayed node mask", replayed.node_mask, g.node_mask)
    r.equal("replayed edge mask", replayed.edge_mask, g.edge_mask)

    finish(r)


if __name__ == "__main__":
    main()
