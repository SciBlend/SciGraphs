# Numpy node/edge arrays become GPUBatch objects, one bundle per graph object,
# cached by content signature so camera moves never rebuild. A bundle holds
# several representations and the draw loop picks one per frame.

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import host
from ...core.render import mesh as mesh_spec
from . import (bundle_gpu, dynamic, edge_styles_gpu, filter_gpu, filters,
               geometry, shaders, simplify, volume)
from .attributes import (
    active_point_color_attribute,
    read_edge_scalar,
    compute_colors,
    normalized_values,
    read_edge_value_channel,
    read_value_channel,
)
from . import state
from .state import SIZE_BUCKETS

_QUAD_CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
_QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

# Past this, no whole-object sphere impostor (4x memory): AUTO falls back.
_IMPOSTOR_MAX = 500_000


def _settings(scene, st=None):
    """Building one costs 99.5 us, then each read is 1.78 us instead of 5.8."""
    return st if st is not None else host.settings_from_scene(scene)


def structure_signature(obj, scene, st=None):
    """Fingerprint of what the community tree depends on, split off so a filter
    threshold does not re-run detection (seconds at 213k edges).
    ``drop_tree_cache`` matches on ``obj.name``, so it comes first."""
    mesh = obj.data
    color_attr = active_point_color_attribute(mesh)
    st = _settings(scene, st)
    return (
        obj.name,
        len(mesh.vertices),
        len(mesh.edges),
        "is_intersection" in mesh.attributes,
        color_attr.name if color_attr else "",
        st.color_mode,
        st.attr_name,
        st.colormap,
        bool(st.reverse_colormap),
        tuple(st.node_color),
        bool(st.size_by_attr),
        round(float(st.size_max_mult), 3),
        bool(st.round_points),
        st.node_style,
        st.edge_style,
        round(float(st.impostor_radius), 5),
        bool(st.use_blocks),
        int(st.block_count),
        bool(st.render_id_pass),
        int(obj.get("scigraphs_preview_epoch", 0)),
        bool(st.heb_vertex_shader),
        round(float(st.edge_radius_scale), 4),
        st.edge_arrows,
        round(float(st.edge_arrow_size), 3),
        st.edge_arrow_style,
        st.edge_size_scale,
        st.edge_taper,
        round(float(st.edge_taper_amount), 3),
        bool(obj.get("is_directed", False)) or bool(obj.get("od_directed", False)),
        bool(st.edge_size_by_attr),
        st.edge_attr_name,
        round(float(st.edge_size_max_mult), 3),
        st.backbone_mode,
        int(st.backbone_k),
        round(float(st.backbone_alpha), 5),
        round(float(st.backbone_sample), 5),
        st.backbone_attr,
        bool(st.coarsen),
        st.coarsen_attr,
        st.coarsen_size_mode,
        st.coarsen_size_attr,
        st.coarsen_size_agg,
        round(float(st.coarsen_size_mult), 3),
        bool(st.adaptive),
        st.adaptive_algorithm,
        st.volume_mode,
        st.volume_when,
        int(st.volume_res),
        round(float(st.volume_smooth), 3),
        round(float(st.volume_threshold), 4),
        round(float(st.volume_falloff), 3),
        round(float(st.volume_hybrid_pct), 2),
    )


def selection_signature(obj, scene, st=None):
    """Fingerprint of which subset is drawn. None of it changes the tree, and
    the vertex-shader path drops the thresholds but keeps the channel names."""
    st = _settings(scene, st)
    if filter_gpu.eligible(obj, scene, st=st)[0]:
        stack = filter_gpu.channel_signature(scene)
    else:
        stack = (
            bool(st.filter_enabled),
            round(float(st.filter_min), 4),
            round(float(st.filter_max), 4),
            filters.slot_signature(scene),
        )
    return (
        stack,
        bool(st.lod_enabled),
        int(st.lod_max_points),
        bool(st.animate),
    )


def content_signature(obj, scene, st=None):
    """Return everything that changes the batches except the live edge style."""
    st = _settings(scene, st)
    return (structure_signature(obj, scene, st)
            + selection_signature(obj, scene, st))


# Last in the signature, so a change to it alone is identifiable: on the
# vertex-shader path the shader reads it every frame.
BETA_PARAM = "heb_beta"


def object_signature(obj, scene, st=None):
    """CPU-baked content as ``(...content..., style minus strength, strength)``."""
    st = _settings(scene, st)
    if not edge_styles_gpu.enabled(scene, st):
        return content_signature(obj, scene, st) + (None, None)
    return content_signature(obj, scene, st) + (
        edge_styles_gpu.params_signature(scene, without=(BETA_PARAM,)),
        round(float(scene.scigraphs.edge_heb_beta), 5),
    )


# The only place that maps a name to a program; a second backend swaps it.
_SHADER_GETTERS = {
    "sphere": lambda ref: shaders.get_sphere_shader(
        filtered=ref.filtered, animated=ref.animated),
    "ribbon": lambda ref: shaders.get_ribbon_shader(
        filtered=ref.filtered, animated=ref.animated),
    "sphere_id": lambda ref: shaders.get_sphere_id_shader(),
    "ribbon_id": lambda ref: shaders.get_ribbon_id_shader(),
    "uniform_color": lambda ref: gpu.shader.from_builtin('UNIFORM_COLOR'),
    "polyline_uniform_color": lambda ref: gpu.shader.from_builtin(
        'POLYLINE_UNIFORM_COLOR'),
    "smooth_color": lambda ref: gpu.shader.from_builtin('SMOOTH_COLOR'),
    "polyline_smooth_color": lambda ref: gpu.shader.from_builtin(
        'POLYLINE_SMOOTH_COLOR'),
    "arrow": lambda ref: shaders.get_arrow_shader(),
    "arrow_flat": lambda ref: shaders.get_arrow_flat_shader(
        animated=ref.animated),
    "line": lambda ref: (shaders.get_line_filter_shader() if ref.filtered
                         else gpu.shader.from_builtin('UNIFORM_COLOR')),
}


def resolve_shader(ref):
    """Compiled program, or None: callers fall back, so a gap is not a crash."""
    getter = _SHADER_GETTERS.get(ref.base)
    return getter(ref) if getter is not None else None


def upload_groups(spec):
    """Grouped MeshSpec to ``([(batch, weight), ...], shader)``. Positions fill
    once, one ``GPUIndexBuf`` per group; ``batch_for_shader`` re-uploads."""
    if spec is None or not spec.groups:
        return None, None
    shader = resolve_shader(spec.shader)
    if shader is None:
        return None, None

    fmt = gpu.types.GPUVertFormat()
    for name, arr in spec.attrs.items():
        fmt.attr_add(id=name, comp_type='F32',
                     len=int(arr.shape[1]) if arr.ndim > 1 else 1,
                     fetch_mode='FLOAT')
    first = next(iter(spec.attrs.values()))
    vbo = gpu.types.GPUVertBuf(len=int(first.shape[0]), format=fmt)
    for name, arr in spec.attrs.items():
        vbo.attr_fill(name, np.ascontiguousarray(arr, dtype=np.float32))

    arity = {'POINTS': 1, 'LINES': 2, 'TRIS': 3}[spec.topology]
    out = []
    for group in spec.groups:
        seq = (np.arange(group.start, group.start + group.count,
                         dtype=np.int32).reshape(-1, arity)
               if group.is_range else group.indices)
        ibo = gpu.types.GPUIndexBuf(type=spec.topology, seq=seq)
        out.append((gpu.types.GPUBatch(type=spec.topology, buf=vbo, elem=ibo),
                    group.weight))
    return (out or None), shader


def upload_with(spec, shader):
    if spec is None or shader is None:
        return None
    return batch_for_shader(shader, spec.topology, dict(spec.attrs),
                            indices=spec.indices)


def upload(spec):
    if spec is None:
        return None, None
    shader = resolve_shader(spec.shader)
    if shader is None:
        return None, None
    return batch_for_shader(shader, spec.topology, dict(spec.attrs),
                            indices=spec.indices), shader


def _radii_for(norm_sub, base_radius, size_by_attr, size_max_mult):
    if size_by_attr and norm_sub is not None:
        scale = 1.0 + norm_sub * (size_max_mult - 1.0)
        return (base_radius * scale).astype(np.float32)
    return np.full(norm_sub.shape[0] if norm_sub is not None else 0,
                   base_radius, dtype=np.float32)


def build_point_batches(shader, coords_sub, colors_sub, norm_sub, size_by_attr,
                        frows=None):
    """One batch per size bucket. ``frows`` comes with a filtered ``shader``."""
    return [(upload_with(spec, shader), weight)
            for spec, weight in mesh_spec.point_specs(
                coords_sub, colors_sub, norm_sub, size_by_attr, frows,
                buckets=SIZE_BUCKETS)]


def build_sphere_batch(coords_sub, colors_sub, radii_sub, frows=None):
    """4 vertices + 2 triangles per node; an unavailable shader skips it."""
    if resolve_shader(mesh_spec.ShaderRef(
            "sphere", filtered=frows is not None)) is None:
        return None
    batch, _ = upload(mesh_spec.sphere_spec(
        coords_sub, colors_sub, radii_sub, frows=frows))
    return batch


# Bit 23 of the 24-bit id marks an edge segment, so nodes and edges share one
# ID pass and one depth test. 23 bits (8.3M) each is far above the cap.
EDGE_ID_FLAG = 0x800000


def encode_ids_to_rgba(ids):
    ids = np.asarray(ids, dtype=np.uint32)
    r = (ids & 0xFF).astype(np.float32) / 255.0
    g = ((ids >> 8) & 0xFF).astype(np.float32) / 255.0
    b = ((ids >> 16) & 0xFF).astype(np.float32) / 255.0
    a = np.ones_like(r)
    return np.stack([r, g, b, a], axis=1)


def build_sphere_id_batch(coords_sub, ids, radii_sub):
    if resolve_shader(mesh_spec.ShaderRef("sphere_id")) is None:
        return None
    batch, _ = upload(mesh_spec.sphere_spec(
        coords_sub, mesh_spec.encode_ids_to_rgba(ids), radii_sub,
        shader_base="sphere_id"))
    return batch


def build_line_batch(coords, edges_sub):
    return upload(mesh_spec.line_spec(coords, edges_sub))


def edge_width_status(obj, scene, st=None):
    """``(applied, reason)``: is edge width by attribute reaching pixels? Every
    failure looks the same on screen, so the panel shows the reason."""
    from .state import is_graph_object
    if not is_graph_object(obj):
        return False, "no graph selected"
    st = _settings(scene, st)
    if not bool(st.edge_size_by_attr):
        return False, "Width by Attribute is off"
    name = st.edge_attr_name
    if not name:
        return False, "no attribute chosen"
    attr = obj.data.attributes.get(name)
    if attr is None:
        return False, f"'{name}' is not on this mesh"
    if attr.domain != 'EDGE':
        return False, f"'{name}' is on {attr.domain}, not EDGE"
    if attr.data_type not in ('FLOAT', 'INT'):
        return False, f"'{name}' is {attr.data_type}, not a number"
    values = read_edge_scalar(obj.data, name)
    if values is None or not np.isfinite(values).any():
        return False, f"'{name}' has no usable values"
    if float(np.nanmax(values)) <= float(np.nanmin(values)):
        return False, f"every edge has the same '{name}'"

    entry = state.CACHE.get(obj.as_pointer())
    if entry is None:
        return True, "ready (not drawn yet)"
    reps = entry.get("reps", {})
    if entry.get("blocks") is not None:
        return False, "spatial blocks draw edges in bulk; turn Blocks off"
    if reps.get("ribbon") is not None:
        if st.edge_style == 'RIBBON':
            return True, ("applied to tube radius (invisible while tubes are "
                          "sub-pixel; use Auto or Line when zoomed out)")
        return True, "applied to tube radius"
    if reps.get("line_buckets"):
        return True, f"applied as {len(reps['line_buckets'])} width classes"
    if reps.get("bundle") is not None:
        return False, "bundling draws its own edges"
    if entry.get("filter") is not None:
        return False, "a GPU filter holds the edges on the per-endpoint path"
    if bool(st.edge_styles_gpu):
        return False, "edge styles tessellate their own segments"
    return False, "no width-carrying batch was built"


def _taper_radii(scene, radii, st=None):
    """``(r_source, r_target)``: a taper changes shape and not thickness."""
    st = _settings(scene, st)
    mode = st.edge_taper
    if mode == 'NONE':
        return radii, radii
    amount = float(np.clip(st.edge_taper_amount, 0.0, 1.0))
    # Never zero: that end is a degenerate quad with no cylinder section.
    thin = (radii * (1.0 - 0.9 * amount)).astype(np.float32)
    return (thin, radii) if mode == 'FORWARD' else (radii, thin)


def _taper_segment_radii(scene, ra, rb, st=None):
    """Scales the radii rather than replacing them, so the shape survives."""
    st = _settings(scene, st)
    mode = st.edge_taper
    if mode == 'NONE':
        return ra, rb
    amount = float(np.clip(st.edge_taper_amount,
                           0.0, 1.0))
    k = np.float32(1.0 - 0.9 * amount)
    return (ra * k, rb) if mode == 'FORWARD' else (ra, rb * k)


def build_segment_line_width_buckets(seg_a, seg_b, widths, col_a=None,
                                     col_b=None, buckets=SIZE_BUCKETS):
    """Width classes over a tessellated segment soup, as
    ``([(batch, mid), ...], shader)``. Without its own split the styled path
    loses edge weighting in the line tier. POLYLINE builtins build width as
    geometry; the rasterizer's line width stops growing inside this range."""
    return upload_groups(mesh_spec.bucketed_segment_spec(
        seg_a, seg_b, widths, col_a, col_b, buckets))


def build_line_width_buckets(coords, edges_sub, widths, buckets=SIZE_BUCKETS):
    """Line batches split by width, as ``[(batch, mid), ...]``, ``mid`` the
    bucket center in [0, 1]. ``gpu.state.line_width_set`` is per draw call, so
    one LINES batch is one thickness. Unsplit, only ribbons got weighting."""
    return upload_groups(
        mesh_spec.bucketed_line_spec(coords, edges_sub, widths, buckets))


def build_filter_line_batch(coords, edges_sub, edge_rows, num_nodes):
    """The line batch carries the filter rows, so the shader can cut it."""
    if edges_sub is None or edges_sub.shape[0] == 0:
        return None, None
    return upload(mesh_spec.filtered_line_spec(
        coords, edges_sub,
        filter_gpu.edge_frows(edges_sub, edge_rows, num_nodes),
        filter_gpu.edge_frows(edges_sub, edge_rows, num_nodes, swap=True)))


def build_segment_line_batch(seg_a, seg_b, col_a=None, col_b=None):
    return upload(mesh_spec.segment_line_spec(seg_a, seg_b, col_a, col_b))


def build_ribbon_batch(seg_a, seg_b, edge_color, radius_a, radius_b,
                       col_a=None, col_b=None, frows=None):
    """4 vertices + 2 tris per segment; per-endpoint radii let a taper thin it."""
    if resolve_shader(mesh_spec.ShaderRef(
            "ribbon", filtered=frows is not None)) is None:
        return None
    batch, _ = upload(mesh_spec.ribbon_spec(
        seg_a, seg_b, edge_color, radius_a, radius_b,
        col_a=col_a, col_b=col_b, frows=frows))
    return batch


def build_ribbon_id_batch(seg_a, seg_b, ids, radius_a, radius_b):
    """Colored by segment id; a per-edge id would miss a half-hidden curve."""
    if resolve_shader(mesh_spec.ShaderRef("ribbon_id")) is None:
        return None
    colors = mesh_spec.encode_ids_to_rgba(ids)
    batch, _ = upload(mesh_spec.ribbon_spec(
        seg_a, seg_b, None, radius_a, radius_b,
        col_a=colors, col_b=colors, shader_base="ribbon_id"))
    return batch


def build_arrow_batch(anchor, direction, edge_radii, node_radii, color,
                      arrow_scale=1.0):
    """Lit 3D cones at directed edge ends, 18 vertices, 16 triangles each."""
    if resolve_shader(mesh_spec.ShaderRef("arrow")) is None:
        return None
    batch, _ = upload(mesh_spec.arrow_spec(
        anchor, direction, edge_radii, node_radii, color, arrow_scale))
    return batch


# In units of its radius, matching the cone: a style switch only changes look.
_FLAT_LEN = 6.0
_FLAT_HALF_WIDTH = 2.4


def build_arrow_flat_batch(anchor_a, anchor_b, edge_radii, node_radii, color,
                           arrow_scale=1.0):
    if resolve_shader(mesh_spec.ShaderRef("arrow_flat")) is None:
        return None
    batch, _ = upload(mesh_spec.arrow_flat_spec(
        anchor_a, anchor_b, edge_radii, node_radii, color, arrow_scale))
    return batch


def coarsen_context(mesh, scene, coords, edges, node_mask, colors,
                    point_shader, point_kind, base_radius, edge_color,
                    want_arrows, arrow_scale, st=None):
    """A plain dict, so the adaptive cut rebuilds without touching the mesh."""
    st = _settings(scene, st)
    logical = edges
    weights = simplify.edge_weights_raw(mesh, st.backbone_attr)
    if node_mask is not None and not node_mask.all():
        rec = edge_styles_gpu.recover_logical_edges(edges, node_mask)
        if rec is None:
            logical, weights = None, None
        else:
            logical, mesh_ids = rec
            weights = weights[mesh_ids] if weights is not None else None

    size_mode = st.coarsen_size_mode
    size_values = None
    if size_mode == 'ATTRIBUTE':
        size_values = simplify.point_scalars_raw(mesh, st.coarsen_size_attr)

    return {
        "coords": coords,
        "logical": logical,
        "weights": weights,
        "colors": colors,
        "base_radius": base_radius,
        "edge_radius": base_radius * float(st.edge_radius_scale),
        "size_mode": size_mode,
        "size_values": size_values,
        "size_agg": st.coarsen_size_agg,
        "size_max_mult": float(st.coarsen_size_mult),
        "point_shader": point_shader,
        "point_kind": point_kind,
        "edge_color": edge_color,
        "want_arrows": want_arrows,
        "arrow_scale": arrow_scale,
        "want_id": bool(st.render_id_pass) or bool(st.adaptive),
    }


_TREE_CACHE = {}
_TREE_CACHE_MAX = 4


def _cached_cluster_tree(obj, mesh, scene, coords, edges, node_mask, colors,
                         base_radius, st=None):
    """Memoized, or every Bundling Strength step re-partitions the graph."""
    key = structure_signature(obj, scene)
    if key in _TREE_CACHE:
        # Membership, not truthiness: a remembered None means "no clustering".
        return _TREE_CACHE[key]
    tree = _cluster_tree(mesh, scene, coords, edges, node_mask, colors,
                         base_radius, st=st)
    if len(_TREE_CACHE) >= _TREE_CACHE_MAX:
        _TREE_CACHE.pop(next(iter(_TREE_CACHE)))
    _TREE_CACHE[key] = tree
    return tree


def clear_tree_cache():
    _TREE_CACHE.clear()


def drop_tree_cache(obj):
    """It holds community centroids, so a new layout changes it silently."""
    name = obj.name
    for key in [k for k in _TREE_CACHE if k and k[0] == name]:
        del _TREE_CACHE[key]


def _bundle_vertex_shader_rep(scene, coords, logical, params, hierarchy, colors,
                              edge_color, edge_style, want_id, edge_norm=None,
                              st=None):
    """Bundling from control points in the vertex shader, or None. Hierarchical
    bundling lands here only when nothing else would consume the tessellated
    segments. The bundle_gpu modes have no numpy fallback."""
    st = _settings(scene, st)
    mode = params["style_type"]
    if mode not in bundle_gpu.PRODUCERS:
        return None
    if want_id or bool(st.adaptive):
        return None
    ctx = {"hierarchy": hierarchy, "scene": scene}
    if not bundle_gpu.usable(mode, params, logical, ctx)[0]:
        return None
    if mode == 'HIERARCHICAL':
        if not st.heb_vertex_shader:
            return None
        segments = max(1, int(params["segments"]))
        past_budget = logical.shape[0] * segments > _IMPOSTOR_MAX
        if not (past_budget or edge_style == 'LINE'):
            return None
    # The width-carrying variant draws quads; one instanced LINES call is one
    # width. Declining here instead cost 130 ms -> 1.34 s on 88k edges.
    return bundle_gpu.build(mode, coords, logical, params, ctx=ctx,
                            node_colors=colors, edge_color=edge_color,
                            edge_widths=edge_norm)


def _cluster_tree(mesh, scene, coords, edges, node_mask, colors, base_radius, st=None):
    """Level one is the attribute the analysis operator wrote. Above it,
    detection re-runs on each level's superedge graph."""
    st = _settings(scene, st)
    labels = simplify.point_int_labels(mesh, st.coarsen_attr)
    if labels is None or edges is None:
        return None
    logical = edges
    weights = simplify.edge_weights_raw(
        mesh, st.backbone_attr)
    if node_mask is not None and not node_mask.all():
        rec = edge_styles_gpu.recover_logical_edges(edges, node_mask)
        if rec is None:
            return None
        logical, mesh_ids = rec
        weights = weights[mesh_ids] if weights is not None else None
    size_mode = st.coarsen_size_mode
    size_values = (simplify.point_scalars_raw(mesh, st.coarsen_size_attr)
                   if size_mode == 'ATTRIBUTE' else None)
    return simplify.build_hierarchy(
        coords, logical, labels, colors, weights, base_radius,
        size_mode=size_mode, size_values=size_values,
        size_agg=st.coarsen_size_agg,
        size_max_mult=float(st.coarsen_size_mult),
        algorithm=st.adaptive_algorithm,
    ) or None


def coarse_from_labels(ctx, labels, stand_in=False):
    """Supernode/superedge batches, entry-shaped so draw.py reuses everything."""
    data = simplify.build_coarse_level(
        ctx["coords"], ctx["logical"], labels, ctx["colors"], ctx["weights"],
        ctx["base_radius"], size_mode=ctx["size_mode"],
        size_values=ctx["size_values"], size_agg=ctx["size_agg"],
        size_max_mult=ctx["size_max_mult"],
        stand_in=stand_in,
    )
    if data is None:
        return None
    point_shader = ctx["point_shader"]
    point_kind = ctx["point_kind"]
    edge_color = ctx["edge_color"]
    want_arrows = ctx["want_arrows"]
    arrow_scale = ctx["arrow_scale"]
    size_mode = ctx["size_mode"]
    size_values = ctx["size_values"]

    centers = data["centers"]
    ccolors = data["colors"]
    cradii = data["radii"]

    creps = {"arrow": None, "ribbon": None}
    creps["sphere"] = build_sphere_batch(centers, ccolors, cradii)
    creps["point"] = build_point_batches(
        point_shader, centers, ccolors, None, False)

    element_idx = np.arange(centers.shape[0], dtype=np.int64)
    creps["sphere_id"] = (
        build_sphere_id_batch(centers, element_idx, cradii)
        if ctx.get("want_id") and centers.shape[0] <= _IMPOSTOR_MAX else None)

    line_shader_c = None
    creps["line"] = None
    n_super = int(data["se_src"].size)
    if n_super:
        seg_a = centers[data["se_src"]]
        seg_b = centers[data["se_dst"]]
        # Against the heaviest superedge; by supernode size a few big
        # collapses would thicken every edge.
        wsum = data["se_weight"]
        wnorm = wsum / wsum.max() if wsum.max() > 0 else np.ones_like(wsum)
        radii_e = (ctx["edge_radius"] * (0.6 + 1.4 * wnorm)).astype(np.float32)
        creps["line"], line_shader_c = build_segment_line_batch(seg_a, seg_b)
        creps["ribbon"] = build_ribbon_batch(
            seg_a, seg_b, edge_color, radii_e, radii_e)

        if want_arrows:
            dom = data["se_dom"] >= 0.6
            if dom.any():
                d = seg_b[dom] - seg_a[dom]
                ln = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
                creps["arrow"] = build_arrow_batch(
                    seg_b[dom], d / ln[:, None], radii_e[dom],
                    cradii[data["se_dst"][dom]], edge_color, arrow_scale,
                )

    return {
        "entry": {
            "reps": creps,
            "point_shader": point_shader,
            "point_kind": point_kind,
            "line_shader": line_shader_c,
            "coords": centers,
            "point_idx": element_idx,
            "radii": cradii,
            "seg_geom": None,
        },
        "radius_world": float(np.median(cradii)),
        "n_comm": data["n_comm"],
        "n_super": n_super,
        "size_mode": size_mode,
        "size_missing": size_mode == 'ATTRIBUTE' and size_values is None,
        "size_range": (
            (float(np.min(data["size_driver"])),
             float(np.max(data["size_driver"])))
            if data["size_driver"] is not None else None
        ),
    }


def build_bundle(obj, scene, st=None):
    """The entry dict draw.py caches: shaders, ``reps``, coords/point_idx/radii,
    and the optional blocks, coarse level, filter, position and density field."""
    st = _settings(scene, st)
    mesh = obj.data
    num_verts = len(mesh.vertices)

    coords = geometry.extract_node_coords(mesh)
    edges = geometry.extract_edges(mesh)
    node_mask = geometry.extract_node_mask(mesh)

    values, vmin, vmax = read_value_channel(mesh, st.attr_name, num_verts)
    norm = normalized_values(values, vmin, vmax, num_verts)
    colors = compute_colors(mesh, num_verts, scene, values, vmin, vmax,
                            st=st)

    # The vertex shader takes it when the drawn set matches. Otherwise decide
    # here, since only here does it reach a density grid or a tessellation.
    gpu_filter = filter_gpu.build(obj, scene) \
        if filter_gpu.eligible(obj, scene, st=st)[0] else None
    if gpu_filter is not None:
        visible = np.ones(num_verts, dtype=bool)
        stack_edges = None
    else:
        # One call for both: only the stack knows if a clause meant nodes.
        visible, stack_edges, _ = filters.masks(obj, scene)

    # Only real nodes (is_intersection == 1), never an edge style's curve points.
    if node_mask is not None:
        node_visible = visible & node_mask
    else:
        node_visible = visible

    point_idx = np.nonzero(node_visible)[0]

    lod = bool(st.lod_enabled)
    lod_max = max(1000, int(st.lod_max_points))
    if lod and point_idx.size > lod_max:
        stride = int(np.ceil(point_idx.size / lod_max))
        point_idx = point_idx[::stride]

    vol = None
    vol_stats = None
    vol_mode = st.volume_mode
    if vol_mode != 'OFF' and point_idx.size:
        weights = norm[point_idx] if (vol_mode == 'WEIGHTED'
                                      and norm is not None) else None
        vol, vol_data = volume.build(
            scene, coords[point_idx], colors[point_idx], weights)
        if vol is not None:
            vol_stats = {
                "shape": vol["shape"],
                "bytes": vol["bytes"],
                "dmax": vol["dmax"],
                "dtrue": vol["dtrue"],
                "nodes": int(point_idx.size),
                "dropped": 0,
                "weighted_missing": vol_mode == 'WEIGHTED' and norm is None,
            }
            # Crowded nodes leave the discrete batches, edges included.
            if st.volume_when == 'HYBRID':
                crowded = volume.crowded_mask(
                    vol_data["node_density"],
                    float(st.volume_hybrid_pct))
                if crowded is not None:
                    dropped = point_idx[crowded]
                    node_visible[dropped] = False
                    visible[dropped] = False
                    point_idx = point_idx[~crowded]
                    vol_stats["dropped"] = int(dropped.size)

    coords_sub = coords[point_idx]
    colors_sub = colors[point_idx]
    norm_sub = norm[point_idx] if norm is not None else None

    # Positions in a texture so a layout moves nodes without rebuilding. Decided
    # here, since it changes which shader every representation asks for.
    animated = bool(st.animate) \
        and dynamic.eligible(obj, scene, st=st)[0]
    pos_entry = dynamic.build(obj, coords) if animated else None
    animated = pos_entry is not None

    # Edge-visible = curve point or visible node, so polylines survive whole.
    if node_mask is not None:
        edge_vis = (~node_mask) | node_visible
    else:
        edge_vis = visible

    unbaked = node_mask is None or bool(node_mask.all())
    bb_mode = st.backbone_mode
    bb_k = int(st.backbone_k)
    bb_alpha = float(st.backbone_alpha)
    bb_sample = float(st.backbone_sample)
    bb_w = None
    bb_mask = None
    bb_stats = None
    if bb_mode != 'ALL' and edges is not None and edges.shape[0]:
        bb_w = simplify.edge_weights_raw(
            mesh, st.backbone_attr)
        if unbaked:
            bb_mask, bb_stats = simplify.backbone_mask(
                edges, bb_w, bb_mode, bb_k, bb_alpha, bb_sample)

    # kept_rows keeps the mesh-edge index, so EDGE attributes stay aligned.
    kept = None
    kept_rows = None
    if edges is not None:
        keep = edge_vis[edges[:, 0]] & edge_vis[edges[:, 1]]
        if bb_mask is not None:
            keep &= bb_mask
        # Edge clauses cut here so the backbone statistics describe what is drawn.
        if stack_edges is not None and stack_edges.size == keep.size:
            keep &= stack_edges
        kept_rows = np.nonzero(keep)[0]
        kept = edges[kept_rows]
        if lod and kept.shape[0] > lod_max:
            stride = int(np.ceil(kept.shape[0] / lod_max))
            kept = kept[::stride]
            kept_rows = kept_rows[::stride]
        if kept.shape[0] == 0:
            kept = None
            kept_rows = None

    ewidth_mesh = None
    enorm = None
    if bool(st.edge_size_by_attr):
        enorm = read_edge_value_channel(
            mesh, st.edge_attr_name,
            scale=st.edge_size_scale)
        if enorm is not None:
            emult = float(st.edge_size_max_mult)
            ewidth_mesh = 1.0 + enorm * (emult - 1.0)

    size_by_attr = bool(st.size_by_attr)
    size_max_mult = float(st.size_max_mult)
    base_radius = float(st.impostor_radius)
    edge_radius = base_radius * float(
        st.edge_radius_scale)
    node_style = st.node_style
    edge_style = st.edge_style
    want_id = bool(st.render_id_pass)
    edge_color = tuple(st.edge_color)

    # Grown once and shared; each level costs a community-detection run.
    want_adaptive = bool(st.adaptive) \
        and bool(st.coarsen)
    hierarchy = None
    if edge_styles_gpu.needs_hierarchy(scene) or want_adaptive:
        hierarchy = _cached_cluster_tree(
            obj, mesh, scene, coords, edges, node_mask, colors, base_radius,
            st=st)

    # Logical edges recovered from baked polylines; ``styled`` replaces them.
    styled = None
    styled_edge_count = 0
    bundle_vs = None
    if edge_styles_gpu.enabled(scene, st) and edges is not None:
        recovered = edge_styles_gpu.recover_logical_edges(edges, node_mask)
        if recovered is not None:
            logical, mesh_ids = recovered
            # The backbone prunes whole logical chains, never single segments.
            if bb_mode != 'ALL' and logical.shape[0]:
                if bb_mask is not None:
                    m = bb_mask[mesh_ids]
                else:
                    w_log = bb_w[mesh_ids] if bb_w is not None else None
                    m, bb_stats = simplify.backbone_mask(
                        logical, w_log, bb_mode, bb_k, bb_alpha, bb_sample)
                if m is not None:
                    logical = logical[m]
                    mesh_ids = mesh_ids[m]
            lkeep = node_visible[logical[:, 0]] & node_visible[logical[:, 1]]
            if stack_edges is not None and mesh_ids.size:
                lkeep &= stack_edges[mesh_ids]
            logical = logical[lkeep]
            mesh_ids = mesh_ids[lkeep]
            if lod and logical.shape[0] > lod_max:
                stride = int(np.ceil(logical.shape[0] / lod_max))
                logical = logical[::stride]
                mesh_ids = mesh_ids[::stride]
            if logical.shape[0]:
                ewidth_logical = ewidth_mesh[mesh_ids] \
                    if ewidth_mesh is not None else None
                style_params = edge_styles_gpu.style_params(scene)
                enorm_logical = enorm[mesh_ids] if enorm is not None else None
                bundle_vs = _bundle_vertex_shader_rep(
                    scene, coords, logical, edge_styles_gpu.bundle_params(scene),
                    hierarchy, colors, edge_color, edge_style, want_id,
                    edge_norm=enorm_logical, st=st)
                if bundle_vs is None:
                    styled = edge_styles_gpu.tessellate(
                        coords, logical, style_params,
                        edge_widths=ewidth_logical, hierarchy=hierarchy,
                        node_colors=colors, edge_color=edge_color,
                    )
                styled_edge_count = int(logical.shape[0])

    # Mandatory when the stack is live: only the filtered variant declares the
    # row attribute, so the flat builtin would draw the whole graph.
    round_pts = bool(st.round_points) \
        or gpu_filter is not None
    round_shader = shaders.get_round_point_shader(
        filtered=gpu_filter is not None, animated=animated) if round_pts else None
    if round_shader is not None:
        point_shader = round_shader
        point_kind = 'ROUND'
    else:
        # FLAT_COLOR takes no node index: better static than all at the origin.
        point_shader = gpu.shader.from_builtin('FLAT_COLOR')
        point_kind = 'FLAT'
        animated = False
        pos_entry = None

    node_frows = filter_gpu.node_frows(point_idx) \
        if gpu_filter is not None else None

    reps = {}
    reps["line_buckets"] = None
    reps["point"] = dynamic.build_point_batches(
        point_shader, point_idx, colors_sub, norm_sub, size_by_attr,
        frows=node_frows, size_buckets=SIZE_BUCKETS,
    ) if animated else build_point_batches(
        point_shader, coords_sub, colors_sub, norm_sub, size_by_attr,
        frows=node_frows,
    )

    want_sphere = node_style in ('SPHERE', 'AUTO') and point_idx.size <= _IMPOSTOR_MAX
    reps["sphere"] = None
    reps["sphere_id"] = None
    radii_sub = None
    if (want_sphere or want_id) and point_idx.size <= _IMPOSTOR_MAX:
        radii_sub = _radii_for(norm_sub if norm_sub is not None
                               else np.zeros(point_idx.size, dtype=np.float32),
                               base_radius, size_by_attr, size_max_mult)
        if want_sphere:
            reps["sphere"] = dynamic.build_sphere_batch(
                point_idx, colors_sub, radii_sub, frows=node_frows) \
                if animated else build_sphere_batch(
                    coords_sub, colors_sub, radii_sub, frows=node_frows)
        if want_id:
            reps["sphere_id"] = build_sphere_id_batch(coords_sub, point_idx, radii_sub)

    reps["ribbon_id"] = None
    seg_geom = None
    if bundle_vs is not None:
        # The curve lives in the shader: no segments to hand out.
        reps["bundle"] = bundle_vs
        reps["ribbon"] = None
        line_batch, line_shader = None, None
        edge_count = styled_edge_count
    elif styled is not None:
        scol_a, scol_b = styled.get("col_a"), styled.get("col_b")
        line_batch, line_shader = build_segment_line_batch(
            styled["seg_a"], styled["seg_b"], scol_a, scol_b
        )
        seg_count = styled["seg_a"].shape[0]
        want_ribbon = edge_style in ('RIBBON', 'AUTO') \
            and seg_count <= _IMPOSTOR_MAX
        seg_ra = (styled["scale_a"] * edge_radius).astype(np.float32)
        seg_rb = (styled["scale_b"] * edge_radius).astype(np.float32)
        # Multiplies the style's own scale; replacing it made the taper a no-op.
        seg_ra, seg_rb = _taper_segment_radii(scene, seg_ra, seg_rb, st=st)
        reps["ribbon"] = build_ribbon_batch(
            styled["seg_a"], styled["seg_b"], edge_color, seg_ra, seg_rb,
            scol_a, scol_b,
        ) if want_ribbon else None
        # Bucketed on the segment radius, which already carries the weighting.
        if enorm is not None:
            reps["line_buckets"], bshader = build_segment_line_width_buckets(
                styled["seg_a"], styled["seg_b"], 0.5 * (seg_ra + seg_rb),
                col_a=scol_a, col_b=scol_b)
            if reps["line_buckets"] is not None:
                line_shader = bshader
        if seg_count <= _IMPOSTOR_MAX:
            owner = styled.get("seg_edge")
            if owner is None:
                owner = np.arange(seg_count, dtype=np.int64)
            seg_geom = (styled["seg_a"], styled["seg_b"], seg_ra, seg_rb,
                        np.asarray(owner, dtype=np.int64))
        edge_count = styled_edge_count
    else:
        if gpu_filter is not None:
            edge_frows = filter_gpu.edge_frows(kept, kept_rows, num_verts) \
                if kept is not None else None
        else:
            edge_frows = None
        if animated:
            # One vertex per node; filtered graphs keep the per-endpoint ones.
            line_batch = dynamic.build_line_batch(kept) \
                if gpu_filter is None else None
            line_shader = dynamic.get_line_shader() if line_batch else None
            if line_batch is None:
                animated = False
                pos_entry = None
        if not animated:
            if gpu_filter is not None:
                line_batch, line_shader = build_filter_line_batch(
                    coords, kept, kept_rows, num_verts)
            else:
                line_batch, line_shader = build_line_batch(coords, kept)
        # Plain indexed only: a split would have to carry per-vertex data too.
        if enorm is not None and kept is not None and gpu_filter is None:
            if animated:
                reps["line_buckets"], bucket_shader = \
                    dynamic.build_line_width_buckets(
                        kept, enorm[kept_rows], num_verts,
                        buckets=SIZE_BUCKETS)
            else:
                reps["line_buckets"], bucket_shader = \
                    build_line_width_buckets(coords, kept, enorm[kept_rows])
            if reps["line_buckets"] is not None:
                line_shader = bucket_shader
        want_ribbon = edge_style in ('RIBBON', 'AUTO') and kept is not None \
            and kept.shape[0] <= _IMPOSTOR_MAX
        reps["ribbon"] = None
        if kept is not None and kept.shape[0] <= _IMPOSTOR_MAX:
            e = kept.shape[0]
            radii = np.full(e, edge_radius, dtype=np.float32)
            if ewidth_mesh is not None:
                radii *= ewidth_mesh[kept_rows]
            r_a, r_b = _taper_radii(scene, radii, st=st)
            if want_ribbon:
                reps["ribbon"] = dynamic.build_ribbon_batch(
                    kept, r_a, r_b, edge_color, filtered_rows=edge_frows,
                ) if animated else build_ribbon_batch(
                    coords[kept[:, 0]], coords[kept[:, 1]], edge_color,
                    r_a, r_b, frows=edge_frows,
                )
            seg_geom = (coords[kept[:, 0]], coords[kept[:, 1]], radii, radii,
                        np.arange(e, dtype=np.int64))
        edge_count = int(kept.shape[0]) if kept is not None else 0
    reps["line"] = line_batch

    if want_id and seg_geom is not None:
        s_a, s_b, s_ra, s_rb, _s_owner = seg_geom
        seg_ids = EDGE_ID_FLAG | np.arange(s_a.shape[0], dtype=np.int64)
        reps["ribbon_id"] = build_ribbon_id_batch(
            s_a, s_b, seg_ids, s_ra, s_rb)

    reps["arrow"] = None
    arrow_mode = st.edge_arrows
    directed = bool(obj.get("is_directed", False)) \
        or bool(obj.get("od_directed", False))
    want_arrows = arrow_mode == 'ON' or (arrow_mode == 'AUTO' and directed)
    arrow_scale = float(st.edge_arrow_size)
    if want_arrows:
        # Target-node radius so the tip rests on the sphere surface.
        node_r = np.full(num_verts, base_radius, dtype=np.float32)
        if size_by_attr and norm is not None:
            node_r *= 1.0 + norm * (size_max_mult - 1.0)

        anchor = direction = eradii = target = None
        # The clamp is a tube constraint (a cone stays wider than what it caps);
        # in the tube-less line tier it squashed a 4x weight range to 1.8x. Ask
        # the batch, not `want_ribbon`, unassigned when bundling (NameError).
        tubes = reps.get("ribbon") is not None

        def _head_radius(w):
            if not tubes:
                return np.asarray(w, dtype=np.float32) * edge_radius
            return np.maximum(np.minimum(w, 1.5), w / 2.2) * edge_radius

        if styled is not None and styled.get("arrow_pos") is not None:
            anchor = styled["arrow_pos"]
            direction = styled["arrow_dir"]
            eradii = _head_radius(styled["arrow_width"])
            target = styled["arrow_target"]
        elif kept is not None and (node_mask is None or node_mask.all()):
            # Stored vertex order; baked curves need GPU Edge Styles for arrows.
            a = coords[kept[:, 0]]
            b = coords[kept[:, 1]]
            d = b - a
            ln = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            sel = ln > 1e-9
            anchor = b[sel]
            direction = (d / ln[:, None])[sel]
            if ewidth_mesh is not None:
                eradii = _head_radius(ewidth_mesh[kept_rows][sel])
            else:
                eradii = np.full(anchor.shape[0], edge_radius,
                                 dtype=np.float32)
            target = kept[sel, 1]
        flat = st.edge_arrow_style == 'FLAT'
        reps["arrow_flat"] = flat
        if animated and kept is not None and kept.shape[0] <= _IMPOSTOR_MAX \
                and (node_mask is None or node_mask.all()):
            # The shader drops edges a live layout degenerated after the build.
            sel_r = _head_radius(ewidth_mesh[kept_rows]) \
                if ewidth_mesh is not None else np.full(
                    kept.shape[0], edge_radius, dtype=np.float32)
            builder = (dynamic.build_arrow_flat_batch if flat
                       else dynamic.build_arrow_batch)
            reps["arrow"] = builder(
                kept, sel_r, node_r[kept[:, 1]], edge_color, arrow_scale,
            )
        elif anchor is not None and anchor.shape[0] <= _IMPOSTOR_MAX:
            if flat:
                # Direction resolves in view space: only the difference counts.
                src = anchor - direction
                reps["arrow"] = build_arrow_flat_batch(
                    src, anchor, eradii, node_r[target], edge_color,
                    arrow_scale,
                )
            else:
                reps["arrow"] = build_arrow_batch(
                    anchor, direction, eradii, node_r[target], edge_color,
                    arrow_scale,
                )

    blocks = None
    if bool(st.use_blocks) and point_idx.size:
        blocks = _build_blocks(
            coords, colors, kept, point_idx, scene, point_shader, styled,
            st=st
        )

    # Drawn instead of the full graph when communities project too small to read.
    coarse = None
    ctx = None
    labels = (simplify.point_int_labels(
        mesh, st.coarsen_attr)
        if bool(st.coarsen) else None)
    if labels is not None:
        ctx = coarsen_context(
            mesh, scene, coords, edges, node_mask, colors, point_shader,
            point_kind, base_radius, edge_color, want_arrows, arrow_scale,
            st=st,
        )
        coarse = coarse_from_labels(ctx, labels)

    return {
        "point_shader": point_shader,
        "point_kind": point_kind,
        "line_shader": line_shader,
        "line_vertex_color": styled is not None and "col_a" in styled,
        # Channel texture, or None when the stack was applied here instead.
        "filter": gpu_filter,
        # Position texture, or None when baked: draw.py refreshes or rebuilds.
        "dynamic": pos_entry,
        "reps": reps,
        "coords": coords,
        "colors": colors,
        "point_idx": point_idx,
        # World-space radius per drawn node, aligned with point_idx.
        "radii": radii_sub,
        # (a, b, radius_a, radius_b, logical edge index) behind ribbon_id.
        "seg_geom": seg_geom if reps["ribbon_id"] is not None else None,
        "edges": kept,
        "visible_count": int(point_idx.size),
        "edge_count": edge_count,
        "blocks": blocks,
        "coarse": coarse,
        # Tree plus the inputs to rebuild batches for any cut draw.py picks.
        "hierarchy": hierarchy,
        "coarsen_ctx": ctx,
        "simplify_stats": bb_stats,
        "volume": vol,
        "volume_stats": vol_stats,
    }


def _build_blocks(coords, colors, kept_edges, point_idx, scene, point_shader,
                  styled=None, st=None):
    """Per-block batches and bounding spheres for the culling path. Styled
    segments bin by their own midpoints, so curved edges cull correctly."""
    st = _settings(scene, st)
    sub_coords = coords[point_idx]
    target = max(64, int(st.block_count))
    seg_mid = 0.5 * (styled["seg_a"] + styled["seg_b"]) \
        if styled is not None else None
    part = geometry.build_spatial_blocks(
        sub_coords, kept_edges if styled is None else None,
        target_blocks=target, seg_mid=seg_mid,
    )

    block_batches = []
    node_counts = []
    for local_idx in part["node_indices"]:
        orig = point_idx[local_idx]
        pb = build_point_batches(
            point_shader, coords[orig], colors[orig], None, False
        )
        block_batches.append(pb)
        node_counts.append(orig.size)

    # draw_blocks binds the uniform color on it, so it survives as a value.
    vertex_color = styled is not None and "col_a" in styled
    line_shader = resolve_shader(mesh_spec.ShaderRef(
        "smooth_color" if vertex_color else "uniform_color"))
    specs = mesh_spec.block_line_specs(
        part["edge_indices"], coords=coords, edges=kept_edges,
        seg_a=None if styled is None else styled["seg_a"],
        seg_b=None if styled is None else styled["seg_b"],
        col_a=styled["col_a"] if vertex_color else None,
        col_b=styled["col_b"] if vertex_color else None,
    )
    line_batches = [upload_with(spec, line_shader) for spec in specs]
    # Off the specs, not the partition: a None spec drew nothing.
    edge_counts = [0 if spec is None else int(np.asarray(eids).size)
                   for eids, spec in zip(part["edge_indices"], specs)]

    return {
        "centers": part["centers"],
        "radii": part["radii"],
        "point_batches": block_batches,
        "line_batches": line_batches,
        "line_shader": line_shader,
        "line_vertex_color": vertex_color,
        "node_counts": np.asarray(node_counts, dtype=np.int64),
        "edge_counts": np.asarray(edge_counts, dtype=np.int64),
        "count": part["count"],
    }
