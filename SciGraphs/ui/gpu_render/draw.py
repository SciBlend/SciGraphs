# The SpaceView3D draw handler and LOD selection. Camera moves reuse batches.

import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent

from . import batches, dynamic, filter_gpu, filters, state, volume
from ...core.render import adaptive, lod
from .state import is_graph_object, tag_redraw

_HEADLIGHT_DIR = (0.0, 0.0, 1.0)  # view space, toward the camera

EDGE_XRAY = 0.0


def _settings(scene, st=None):
    """22.9 us to build, ~0.05 us per read, so ~460 us saved per redraw."""
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)

def _luminance(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def compute_lighting(scene, view_matrix, st=None):
    """Scene lights as a key + fill + ambient rig. Python sets no arrays."""
    st = _settings(scene, st)
    from mathutils import Vector

    strength = float(st.light_strength)
    ambient_amt = float(st.ambient)
    rim = float(st.rim)

    world = scene.world
    if world is not None and not getattr(world, "use_nodes", False):
        wc = world.color
        ambient = (wc[0] * ambient_amt, wc[1] * ambient_amt, wc[2] * ambient_amt)
    else:
        ambient = (ambient_amt, ambient_amt, ambient_amt)

    use_scene = bool(st.use_scene_lights)

    lights = []
    if use_scene and view_matrix is not None:
        rot = np.asarray(view_matrix, dtype=np.float64)[:3, :3]
        for obj in scene.objects:
            if obj.type != 'LIGHT' or obj.hide_render:
                continue
            data = obj.data
            color = np.array(data.color, dtype=np.float64)
            energy = float(getattr(data, "energy", 1.0))
            mw = obj.matrix_world
            if data.type == 'SUN':
                # A sun shines along local -Z, so toward it is local +Z.
                world_dir = np.array(mw.to_3x3() @ Vector((0.0, 0.0, 1.0)))
                intensity = energy
            else:
                # Treat as directional, origin toward the lamp. No falloff.
                pos = np.array(mw.translation)
                world_dir = pos
                intensity = energy * 0.001
            norm = np.linalg.norm(world_dir)
            if norm < 1e-9:
                continue
            dir_view = rot @ (world_dir / norm)
            dv = np.linalg.norm(dir_view)
            if dv < 1e-9:
                continue
            dir_view = dir_view / dv
            lights.append((dir_view, color * (intensity * strength)))

    if not lights:
        return {
            "key_dir": _HEADLIGHT_DIR,
            "key_col": (0.9 * strength, 0.9 * strength, 0.9 * strength),
            "fill_dir": _HEADLIGHT_DIR,
            "fill_col": (0.0, 0.0, 0.0),
            "ambient": ambient,
            "rim": rim,
        }

    lights.sort(key=lambda item: _luminance(item[1]), reverse=True)
    key_dir, key_col = lights[0]

    if len(lights) > 1:
        fill_col = np.sum([c for _, c in lights[1:]], axis=0)
        weighted = np.sum([d * _luminance(c) for d, c in lights[1:]], axis=0)
        wn = np.linalg.norm(weighted)
        fill_dir = weighted / wn if wn > 1e-9 else key_dir
    else:
        fill_col = np.zeros(3)
        fill_dir = key_dir

    return {
        "key_dir": tuple(float(v) for v in key_dir),
        "key_col": tuple(float(v) for v in key_col),
        "fill_dir": tuple(float(v) for v in fill_dir),
        "fill_col": tuple(float(v) for v in fill_col),
        "ambient": ambient,
        "rim": rim,
    }


def get_cache_entry(obj, scene):
    key = obj.as_pointer()
    sig = batches.object_signature(obj, scene)
    entry = state.CACHE.get(key)
    if entry is not None and entry.get("sig") != sig \
            and entry["sig"][:-1] == sig[:-1] \
            and entry.get("reps", {}).get("bundle") is not None:
        # Only the strength moved, which the shader reads per frame. A numpy
        # fallback has no "bundle" rep and rebuilds instead.
        entry["sig"] = sig
    if entry is None or entry.get("sig") != sig:
        live = _live_coords(obj, entry)
        try:
            data = batches.build_bundle(obj, scene)
        except Exception:  # noqa: BLE001 - never let drawing crash the viewport
            return None
        data["sig"] = sig
        _carry_live_coords(data, live)
        state.CACHE[key] = data
        entry = data
    return entry


def _live_coords(obj, entry):
    """The positions an animated layout holds in its texture, which a rebuild
    would otherwise lose. Only while playback owns the object: any other time
    the mesh is the truth and a stale copy would undo a real edit."""
    if entry is None or entry.get("dynamic") is None:
        return None
    from . import playback
    if not playback.is_running(obj):
        return None
    pending = entry.get("pending_coords")
    return pending if pending is not None else entry.get("coords")


def _carry_live_coords(data, live):
    """A rebuild reads positions from the mesh, but an animated layout does not
    write the mesh until it stops. So anything that moves the signature
    mid-playback pinned the nodes to their mesh coordinates while the
    simulation ran on, and the graph only snapped into place on Stop. A
    keyframed traversal clock left over from an earlier traversal is enough to
    move it, on every single frame."""
    if live is None:
        return
    dyn = data.get("dynamic")
    if dyn is None or live.shape[0] != dyn["count"]:
        return
    data["pending_coords"] = live
    data["coords"] = live


def invalidate(obj=None):
    """Mark the cached batches stale. The tree and filter channels stay: clearing
    them here cost 6.2 s per filter step on a bundled 30k-node graph."""
    if obj is None:
        state.CACHE.clear()
    else:
        state.CACHE.pop(obj.as_pointer(), None)
    tag_redraw()


def refresh_positions(obj, coords=None):
    """Swap the position texture only, returning True when that was enough. What
    an animation loop wants; ``invalidate_geometry`` costs about a second."""
    if obj is None:
        return False
    entry = state.CACHE.get(obj.as_pointer())
    if entry is None or entry.get("dynamic") is None:
        return False
    if coords is None:
        coords = batches.geometry.extract_node_coords(obj.data)
    if coords.shape[0] != entry["dynamic"]["count"]:
        return False
    # Recorded, not uploaded: a frame-change handler has no GPU context bound
    # during a render, and ``dynamic.texture`` swallowed that and froze the render.
    entry["pending_coords"] = coords
    entry["coords"] = coords
    volume.refresh_field(entry.get("volume"), coords)
    tag_redraw()
    return True


def flush_positions(entry):
    """Upload what ``refresh_positions`` parked. Draw time only."""
    pending = entry.get("pending_coords")
    if pending is None:
        return
    if dynamic.refresh(entry.get("dynamic"), pending):
        entry["pending_coords"] = None
        entry["upload_failed"] = False
    else:
        entry["upload_failed"] = True
    volume.flush(entry.get("volume"))


def invalidate_geometry(obj=None):
    """``invalidate`` plus the tree and channel caches. Moved coordinates shift
    centroids and edge lengths while the signature stays identical."""
    if obj is None:
        batches.clear_tree_cache()
        batches.drop_geometry_cache()
        filters.drop_channel_cache()
        filter_gpu.drop_cache()
    else:
        batches.drop_tree_cache(obj)
        batches.drop_geometry_cache(obj)
        filters.drop_channel_cache(obj)
        filter_gpu.drop_cache(obj)
    invalidate(obj)


def _pixel_radius(obj, center, base_radius, persp_mat, height):
    """On-screen radius (px) of a typical node; ``persp_mat`` maps world to clip.
    *center* comes from :func:`_entry_center`, not from the coordinates, because
    this runs per frame and the bounding box does not."""
    if center is None or persp_mat is None or not height:
        return 999.0
    center_world = obj.matrix_world @ center
    persp = np.asarray(persp_mat, dtype=np.float64)
    p = np.array([center_world[0], center_world[1], center_world[2], 1.0])
    clip = persp @ p
    w = abs(clip[3]) + 1e-9
    return (base_radius / w) * 0.5 * height


def draw_graph_object(entry, scene, obj, modelview, proj, persp_mat, height,
                      view_matrix=None, st=None):
    """Draw one graph object at the LOD-selected reps. The caller owns GPU state
    and must have ``modelview``/``proj`` in ``gpu.matrix`` for builtin shaders.
    ``persp_mat``/``height`` size the LOD in reference px of a 1080-tall image;
    ``None``/``0`` forces full detail."""
    st = _settings(scene, st)
    flush_positions(entry)
    base_radius = float(st.impostor_radius)
    show_edges = bool(st.show_edges)

    px = _pixel_radius(obj, _entry_center(entry), base_radius, persp_mat, height)
    scale = lod.px_scale(height)
    px_ref = px / scale
    edge_rscale = float(st.edge_radius_scale)

    entry["_coarse_active"] = False
    entry["_adaptive_active"] = False
    # Cloud Only skips both cuts below; it stands in for unresolvable nodes.
    volume_only = entry.get("volume") is not None \
        and st.volume_mode != 'OFF' \
        and st.volume_when == 'ONLY'

    adaptive_entry = None if volume_only else _adaptive_cut_entry(
        entry, scene, obj, proj, persp_mat, height, st=st)
    if adaptive_entry is not None:
        entry["_adaptive_active"] = True
        reps = _draw_level(adaptive_entry, scene, modelview, proj, view_matrix,
                           entry["_adaptive_px"], scale, show_edges, st=st)
        _maybe_draw_volume(entry, scene, modelview, proj, reps[0], st=st)
        return reps

    # Below the threshold, draw the aggregated level, unless the cut already
    # ruled: "draw every node" returns no batches, and that reads like none.
    coarse = entry.get("coarse")
    if coarse is not None and not entry.get("_cut_ruled") and not volume_only:
        ratio = coarse["radius_world"] / max(base_radius, 1e-9)
        px_c = px * ratio
        if (px_c / scale) < float(st.coarsen_px):
            entry["_coarse_active"] = True
            reps = _draw_level(coarse["entry"], scene, modelview, proj,
                               view_matrix, px_c, scale, show_edges, st=st)
            _maybe_draw_volume(entry, scene, modelview, proj, reps[0], st=st)
            return reps

    node_rep = _select_node_rep(scene, px_ref, st=st)
    edge_rep = _select_edge_rep(scene, px_ref * edge_rscale, st=st)

    # Composited last, over the opaque geometry whose depth clips it.
    want_volume = _volume_wanted(entry, scene, node_rep, st=st)
    if want_volume and (node_rep == 'DENSITY' or volume_only):
        node_rep = 'HIDE'

    lighting = None
    if node_rep == 'SPHERE' or edge_rep == 'RIBBON' \
            or entry["reps"].get("arrow") is not None:
        lighting = compute_lighting(scene, view_matrix, st=st)

    blended = (show_edges and edge_rep not in ('NONE', 'RIBBON')
               and bool(st.additive_edges))
    if show_edges and edge_rep != 'NONE' and not blended:
        _draw_edges(entry, scene, edge_rep, modelview, proj, lighting,
                    px * edge_rscale, scale, st=st)
    if node_rep != 'HIDE':
        _draw_nodes(entry, scene, node_rep, modelview, proj, lighting,
                    px, scale, st=st)
    if blended:
        prev_mask = gpu.state.depth_mask_get()
        gpu.state.depth_mask_set(False)
        try:
            _draw_edges(entry, scene, edge_rep, modelview, proj, lighting,
                        px * edge_rscale, scale, st=st)
        finally:
            gpu.state.depth_mask_set(prev_mask)
    if show_edges and edge_rep != 'NONE':
        _draw_edges_xray(entry, scene, modelview, proj,
                         px * edge_rscale, scale, st=st)
    if want_volume:
        entry["_volume_active"] = volume.draw(entry["volume"], scene,
                                              modelview, proj, st=st)
    return node_rep, edge_rep


def _maybe_draw_volume(entry, scene, modelview, proj, node_rep, st=None):
    if _volume_wanted(entry, scene, node_rep, st=st):
        entry["_volume_active"] = volume.draw(entry["volume"], scene,
                                             modelview, proj, st=st)


def _volume_wanted(entry, scene, node_rep, st=None):
    """``FAR`` ties the cloud to the LOD verdict; hybrid always draws."""
    st = _settings(scene, st)
    entry["_volume_active"] = False
    if entry.get("volume") is None \
            or st.volume_mode == 'OFF':
        return False
    when = st.volume_when
    if when == 'FAR':
        return node_rep in ('DENSITY', 'HIDE')
    return True


def _draw_level(level_entry, scene, modelview, proj, view_matrix,
                px_nodes, scale, show_edges, st=None):
    node_rep = _select_node_rep(scene, px_nodes / scale, st=st)
    px_e = px_nodes * 0.2
    edge_rep = _select_edge_rep(scene, px_e / scale, st=st)
    lighting = None
    if node_rep == 'SPHERE' or edge_rep == 'RIBBON' \
            or level_entry["reps"].get("arrow") is not None:
        lighting = compute_lighting(scene, view_matrix, st=st)
    if show_edges and edge_rep != 'NONE':
        _draw_edges(level_entry, scene, edge_rep, modelview, proj, lighting,
                    px_e, scale, st=st)
    if node_rep != 'HIDE':
        _draw_nodes(level_entry, scene, node_rep, modelview, proj, lighting,
                    px_nodes, scale, st=st)
    if show_edges and edge_rep != 'NONE':
        _draw_edges_xray(level_entry, scene, modelview, proj, px_e, scale, st=st)
    return node_rep, edge_rep


def _adaptive_cut_entry(entry, scene, obj, proj, persp_mat, height, st=None):
    """This frame's cut, or None to draw normally. Batches rebuild on a change."""
    st = _settings(scene, st)
    entry["_cut_ruled"] = False
    hierarchy = entry.get("hierarchy")
    ctx = entry.get("coarsen_ctx")
    if not hierarchy or ctx is None or persp_mat is None \
            or not bool(st.adaptive):
        return None

    if bool(st.adaptive_freeze) \
            and "_cut_entry" in entry:
        entry["_cut_ruled"] = True
        cached = entry["_cut_entry"]
        return cached["entry"] if cached is not None else None

    state = entry.get("_cut_state")
    if state is None or len(state.hidden) != len(hierarchy):
        state = adaptive.CutState(hierarchy)
        entry["_cut_state"] = state

    coords = ctx["coords"]
    obj_to_clip = np.asarray(persp_mat, dtype=np.float64) @ np.asarray(
        obj.matrix_world, dtype=np.float64)
    camera = adaptive.camera_terms(obj_to_clip, proj, height)
    leaf = {"coords": coords, "radii": ctx["base_radius"]}

    drawn, changes = adaptive.select_cut(
        hierarchy, leaf, state, camera,
        min_predicted=float(st.adaptive_predicted),
        min_px=float(st.adaptive_min_px)
        * lod.px_scale(height),
    )
    entry["_adaptive_drawn"] = drawn
    entry["_cut_ruled"] = True
    cached = entry.get("_cut_entry")
    if changes or "_cut_entry" not in entry:
        labels = adaptive.cut_labels(hierarchy, drawn, coords.shape[0])
        cached = batches.coarse_from_labels(ctx, labels, stand_in=True)
        entry["_cut_entry"] = cached
    if cached is None:
        return None

    entry["_adaptive_px"] = _pixel_radius(
        obj, _cached_center(entry, coords, "_cut_center"),
        cached["radius_world"], persp_mat, height)
    return cached["entry"]


def _np_center(coords):
    from mathutils import Vector
    if coords is None or coords.shape[0] == 0:
        return None
    c = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
    return Vector((float(c[0]), float(c[1]), float(c[2])))


def _cached_center(store, coords, key="_center"):
    """Cached bounding-box center, keyed on the coordinate array itself so a
    swapped array recomputes and an animated frame that reuses one does not.
    Uncached this walked every coordinate per frame: 41 ms at 1M nodes, against
    0.9 ms for the draw it was sizing."""
    cached = store.get(key)
    if cached is not None and cached[0] is coords:
        return cached[1]
    center = _np_center(coords)
    store[key] = (coords, center)
    return center


def _entry_center(entry):
    return _cached_center(entry, entry.get("coords"))


def _select_node_rep(scene, pixel_radius, st=None):
    st = _settings(scene, st)
    style = st.node_style
    if style != 'AUTO':
        return style  # 'POINT' / 'DISK' / 'SPHERE'
    # AUTO holds the impostor to the disk threshold: depth where tubes converge.
    tier = lod.node_tier(pixel_radius, sphere_min=4.0)
    if tier == lod.NODE_HIDE \
            and bool(st.density_fallback):
        return 'DENSITY'
    return {
        lod.NODE_SPHERE: 'SPHERE',
        lod.NODE_DISK: 'DISK',
        lod.NODE_POINT: 'POINT',
        lod.NODE_HIDE: 'HIDE',
    }[tier]


def _select_edge_rep(scene, pixel_radius, st=None):
    st = _settings(scene, st)
    style = st.edge_style
    if style != 'AUTO':
        return style  # 'LINE' / 'RIBBON' / 'NONE'
    # Hairlines to 0.4 ref px, so structure outlives visible nodes.
    tier = lod.edge_tier(pixel_radius, line_min=0.4)
    return {
        lod.EDGE_RIBBON: 'RIBBON',
        lod.EDGE_LINE: 'LINE',
        lod.EDGE_HIDE: 'NONE',
    }[tier]


def _lighting_ubo(lighting, with_rim):
    """A std140 LightRig buffer. Keep it referenced or it frees mid-draw.

    Only key_col.w carries a value; the other four .w are padding."""
    rim = float(lighting["rim"]) if with_rim else 0.0
    kd = lighting["key_dir"]
    kc = lighting["key_col"]
    fd = lighting["fill_dir"]
    fc = lighting["fill_col"]
    amb = lighting["ambient"]
    data = np.array([
        kd[0], kd[1], kd[2], 0.0,
        kc[0], kc[1], kc[2], rim,
        fd[0], fd[1], fd[2], 0.0,
        fc[0], fc[1], fc[2], 0.0,
        amb[0], amb[1], amb[2], 0.0,
    ], dtype=np.float32)
    buf = gpu.types.Buffer('FLOAT', data.size, data)
    return gpu.types.GPUUniformBuf(buf)


# Hardcoded: the driver clamps silently, and probing in a bound framebuffer
# returns 1. Measured, 1/8/16/32/64 arrive exact. Device pixels, so at 5.33x
# supersampling the thickest class capped at 1.8 final px against 1.0 unweighted.
MAX_LINE_WIDTH = 64.0


def max_line_width():
    return MAX_LINE_WIDTH


# Below this tube radius (device px) a ribbon shows no thickness; use lines.
_RIBBON_MIN_PX = 0.75


def _width_range(scene, base_w, scale, st=None):
    """Device-pixel widths for the lightest and heaviest edge. Past the ceiling
    the range slides down instead of clipping; the floor is half a final pixel."""
    st = _settings(scene, st)
    wmax = max(1.0, float(st.edge_size_max_mult))
    ceiling = max_line_width()
    w_lo, w_hi = base_w, base_w * wmax
    if w_hi > ceiling:
        w_hi = ceiling
        w_lo = ceiling / wmax
    floor = 0.5 * max(scale, 1.0)
    if w_lo < floor:
        w_lo = floor
        w_hi = max(w_hi, w_lo * 1.5)
    return w_lo, w_hi


def _filtered(entry):
    return entry.get("filter") is not None


def _animated(entry):
    return entry.get("dynamic") is not None


def _bind_pos(shader, entry):
    if _animated(entry):
        dynamic.bind(shader, entry["dynamic"])


def _bind_filter(shader, entry, scene):
    """Thresholds for this draw, None if unfiltered. The buffer must outlive it."""
    if not _filtered(entry):
        return None
    return filter_gpu.bind(shader, scene, entry["filter"])


def _toon_pass(st, getter, edges=False, **kwargs):
    """The stylized program for one rep, plus its param buffer. ``(None, None)``
    means draw as before, so this must never raise. ``getter`` is a name because
    importing toon.shaders builds the whole GLSL library."""
    from . import toon
    from .toon import params as toon_params
    if not (toon_params.stylized_edges(st) if edges else toon.is_stylized(st)):
        return None, None
    from .toon import shaders as toon_shaders
    shader = getattr(toon_shaders, getter)(**kwargs)
    if shader is None:
        return None, None
    return shader, toon_params.toon_ubo(st)


def _draw_nodes(entry, scene, node_rep, modelview, proj, lighting=None,
                px=None, scale=1.0, st=None):
    st = _settings(scene, st)
    reps = entry["reps"]
    node_size = float(st.node_size)
    size_max_mult = float(st.size_max_mult)

    if node_rep == 'DENSITY':
        # 1-px additive, no depth writes: sub-pixel nodes make a brightness map.
        prev_mask = gpu.state.depth_mask_get()
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set('ADDITIVE')
        shader = entry["point_shader"]
        shader.bind()
        if entry["point_kind"] == 'ROUND':
            shader.uniform_float("u_mvp", proj @ modelview)
        fbuf = _bind_filter(shader, entry, scene)  # noqa: F841 - kept alive
        _bind_pos(shader, entry)
        gpu.state.point_size_set(max(1.0, scale))
        for batch, _mid in reps["point"]:
            batch.draw(shader)
        gpu.state.point_size_set(1.0)
        gpu.state.depth_mask_set(prev_mask)
        return

    if node_rep == 'SPHERE' and reps.get("sphere") is not None:
        gpu.state.blend_set('NONE')
        from . import shaders as _sh
        # The stylized sphere declares _build_sphere's layout, so no rebuild.
        shader, tbuf = _toon_pass(st, "get_toon_sphere_shader",
                                  filtered=_filtered(entry),
                                  animated=_animated(entry))
        if shader is None:
            shader = _sh.get_sphere_shader(filtered=_filtered(entry),
                                           animated=_animated(entry))
        shader.bind()
        shader.uniform_float("u_view", modelview)
        shader.uniform_float("u_proj", proj)
        if lighting is None:
            lighting = compute_lighting(scene, None, st=st)
        # The toon LIT branch matches term for term, so the rig needs its rim.
        ubo = _lighting_ubo(lighting, with_rim=True)
        shader.uniform_block("u_light", ubo)
        if tbuf is not None:  # kept alive until after the draw, like ubo
            shader.uniform_block("u_toon", tbuf)
        fbuf = _bind_filter(shader, entry, scene)  # noqa: F841 - kept alive
        _bind_pos(shader, entry)
        reps["sphere"].draw(shader)
        return

    # POINT / DISK, and SPHERE with no impostor batch. AUTO follows the projected
    # world radius, so transitions are continuous.
    gpu.state.blend_set('ALPHA')
    shader = entry["point_shader"]
    # Glyphs reach this tier only via the round point; an undeclared builtin
    # layout draws silently wrong.
    round_point = entry["point_kind"] == 'ROUND'
    tshader, tbuf = _toon_pass(st, "get_toon_point_shader",
                               filtered=_filtered(entry),
                               animated=_animated(entry)) \
        if round_point else (None, None)
    if tshader is not None:
        shader = tshader
    shader.bind()
    if round_point:
        mvp = proj @ modelview
        shader.uniform_float("u_mvp", mvp)
        if tbuf is not None:  # kept alive until after the draws below
            shader.uniform_block("u_toon", tbuf)
    fbuf = _bind_filter(shader, entry, scene)  # noqa: F841 - kept alive
    _bind_pos(shader, entry)
    if st.node_style == 'AUTO' and px is not None:
        base_px = min(max(2.0 * px, 1.0), 255.0)
    else:
        base_px = node_size * scale
        if node_rep == 'POINT':
            base_px = max(1.0, base_px * 0.5)
    for batch, mid in reps["point"]:
        size = base_px if mid is None \
            else base_px * (1.0 + mid * (size_max_mult - 1.0))
        gpu.state.point_size_set(min(max(size, 1.0), 255.0))
        batch.draw(shader)
    gpu.state.point_size_set(1.0)


def _blend_edges(additive, xray=None):
    """Blend state for one blended edge tier.

    The x-ray pass forces ALPHA whatever the scene says. Dimming an additive
    draw still *adds* light, and light added on top of an occluder reads as in
    front of it, the opposite of what the pass is for.

    Note what is deliberately absent. These tiers blend and still leave the
    depth mask on, where ``_draw_nodes`` clears it for its DENSITY tier. The
    mask is a real defect (additive is order-independent only while nothing
    writes depth, so the first edge over a pixel hides every edge behind it),
    but it is half of one. Edges draw *before* nodes, so clearing it here moves
    the error instead of removing it: every node behind an edge starts painting
    over that edge. Measured on a 180-node, 529-edge cluster, clearing it for
    the additive tiers changed 1059 pixels, 939 of them a node erasing the edge
    in front of it. Correct blended edges need the mask off *and* the blended
    tiers drawn after the nodes, which is a reorder of ``draw_graph_object``."""
    if xray is not None:
        gpu.state.blend_set('ALPHA')
    else:
        gpu.state.blend_set('ADDITIVE' if additive else 'ALPHA')


def _edge_color(st, xray=None):
    """The scene edge color, its alpha scaled for the x-ray pass."""
    c = tuple(st.edge_color)
    if xray is None:
        return c
    return (c[0], c[1], c[2], c[3] * xray)


def _draw_styled(entry, scene, data, edge_rep, modelview, proj, lighting,
                 line_width, scale, st=None, xray=None):
    """The six closed-form edge styles, one instance per edge over the position
    texture. Self-loops are a second pass with their own style code and span
    count, as ``tessellate`` gives them their own branch."""
    from . import shaders as _sh
    st = _settings(scene, st)
    ribbon = edge_rep == 'RIBBON'
    wide = bool(data["weighted"])
    shader = _sh.get_style_ribbon_shader() if ribbon \
        else _sh.get_style_line_shader(wide=wide)
    if shader is None:
        return False

    light = None
    if ribbon:
        amount = float(np.clip(st.edge_taper_amount, 0.0, 1.0))
        width = (data["radius"], 0.0)
        end_taper = (1.0 - 0.9 * amount, st.edge_taper)
        gpu.state.blend_set('NONE')
        if lighting is None:
            lighting = compute_lighting(scene, None, st=st)
        light = _lighting_ubo(lighting, with_rim=False)
    else:
        end_taper = (1.0, 'NONE')
        _blend_edges(bool(st.additive_edges), xray)
        if wide:
            width = _width_range(scene, line_width, scale, st=st)
            entry["_width_range"] = (width[0], width[1], 0)
        else:
            width = (line_width, line_width)
            gpu.state.line_width_set(line_width)

    mvp = proj @ modelview
    vp = gpu.state.viewport_get()
    color = _edge_color(st, xray)
    drew = False
    for key, style in (("plain", data["style"]), ("loops", 'SELF_LOOP')):
        sub = data.get(key)
        if sub is None:
            continue
        ubo = _sh.style_uniform_block(data["params"], color, width=width,
                                      end_taper=end_taper, style=style)
        shader.bind()
        shader.uniform_block("u_style", ubo)
        shader.uniform_sampler("u_edge", sub["tex"])
        dynamic.bind(shader, entry["dynamic"])
        if ribbon:
            shader.uniform_float("u_view", modelview)
            shader.uniform_float("u_proj", proj)
            shader.uniform_block("u_light", light)
        else:
            shader.uniform_float("u_mvp", mvp)
            if wide:
                shader.uniform_float("u_viewport", (float(vp[2]), float(vp[3])))
        batch = sub["quad"] if (ribbon or wide) else sub["line"]
        batch.draw_instanced(shader, instance_count=sub["instances"])
        drew = True

    if not (ribbon or wide):
        gpu.state.line_width_set(1.0)
    return drew


def _draw_edges(entry, scene, edge_rep, modelview, proj, lighting=None,
                px=None, scale=1.0, st=None, xray=None):
    """``xray`` is the opacity multiplier for the occluded-stretch pass, or None
    for the ordinary draw. It only reaches the tiers whose color is a shader
    uniform; see :func:`_draw_edges_xray`."""
    st = _settings(scene, st)
    reps = entry["reps"]
    edge_color = _edge_color(st, xray)
    edge_width = float(st.edge_width)
    additive = bool(st.additive_edges)
    vertex_colored = bool(entry.get("line_vertex_color"))

    def _line_width():
        # AUTO matches the projected tube thickness, so LINE is a thin ribbon.
        if st.edge_style == 'AUTO' and px is not None:
            return min(max(2.0 * px, 0.75), 10.0 * scale)
        return edge_width * scale

    # A sub-pixel tube shows no thickness; width-class lines still do. Viewport
    # and render get different `px`, so one scene sits on either side.
    ribbon_too_thin = (
        px is not None
        and (reps.get("line_buckets") or reps.get("style"))
        and float(px) < _RIBBON_MIN_PX
    )
    if ribbon_too_thin:
        edge_rep = 'LINE'

    drew_edges = False
    if edge_rep == 'RIBBON' and reps.get("ribbon") is not None:
        # Opaque: the edge alpha is a density hint for flat LINE only.
        gpu.state.blend_set('NONE')
        from . import shaders as _sh
        shader, tbuf = _toon_pass(st, "get_toon_ribbon_shader", edges=True,
                                  filtered=_filtered(entry),
                                  animated=_animated(entry))
        if shader is None:
            shader = _sh.get_ribbon_shader(filtered=_filtered(entry),
                                           animated=_animated(entry))
        shader.bind()
        shader.uniform_float("u_view", modelview)
        shader.uniform_float("u_proj", proj)
        if lighting is None:
            lighting = compute_lighting(scene, None, st=st)
        # The lit ribbon has no rim term, so the LIT branch matches only at zero.
        ubo = _lighting_ubo(lighting, with_rim=False)
        shader.uniform_block("u_light", ubo)
        if tbuf is not None:  # kept alive until after the draw, like ubo
            shader.uniform_block("u_toon", tbuf)
        fbuf = _bind_filter(shader, entry, scene)  # noqa: F841 - kept alive
        _bind_pos(shader, entry)
        reps["ribbon"].draw(shader)
        drew_edges = True

    if not drew_edges and reps.get("style") is not None:
        drew_edges = _draw_styled(entry, scene, reps["style"], edge_rep,
                                  modelview, proj, lighting, _line_width(),
                                  scale, st=st, xray=xray)

    if not drew_edges and reps.get("bundle") is not None and xray is None:
        _blend_edges(additive, xray)
        gpu.state.line_width_set(_line_width())
        from . import bundle_gpu as _bundle
        bundle_data = reps["bundle"]
        # Holten's equation 1 when hierarchical, elsewhere the pull straight.
        beta = float(scene.scigraphs.edge_heb_beta)
        # The only setting reading the camera without invalidating anything.
        view_amount = float(getattr(scene.scigraphs, "edge_bundle_view_beta", 0.0))
        if view_amount > 0.0:
            vp = gpu.state.viewport_get()
            beta = _bundle.view_beta(
                bundle_data, beta, view_amount, proj @ modelview,
                (float(vp[2]), float(vp[3])))
        if bundle_data.get("width_tex") is not None:
            w_lo, w_hi = _width_range(scene, _line_width(), scale, st=st)
            entry["_width_range"] = (w_lo, w_hi, 0)
            vp = gpu.state.viewport_get()
            drew_edges = _bundle.draw(
                bundle_data, proj @ modelview, beta,
                width_range=(w_lo, w_hi),
                viewport=(float(vp[2]), float(vp[3])))
        else:
            drew_edges = _bundle.draw(bundle_data, proj @ modelview, beta)
        gpu.state.line_width_set(1.0)

    if not drew_edges and reps.get("line_buckets") \
            and not (xray is not None and vertex_colored):
        _blend_edges(additive, xray)
        w_lo, w_hi = _width_range(scene, _line_width(), scale, st=st)
        entry["_width_range"] = (w_lo, w_hi, len(reps["line_buckets"]))

        # Not line_width_set: it flattened weighted graphs. Both take a uniform.
        vp = gpu.state.viewport_get()
        viewport = (float(vp[2]), float(vp[3]))
        shader = entry["line_shader"]
        shader.bind()
        animated = _animated(entry)
        if animated:
            shader.uniform_float("u_mvp", proj @ modelview)
            shader.uniform_float("u_color", edge_color)
            shader.uniform_float("u_viewport", viewport)
            _bind_pos(shader, entry)
        else:
            if not entry.get("line_vertex_color"):
                shader.uniform_float("color", edge_color)
            shader.uniform_float("viewportSize", viewport)
        width_uniform = "u_width" if animated else "lineWidth"
        for batch, mid in reps["line_buckets"]:
            shader.uniform_float(width_uniform, w_lo + mid * (w_hi - w_lo))
            batch.draw(shader)
        gpu.state.line_width_set(1.0)
        drew_edges = True

    if not drew_edges and reps.get("line") is not None \
            and not (xray is not None and vertex_colored):
        _blend_edges(additive, xray)
        gpu.state.line_width_set(_line_width())
        shader = entry["line_shader"]
        shader.bind()
        # Self-shading styles baked the color in, the rest take the scene edge
        # color. Neither shader is a builtin, so both need the matrix.
        if _filtered(entry) or _animated(entry):
            shader.uniform_float("u_mvp", proj @ modelview)
            shader.uniform_float("u_color", edge_color)
            fbuf = _bind_filter(shader, entry, scene)  # noqa: F841
            _bind_pos(shader, entry)
        elif not entry.get("line_vertex_color"):
            shader.uniform_float("color", edge_color)
        reps["line"].draw(shader)
        gpu.state.line_width_set(1.0)

    if reps.get("arrow") is not None and xray is None:
        gpu.state.blend_set('NONE')
        from . import shaders as _sh
        flat = bool(reps.get("arrow_flat"))
        tbuf = None
        if flat:
            shader = _sh.get_arrow_flat_shader(animated=_animated(entry))
        else:
            # The animated getter returns None, so moving graphs get the lit cone.
            shader, tbuf = _toon_pass(st, "get_toon_arrow_shader", edges=True,
                                      animated=_animated(entry))
            if shader is None:
                shader = _sh.get_arrow_dynamic_shader() if _animated(entry) \
                    else _sh.get_arrow_shader()
        if shader is not None:
            shader.bind()
            shader.uniform_float("u_view", modelview)
            shader.uniform_float("u_proj", proj)
            # A mark, not a surface: no normals, so no lights.
            if not flat:
                if lighting is None:
                    lighting = compute_lighting(scene, None, st=st)
                ubo = _lighting_ubo(lighting, with_rim=False)
                shader.uniform_block("u_light", ubo)
                if tbuf is not None:  # kept alive until after the draw
                    shader.uniform_block("u_toon", tbuf)
            _bind_pos(shader, entry)
            reps["arrow"].draw(shader)


def _draw_edges_xray(entry, scene, modelview, proj, px, scale, st=None):
    """Redraw the stretches that lost the depth test, dimmed instead of gone.

    An edge that dives behind a cluster otherwise just stops and reappears
    somewhere else, and nothing tells the viewer the two stubs are one edge. It
    also makes two edges crossing in projection identical to two edges that
    meet. A faint hidden stretch restores both: you can trace an edge through a
    cluster, and a crossing reads as over-and-under.

    ``GREATER`` passes exactly where the edge is behind what is already in the
    depth buffer, which is exactly the hidden stretch. It has to run after
    ``_draw_nodes``, because the nodes are the geometry doing the occluding.

    Two things this is not. With depth writes off and no sorting, the dim
    fragments accumulate in draw order, so hidden edges get darker where they
    pile up: it reads as density, not as correct transparency. And the pass
    always draws the LINE tier, so a hidden tube comes back as a hairline ghost
    rather than a dim tube, because the impostor writes alpha 1.0 and has no
    uniform to dim.

    Bundled edges and vertex-colored lines are skipped for the same reason; see
    :func:`_draw_edges`."""
    st = _settings(scene, st)
    alpha = float(getattr(st, "edge_xray", EDGE_XRAY))
    prev_test = gpu.state.depth_test_get()
    if alpha <= 0.0 or prev_test == 'NONE':
        return
    prev_mask = gpu.state.depth_mask_get()
    gpu.state.depth_test_set('GREATER')
    gpu.state.depth_mask_set(False)
    try:
        _draw_edges(entry, scene, 'LINE', modelview, proj, None, px, scale,
                    st=st, xray=min(max(alpha, 0.0), 1.0))
    finally:
        gpu.state.depth_test_set(prev_test)
        gpu.state.depth_mask_set(prev_mask)


def draw_blocks(entry, scene, obj, modelview, proj, persp_mat, height, st=None):
    """Frustum cull and budget whole blocks, then draw the cheap reps. This path
    skips ``_draw_nodes``/``_draw_edges``, so Spatial Blocks gets no filter stack,
    position texture, impostor or stylized shading, only lit round points."""
    st = _settings(scene, st)
    flush_positions(entry)
    blocks = entry["blocks"]
    if blocks["count"] == 0 or persp_mat is None:
        return

    mw = np.asarray(obj.matrix_world, dtype=np.float64)
    centers_local = blocks["centers"]
    homog = np.empty((centers_local.shape[0], 4))
    homog[:, :3] = centers_local
    homog[:, 3] = 1.0
    centers_world = (homog @ mw.T)[:, :3]

    persp = np.asarray(persp_mat, dtype=np.float64)
    keep = lod.frustum_cull_spheres(centers_world, blocks["radii"], persp)

    node_budget = int(st.node_budget)
    if node_budget > 0 and height:
        px = lod.projected_pixel_radius(centers_world, blocks["radii"], persp, height)
        px[~keep] = -1.0
        budget_mask = lod.apply_budget(px, blocks["node_counts"], node_budget)
        keep &= budget_mask

    scale = lod.px_scale(height)
    node_size = float(st.node_size) * scale
    edge_color = tuple(st.edge_color)
    edge_width = float(st.edge_width) * scale
    show_edges = bool(st.show_edges)

    visible_idx = np.nonzero(keep)[0]

    if show_edges:
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(max(0.75, edge_width))
        lshader = blocks["line_shader"]
        lshader.bind()
        if not blocks.get("line_vertex_color"):
            lshader.uniform_float("color", edge_color)
        for b in visible_idx:
            lb = blocks["line_batches"][b]
            if lb is not None:
                lb.draw(lshader)
        gpu.state.line_width_set(1.0)

    gpu.state.blend_set('ALPHA')
    pshader = entry["point_shader"]
    pshader.bind()
    if entry["point_kind"] == 'ROUND':
        pshader.uniform_float("u_mvp", proj @ modelview)
    gpu.state.point_size_set(min(max(node_size, 1.0), 255.0))
    for b in visible_idx:
        for batch, _mid in blocks["point_batches"][b]:
            batch.draw(pshader)
    gpu.state.point_size_set(1.0)

    entry["_blocks_drawn"] = int(visible_idx.size)

    base_radius = float(st.impostor_radius)
    px_ref = _pixel_radius(obj, _entry_center(entry), base_radius,
                           persp_mat, height) / max(scale, 1e-9)
    _maybe_draw_volume(entry, scene, modelview, proj,
                       _select_node_rep(scene, px_ref, st=st), st=st)


def _draw_callback():
    context = bpy.context
    if context.area is None or context.area.type != 'VIEW_3D':
        return

    scene = context.scene
    if scene is None:
        return
    st = _settings(scene)  # built once, threaded down the whole chain
    if not st.enabled:
        return

    obj = context.active_object
    if not is_graph_object(obj):
        return

    entry = get_cache_entry(obj, scene)
    if entry is None:
        return

    region = context.region
    rv3d = context.region_data
    depth_test = bool(st.depth_test)

    prev_blend = gpu.state.blend_get()
    if depth_test:
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)

    with gpu.matrix.push_pop():
        gpu.matrix.multiply_matrix(obj.matrix_world)
        modelview = gpu.matrix.get_model_view_matrix()
        proj = gpu.matrix.get_projection_matrix()

        if entry.get("blocks") is not None and region is not None and rv3d is not None:
            draw_blocks(entry, scene, obj, modelview, proj,
                        rv3d.perspective_matrix, region.height, st=st)
        else:
            persp = rv3d.perspective_matrix if rv3d is not None else None
            height = region.height if region is not None else 0
            view_mat = rv3d.view_matrix if rv3d is not None else None
            draw_graph_object(entry, scene, obj, modelview, proj, persp, height,
                              view_matrix=view_mat, st=st)

    if depth_test:
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
    gpu.state.blend_set(prev_blend)


def sync_native_display(context):
    if state.SYNCING or context is None:
        return

    scene = getattr(context, "scene", None)
    st = _settings(scene) if scene is not None else None
    hide = bool(st.hide_mesh) if st is not None else False
    active_on = state.ENABLED and (st is not None and bool(st.enabled))

    desired = None
    if active_on and hide:
        obj = getattr(context, "active_object", None)
        if is_graph_object(obj):
            desired = obj
    desired_name = desired.name if desired is not None else None

    state.SYNCING = True
    try:
        for name in list(state.HIDDEN.keys()):
            if name == desired_name:
                continue
            obj = bpy.data.objects.get(name)
            original = state.HIDDEN.pop(name)
            if obj is not None and obj.display_type != original:
                obj.display_type = original

        if desired is not None and desired_name not in state.HIDDEN:
            state.HIDDEN[desired_name] = desired.display_type
            if desired.display_type != 'BOUNDS':
                desired.display_type = 'BOUNDS'
    finally:
        state.SYNCING = False


def _ensure_depsgraph_handler(add):
    handlers = bpy.app.handlers.depsgraph_update_post
    present = _depsgraph_update_handler in handlers
    if add and not present:
        handlers.append(_depsgraph_update_handler)
    elif not add and present:
        handlers.remove(_depsgraph_update_handler)


def enable_preview():
    if not state.ENABLED:
        state.HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, (), 'WINDOW', 'POST_VIEW'
        )
        state.ENABLED = True
    _ensure_depsgraph_handler(True)
    sync_native_display(bpy.context)
    tag_redraw()


def disable_preview():
    if state.ENABLED and state.HANDLE:
        bpy.types.SpaceView3D.draw_handler_remove(state.HANDLE, 'WINDOW')
        state.HANDLE = None
        state.ENABLED = False
    _ensure_depsgraph_handler(False)
    sync_native_display(bpy.context)
    state.CACHE.clear()
    tag_redraw()


_SYNCING = False


@persistent
def _sync_display_to_render(scene, _depsgraph=None):
    """Point the display path at whatever the render engine needs.

    Persistent because opening a .blend clears every handler that is not, and
    without it the sync worked until the first file load and then silently
    stopped, leaving the mesh drawn as bounds under EEVEE.

    There used to be a second switch in the sidebar for this, which meant two
    controls for one decision: leave it on GPU under EEVEE and the final render
    is empty, because the Geometry Nodes modifier is the thing EEVEE can see.
    depsgraph_update_post fires on a render-engine change, which is the only
    notification Blender offers for it.

    The guard is not cosmetic: assigning the property runs
    apply_display_engine, which writes the preview flag and a modifier, and
    each of those schedules another depsgraph update.
    """
    global _SYNCING
    if _SYNCING or scene is None:
        return
    want = ('GPU' if getattr(scene.render, "engine", "") == 'SCIGRAPHS'
            else 'GEOMETRY_NODES')
    if getattr(scene, "scigraphs_display_engine", want) == want:
        return
    _SYNCING = True
    try:
        scene.scigraphs_display_engine = want
    finally:
        _SYNCING = False



def ensure_display_sync(add):
    handlers = bpy.app.handlers.depsgraph_update_post
    present = _sync_display_to_render in handlers
    if add and not present:
        handlers.append(_sync_display_to_render)
    elif not add and present:
        handlers.remove(_sync_display_to_render)


@persistent
def _on_frame_change(scene):
    """Colours baked from a mesh attribute go stale when something animates it.

    Nothing else notices. The batch signature carries the attribute's *name*,
    not its values, and fingerprinting the values costs 56 ms at 262k nodes,
    far too much to pay on every draw. A frame change is the only moment they
    can have moved, so the cost is paid there and only for objects something
    actually animates.

    Not gated on the colour mode. It was, and that made the whole thing a no-op
    for anyone who had not switched to Attribute + Colormap by hand, which is
    not the default: the colours only refreshed when some other setting forced
    a rebuild.
    """
    if not state.ENABLED or not state.CACHE:
        return
    for obj in scene.objects:
        if not is_graph_object(obj):
            continue
        if "traversal_time" in obj:
            continue
        if obj.animation_data is None:
            continue
        try:
            from . import playback
            if playback.is_running(obj):
                continue
        except Exception:  # noqa: BLE001 - a handler must not break playback
            pass
        invalidate(obj)


def ensure_frame_handler(add):
    handlers = bpy.app.handlers.frame_change_post
    present = _on_frame_change in handlers
    if add and not present:
        handlers.append(_on_frame_change)
    elif not add and present:
        handlers.remove(_on_frame_change)


@persistent
def _on_load_post(*_args):
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return
    st = _settings(scene)
    if bool(st.enabled):
        enable_preview()
    else:
        disable_preview()


@persistent
def _depsgraph_update_handler(scene, depsgraph):
    if not state.ENABLED:
        return

    sync_native_display(bpy.context)

    if not state.CACHE:
        return

    dirty = False
    for update in depsgraph.updates:
        id_data = update.id
        if not isinstance(id_data, bpy.types.Object) or id_data.type != 'MESH':
            continue
        if update.is_updated_geometry or update.is_updated_transform:
            original = id_data.original
            if update.is_updated_geometry:
                batches.drop_geometry_cache(original)
            # A bundle reading positions from a texture absorbs this. The branch
            # below drops batches, tree and channels: over 1 s/frame at 2M nodes.
            if refresh_positions(original):
                dirty = True
                continue
            state.CACHE.pop(original.as_pointer(), None)
            batches.drop_tree_cache(original)
            # Not in ``invalidate``: only this says the graph itself moved, and
            # edge length is measured on that.
            filters.drop_channel_cache(original)
            filter_gpu.drop_cache(original)
            dirty = True

    if dirty:
        tag_redraw()


def apply_display_engine(context):
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    engine = getattr(scene, "scigraphs_display_engine", 'GPU')
    obj = getattr(context, "active_object", None)

    if engine == 'GPU':
        if is_graph_object(obj):
            mod = obj.modifiers.get("SciGraphs_Viz")
            if mod is not None and mod.show_viewport:
                mod.show_viewport = False
        if not scene.scigraphs_preview_enabled:
            scene.scigraphs_preview_enabled = True
        else:
            enable_preview()
    else:  # GEOMETRY_NODES / CPU
        if scene.scigraphs_preview_enabled:
            scene.scigraphs_preview_enabled = False
        else:
            disable_preview()
        if is_graph_object(obj):
            mod = obj.modifiers.get("SciGraphs_Viz")
            if mod is None:
                try:
                    from ...core.mesh.geometry import setup_geometry_nodes_visualization
                    setup_geometry_nodes_visualization(obj)
                    mod = obj.modifiers.get("SciGraphs_Viz")
                except Exception:  # noqa: BLE001
                    mod = None
            if mod is not None and not mod.show_viewport:
                mod.show_viewport = True
    tag_redraw()
