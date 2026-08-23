# Per-element coverage counting from an ID pass. The sphere impostor writes
# depth and encodes its index in 24 bits, so it occludes exactly as the beauty
# pass does. Depth comes from a prepass over the whole scene, since node-vs-node
# underestimates badly, and radii add the term lod.projected_pixel_radius omits.

import sys
import time

import bpy
import gpu
import numpy as np

from . import batches, shaders
from ...core.render import adaptive
from .draw import draw_graph_object, get_cache_entry, invalidate
from .state import is_graph_object, tag_redraw
def _settings(scene, st=None):
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)


# Below this radius pi*r^2 is not a meaningful denominator, so count binary.
MIN_RATIO_PX = 2.0

MIN_EDGE_PX = 12.0

# Occlusion is low-frequency, and each halving quarters the readback.
DEFAULT_DIVISOR = 2


def graph_objects(scene):
    objs = []
    for obj in scene.objects:
        if not is_graph_object(obj) or obj.hide_viewport:
            continue
        try:
            if obj.hide_get():
                continue
        except RuntimeError:
            pass  # not in the active view layer; treat as drawable
        objs.append(obj)
    return objs


def impostor_radius_px(radii, w_clip, focal_y, height):
    """Impostor radius in px. The NDC half-extent is ``proj[1][1] * radius / |w|``."""
    return focal_y * np.asarray(radii, dtype=np.float64) / w_clip * 0.5 * height


def _decode_ids(raw, count):
    """Foreground ids from an RGBA8 readback. Every fragment writes alpha 1, so
    a nonzero word is foreground and its low 24 bits are the id."""
    flat = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8).reshape(-1))
    if flat.size < count * 4:
        return np.empty(0, dtype=np.int64)
    if sys.byteorder == 'little':
        u32 = flat[:count * 4].view(np.uint32)
        fg = u32 != 0
        return (u32[fg] & 0x00FFFFFF).astype(np.int64)
    px = flat[:count * 4].reshape(count, 4).astype(np.uint32)
    ids = px[:, 0] | (px[:, 1] << 8) | (px[:, 2] << 16)
    return ids[px[:, 3] != 0].astype(np.int64)


def _id_image(raw, count, width, height):
    """Same decode as ``_decode_ids``, but keeps the image shape; background -1."""
    flat = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8).reshape(-1))
    if flat.size < count * 4:
        return np.full((height, width), -1, dtype=np.int64)
    px = flat[:count * 4].reshape(count, 4).astype(np.int64)
    ids = px[:, 0] | (px[:, 1] << 8) | (px[:, 2] << 16)
    ids[px[:, 3] == 0] = -1
    return ids.reshape(height, width)


def _project_px(pts_view, proj, width, height):
    """View-space points to pixels, plus a front-of-eye mask callers must read."""
    homog = np.empty((pts_view.shape[0], 4), dtype=np.float64)
    homog[:, :3] = pts_view
    homog[:, 3] = 1.0
    clip = homog @ np.asarray(proj, dtype=np.float64).T
    w = clip[:, 3]
    front = w > 1e-6
    safe = np.where(front, w, 1.0)
    px = np.empty((pts_view.shape[0], 2), dtype=np.float64)
    px[:, 0] = (clip[:, 0] / safe * 0.5 + 0.5) * width
    px[:, 1] = (clip[:, 1] / safe * 0.5 + 0.5) * height
    return px, front


def _edge_stats(entry, counts, modelview, proj, width, height,
                min_px=MIN_EDGE_PX):
    """Per-edge visibility from the segment-id histogram. The denominator is the
    projected quad's shoelace area, rebuilt the way the vertex shader builds it.
    It is exact because the ribbon fragment shader discards nothing."""
    geom = entry.get("seg_geom")
    if geom is None:
        return None
    a_world, b_world, radius_a, radius_b, owner = geom
    n_seg = a_world.shape[0]
    if n_seg == 0:
        return None

    mv = np.asarray(modelview, dtype=np.float64)
    av = a_world @ mv[:3, :3].T + mv[:3, 3]
    bv = b_world @ mv[:3, :3].T + mv[:3, 3]
    ra = np.asarray(radius_a, dtype=np.float64)[:, None]
    rb = np.asarray(radius_b, dtype=np.float64)[:, None]

    direction = bv - av
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.divide(direction, np.maximum(length, 1e-12))

    # offset = normalize(cross(dir, view_z)), with the shader's end-on fallback.
    offset = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    olen = np.linalg.norm(offset, axis=1, keepdims=True)
    fallback = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    flen = np.linalg.norm(fallback, axis=1, keepdims=True)
    offset = np.where(olen > 1e-6, offset / np.maximum(olen, 1e-12),
                      fallback / np.maximum(flen, 1e-12))

    cap_a = av - direction * (ra * 0.5)
    cap_b = bv + direction * (rb * 0.5)
    corners = [cap_a - offset * ra, cap_b - offset * rb,
               cap_b + offset * rb, cap_a + offset * ra]

    px = np.empty((4, n_seg, 2), dtype=np.float64)
    front = np.ones(n_seg, dtype=bool)
    for i, c in enumerate(corners):
        px[i], ok = _project_px(c, proj, width, height)
        front &= ok

    area = np.zeros(n_seg, dtype=np.float64)
    for i in range(4):
        j = (i + 1) % 4
        area += px[i][:, 0] * px[j][:, 1] - px[j][:, 0] * px[i][:, 1]
    area = np.abs(area) * 0.5

    inside = front.copy()
    for i in range(4):
        inside &= ((px[i][:, 0] >= 0.0) & (px[i][:, 0] <= width)
                   & (px[i][:, 1] >= 0.0) & (px[i][:, 1] <= height))

    visible_seg = counts[:n_seg] if counts.size >= n_seg else \
        np.pad(counts, (0, n_seg - counts.size))

    n_edges = int(owner.max()) + 1 if owner.size else 0
    keep = inside & (area >= min_px)
    visible = np.bincount(owner, weights=np.where(keep, visible_seg, 0.0),
                          minlength=n_edges)
    potential = np.bincount(owner, weights=np.where(keep, area, 0.0),
                            minlength=n_edges)
    measurable = potential > 0.0
    n_measurable = int(measurable.sum())
    if not n_measurable:
        return None

    ratio = np.full(n_edges, np.nan)
    np.divide(visible, potential, out=ratio, where=measurable)
    np.minimum(ratio, 1.0, out=ratio, where=measurable)
    m_ratio = ratio[measurable]

    return {
        "segments": n_seg,
        "edges": n_edges,
        "measurable": n_measurable,
        "fully_occluded": int(np.count_nonzero(visible[measurable] == 0)),
        "below_half": int(np.count_nonzero(m_ratio < 0.5)),
        "mean_ratio": float(m_ratio.mean()),
        "median_ratio": float(np.median(m_ratio)),
        "area_ratio": float(visible[measurable].sum()
                            / potential[measurable].sum()),
        "visible_px": float(visible[measurable].sum()),
        "potential_px": float(potential[measurable].sum()),
        "ratio": ratio,
    }


def _stats(obj, entry, counts, persp, focal_y, width, height,
           min_ratio_px=MIN_RATIO_PX, full=True):
    """Per-node visibility from the id-pixel histogram. Runs every frame."""
    point_idx = entry["point_idx"]
    radii = entry.get("radii")
    if radii is None or point_idx.size == 0:
        return None

    # Cached on the entry; gathering and padding this every frame was the cost.
    homog = entry.get("_vis_homog")
    if homog is None or homog.shape[0] != point_idx.size:
        homog = np.empty((point_idx.size, 4), dtype=np.float32)
        homog[:, :3] = entry["coords"][point_idx]
        homog[:, 3] = 1.0
        entry["_vis_homog"] = homog

    mvp = (np.asarray(persp, dtype=np.float64)
           @ np.asarray(obj.matrix_world, dtype=np.float64))
    clip = homog @ mvp.T.astype(np.float32)

    w_clip = clip[:, 3]
    front = w_clip > 1e-6
    aw = np.where(front, w_clip, np.float32(1.0))

    r_px = impostor_radius_px(radii, aw, focal_y, height).astype(np.float32)
    cx = (clip[:, 0] / aw * 0.5 + 0.5) * width
    cy = (clip[:, 1] / aw * 0.5 + 0.5) * height
    inside = (front
              & (cx - r_px >= 0.0) & (cx + r_px <= width)
              & (cy - r_px >= 0.0) & (cy + r_px <= height))

    visible = counts[point_idx]
    potential = np.square(r_px) * np.float32(np.pi)
    measurable = inside & (r_px >= min_ratio_px)
    ratio = np.full(point_idx.size, np.nan, dtype=np.float32)
    np.divide(visible, potential, out=ratio, where=measurable)
    # A rasterized disk can win marginally more pixel centers than pi*r^2.
    np.minimum(ratio, np.float32(1.0), out=ratio, where=measurable)

    n_measurable = int(measurable.sum())
    out = {
        "name": obj.name,
        "nodes": int(point_idx.size),
        "onscreen": int(inside.sum()),
        "measurable": n_measurable,
        "fully_occluded": int((inside & (visible == 0)).sum()),
        "visible_px": float(visible.sum()),
        # Area-weighted, so the many tiny nodes do not dominate the mean.
        "visible_px_m": float(np.where(measurable, visible, 0).sum()),
        "potential_px_m": float(np.where(measurable, potential, 0.0).sum()),
        "below_half": int(np.count_nonzero(ratio < 0.5)),
        "mean_ratio": (float(np.nansum(ratio)) / n_measurable
                       if n_measurable else float('nan')),
        "coarse_active": bool(entry.get("_coarse_active")),
    }
    out["offscreen"] = out["nodes"] - out["onscreen"]
    if not full:
        return out

    m_ratio = ratio[measurable]
    out["median_ratio"] = (float(np.median(m_ratio)) if m_ratio.size
                           else float('nan'))
    out["median_radius_px"] = (float(np.median(r_px[inside])) if inside.any()
                               else 0.0)
    out["ratio"] = ratio
    out["radius_px"] = r_px
    out["inside"] = inside
    out["center_px"] = np.stack([cx, cy], axis=1)
    return out


def feed_back(scene, result, view, st=None):
    """Turn a measurement into collapses where the cut is active. It sees cover
    by unrelated parts of the graph, which a local test cannot. Only collapses,
    and only for this camera's direction."""
    st = _settings(scene, st)
    view = np.asarray(view, dtype=np.float64)
    forward = -view[2, :3]
    added = 0
    for obj, entry in result.get("pairs", ()):
        drawn = entry.get("_adaptive_drawn")
        state = entry.get("_cut_state")
        hierarchy = entry.get("hierarchy")
        if drawn is None or state is None or not hierarchy:
            continue
        stats = next((s for s in result["objects"] if s["name"] == obj.name), None)
        if stats is None or stats.get("ratio") is None:
            continue

        # The stats cover a strided subset; scatter back onto the population.
        measured = entry.get("_cut_entry") if entry.get("_adaptive_active") \
            else None
        measured = (measured or {}).get("entry") or entry
        total = sum(int(d.sum()) for d in drawn)
        ratio = np.full(total, np.nan, dtype=np.float64)
        point_idx = np.asarray(measured["point_idx"], dtype=np.int64)
        keep = point_idx < total
        ratio[point_idx[keep]] = np.asarray(stats["ratio"], dtype=np.float64)[keep]

        # The controller works in object space, so bring the camera along.
        to_obj = np.asarray(obj.matrix_world.inverted(), dtype=np.float64)
        obj_forward = to_obj[:3, :3] @ forward
        obj_forward /= np.linalg.norm(obj_forward) + 1e-12

        # Too-small or never-drawn come back NaN, which is not "hidden".
        indices, per_level = adaptive.split_by_level(
            hierarchy, drawn, np.nan_to_num(ratio, nan=1.0))
        added += adaptive.apply_measurement(
            hierarchy, state, indices, per_level, obj_forward,
            max_measured=float(st.adaptive_collapse),
            max_merge_loss=float(st.adaptive_merge_loss),
        )
    return added


def measure(scene, view, proj, persp, width, height, divisor=DEFAULT_DIVISOR,
            min_ratio_px=MIN_RATIO_PX, full=True, keep_ids=False, st=None):
    """Per-node visibility for every graph object in ``scene``, the ID buffer at
    ``1/divisor`` of width x height. Returns ``objects``, ``timings`` (ms) and
    the buffer size, or None with no GPU context. Readback is the sync point,
    so it absorbs the GPU time."""
    st = _settings(scene, st)
    id_shader = shaders.get_sphere_id_shader()
    edge_id_shader = shaders.get_ribbon_id_shader()
    if id_shader is None:
        return None

    pairs = []
    for obj in graph_objects(scene):
        entry = get_cache_entry(obj, scene)
        if entry is None:
            continue
        pairs.append((obj, entry))
    if not pairs:
        return None

    div = max(1, int(divisor))
    bw, bh = max(2, int(width) // div), max(2, int(height) // div)
    focal_y = float(np.asarray(proj, dtype=np.float64)[1][1])
    npx = bw * bh

    prepass_ms = id_ms = read_ms = decode_ms = stats_ms = 0.0
    results = []
    t_start = time.perf_counter()

    offscreen = gpu.types.GPUOffScreen(bw, bh)
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            prev_blend = gpu.state.blend_get()

            t0 = time.perf_counter()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            # Draw what the beauty pass picked, or fat tubes steal node pixels.
            drawn_rep = {}
            for obj, entry in pairs:
                with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                    gpu.matrix.load_matrix(view @ obj.matrix_world)
                    gpu.matrix.load_projection_matrix(proj)
                    drawn_rep[obj.name] = draw_graph_object(
                        entry, scene, obj,
                        gpu.matrix.get_model_view_matrix(),
                        gpu.matrix.get_projection_matrix(),
                        persp, bh, view_matrix=view,
                    )
            gpu.state.depth_mask_set(False)
            gpu.state.blend_set('NONE')
            prepass_ms = (time.perf_counter() - t0) * 1000.0

            # One pass per object, since ids are per-object indices. Nodes and
            # edges share the pass through EDGE_ID_FLAG and resolve by depth.
            show_edges = bool(st.show_edges)
            for obj, entry in pairs:
                # With a cut on screen the original nodes are not in the image.
                measured = entry.get("_cut_entry") if \
                    entry.get("_adaptive_active") else None
                measured = (measured or {}).get("entry") or entry
                node_batch = measured["reps"].get("sphere_id")
                edge_rep = (drawn_rep.get(obj.name) or (None, None))[1]
                edge_batch = measured["reps"].get("ribbon_id") \
                    if show_edges and edge_rep == 'RIBBON' else None
                if node_batch is None and edge_batch is None:
                    continue
                t0 = time.perf_counter()
                fb.clear(color=(0.0, 0.0, 0.0, 0.0))  # keeps the depth buffer
                id_passes = [(shader, batch) for shader, batch
                             in ((id_shader, node_batch),
                                 (edge_id_shader, edge_batch))
                             if shader is not None and batch is not None]
                with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                    gpu.matrix.load_matrix(view @ obj.matrix_world)
                    gpu.matrix.load_projection_matrix(proj)
                    modelview = gpu.matrix.get_model_view_matrix()
                    projection = gpu.matrix.get_projection_matrix()
                    for shader, batch in id_passes:
                        shader.bind()
                        shader.uniform_float("u_view", modelview)
                        shader.uniform_float("u_proj", projection)
                        batch.draw(shader)
                id_ms += (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                raw = fb.read_color(0, 0, bw, bh, 4, 0, 'UBYTE')
                t1 = time.perf_counter()
                # Ids index whatever was drawn: vertices, or the cut's elements.
                n_ids = int(measured["point_idx"].size) if measured is not entry \
                    else len(obj.data.vertices)
                ids = _decode_ids(raw, npx)
                is_edge = (ids & batches.EDGE_ID_FLAG) != 0
                node_ids = ids[~is_edge]
                seg_ids = ids[is_edge] & (batches.EDGE_ID_FLAG - 1)
                counts = np.bincount(node_ids, minlength=n_ids) \
                    if node_ids.size else np.zeros(n_ids, dtype=np.int64)
                t2 = time.perf_counter()
                stats = _stats(obj, measured, counts, persp, focal_y, bw, bh,
                               min_ratio_px, full=full)
                if stats is not None:
                    stats["cut_active"] = measured is not entry
                    if keep_ids:
                        stats["ids"] = _id_image(raw, npx, bw, bh)
                if stats is not None and edge_batch is not None \
                        and measured.get("seg_geom") is not None:
                    seg_counts = np.bincount(seg_ids) if seg_ids.size \
                        else np.zeros(1, dtype=np.int64)
                    stats["edge_stats"] = _edge_stats(
                        measured, seg_counts, modelview, projection, bw, bh)
                t3 = time.perf_counter()
                read_ms += (t1 - t0) * 1000.0
                decode_ms += (t2 - t1) * 1000.0
                stats_ms += (t3 - t2) * 1000.0
                if stats is not None:
                    results.append(stats)

            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set(prev_blend)
    finally:
        offscreen.free()

    return {
        "objects": results,
        "pairs": pairs,
        "buffer": (bw, bh),
        "divisor": div,
        "timings": {
            "prepass_ms": prepass_ms,
            "id_ms": id_ms,
            "draw_ms": prepass_ms + id_ms,
            "readback_ms": read_ms,
            "decode_ms": decode_ms,
            "stats_ms": stats_ms,
            "total_ms": (time.perf_counter() - t_start) * 1000.0,
        },
    }


def measure_region(context, divisor=DEFAULT_DIVISOR):
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None
    return measure(context.scene, rv3d.view_matrix, rv3d.window_matrix,
                   rv3d.perspective_matrix, region.width, region.height,
                   divisor=divisor)


def format_report(result):
    if result is None:
        return "visibility: unavailable (no ID batch or no GPU support)"
    bw, bh = result["buffer"]
    t = result["timings"]
    lines = [f"ID buffer {bw}x{bh} (1/{result['divisor']}) | "
             f"prepass {t['prepass_ms']:.1f} + id {t['id_ms']:.1f} + "
             f"readback {t['readback_ms']:.1f} + decode {t['decode_ms']:.1f} + "
             f"stats {t['stats_ms']:.1f} = {t['total_ms']:.1f} ms"]
    for st in result["objects"]:
        note = (" [adaptive cut]" if st.get("cut_active")
                else " [coarse level active]" if st["coarse_active"] else "")
        median = st.get("median_ratio")
        median_txt = f"median visibility {median:.2f}, " if median is not None \
            else ""
        lines.append(
            f"{st['name']}: {st['nodes']} nodes, {st['onscreen']} on-screen, "
            f"{st['fully_occluded']} fully occluded, "
            f"{st['measurable']} measurable (>= {MIN_RATIO_PX:g} px), "
            f"{median_txt}mean visibility {st['mean_ratio']:.2f}, "
            f"{st['below_half']} below 50%{note}"
        )
        es = st.get("edge_stats")
        if es:
            lines.append(
                f"{' ' * len(st['name'])}  edges: {es['measurable']} of "
                f"{es['edges']} measurable, {es['fully_occluded']} fully "
                f"occluded, mean visibility {es['mean_ratio']:.2f}, "
                f"area-weighted {es['area_ratio']:.2f}, "
                f"{es['below_half']} below 50%"
            )
    return "\n".join(lines)


def _ensure_id_batches(scene, st=None):
    st = _settings(scene, st)
    if bool(st.render_id_pass):
        return False
    scene.scigraphs_preview_render_id_pass = True
    invalidate()
    return True


class SCIGRAPHS_OT_measure_visibility(bpy.types.Operator):
    """Report the fraction of each node's screen footprint that is not covered"""
    bl_idname = "scigraphs.measure_visibility"
    bl_label = "Measure Visibility"
    bl_description = (
        "Render the node ID pass for the current camera and report, per node, "
        "the fraction of its screen footprint that is not covered by anything"
    )

    divisor: bpy.props.IntProperty(
        name="Resolution Divisor", default=DEFAULT_DIVISOR, min=1, max=8,
        description="Render the ID buffer at 1/N of the viewport resolution",
    )

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def execute(self, context):
        rebuilt = _ensure_id_batches(context.scene)
        result = measure_region(context, divisor=self.divisor)
        if result is None:
            self.report({'WARNING'}, "SciGraphs: no measurable graph in view"
                        + (" (ID batches rebuilding, run again)" if rebuilt else ""))
            return {'CANCELLED'}
        text = format_report(result)
        print("[SciGraphs] visibility\n" + text)
        self.report({'INFO'}, text.replace("\n", " | "))
        return {'FINISHED'}


class SCIGRAPHS_OT_refine_cut(bpy.types.Operator):
    """Measure the current image and merge whatever it shows to be hidden"""
    bl_idname = "scigraphs.refine_cut"
    bl_label = "Refine Cut from Image"
    bl_description = (
        "Count how many pixels each drawn element wins in the current image "
        "and merge the ones it covers into their parent. Run it again until it "
        "reports no further change"
    )

    # A community only reads as hidden once its children merged, so the worst
    # case is one pass per hierarchy level. Eight covers the deepest one.
    passes: bpy.props.IntProperty(
        name="Passes", default=8, min=1, max=32,
        description="Measure and merge this many times, or until nothing changes",
    )

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def execute(self, context):
        scene = context.scene
        st = _settings(scene)
        if not bool(st.adaptive):
            self.report({'WARNING'}, "SciGraphs: enable the adaptive cut first")
            return {'CANCELLED'}
        rebuilt = _ensure_id_batches(scene)

        total, rounds = 0, 0
        for _ in range(self.passes):
            result = measure_region(context)
            if result is None:
                self.report({'WARNING'}, "SciGraphs: nothing measurable in view"
                            + (" (rebuilding, run again)" if rebuilt else ""))
                return {'CANCELLED'}
            rounds += 1
            added = feed_back(scene, result, context.region_data.view_matrix)
            total += added
            if not added:
                break
        tag_redraw()
        self.report({'INFO'}, f"SciGraphs: {total} merges over {rounds} passes"
                    + (" (settled)" if total == 0 or rounds < self.passes
                       else ""))
        return {'FINISHED'}


class SCIGRAPHS_OT_benchmark_visibility(bpy.types.Operator):
    """Time the ID pass, readback and decode across resolution divisors"""
    bl_idname = "scigraphs.benchmark_visibility"
    bl_label = "Benchmark ID Pass"
    bl_description = (
        "Time the ID pass, readback and decode at several resolutions to see "
        "whether they fit in an interactive frame budget"
    )

    repeats: bpy.props.IntProperty(name="Repeats", default=5, min=1, max=50)

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def execute(self, context):
        rebuilt = _ensure_id_batches(context.scene)
        region = context.region
        rows = []
        nodes = 0
        for div in (1, 2, 3, 4):
            samples = []
            for _ in range(self.repeats):
                result = measure_region(context, divisor=div)
                if result is None:
                    break
                samples.append(result)
            if not samples:
                continue
            nodes = max(nodes, sum(st["nodes"] for st in samples[0]["objects"]))
            mean = {k: float(np.mean([s["timings"][k] for s in samples]))
                    for k in ("draw_ms", "readback_ms", "decode_ms",
                              "stats_ms", "total_ms")}
            rows.append((div, samples[0]["buffer"], mean))

        if not rows:
            self.report({'WARNING'}, "SciGraphs: nothing to benchmark"
                        + (" (ID batches rebuilding, run again)" if rebuilt else ""))
            return {'CANCELLED'}

        header = (f"[SciGraphs] visibility benchmark - region "
                  f"{region.width}x{region.height}, {nodes} drawn nodes, "
                  f"{self.repeats} repeats (mean ms)")
        lines = [header,
                 f"{'div':>4} {'buffer':>12} {'draw':>8} {'readback':>10} "
                 f"{'decode':>8} {'stats':>8} {'total':>8} {'fps':>7}"]
        for div, (bw, bh), m in rows:
            fps = 1000.0 / m["total_ms"] if m["total_ms"] > 0 else float('inf')
            lines.append(
                f"{div:>4} {f'{bw}x{bh}':>12} {m['draw_ms']:>8.2f} "
                f"{m['readback_ms']:>10.2f} {m['decode_ms']:>8.2f} "
                f"{m['stats_ms']:>8.2f} {m['total_ms']:>8.2f} {fps:>7.1f}")
        print("\n".join(lines))
        fastest = min(rows, key=lambda r: r[2]["total_ms"])
        self.report({'INFO'},
                    f"Visibility: {fastest[2]['total_ms']:.1f} ms at 1/"
                    f"{fastest[0]} - see the system console for the table")
        return {'FINISHED'}
