# A RenderEngine that draws straight from the buffers batches.py built, so F12
# and animation match the preview. Materials and lighting do nothing here.

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import dynamic, filter_gpu, labels, shaders
from .draw import draw_blocks, draw_graph_object, get_cache_entry
from .state import is_graph_object
def _settings(scene, st=None):
    """The caller's snapshot when given; the host import stays local."""
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)


_VIEW_DOF_MAX_DIM = 1280


def _graph_objects(scene):
    objs = []
    for obj in scene.objects:
        if is_graph_object(obj) and not obj.hide_render:
            objs.append(obj)
    return objs


def _render_size(scene):
    scale = scene.render.resolution_percentage / 100.0
    sx = max(1, int(scene.render.resolution_x * scale))
    sy = max(1, int(scene.render.resolution_y * scale))
    return sx, sy


# Hard cap so a big supersample factor cannot exceed GPU texture limits.
_MAX_DIM = 16384


def _aa_factor(scene, sx, sy, st=None):
    st = _settings(scene, st)
    s = max(1, int(st.render_aa))
    while s > 1 and (sx * s > _MAX_DIM or sy * s > _MAX_DIM):
        s -= 1
    return s


def _store(layer, name, array):
    """Assigning ``rect`` unpacks 921,600 four-float lists at 720p."""
    if name not in layer.passes:
        return False
    flat = np.ascontiguousarray(array, dtype=np.float32).ravel()
    layer.passes[name].rect.foreach_set(flat)
    return True


def _reduce_blocks(image, s, combine):
    """Reduce each s x s block of ``image`` with ``combine``. One accumulator over
    s*s strided views, not an (h, s, w, s) reshape: 5x faster on color and 17x on
    depth at 720p, s=2. Order differs from numpy's pairwise sum, so at 3x the
    float32 result lands ~1 ULP away and three channels move one 8-bit step."""
    h, w = image.shape[0] // s, image.shape[1] // s
    acc = np.array(image[0:h * s:s, 0:w * s:s])
    for i in range(s):
        for j in range(s):
            if i or j:
                combine(acc, image[i:h * s:s, j:w * s:s], out=acc)
    return acc


def _downsample_color(color, s):
    if s <= 1:
        return color
    acc = _reduce_blocks(color, s, np.add)
    acc *= np.float32(1.0 / (s * s))
    return acc


def _gpu_resolve(texture, sx, sy, s):
    """Box-filter on the GPU, or None. 32F, since 8-bit would round back."""
    from . import shaders

    sh = shaders.get_resolve_shader()
    if sh is None:
        return None
    batch = batch_for_shader(
        sh, 'TRI_FAN', {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))})
    offscreen = gpu.types.GPUOffScreen(sx, sy, format='RGBA32F')
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            prev_blend = gpu.state.blend_get()
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
            gpu.state.depth_mask_set(False)
            sh.bind()
            sh.uniform_sampler("src", texture)
            sh.uniform_int("u_s", int(s))
            batch.draw(sh)
            gpu.state.blend_set(prev_blend)
            return np.array(fb.read_color(0, 0, sx, sy, 4, 0, 'FLOAT'),
                            dtype=np.float32).reshape(sy, sx, 4)
    except Exception:  # noqa: BLE001 - fall back to reading the whole buffer
        return None
    finally:
        offscreen.free()


def _antialiased_color(offscreen, rsx, rsy, sx, sy, s):
    """The color buffer at final resolution. Readback dominates (42 ms against
    18 ms of numpy averaging), so filtering first moves 130 MB across the bus
    instead of 1.2 GB at 4K, s=3. Call outside ``offscreen.bind()``: a nested
    bind leaves the outer one unbound and ``clear`` then fails, "returned NULL"."""
    if s > 1:
        resolved = _gpu_resolve(offscreen.texture_color, sx, sy, s)
        if resolved is not None:
            return resolved
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        color = np.array(fb.read_color(0, 0, rsx, rsy, 4, 0, 'FLOAT'),
                         dtype=np.float32).reshape(rsy, rsx, 4)
    return _downsample_color(color, s)


def _downsample_depth(depth, s):
    if s <= 1:
        return depth
    return _reduce_blocks(depth, s, np.minimum)


def _linearize_depth(depth01, proj):
    """OpenGL depth [0,1] to eye-space distance: B / (ndc_z + A), background large."""
    a = float(proj[2][2])
    b = float(proj[2][3])
    ndc = depth01.astype(np.float64) * 2.0 - 1.0
    denom = ndc + a
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = np.where(np.abs(denom) > 1e-12, b / denom, 1.0e10)
    dist = np.abs(dist).astype(np.float32)
    dist[depth01 >= 1.0] = 1.0e10
    return dist


# Cap on the DoF blur radius (final px); a tiny f-stop asks for thousands.
_DOF_MAX_RADIUS = 96

# Gather cost grows with radius squared, so the preview clamps harder.
_VIEW_DOF_MAX_RADIUS = 48.0


def _dof_shape(cam, scene, st=None):
    """(blades, rotation, ratio, boost): camera aperture plus Bokeh Highlights."""
    st = _settings(scene, st)
    dof = cam.data.dof
    return (
        float(dof.aperture_blades),
        float(dof.aperture_rotation),
        max(float(dof.aperture_ratio), 1e-3),
        max(float(st.dof_highlights), 0.0),
    )


def _dof_rad_scale(max_radius, target_samples):
    """Spiral step (px): ``radius += step / radius`` gives r^2/(2*step) samples."""
    return max(0.75, float(max_radius) ** 2 / (2.0 * float(target_samples)))


def _dof_focus_distance(cam):
    from mathutils import Vector

    dof = cam.data.dof
    focus_obj = dof.focus_object
    if focus_obj is not None:
        cam_mw = cam.matrix_world
        view_dir = (cam_mw.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        delta = focus_obj.matrix_world.translation - cam_mw.translation
        return max(float(delta.dot(view_dir)), 1e-4)
    return max(float(dof.focus_distance), 1e-4)


def _dof_params(cam, width_px):
    """``(focus, coc_scale)`` or None; CoC px = coc_scale * |d - focus| / d."""
    data = cam.data
    dof = data.dof
    if not dof.use_dof:
        return None

    n_stop = max(float(dof.aperture_fstop), 1e-3)
    focal_m = max(float(data.lens), 1e-3) / 1000.0
    focus = max(_dof_focus_distance(cam), focal_m * 1.001)

    sensor_mm = float(data.sensor_width)
    if data.sensor_fit == 'VERTICAL':
        sensor_mm = float(data.sensor_height)
    sensor_mm = max(sensor_mm, 1e-3)

    # CoC diameter (m) = f^2 / (N (S1 - f)) * |S2 - S1| / S2, then -> px.
    coc_scale = (focal_m * focal_m) / (n_stop * (focus - focal_m))
    coc_scale = coc_scale * 1000.0 / sensor_mm * float(width_px)
    return focus, coc_scale


def _gpu_depth_of_field(color, depth01, proj, cam, scene, sx, sy, seeds=4,
                        target_samples=2000):
    """The viewport's gather shader, averaging ``seeds`` rotated passes."""
    from . import shaders

    params = _dof_params(cam, sx)
    if params is None:
        return color
    focus, coc_scale = params

    sh = shaders.get_dof_tex_shader()
    if sh is None:
        return color

    # Global CoC bound: walk only as far as anything scatters.
    dist = _linearize_depth(depth01, proj)
    coc = 0.5 * coc_scale * np.abs(dist - focus) / np.maximum(dist, 1e-6)
    max_radius = float(min(np.max(coc), _DOF_MAX_RADIUS))
    if max_radius < 0.75:
        return color

    color = np.ascontiguousarray(color, dtype=np.float32)
    depth01 = np.ascontiguousarray(depth01, dtype=np.float32)
    col_tex = gpu.types.GPUTexture(
        (sx, sy), format='RGBA32F',
        data=gpu.types.Buffer('FLOAT', color.shape, color))
    dep_tex = gpu.types.GPUTexture(
        (sx, sy), format='R32F',
        data=gpu.types.Buffer('FLOAT', depth01.shape, depth01))

    batch = batch_for_shader(
        sh, 'TRI_FAN',
        {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1)),
         "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1))},
    )

    blades, blade_rot, ratio, boost = _dof_shape(cam, scene)

    acc = np.zeros((sy, sx, 4), np.float64)
    offscreen = gpu.types.GPUOffScreen(sx, sy, format='RGBA32F')
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            prev_blend = gpu.state.blend_get()
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')
            gpu.state.depth_mask_set(False)
            for i in range(max(1, seeds)):
                fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                sh.bind()
                sh.uniform_sampler("color", col_tex)
                sh.uniform_sampler("depth", dep_tex)
                sh.uniform_float("u_texel", (1.0 / sx, 1.0 / sy))
                sh.uniform_float("u_a", float(proj[2][2]))
                sh.uniform_float("u_b", float(proj[2][3]))
                sh.uniform_float("u_focus", float(focus))
                sh.uniform_float("u_coc_scale", float(coc_scale))
                sh.uniform_float("u_max_radius", max_radius)
                sh.uniform_float("u_rad_scale",
                                 _dof_rad_scale(max_radius, target_samples))
                sh.uniform_float("u_seed", i / max(1, seeds))
                sh.uniform_float("u_blades", blades)
                sh.uniform_float("u_blade_rot", blade_rot)
                sh.uniform_float("u_ratio", ratio)
                sh.uniform_float("u_boost", boost)
                batch.draw(sh)
                acc += np.array(fb.read_color(0, 0, sx, sy, 4, 0, 'FLOAT'),
                                dtype=np.float32).reshape(sy, sx, 4)
            gpu.state.blend_set(prev_blend)
    finally:
        offscreen.free()

    return (acc / max(1, seeds)).astype(np.float32)


def _composite_labels(scene, persp, sx, sy, dist_buf, color):
    items = []
    for obj in _graph_objects(scene):
        items.extend(labels.collect_label_items(obj, scene, persp, sx, sy,
                                                dist_buf))
    if not items:
        return color

    offscreen = gpu.types.GPUOffScreen(sx, sy)
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            labels.draw_label_items(items, scene, sx, sy)
            lbl = np.array(fb.read_color(0, 0, sx, sy, 4, 0, 'FLOAT'),
                           dtype=np.float32).reshape(sy, sx, 4)
    finally:
        offscreen.free()

    la = lbl[..., 3:4]
    ca = color[..., 3:4]
    out_a = la + ca * (1.0 - la)
    out_rgb = lbl[..., :3] * la + color[..., :3] * ca * (1.0 - la)
    rgb = np.divide(out_rgb, out_a, out=np.zeros_like(out_rgb),
                    where=out_a > 1e-6)
    return np.concatenate([rgb, out_a], axis=-1).astype(np.float32)


_VIEW_FBO = {"size": None, "color": None, "depth": None, "fb": None,
             "blur": None, "blur_fb": None}


def _get_view_fbo(rw, rh):
    if _VIEW_FBO["size"] != (rw, rh):
        color_tex = gpu.types.GPUTexture((rw, rh), format='RGBA16F')
        depth_tex = gpu.types.GPUTexture((rw, rh), format='DEPTH_COMPONENT32F')
        fb = gpu.types.GPUFrameBuffer(depth_slot=depth_tex, color_slots=color_tex)
        bw, bh = max(2, rw // 2), max(2, rh // 2)
        blur_tex = gpu.types.GPUTexture((bw, bh), format='RGBA16F')
        blur_fb = gpu.types.GPUFrameBuffer(color_slots=blur_tex)
        _VIEW_FBO.update(size=(rw, rh), color=color_tex, depth=depth_tex,
                         fb=fb, blur=blur_tex, blur_fb=blur_fb)
    return (_VIEW_FBO["color"], _VIEW_FBO["depth"], _VIEW_FBO["fb"],
            _VIEW_FBO["blur"], _VIEW_FBO["blur_fb"])


def _background_color(scene, st=None):
    st = _settings(scene, st)
    world = scene.world
    if world is not None and getattr(world, "use_nodes", False) is False:
        c = world.color
        return (c[0], c[1], c[2], 1.0)
    return tuple(st.render_bg)


class SciGraphsRenderEngine(bpy.types.RenderEngine):
    bl_idname = "SCIGRAPHS"
    bl_label = "SciGraphs"
    bl_use_preview = False
    bl_use_gpu_context = True

    def update_render_passes(self, scene=None, renderlayer=None):
        self.register_pass(scene, renderlayer, "Combined", 4, "RGBA", 'COLOR')
        self.register_pass(scene, renderlayer, "Depth", 1, "Z", 'VALUE')
        self.register_pass(scene, renderlayer, "NodeID", 4, "RGBA", 'COLOR')
        self.register_pass(scene, renderlayer, "Overdraw", 1, "X", 'VALUE')


    def render(self, depsgraph):
        scene = depsgraph.scene
        st = _settings(scene)
        sx, sy = _render_size(scene)
        result = self.begin_result(0, 0, sx, sy)
        try:
            combined, depth = self._render_beauty(depsgraph, scene, sx, sy)
            layer = result.layers[0]
            if combined is not None:
                _store(layer, "Combined", combined)
            if depth is not None:
                _store(layer, "Depth", depth)
            if bool(st.render_id_pass):
                node_id = self._render_id(depsgraph, scene, sx, sy)
                if node_id is not None:
                    _store(layer, "NodeID", node_id)
            if bool(st.render_overdraw):
                over = self._render_overdraw(depsgraph, scene, sx, sy)
                if over is not None:
                    _store(layer, "Overdraw", over)
        except Exception as exc:  # noqa: BLE001 - report instead of crashing
            self.report({'ERROR'}, f"SciGraphs render failed: {exc}")
        finally:
            self.end_result(result)

    def _camera_matrices(self, depsgraph, scene, sx, sy):
        cam = scene.camera
        if cam is None:
            return None
        view = cam.matrix_world.inverted()
        proj = cam.calc_matrix_camera(depsgraph, x=sx, y=sy)
        persp = np.asarray(proj, dtype=np.float64) @ np.asarray(view, dtype=np.float64)
        return view, proj, persp

    def _render_beauty(self, depsgraph, scene, sx, sy, st=None):
        """Combined as (color, linear_depth), supersampled then averaged down."""
        st = _settings(scene, st)
        mats = self._camera_matrices(depsgraph, scene, sx, sy)
        if mats is None:
            self.report({'WARNING'}, "SciGraphs: no active camera")
            return None, None
        view, proj, persp = mats
        bg = _background_color(scene)
        s = _aa_factor(scene, sx, sy)
        rsx, rsy = sx * s, sy * s

        depth_test = bool(st.depth_test)
        cam = scene.camera
        dof_on = (cam is not None and cam.type == 'CAMERA'
                  and _dof_params(cam, sx) is not None)

        def _draw_all():
            for obj in _graph_objects(scene):
                entry = get_cache_entry(obj, scene)
                if entry is None:
                    continue
                self._draw_object(entry, scene, obj, view, proj, persp, rsy)

        offscreen = gpu.types.GPUOffScreen(rsx, rsy)
        try:
            with offscreen.bind():
                fb = gpu.state.active_framebuffer_get()

                # Depth Test off draws the graph see-through.
                fb.clear(color=bg, depth=1.0)
                if depth_test:
                    gpu.state.depth_test_set('LESS_EQUAL')
                    gpu.state.depth_mask_set(True)
                _draw_all()
                gpu.state.depth_test_set('NONE')
                gpu.state.depth_mask_set(False)
                depth = (np.array(fb.read_depth(0, 0, rsx, rsy),
                                  dtype=np.float32).reshape(rsy, rsx)
                         if depth_test else None)

            # Outside the bind on purpose. See _antialiased_color.
            color = _antialiased_color(offscreen, rsx, rsy, sx, sy, s)

            if not depth_test:
                # DoF and the Depth AOV still need depth. This pass wipes the
                # color resolved above, so it runs after that read.
                with offscreen.bind():
                    fb = gpu.state.active_framebuffer_get()
                    fb.clear(color=bg, depth=1.0)
                    gpu.state.depth_test_set('LESS_EQUAL')
                    gpu.state.depth_mask_set(True)
                    _draw_all()
                    gpu.state.depth_test_set('NONE')
                    gpu.state.depth_mask_set(False)
                    depth = np.array(fb.read_depth(0, 0, rsx, rsy),
                                     dtype=np.float32).reshape(rsy, rsx)
        finally:
            offscreen.free()

        depth_min = _downsample_depth(depth, s)
        linear = _linearize_depth(depth_min, proj)

        if dof_on:
            color = _gpu_depth_of_field(color, depth_min, proj, cam, scene,
                                        sx, sy)

        # After DoF, so labels stay crisp; occlusion comes from the depth.
        if labels.labels_enabled(scene, st=st):
            color = _composite_labels(scene, persp, sx, sy, linear, color)

        # Blender stores Combined premultiplied and this path is straight alpha,
        # so associate here or transparency shows halos in the compositor.
        color = np.ascontiguousarray(color, dtype=np.float32)
        np.clip(color, 0.0, None, out=color)
        color[..., :3] *= color[..., 3:4]

        return color, linear

    def _render_id(self, depsgraph, scene, sx, sy):
        mats = self._camera_matrices(depsgraph, scene, sx, sy)
        if mats is None:
            return None
        view, proj, _persp = mats
        from . import shaders

        id_shader = shaders.get_sphere_id_shader()
        if id_shader is None:
            return None

        offscreen = gpu.types.GPUOffScreen(sx, sy)
        try:
            with offscreen.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(True)
                for obj in _graph_objects(scene):
                    entry = get_cache_entry(obj, scene)
                    if entry is None or entry["reps"].get("sphere_id") is None:
                        continue
                    with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                        gpu.matrix.load_matrix(view @ obj.matrix_world)
                        gpu.matrix.load_projection_matrix(proj)
                        modelview = gpu.matrix.get_model_view_matrix()
                        projm = gpu.matrix.get_projection_matrix()
                        id_shader.bind()
                        id_shader.uniform_float("u_view", modelview)
                        id_shader.uniform_float("u_proj", projm)
                        entry["reps"]["sphere_id"].draw(id_shader)
                gpu.state.depth_test_set('NONE')
                gpu.state.depth_mask_set(False)
                idbuf = np.array(fb.read_color(0, 0, sx, sy, 4, 0, 'FLOAT'),
                                 dtype=np.float32).reshape(sy, sx, 4)
        finally:
            offscreen.free()
        return idbuf

    def _render_overdraw(self, depsgraph, scene, sx, sy):
        """Edge segments per pixel, additive into a float buffer; count in R."""
        mats = self._camera_matrices(depsgraph, scene, sx, sy)
        if mats is None:
            return None
        view, proj, _persp = mats

        offscreen = gpu.types.GPUOffScreen(sx, sy, format='RGBA32F')
        try:
            with offscreen.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                prev_blend = gpu.state.blend_get()
                gpu.state.blend_set('ADDITIVE')
                gpu.state.depth_test_set('NONE')
                gpu.state.depth_mask_set(False)
                gpu.state.line_width_set(1.0)
                for obj in _graph_objects(scene):
                    entry = get_cache_entry(obj, scene)
                    if entry is None or entry["reps"].get("line") is None \
                            or entry.get("line_shader") is None:
                        continue
                    with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                        gpu.matrix.load_matrix(view @ obj.matrix_world)
                        gpu.matrix.load_projection_matrix(proj)
                        # A vertex-colored batch would weight fragments by
                        # opacity, so it takes the color-ignoring shader.
                        if entry.get("line_vertex_color"):
                            shader = shaders.get_line_count_shader()
                            if shader is None:
                                continue
                            shader.bind()
                            shader.uniform_float(
                                "u_mvp",
                                gpu.matrix.get_projection_matrix()
                                @ gpu.matrix.get_model_view_matrix())
                        elif entry.get("filter") is not None \
                                or entry.get("dynamic") is not None:
                            # The filter stack lives in the shader, so the
                            # count goes through it. So does the animated
                            # line shader; else "uniform color not found".
                            shader = entry["line_shader"]
                            shader.bind()
                            shader.uniform_float(
                                "u_mvp",
                                gpu.matrix.get_projection_matrix()
                                @ gpu.matrix.get_model_view_matrix())
                            shader.uniform_float("u_color",
                                                 (1.0, 1.0, 1.0, 1.0))
                            if entry.get("filter") is not None:
                                _fbuf = filter_gpu.bind(
                                    shader, scene, entry["filter"])
                            if entry.get("dynamic") is not None:
                                dynamic.bind(shader, entry["dynamic"])
                        else:
                            shader = entry["line_shader"]
                            shader.bind()
                            shader.uniform_float("color",
                                                 (1.0, 1.0, 1.0, 1.0))
                        entry["reps"]["line"].draw(shader)
                gpu.state.blend_set(prev_blend)
                buf = np.array(fb.read_color(0, 0, sx, sy, 4, 0, 'FLOAT'),
                               dtype=np.float32).reshape(sy, sx, 4)
        finally:
            offscreen.free()
        return buf[..., 0]

    @staticmethod
    def _draw_object(entry, scene, obj, view, proj, persp, height):
        with gpu.matrix.push_pop():
            with gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(view @ obj.matrix_world)
                gpu.matrix.load_projection_matrix(proj)
                modelview = gpu.matrix.get_model_view_matrix()
                projm = gpu.matrix.get_projection_matrix()
                if entry.get("blocks") is not None:
                    draw_blocks(entry, scene, obj, modelview, projm, persp, height)
                else:
                    draw_graph_object(entry, scene, obj, modelview, projm, persp,
                                      height, view_matrix=view)

    def view_update(self, context, depsgraph):
        # Required override. Batches rebuild lazily in get_cache_entry.
        pass

    def view_draw(self, context, depsgraph):
        scene = depsgraph.scene
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return

        # A camera effect, so preview it only through the scene camera.
        cam = scene.camera
        dof_on = (
            rv3d.view_perspective == 'CAMERA'
            and cam is not None and cam.type == 'CAMERA'
            and cam.data.dof.use_dof
        )
        if dof_on:
            try:
                self._view_draw_dof(scene, region, rv3d, cam)
                return
            except Exception:  # noqa: BLE001 - fall back to the crisp preview
                pass

        self._view_draw_direct(scene, region, rv3d)

    def _view_draw_direct(self, scene, region, rv3d, st=None):
        st = _settings(scene, st)
        prev_blend = gpu.state.blend_get()
        depth_test = bool(st.depth_test)
        if depth_test:
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)

        persp = rv3d.perspective_matrix
        for obj in _graph_objects(scene):
            entry = get_cache_entry(obj, scene)
            if entry is None:
                continue
            with gpu.matrix.push_pop():
                with gpu.matrix.push_pop_projection():
                    gpu.matrix.load_projection_matrix(rv3d.window_matrix)
                    gpu.matrix.load_matrix(rv3d.view_matrix)
                    gpu.matrix.multiply_matrix(obj.matrix_world)
                    modelview = gpu.matrix.get_model_view_matrix()
                    projm = gpu.matrix.get_projection_matrix()
                    if entry.get("blocks") is not None:
                        draw_blocks(entry, scene, obj, modelview, projm,
                                    persp, region.height)
                    else:
                        draw_graph_object(entry, scene, obj, modelview, projm,
                                          persp, region.height,
                                          view_matrix=rv3d.view_matrix)

        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set(prev_blend)

        if not labels.labels_enabled(scene, st=st):
            return

        dist_buf = None
        props = getattr(scene, "scigraphs", None)
        if depth_test and props is not None and props.text_depth_occlusion:
            w, h = region.width, region.height
            fb = gpu.state.active_framebuffer_get()
            depth01 = np.array(fb.read_depth(0, 0, w, h),
                               dtype=np.float32).reshape(h, w)
            dist_buf = _linearize_depth(depth01, rv3d.window_matrix)
        self._draw_view_labels(scene, region, rv3d, dist_buf)

    def _draw_view_labels(self, scene, region, rv3d, dist_buf):
        w, h = region.width, region.height
        items = []
        for obj in _graph_objects(scene):
            items.extend(labels.collect_label_items(
                obj, scene, rv3d.perspective_matrix, w, h, dist_buf))
        labels.draw_label_items(items, scene, w, h)

    def _view_draw_dof(self, scene, region, rv3d, cam, st=None):
        st = _settings(scene, st)
        from . import shaders

        dof_sh = shaders.get_dof_shader()
        comp_sh = shaders.get_dof_comp_shader()
        if dof_sh is None or comp_sh is None:
            self._view_draw_direct(scene, region, rv3d)
            return

        w, h = region.width, region.height
        if w < 2 or h < 2:
            self._view_draw_direct(scene, region, rv3d)
            return

        scale = min(1.0, _VIEW_DOF_MAX_DIM / float(max(w, h)))
        rw, rh = max(2, int(w * scale)), max(2, int(h * scale))

        params = _dof_params(cam, rw)
        if params is None:
            self._view_draw_direct(scene, region, rv3d)
            return
        focus, coc_scale = params
        blades, blade_rot, ratio, boost = _dof_shape(cam, scene)

        view = rv3d.view_matrix
        proj = rv3d.window_matrix
        persp = rv3d.perspective_matrix
        bg = _background_color(scene)
        bg = (bg[0], bg[1], bg[2], 1.0)

        color_tex, depth_tex, fb, blur_tex, blur_fb = _get_view_fbo(rw, rh)
        depth_test = bool(st.depth_test)

        def _draw_all():
            for obj in _graph_objects(scene):
                entry = get_cache_entry(obj, scene)
                if entry is None:
                    continue
                self._draw_object(entry, scene, obj, view, proj, persp, rh)

        with fb.bind():
            fb.clear(color=bg, depth=1.0)
            if depth_test:
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(True)
                _draw_all()
                gpu.state.depth_test_set('NONE')
                gpu.state.depth_mask_set(False)
            else:
                # Fill the depth buffer the CoC needs, then redraw color flat.
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(True)
                _draw_all()
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set('NONE')
                fb.clear(color=bg)
                _draw_all()

        fullscreen = {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1)),
                      "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1))}

        def _set_common(sh):
            sh.uniform_float("u_texel", (1.0 / rw, 1.0 / rh))
            sh.uniform_float("u_a", float(proj[2][2]))
            sh.uniform_float("u_b", float(proj[2][3]))
            sh.uniform_float("u_focus", float(focus))
            sh.uniform_float("u_coc_scale", float(coc_scale))
            sh.uniform_float("u_max_radius", float(_VIEW_DOF_MAX_RADIUS))

        prev_blend = gpu.state.blend_get()
        gpu.state.blend_set('NONE')

        with blur_fb.bind():
            batch = batch_for_shader(dof_sh, 'TRI_FAN', fullscreen)
            dof_sh.bind()
            dof_sh.uniform_sampler("color", color_tex)
            dof_sh.uniform_sampler("depth", depth_tex)
            _set_common(dof_sh)
            dof_sh.uniform_float(
                "u_rad_scale", _dof_rad_scale(_VIEW_DOF_MAX_RADIUS, 384))
            dof_sh.uniform_float("u_seed", 0.0)
            dof_sh.uniform_float("u_blades", blades)
            dof_sh.uniform_float("u_blade_rot", blade_rot)
            dof_sh.uniform_float("u_ratio", ratio)
            dof_sh.uniform_float("u_boost", boost)
            batch.draw(dof_sh)

        batch = batch_for_shader(comp_sh, 'TRI_FAN', fullscreen)
        comp_sh.bind()
        comp_sh.uniform_sampler("color", color_tex)
        comp_sh.uniform_sampler("blur", blur_tex)
        comp_sh.uniform_sampler("depth", depth_tex)
        _set_common(comp_sh)
        batch.draw(comp_sh)
        gpu.state.blend_set(prev_blend)

        # On top of the DoF result. Labels are never blurred.
        if labels.labels_enabled(scene, st=st):
            dist_buf = None
            props = getattr(scene, "scigraphs", None)
            if props is not None and props.text_depth_occlusion:
                with fb.bind():
                    depth01 = np.array(fb.read_depth(0, 0, rw, rh),
                                       dtype=np.float32).reshape(rh, rw)
                dist_buf = _linearize_depth(depth01, proj)
            self._draw_view_labels(scene, region, rv3d, dist_buf)
