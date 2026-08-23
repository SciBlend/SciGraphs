# `queue.write_buffer` is ordered against the submit, not against the commands
# inside the buffer. Two writes around two draws give both draws the second
# value, and every bucket comes out the size of the last. Hence `_DrawRing`:
# each draw reads its own region of one buffer at a dynamic offset.

import numpy as np
import wgpu

from .camera import CAMERA_BLOCK_SIZE
from .buffers import _merge, upload
from .pipelines import PipelineCache
from .shaders import CAMERA_BLOCK, DRAW_BLOCK, ShaderLibrary

# f32 point_size, line_width, weight, flags + vec4 tint. See README 1.3.
DRAW_BLOCK_SIZE = 32

# No MeshGroup class: -1, not 0, since 0 is a legitimate class center.
NO_WEIGHT = -1.0

DEFAULT_POINT_SIZE = 4.0
DEFAULT_LINE_WIDTH = 1.0


class _DrawRing:
    """N draw blocks in one uniform buffer, at the device's alignment."""

    def __init__(self, ctx, capacity=64):
        self.ctx = ctx
        # 64 on NVIDIA/Vulkan, but read it: a 256-alignment device rejects a
        # 64-byte stride and the error never names the stride.
        self.stride = max(int(ctx.limit("min-uniform-buffer-offset-alignment",
                                        256)), DRAW_BLOCK_SIZE)
        self.capacity = 0
        self.buffer = None
        self._grow(capacity)

    def _grow(self, capacity):
        if capacity <= self.capacity:
            return
        capacity = max(capacity, self.capacity * 2, 8)
        if self.buffer is not None:
            self.buffer.destroy()
        self.buffer = self.ctx.device.create_buffer(
            label="scigraphs-draw-ring", size=capacity * self.stride,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.capacity = capacity

    def write(self, blocks):
        n = len(blocks)
        self._grow(n)
        staging = np.zeros(n * self.stride // 4, dtype=np.float32)
        step = self.stride // 4
        for i, block in enumerate(blocks):
            staging[i * step:i * step + DRAW_BLOCK_SIZE // 4] = block
        self.ctx.queue.write_buffer(self.buffer, 0, staging.tobytes())
        return [i * self.stride for i in range(n)]


def draw_block(state, weight, tint=(1.0, 1.0, 1.0, 1.0)):
    """The 8 floats of one `draw` uniform block."""
    point_size = DEFAULT_POINT_SIZE
    line_width = DEFAULT_LINE_WIDTH
    if state is not None:
        if state.point_size is not None:
            point_size = float(state.point_size)
        if state.line_width is not None:
            line_width = float(state.line_width)
    out = np.zeros(8, dtype=np.float32)
    out[0] = point_size
    out[1] = line_width
    out[2] = NO_WEIGHT if weight is None else float(weight)
    out[3] = 0.0
    out[4:8] = tint
    return out


class Renderer:
    """Draw MeshSpecs into a Target. The camera block carries the viewport,
    so one Renderer can serve targets of different sizes."""

    def __init__(self, ctx, target, library=None, default_state=None,
                 pipelines=None):
        self.ctx = ctx
        self.target = target
        self.library = library if library is not None else ShaderLibrary(ctx)
        self.default_state = default_state
        self.pipelines = pipelines if pipelines is not None \
            else PipelineCache(ctx)
        self._camera_buffer = ctx.device.create_buffer(
            label="scigraphs-camera", size=CAMERA_BLOCK_SIZE,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._ring = _DrawRing(ctx)
        self._bind_groups = {}
        self.last_stats = {}

    def upload(self, spec, label=""):
        """An UploadedMesh for the spec, or None if the shader has no port
        yet, so a build with only round_point still renders."""
        if spec is None:
            return None
        shader = self.library.get(spec.shader)
        if shader is None:
            return None
        # Inheritance chain: group -> spec -> shader -> renderer.
        fallback = _merge(shader.default_state, self.default_state)
        return upload(self.ctx, spec, shader, fallback, label)

    def _bind_group(self, shader, extra):
        key = (shader.name, tuple(sorted(extra)) if extra else ())
        got = self._bind_groups.get(key)
        if got is not None:
            return got
        layouts = self.pipelines.bind_group_layouts(shader)
        entries = []
        for b in shader.bindings:
            if b.group != 0:
                raise NotImplementedError(
                    f"{shader.name} declares a binding in group {b.group}; the "
                    f"renderer supplies group 0 only. Add the group before "
                    f"using it, do not silently drop it.")
            if b.name == CAMERA_BLOCK:
                # The declared size, not this module's: a 208-byte Camera
                # fails validation if it is bound at 224.
                buf = self._camera_buffer
                size = b.size or CAMERA_BLOCK_SIZE
            elif b.name == DRAW_BLOCK:
                buf, size = self._ring.buffer, b.size or DRAW_BLOCK_SIZE
            else:
                if extra is None or b.name not in extra:
                    raise KeyError(
                        f"{shader.name} declares binding {b.name!r}; pass it as "
                        f"render(blocks={{{b.name!r}: array}})")
                buf = extra[b.name]
                size = buf.size
            entries.append({"binding": b.binding,
                            "resource": {"buffer": buf, "offset": 0,
                                         "size": size}})
        group = self.ctx.device.create_bind_group(
            label=f"{shader.name}:bind0", layout=layouts[0], entries=entries)
        self._bind_groups[key] = group
        return group

    def block_buffer(self, name, data):
        """A uniform block for ``render(blocks=...)``, uploaded once."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        return self.ctx.device.create_buffer_with_data(
            label=f"scigraphs-block-{name}", data=data,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    def render(self, meshes, camera, clear_color=(0.0, 0.0, 0.0, 1.0),
               clear=True, blocks=None, tint=(1.0, 1.0, 1.0, 1.0)):
        """Draw uploaded meshes into the target and return the stats dict.
        ``meshes`` is one UploadedMesh or an iterable. None entries are
        mesh.py's "nothing to draw" and get skipped."""
        self.ctx.check()
        if meshes is None:
            meshes = []
        elif not isinstance(meshes, (list, tuple)):
            meshes = [meshes]
        meshes = [m for m in meshes if m is not None]

        self.ctx.queue.write_buffer(
            self._camera_buffer, 0,
            camera.block(self.target.width, self.target.height).tobytes())

        # All in one go before recording; see the module header.
        plan = []
        for mesh in meshes:
            for call in mesh.draws:
                plan.append((mesh, call))
        offsets = self._ring.write(
            [draw_block(call.state, call.weight, tint) for _, call in plan]
        ) if plan else []
        # The write may have reallocated the ring, and `_bind_group` keys on
        # binding names rather than buffer identity, so a fresh buffer gets
        # served the stale group. heb_line does that, and it aborts the device.
        if plan:
            self._bind_groups.clear()

        enc = self.ctx.device.create_command_encoder(label="scigraphs-render")
        colors, depth = self.target.attachments(clear_color, clear)
        rp = enc.begin_render_pass(color_attachments=colors,
                                   depth_stencil_attachment=depth)

        pipeline_before = self.pipelines.misses
        vertices = 0
        current = None
        for (mesh, call), offset in zip(plan, offsets):
            pipeline = self.pipelines.get(mesh.shader, mesh.topology,
                                          call.state)
            if pipeline is not current:
                rp.set_pipeline(pipeline)
                current = pipeline
            group = self._bind_group(mesh.shader, blocks)
            dynamic = [offset] if mesh.shader.wants_draw else []
            rp.set_bind_group(0, group, dynamic, 0, 999999)
            for slot, buf in enumerate(mesh.vertex_buffers):
                rp.set_vertex_buffer(slot, buf)
            if call.indexed:
                rp.set_index_buffer(mesh.index_buffer, "uint32")
                rp.draw_indexed(call.count, call.instances, call.first, 0,
                                call.first_instance)
                vertices += call.count
            else:
                rp.draw(call.count, call.instances, call.first,
                        call.first_instance)
                vertices += call.count * call.instances
        rp.end()
        self.ctx.queue.submit([enc.finish()])

        self.last_stats = {
            "meshes": len(meshes),
            "draws": len(plan),
            "vertices": vertices,
            "pipelines_created": self.pipelines.misses - pipeline_before,
            "pipelines_cached": len(self.pipelines),
        }
        return self.last_stats

    def sync(self):
        """Block until the GPU drains; returns which path ran. submit()
        returns immediately, so timing around it measures CPU encoding.

        on_submitted_work_done_sync() is broken in wgpu-py 0.32.0: it passes a
        four-parameter callback where the struct has three, and raises
        TypeError before waiting. The 1x1 readback fallback costs 0.13 ms on an
        idle queue against 0.013 ms for the internal poll.
        """
        try:
            self.ctx.queue.on_submitted_work_done_sync()
            return "queue"
        except Exception:       # noqa: BLE001 - wgpu-py 0.32.0, see docstring
            pass
        poll = getattr(self.ctx.device, "_poll_wait", None)
        if poll is not None:
            poll()
            return "poll"
        self.ctx.queue.read_texture(
            {"texture": self.target.color, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": 4, "rows_per_image": 1}, (1, 1, 1))
        return "readback"

    def render_specs(self, specs, camera, **kwargs):
        """Upload, draw and destroy in one call. Wrong in a loop: a million
        points cost 8 ms to upload and 0.3 ms to draw, so hold the mesh."""
        if specs is None:
            specs = []
        elif not isinstance(specs, (list, tuple)):
            specs = [specs]
        meshes = [self.upload(s) for s in specs]
        try:
            return self.render([m for m in meshes if m is not None], camera,
                               **kwargs)
        finally:
            for m in meshes:
                if m is not None:
                    m.destroy()
