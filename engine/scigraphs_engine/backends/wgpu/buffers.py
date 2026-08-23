# MeshSpec -> GPU buffers and a list of draw calls.
#
# A range-grouped spec builds no index buffer, since WebGPU can express a draw
# offset. The Blender counterpart materializes an arange index buffer per group:
# 11 MB of int32 on the 2.8M-segment bundled graph. One vertex buffer per
# attribute, not interleaved, since mesh.py already builds each contiguous.

import numpy as np
import wgpu

from ...mesh import LINES, POINTS, TRIS

# A shader's `expand`, when it declares one, overrides this.
TOPOLOGY = {
    POINTS: "point-list",
    LINES: "line-list",
    TRIS: "triangle-list",
}

# Vertices per primitive, for checking a range group divides evenly.
ARITY = {POINTS: 1, LINES: 2, TRIS: 3}


class UploadError(RuntimeError):
    """The spec and the shader disagree about something that must match."""


class DrawCall:
    """One recorded draw over an UploadedMesh's buffers. Holds no pipeline
    reference, so the cache can key off it without a circular import."""

    __slots__ = ("indexed", "count", "first", "instances", "first_instance",
                 "state", "weight")

    def __init__(self, indexed, count, first=0, instances=1, first_instance=0,
                 state=None, weight=None):
        self.indexed = bool(indexed)
        self.count = int(count)
        self.first = int(first)
        self.instances = int(instances)
        self.first_instance = int(first_instance)
        self.state = state
        self.weight = weight

    def __repr__(self):
        kind = "draw_indexed" if self.indexed else "draw"
        return (f"<{kind} count={self.count} first={self.first} "
                f"instances={self.instances} first_instance={self.first_instance}>")


class UploadedMesh:
    """The GPU-side half of a MeshSpec: buffers plus the draws over them."""

    __slots__ = ("shader", "topology", "vertex_buffers", "index_buffer",
                 "index_count", "draws", "vertex_count", "spec")

    def __init__(self, shader, topology, vertex_buffers, index_buffer,
                 index_count, draws, vertex_count, spec):
        self.shader = shader
        self.topology = topology
        # Parallel to shader.attributes: slot i is attributes[i].
        self.vertex_buffers = vertex_buffers
        self.index_buffer = index_buffer
        self.index_count = index_count
        self.draws = draws
        self.vertex_count = vertex_count
        self.spec = spec

    @property
    def bytes_uploaded(self):
        total = sum(b.size for b in self.vertex_buffers)
        if self.index_buffer is not None:
            total += self.index_buffer.size
        return int(total)

    def destroy(self):
        for buf in self.vertex_buffers:
            try:
                buf.destroy()
            except Exception:   # noqa: BLE001 - best-effort
                pass
        if self.index_buffer is not None:
            try:
                self.index_buffer.destroy()
            except Exception:   # noqa: BLE001
                pass


def _column_array(arr, columns, name, shader_name):
    """(N, columns) contiguous float32, or UploadError. wgpu takes a float32x4
    declaration over an (N, 3) array and draws it one row shifted, nearly right."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        got = 1
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        got = int(arr.shape[1])
    else:
        raise UploadError(
            f"{shader_name}: attribute {name!r} has shape {arr.shape}; "
            f"expected (N,) or (N, k)")
    if got != columns:
        raise UploadError(
            f"{shader_name}: attribute {name!r} declares {columns} columns in "
            f"the JSON but the MeshSpec array has {got}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _index_array(idx):
    """Flat contiguous uint32, always: texas-roads alone has 2M vertices."""
    return np.ascontiguousarray(np.asarray(idx).ravel(), dtype=np.uint32)


def _plan_draws(spec, expand, state_default):
    n = spec.vertex_count
    arity = ARITY.get(spec.topology, 1)
    # Merge, not replace: spec.state whole drops the renderer's defaults.
    base_state = _merge(spec.state, state_default)

    if expand is not None:
        if spec.groups:
            for i, g in enumerate(spec.groups):
                if not g.is_range:
                    raise UploadError(
                        f"shader expands to instances, so group {i} must be a "
                        f"start/count range; it carries explicit indices, "
                        f"which would name instances rather than vertices")
        if spec.groups:
            draws = [DrawCall(False, expand.vertex_count,
                              instances=int(g.count), first_instance=int(g.start),
                              state=_merge(g.state, base_state), weight=g.weight)
                     for g in spec.groups]
        else:
            draws = [DrawCall(False, expand.vertex_count, instances=n,
                              state=base_state)]
        return draws, []

    if spec.groups:
        draws, index_arrays, offset = [], [], 0
        for i, g in enumerate(spec.groups):
            state = _merge(g.state, base_state)
            if g.is_range:
                if g.count % arity:
                    raise UploadError(
                        f"group {i}: {g.count} vertices is not a whole number "
                        f"of {spec.topology} primitives")
                draws.append(DrawCall(False, int(g.count), first=int(g.start),
                                      state=state, weight=g.weight))
            else:
                idx = _index_array(g.indices)
                index_arrays.append(idx)
                draws.append(DrawCall(True, int(idx.size), first=offset,
                                      state=state, weight=g.weight))
                offset += int(idx.size)
        return draws, index_arrays

    if spec.indices is not None:
        idx = _index_array(spec.indices)
        return [DrawCall(True, int(idx.size), state=base_state)], [idx]
    return [DrawCall(False, n, state=base_state)], []


def _merge(state, fallback):
    """A DrawState with its None fields filled from ``fallback``."""
    if state is None:
        return fallback
    if fallback is None:
        return state
    from ...mesh import DrawState
    return DrawState(
        point_size=state.point_size if state.point_size is not None
        else fallback.point_size,
        line_width=state.line_width if state.line_width is not None
        else fallback.line_width,
        blend=state.blend if state.blend is not None else fallback.blend,
        depth_test=state.depth_test if state.depth_test is not None
        else fallback.depth_test,
        depth_write=state.depth_write if state.depth_write is not None
        else fallback.depth_write,
    )


def plan(spec, shader, state_default=None):
    """(draws, index_arrays) for a spec, no GPU involved. Split out of ``upload``
    because a wrong range-group offset still renders and needs a test."""
    reason = spec.validate()
    if reason:
        raise UploadError(f"invalid MeshSpec: {reason}")
    expand = getattr(shader, "expand", None) if shader is not None else None
    return _plan_draws(spec, expand, state_default)


def upload(ctx, spec, shader, state_default=None, label=""):
    """An UploadedMesh for a MeshSpec plus its Shader, None if empty."""
    if spec is None or shader is None:
        return None
    n = spec.vertex_count
    if n == 0:
        return None

    draws, index_arrays = plan(spec, shader, state_default)

    vertex_buffers = []
    for attr in shader.attributes:
        if attr.name not in spec.attrs:
            raise UploadError(
                f"{shader.name} declares attribute {attr.name!r}; the MeshSpec "
                f"has {sorted(spec.attrs)}")
        data = _column_array(spec.attrs[attr.name], attr.columns, attr.name,
                             shader.name)
        vertex_buffers.append(ctx.device.create_buffer_with_data(
            label=f"{label or shader.name}:{attr.name}",
            data=data, usage=wgpu.BufferUsage.VERTEX))

    index_buffer, index_count = None, 0
    if index_arrays:
        joined = (index_arrays[0] if len(index_arrays) == 1
                  else np.concatenate(index_arrays))
        if joined.size and int(joined.max()) >= n:
            raise UploadError(
                f"index {int(joined.max())} out of range for {n} vertices")
        index_count = int(joined.size)
        index_buffer = ctx.device.create_buffer_with_data(
            label=f"{label or shader.name}:index",
            data=joined, usage=wgpu.BufferUsage.INDEX)

    expand = shader.expand
    topology = expand.topology if expand is not None \
        else TOPOLOGY[spec.topology]

    return UploadedMesh(shader, topology, tuple(vertex_buffers), index_buffer,
                        index_count, tuple(draws), n, spec)
