# DrawState + shader + vertex layout -> a cached GPURenderPipeline. WebGPU bakes
# blend and depth state into the pipeline, and building one costs 0.33 ms against
# 1.1 us for a cache hit (tests/bench.py, RTX 4060 over Vulkan), so a pipeline
# per draw would put 4 ms in front of a frame that draws 100k points in 0.36 ms.
# point_size, line_width and weight are uniforms, so they stay out of the key.

import wgpu

from .target import COLOR_FORMAT, DEPTH_FORMAT

# Factor names must be hyphenated. wgpu.BlendFactor's attributes are underscored
# but its values are not, so `one_minus_src_alpha` looks right and dies with a
# bare cffi KeyError that never mentions blending.
_ALPHA = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha",
              "operation": "add"},
    "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha",
              "operation": "add"},
}
_PREMULT = {
    "color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha",
              "operation": "add"},
    "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha",
              "operation": "add"},
}
_ADDITIVE = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one",
              "operation": "add"},
    "alpha": {"src_factor": "one", "dst_factor": "one", "operation": "add"},
}

BLEND = {
    "NONE": None,
    "ALPHA": _ALPHA,
    "ALPHA_PREMULT": _PREMULT,
    "ADDITIVE": _ADDITIVE,
}

# Used when neither the group, the spec nor the renderer set the field.
DEFAULT_BLEND = "NONE"
DEFAULT_DEPTH_TEST = True
DEFAULT_DEPTH_WRITE = True


class PipelineError(RuntimeError):
    pass


def state_key(state):
    """(blend, depth_test, depth_write), None resolved to the defaults above so
    two DrawStates differing only in an unset field share a cache entry."""
    if state is None:
        return (DEFAULT_BLEND, DEFAULT_DEPTH_TEST, DEFAULT_DEPTH_WRITE)
    blend = state.blend if state.blend is not None else DEFAULT_BLEND
    if blend not in BLEND:
        raise PipelineError(
            f"blend {blend!r} is not one of {sorted(BLEND)}; the engine only "
            f"emits those, so this is a typo rather than a missing feature")
    test = state.depth_test if state.depth_test is not None \
        else DEFAULT_DEPTH_TEST
    write = state.depth_write if state.depth_write is not None \
        else DEFAULT_DEPTH_WRITE
    return (blend, bool(test), bool(write))


class PipelineCache:
    def __init__(self, ctx, color_format=COLOR_FORMAT, depth_format=DEPTH_FORMAT,
                 sample_count=1):
        self.ctx = ctx
        self.color_format = color_format
        self.depth_format = depth_format
        self.sample_count = int(sample_count)
        self._pipelines = {}
        self._layouts = {}
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._pipelines)

    def key(self, shader, topology, state):
        return (shader.name, topology, shader.layout_key, shader.cull,
                shader.writes_depth, state_key(state),
                self.color_format, self.depth_format, self.sample_count)

    def bind_group_layouts(self, shader):
        """Cached by name, apart from the pipeline: three blend modes share one
        layout, and a rebuilt layout may no longer compare equal."""
        cache_key = (shader.name,)
        got = self._layouts.get(cache_key)
        if got is not None:
            return got

        by_group = {}
        for b in shader.bindings:
            entry = {
                "binding": b.binding,
                "visibility": b.visibility,
                "buffer": {"type": b.type,
                           "has_dynamic_offset": bool(b.dynamic)},
            }
            by_group.setdefault(b.group, []).append(entry)

        layouts = []
        for group in range(max(by_group) + 1 if by_group else 0):
            layouts.append(self.ctx.device.create_bind_group_layout(
                label=f"{shader.name}:group{group}",
                entries=by_group.get(group, [])))
        layouts = tuple(layouts)
        self._layouts[cache_key] = layouts
        return layouts

    def get(self, shader, topology, state):
        key = self.key(shader, topology, state)
        got = self._pipelines.get(key)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        pipeline = self._build(shader, topology, state)
        self._pipelines[key] = pipeline
        return pipeline

    def _build(self, shader, topology, state):
        blend_name, depth_test, depth_write = state_key(state)

        step = "instance" if shader.expand is not None else "vertex"
        buffers = [{
            "array_stride": attr.columns * 4,
            "step_mode": step,
            "attributes": [{"format": attr.format, "offset": 0,
                            "shader_location": attr.location}],
        } for attr in shader.attributes]

        layouts = self.bind_group_layouts(shader)
        layout = self.ctx.device.create_pipeline_layout(
            label=f"{shader.name}:layout", bind_group_layouts=list(layouts))

        target = {"format": self.color_format,
                  "blend": BLEND[blend_name],
                  "write_mask": wgpu.ColorWrite.ALL}

        # A strip needs a strip index format even when nothing is drawn
        # indexed. Omit it and the validation error never mentions strips.
        primitive = {
            "topology": topology,
            "front_face": "ccw",
            "cull_mode": shader.cull,
        }
        if topology.endswith("-strip"):
            primitive["strip_index_format"] = "uint32"

        try:
            return self.ctx.device.create_render_pipeline(
                label=f"{shader.name}[{blend_name},"
                      f"{'test' if depth_test else 'notest'},"
                      f"{'write' if depth_write else 'nowrite'}]",
                layout=layout,
                vertex={"module": shader.module,
                        "entry_point": shader.vertex_entry,
                        "buffers": buffers},
                primitive=primitive,
                depth_stencil={
                    "format": self.depth_format,
                    "depth_write_enabled": bool(depth_write
                                                and shader.writes_depth),
                    # 'always' when the DrawState turned the test off.
                    "depth_compare": (shader.depth_compare if depth_test
                                      else "always"),
                },
                multisample={"count": self.sample_count, "mask": 0xFFFFFFFF,
                             "alpha_to_coverage_enabled": False},
                fragment={"module": shader.module,
                          "entry_point": shader.fragment_entry,
                          "targets": [target]},
            )
        except Exception as exc:    # noqa: BLE001 - wgpu raises several types
            raise PipelineError(
                f"{shader.name}: pipeline creation failed ({blend_name}, "
                f"{topology}): {exc}") from exc
