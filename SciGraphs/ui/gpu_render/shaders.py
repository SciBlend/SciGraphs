# GLSL shaders, compiled lazily and cached. A getter returns None when the
# backend rejects its shader, and the caller falls back. Blender 5.x removed
# ``gpu.types.GPUShader(vertexcode, fragcode)`` ("TypeError: cannot create
# 'GPUShader' instances"), so shaders go through ``GPUShaderCreateInfo``, and
# uniforms became push constants but are still set with ``uniform_float``.

import gpu

from .glsl_source import glsl

# A UBO, not push constants: two mat4 already fill the guaranteed 128 bytes.
# All fields vec4 for trivial std140; key_col.w is rim strength.
_LIGHT_RIG_STRUCT = glsl("light_rig_struct")


# Filter stack, evaluated in the vertex shader. As a numpy predicate a threshold
# move rebuilds every batch (33 ms at 31k nodes, 175 ms at 262k, 1.2 s at 2M);
# as a uniform it is one buffer update. Four clauses per domain: Blender cannot
# set uniform arrays from Python, and 8 vec4 is exactly a 128-byte block.
# lo/hi bound each range, inv inverts it, act enables it; e* is the edge domain.
_FILTER_STRUCT = glsl("filter_struct")

# Prepended to each filtered vertex source. ``typedef_source`` takes one string
# and lands ahead of the sampler and UBO declarations, so it cannot see them.
# ``frow`` = (own node, other endpoint, own edge), CPU-offset; .z of -1 = no edge.
# glsl/filter.glsl hardcodes this same 4096 in ``scig_frow``. Nothing
# interpolates it, so changing one without the other silently misreads the
# texture from the second row on.
FILTER_TEX_ROW = 4096

_FILTER_GLSL = glsl("filter")

# A degenerate zero-length line is not reliably discarded, so push it off-clip.
_FILTER_CULL = "gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return;"


def _add_filter(info, sampler_slot, attr_slot, ubo_slot, typedefs=""):
    info.typedef_source(typedefs + _FILTER_STRUCT)
    info.uniform_buf(ubo_slot, "FilterStack", "u_filter")
    info.sampler(sampler_slot, 'FLOAT_2D', "u_fchan")
    if attr_slot is not None:
        info.vertex_in(attr_slot, 'VEC3', "frow")


def _guard(filtered):
    return f"if (!scig_keep(frow)) {{ {_FILTER_CULL} }}" if filtered else ""


def _prelude(filtered):
    return _FILTER_GLSL if filtered else ""


# Positions in a texture, not an attribute: an animated layout costs one upload
# per frame, not a batch rebuild. +4% at 100k, -0.5% at 1M (RTX 4060, Vulkan).
POS_TEX_ROW = 4096

_POS_GLSL = f"""
vec3 scig_pos(int i)
{{
  return texelFetch(u_pos, ivec2(i % {POS_TEX_ROW}, i / {POS_TEX_ROW}), 0).xyz;
}}
"""


def _add_pos(info, sampler_slot):
    info.sampler(sampler_slot, 'FLOAT_2D', "u_pos")


def _pos_prelude(animated):
    return _POS_GLSL if animated else ""


# Round point (screen-space disk). Point size comes from the fixed-function
# gpu.state.point_size_set; gl_PointSize is unreliable across backends.

def _build_round_point(filtered=False, animated=False):
    vout = gpu.types.GPUStageInterfaceInfo("scig_round_iface")
    vout.smooth('VEC4', "f_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    # Slot 0 is reused so the two formats stay the same width.
    if animated:
        info.vertex_in(0, 'FLOAT', "n_idx")
        _add_pos(info, sampler_slot=1 if filtered else 0)
    else:
        info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC4', "color")
    if filtered:
        _add_filter(info, sampler_slot=0, attr_slot=2, ubo_slot=0)
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    src = "scig_pos(int(n_idx + 0.5))" if animated else "pos"
    info.vertex_source(
        _prelude(filtered) + _pos_prelude(animated) +
        "void main()"
        "{"
        "  f_color = color;"
        "  " + _guard(filtered) +
        f"  gl_Position = u_mvp * vec4({src}, 1.0);"
        "}"
    )
    info.fragment_source(glsl("round_point_frag"))
    return gpu.shader.create_from_info(info)


# The builtin UNIFORM_COLOR takes no extra attribute, and an indexed batch shares
# a node between its edges, so ``frow`` cannot name the edge: a vertex per end.

def _build_line_dynamic(filtered=False):
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('VEC4', "u_color")
    info.vertex_in(0, 'FLOAT', "n_idx")
    _add_pos(info, sampler_slot=1 if filtered else 0)
    if filtered:
        _add_filter(info, sampler_slot=0, attr_slot=1, ubo_slot=0)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(
        _prelude(filtered) + _POS_GLSL +
        "void main()"
        "{"
        "  " + _guard(filtered) +
        "  gl_Position = u_mvp * vec4(scig_pos(int(n_idx + 0.5)), 1.0);"
        "}"
    )
    info.fragment_source(glsl("line_dynamic_frag"))
    return gpu.shader.create_from_info(info)


def _build_line_dynamic_wide():
    """Animated line as a screen-space quad: the rasterizer clamps line width, and
    the POLYLINE builtins that avoid the clamp cannot sample a position texture."""
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('VEC4', "u_color")
    info.push_constant('VEC2', "u_viewport")
    info.push_constant('FLOAT', "u_width")
    info.vertex_in(0, 'VEC4', "edge")
    _add_pos(info, sampler_slot=0)
    info.fragment_out(0, 'VEC4', "fragColor")
    # Behind the camera the divide is meaningless; leave it to clipping.
    # NDC spans two units, so a half-width of w/2 pixels is perp*w/viewport.
    info.vertex_source(_POS_GLSL + glsl("line_dynamic_wide_vert"))
    info.fragment_source(glsl("line_dynamic_wide_frag"))
    return gpu.shader.create_from_info(info)


def get_line_dynamic_wide_shader():
    return _compile("line_wide_d")


def _build_line_filter():
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('VEC4', "u_color")
    info.vertex_in(0, 'VEC3', "pos")
    _add_filter(info, sampler_slot=0, attr_slot=1, ubo_slot=0)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(
        _prelude(True) +
        "void main()"
        "{"
        "  " + _guard(True) +
        "  gl_Position = u_mvp * vec4(pos, 1.0);"
        "}"
    )
    info.fragment_source(glsl("line_filter_frag"))
    return gpu.shader.create_from_info(info)


# Sphere impostor: a view-space quad per node, writing true depth so they sort.

def _build_sphere(filtered=False, animated=False):
    vout = gpu.types.GPUStageInterfaceInfo("scig_sphere_iface")
    vout.smooth('VEC2', "v_uv")
    vout.smooth('VEC3', "v_view_center")
    vout.smooth('FLOAT', "v_radius")
    vout.smooth('VEC4', "v_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    info.uniform_buf(0, "LightRig", "u_light")
    if filtered:
        _add_filter(info, sampler_slot=0, attr_slot=4, ubo_slot=1,
                    typedefs=_LIGHT_RIG_STRUCT)
    else:
        info.typedef_source(_LIGHT_RIG_STRUCT)
    if animated:
        info.vertex_in(0, 'FLOAT', "n_idx")
        _add_pos(info, sampler_slot=1 if filtered else 0)
    else:
        info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC2', "corner")
    info.vertex_in(2, 'FLOAT', "radius")
    info.vertex_in(3, 'VEC4', "color")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    src = "scig_pos(int(n_idx + 0.5))" if animated else "pos"
    info.vertex_source(
        _prelude(filtered) + _pos_prelude(animated) +
        "void main()"
        "{"
        "  " + _guard(filtered) +
        f"  vec4 view_center = u_view * vec4({src}, 1.0);"
        "  vec3 vc = view_center.xyz;"
        "  vec3 view_pos = vc + vec3(corner * radius, 0.0);"
        "  gl_Position = u_proj * vec4(view_pos, 1.0);"
        "  v_uv = corner;"
        "  v_view_center = vc;"
        "  v_radius = radius;"
        "  v_color = color;"
        "}"
    )
    info.fragment_source(glsl("sphere_frag"))
    return gpu.shader.create_from_info(info)


def _build_sphere_id():
    """Unlit ID-pass impostor. It writes gl_FragDepth so it occludes like the
    beauty pass, and ``color`` is an encoded node id."""
    vout = gpu.types.GPUStageInterfaceInfo("scig_sphere_id_iface")
    vout.smooth('VEC2', "v_uv")
    vout.smooth('VEC3', "v_view_center")
    vout.smooth('FLOAT', "v_radius")
    vout.smooth('VEC4', "v_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC2', "corner")
    info.vertex_in(2, 'FLOAT', "radius")
    info.vertex_in(3, 'VEC4', "color")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("sphere_id_vert"))
    info.fragment_source(glsl("sphere_id_frag"))
    return gpu.shader.create_from_info(info)


# Cylinder impostor: one quad per edge, offset perpendicular to the projected
# edge direction. True surface depth, so tubes occlude spheres and each other.

def _build_ribbon(filtered=False, animated=False):
    vout = gpu.types.GPUStageInterfaceInfo("scig_ribbon_iface")
    vout.smooth('FLOAT', "v_side")
    vout.smooth('VEC4', "v_color")
    vout.smooth('VEC3', "v_normal_basis")
    vout.smooth('VEC3', "v_view_pos")
    vout.smooth('FLOAT', "v_radius")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    info.uniform_buf(0, "LightRig", "u_light")
    if filtered:
        _add_filter(info, sampler_slot=0, attr_slot=6, ubo_slot=1,
                    typedefs=_LIGHT_RIG_STRUCT)
    else:
        info.typedef_source(_LIGHT_RIG_STRUCT)
    # One node pair replaces eight floats of position per segment.
    if animated:
        info.vertex_in(0, 'VEC2', "n_pair")
        _add_pos(info, sampler_slot=1 if filtered else 0)
    else:
        info.vertex_in(0, 'VEC3', "pos_a")
        info.vertex_in(1, 'VEC3', "pos_b")
    info.vertex_in(2, 'FLOAT', "side")
    info.vertex_in(3, 'FLOAT', "endsel")
    info.vertex_in(4, 'FLOAT', "radius")
    info.vertex_in(5, 'VEC4', "color")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    src_a = "scig_pos(int(n_pair.x + 0.5))" if animated else "pos_a"
    src_b = "scig_pos(int(n_pair.y + 0.5))" if animated else "pos_b"
    info.vertex_source(
        _prelude(filtered) + _pos_prelude(animated) +
        "void main()"
        "{"
        "  " + _guard(filtered) +
        f"  vec3 a = (u_view * vec4({src_a}, 1.0)).xyz;"
        f"  vec3 b = (u_view * vec4({src_b}, 1.0)).xyz;"
        "  vec3 center = mix(a, b, endsel);"
        "  vec3 dir = b - a;"
        "  float len = length(dir);"
        "  dir = (len > 1e-8) ? dir / len : vec3(1.0, 0.0, 0.0);"
        # Square cap: without the half-radius overlap, bends show dark ticks.
        "  center += dir * ((endsel * 2.0 - 1.0) * radius * 0.5);"
        "  vec3 view_axis = vec3(0.0, 0.0, 1.0);"
        "  vec3 offset = cross(dir, view_axis);"
        "  float ol = length(offset);"
        "  offset = (ol > 1e-6) ? offset / ol : normalize(cross(dir, vec3(0.0, 1.0, 0.0)));"
        "  vec3 view_pos = center + offset * (side * radius);"
        "  gl_Position = u_proj * vec4(view_pos, 1.0);"
        "  v_side = side;"
        "  v_color = color;"
        "  v_normal_basis = offset;"
        "  v_view_pos = view_pos;"
        "  v_radius = radius;"
        "}"
    )
    info.fragment_source(glsl("ribbon_frag"))
    return gpu.shader.create_from_info(info)


def _build_ribbon_id():
    """Unlit ID ribbon; geometry matches ``_build_ribbon`` fragment for fragment."""
    vout = gpu.types.GPUStageInterfaceInfo("scig_ribbon_id_iface")
    vout.smooth('FLOAT', "v_side")
    vout.smooth('VEC4', "v_color")
    vout.smooth('VEC3', "v_normal_basis")
    vout.smooth('VEC3', "v_view_pos")
    vout.smooth('FLOAT', "v_radius")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    info.vertex_in(0, 'VEC3', "pos_a")
    info.vertex_in(1, 'VEC3', "pos_b")
    info.vertex_in(2, 'FLOAT', "side")
    info.vertex_in(3, 'FLOAT', "endsel")
    info.vertex_in(4, 'FLOAT', "radius")
    info.vertex_in(5, 'VEC4', "color")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("ribbon_id_vert"))
    info.fragment_source(glsl("ribbon_id_frag"))
    return gpu.shader.create_from_info(info)


STYLE_TEX_ROW = 4096

STYLE_CODE = {
    'STRAIGHT': 0, 'CURVED': 1, 'QUADRATIC': 2, 'ARC': 3, 'TAPERED': 4,
    'ORTHOGONAL': 5,
    'SELF_LOOP': 6,
}
STYLE_ANIMATED = frozenset(k for k in STYLE_CODE if k != 'SELF_LOOP')

DIRECTION_CODE = {'AUTO': 0, 'CLOCKWISE': 1, 'COUNTER_CLOCKWISE': 2,
                  'ALTERNATING': 3}
ORTHOGONAL_CODE = {'CENTERED': 0, 'HORIZONTAL_FIRST': 1, 'VERTICAL_FIRST': 2,
                   'SHORTEST': 3}
TAPER_MODE_CODE = {'NONE': 0, 'FORWARD': 1, 'BACKWARD': 2}

_STYLE_STRUCT = glsl("edge_style_struct")
_STYLE_GLSL = glsl("edge_style")

_STYLE_KEEP = {
    False: "bool scig_style_keep(int e) { return true; }\n",
    True: "bool scig_style_keep(int e) { return scig_keep(scig_edge_frow(e)); }\n",
}


def _add_style(info, filtered, ubo_slot, typedefs=""):
    """Position texture, per-edge texture and the settings block. Sampler slot 0
    is the filter's when there is one, as everywhere else in this file."""
    if filtered:
        _add_filter(info, sampler_slot=0, attr_slot=None, ubo_slot=ubo_slot + 1,
                    typedefs=typedefs + _STYLE_STRUCT)
    else:
        info.typedef_source(typedefs + _STYLE_STRUCT)
    info.uniform_buf(ubo_slot, "EdgeStyle", "u_style")
    _add_pos(info, sampler_slot=1 if filtered else 0)
    info.sampler(2 if filtered else 1, 'FLOAT_2D', "u_edge")


def _style_prelude(filtered):
    return (_prelude(filtered) + _POS_GLSL + _STYLE_GLSL
            + _STYLE_KEEP[bool(filtered)])


def _build_style_line(filtered=False):
    """Thin styled lines. One LINES instance per edge over the sample points."""
    vout = gpu.types.GPUStageInterfaceInfo(
        "scig_style_line_f_iface" if filtered else "scig_style_line_iface")
    vout.smooth('VEC4', "f_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    _add_style(info, filtered, ubo_slot=0)
    info.vertex_in(0, 'FLOAT', "sample_t")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(_style_prelude(filtered) + glsl("style_line_vert"))
    info.fragment_source(glsl("style_line_frag"))
    return gpu.shader.create_from_info(info)


def _build_style_line_wide(filtered=False):
    """Styled lines as screen-space quads, so an edge can carry its own width."""
    vout = gpu.types.GPUStageInterfaceInfo(
        "scig_style_wide_f_iface" if filtered else "scig_style_wide_iface")
    vout.smooth('VEC4', "f_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('VEC2', "u_viewport")
    _add_style(info, filtered, ubo_slot=0)
    info.vertex_in(0, 'VEC4', "seg")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(_style_prelude(filtered) + glsl("style_line_wide_vert"))
    info.fragment_source(glsl("style_line_frag"))
    return gpu.shader.create_from_info(info)


def _build_style_ribbon(filtered=False):
    """Cylinder impostors along the curve. Two mat4 fill the 128-byte push block,
    so the color and the widths ride in the settings UBO rather than beside them."""
    vout = gpu.types.GPUStageInterfaceInfo(
        "scig_style_ribbon_f_iface" if filtered else "scig_style_ribbon_iface")
    vout.smooth('FLOAT', "v_side")
    vout.smooth('VEC4', "v_color")
    vout.smooth('VEC3', "v_normal_basis")
    vout.smooth('VEC3', "v_view_pos")
    vout.smooth('FLOAT', "v_radius")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    info.uniform_buf(0, "LightRig", "u_light")
    _add_style(info, filtered, ubo_slot=1, typedefs=_LIGHT_RIG_STRUCT)
    info.vertex_in(0, 'VEC4', "seg")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(_style_prelude(filtered) + glsl("style_ribbon_vert"))
    info.fragment_source(glsl("ribbon_frag"))
    return gpu.shader.create_from_info(info)


def style_segments(params, style=None):
    """Spans per edge the batch must carry, following tessellate's own branch:
    below 1e-3 curvature it emits a single straight segment, and ORTHOGONAL is
    always the three spans of a four-point routing."""
    style = style or params["style_type"]
    if style == 'SELF_LOOP':
        return max(int(params["segments"]), 8)
    if style == 'ORTHOGONAL':
        return 3
    if style == 'STRAIGHT' or float(params["curvature"]) < 1e-3:
        return 1
    return max(1, int(params["segments"]))


def style_uniform_block(params, color, width=(1.0, 1.0),
                        end_taper=(1.0, 'NONE'), style=None):
    """Settings block for the styled shaders; the caller holds it until after
    the draw, as with the filter and light blocks. ``width`` is (lo, hi) full
    line widths in device pixels for the line tiers, and (base radius, unused)
    for the ribbon. ``end_taper`` is batches._taper_segment_radii's factor and
    mode. ``style`` overrides the params' own, which the self-loop pass needs."""
    style = style or params["style_type"]
    values = [
        float(STYLE_CODE[style]), float(params["curvature"]),
        float(DIRECTION_CODE[params["direction"]]),
        float(ORTHOGONAL_CODE[params["orthogonal_style"]]),
        float(params["self_loop_radius"]), float(params["taper_start"]),
        float(params["taper_end"]), 0.0,
        float(color[0]), float(color[1]), float(color[2]), float(color[3]),
        float(width[0]), float(width[1]), float(end_taper[0]),
        float(TAPER_MODE_CODE[end_taper[1]]),
    ]
    return gpu.types.GPUUniformBuf(
        gpu.types.Buffer('FLOAT', len(values), values))


def get_style_line_shader(filtered=False, wide=False):
    return _compile(("style_wide" if wide else "style_line")
                    + ("_f" if filtered else ""))


def get_style_ribbon_shader(filtered=False):
    return _compile("style_ribbon_f" if filtered else "style_ribbon")


# Must agree with scigraphs_engine.mesh.ARROW_SIDES, which indexes the vertices
# placed here; dynamic.py reads it for the vertex count (2*SIDES + 2).
ARROW_SIDES = 8


def _build_arrow_flat(animated=False):
    """Flat arrowhead: three vertices instead of the cone's eighteen, billboarded
    along the edge's view-space direction."""
    vout = gpu.types.GPUStageInterfaceInfo(
        "scig_arrow_flat_dyn_iface" if animated else "scig_arrow_flat_iface")
    vout.smooth('VEC4', "v_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    if animated:
        _add_pos(info, sampler_slot=0)
        info.vertex_in(0, 'VEC2', "n_pair")
    else:
        info.vertex_in(0, 'VEC3', "pos_a")
        info.vertex_in(1, 'VEC3', "pos_b")
    info.vertex_in(2, 'FLOAT', "corner")     # 0 = tip, 1 and 2 = base
    info.vertex_in(3, 'VEC3', "cone")        # (length, half width, node radius)
    info.vertex_in(4, 'VEC4', "color")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")

    src_a = "scig_pos(int(n_pair.x + 0.5))" if animated else "pos_a"
    src_b = "scig_pos(int(n_pair.y + 0.5))" if animated else "pos_b"
    info.vertex_source(
        (_POS_GLSL if animated else "") +
        "void main()"
        "{"
        f"  vec3 a = (u_view * vec4({src_a}, 1.0)).xyz;"
        f"  vec3 b = (u_view * vec4({src_b}, 1.0)).xyz;"
        "  vec2 d = b.xy - a.xy;"
        "  float L = length(d);"
        "  if (L < 1e-9) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }"
        "  vec2 dir = d / L;"
        "  vec2 perp = vec2(-dir.y, dir.x);"
        "  vec2 tip = b.xy - dir * cone.z;"
        "  int k = int(corner + 0.5);"
        "  vec2 p = tip;"
        "  if (k == 1) { p = tip - dir * cone.x + perp * cone.y; }"
        "  else if (k == 2) { p = tip - dir * cone.x - perp * cone.y; }"
        # One depth for all three vertices: the triangle cannot be seen edge-on.
        "  gl_Position = u_proj * vec4(p, b.z, 1.0);"
        "  v_color = color;"
        "}"
    )
    # Opaque, like the cone: the edge alpha (0.35 by default, a density hint for
    # flat lines) made heads translucent in renders, though not in the viewport.
    info.fragment_source(glsl("arrow_flat_frag"))
    return gpu.shader.create_from_info(info)


def _build_arrow_dynamic():
    """Cone rebuilt from its two endpoints; the baked path redoes eighteen numpy
    positions and normals per arrow on every node move. Scale folds into the
    lengths on the CPU because two mat4 fill the 128-byte block."""
    vout = gpu.types.GPUStageInterfaceInfo("scig_arrow_dyn_iface")
    vout.smooth('VEC3', "v_normal")
    vout.smooth('VEC4', "v_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    info.typedef_source(_LIGHT_RIG_STRUCT)
    info.uniform_buf(0, "LightRig", "u_light")
    _add_pos(info, sampler_slot=0)
    info.vertex_in(0, 'VEC2', "n_pair")   # (source node, target node)
    info.vertex_in(1, 'FLOAT', "a_vert")  # 0 .. 2*SIDES+1
    info.vertex_in(2, 'VEC3', "cone")     # (length, base radius, node radius)
    info.vertex_in(3, 'VEC4', "color")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(
        _POS_GLSL +
        "void main()"
        "{"
        "  vec3 a = scig_pos(int(n_pair.x + 0.5));"
        "  vec3 b = scig_pos(int(n_pair.y + 0.5));"
        "  vec3 ab = b - a;"
        "  float seg = length(ab);"
        # An edge can go degenerate mid-motion; the baked path drops those early.
        "  if (seg < 1e-9) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }"
        "  vec3 d = ab / seg;"
        "  float len = cone.x;"
        "  float base_r = cone.y;"
        "  vec3 tip = b - d * cone.z;"
        "  vec3 base_c = tip - d * len;"
        "  vec3 up = (abs(d.z) < 0.9) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);"
        "  vec3 uu = normalize(cross(d, up));"
        "  vec3 vv = cross(d, uu);"
        f"  const int S = {ARROW_SIDES};"
        "  int k = int(a_vert + 0.5);"
        "  vec3 p; vec3 n;"
        "  if (k == 0) { p = tip; n = d; }"
        "  else if (k == S + 1) { p = base_c; n = -d; }"
        "  else {"
        "    int ri = (k <= S) ? (k - 1) : (k - S - 2);"
        "    float ang = 6.28318530717958647 * float(ri) / float(S);"
        "    vec3 radial = cos(ang) * uu + sin(ang) * vv;"
        "    p = base_c + radial * base_r;"
        "    n = (k <= S) ? normalize(radial * len + d * base_r) : -d;"
        "  }"
        "  gl_Position = u_proj * (u_view * vec4(p, 1.0));"
        "  v_normal = normalize((u_view * vec4(n, 0.0)).xyz);"
        "  v_color = color;"
        "}"
    )
    info.fragment_source(glsl("arrow_dynamic_frag"))
    return gpu.shader.create_from_info(info)


def _build_arrow():
    vout = gpu.types.GPUStageInterfaceInfo("scig_arrow_iface")
    vout.smooth('VEC3', "v_normal")
    vout.smooth('VEC4', "v_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    info.typedef_source(_LIGHT_RIG_STRUCT)
    info.uniform_buf(0, "LightRig", "u_light")
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC3', "nrm")
    info.vertex_in(2, 'VEC4', "color")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("arrow_vert"))
    info.fragment_source(glsl("arrow_frag"))
    return gpu.shader.create_from_info(info)


# Hierarchical bundling on the GPU. The numpy tessellator uploads the evaluated
# spline (8.5M vertices, 239 MB at 213k edges) and rebuilds it on every strength
# move. Here the control polygon (2*depth+3 points per edge) is a texture, one
# instance per edge, strength a push constant, count in the .w of the first point.
_HEB_GLSL = glsl("heb")


def _build_heb_line():
    vout = gpu.types.GPUStageInterfaceInfo("scig_heb_iface")
    vout.smooth('VEC4', "f_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "u_ctrl")
    info.sampler(1, 'FLOAT_2D', "u_meta")
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('FLOAT', "u_beta")
    info.push_constant('INT', "u_stride")
    info.push_constant('INT', "u_ctrl_row")
    info.push_constant('INT', "u_meta_row")
    info.push_constant('INT', "u_meta_stride")
    info.vertex_in(0, 'FLOAT', "sample_t")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(_HEB_GLSL + glsl("heb_line_vert"))
    info.fragment_source(glsl("heb_line_frag"))
    return gpu.shader.create_from_info(info)


def _build_heb_line_wide():
    """Bundling as a quad so edges can carry their own width. Handing 88k weighted
    edges to the numpy tessellator took a parameter change from 130 ms to 1.34 s."""
    vout = gpu.types.GPUStageInterfaceInfo("scig_heb_wide_iface")
    vout.smooth('VEC4', "f_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "u_ctrl")
    info.sampler(1, 'FLOAT_2D', "u_meta")
    info.sampler(2, 'FLOAT_2D', "u_width")
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('FLOAT', "u_beta")
    info.push_constant('INT', "u_stride")
    info.push_constant('INT', "u_ctrl_row")
    info.push_constant('INT', "u_meta_row")
    info.push_constant('INT', "u_meta_stride")
    info.push_constant('INT', "u_width_row")
    info.push_constant('VEC2', "u_viewport")
    info.push_constant('FLOAT', "u_w_lo")
    info.push_constant('FLOAT', "u_w_hi")
    info.vertex_in(0, 'VEC4', "seg")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(_HEB_GLSL + glsl("heb_line_wide_vert"))
    info.fragment_source(glsl("heb_line_wide_frag"))
    return gpu.shader.create_from_info(info)


def get_heb_line_wide_shader():
    return _compile("heb_line_wide")


def _build_heb_count():
    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "u_ctrl")
    info.sampler(1, 'FLOAT_2D', "u_meta")
    info.push_constant('MAT4', "u_mvp")
    info.push_constant('FLOAT', "u_beta")
    info.push_constant('INT', "u_stride")
    info.push_constant('INT', "u_ctrl_row")
    info.push_constant('INT', "u_meta_row")
    info.push_constant('INT', "u_meta_stride")
    info.vertex_in(0, 'FLOAT', "sample_t")
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(_HEB_GLSL + glsl("heb_count_vert"))
    info.fragment_source(glsl("heb_count_frag"))
    return gpu.shader.create_from_info(info)


# Line counter for the Overdraw pass: every fragment must weigh exactly one, and
# a vertex-colored batch would contribute its alpha and draw a picture instead.

def _build_line_count():
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC4', "color")
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("line_count_vert"))
    info.fragment_source(glsl("line_count_frag"))
    return gpu.shader.create_from_info(info)


# Volume slice: view-aligned slabs of the node density field, drawn back to front
# with alpha blending and depth-tested against the opaque graph. u_clip_to_vol
# maps (ndc.xy, ndc.z, 1) to [0,1]^3 field coordinates, and at ndc.z + u_slab.y
# it gives slab thickness in voxels, so the look does not follow the slice count.
# Shell width is in density, not pixels: fwidth() speckles next to the discards.

def _build_volume_slice():
    vout = gpu.types.GPUStageInterfaceInfo("scig_volume_iface")
    vout.smooth('VEC2', "v_ndc")

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_3D', "field")
    info.sampler(1, 'FLOAT_2D', "lut")
    info.push_constant('MAT4', "u_clip_to_vol")
    info.push_constant('VEC4', "u_rect")
    info.push_constant('VEC3', "u_shape")
    # Two vectors, not six scalars: 8 bytes each blew the 128-byte block; packed,
    # the shader is 120.
    #   u_slab  = (slice NDC depth, slab thickness in NDC)
    #   u_field = (opacity gain, color mode, isosurface levels, shell width)
    info.push_constant('VEC2', "u_slab")
    info.push_constant('VEC4', "u_field")
    info.vertex_in(0, 'VEC2', "corner")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("volume_slice_vert"))
    # Stored unclipped so the field still integrates to the node count.
    # Extinction into shells around d = k/levels, a countable scale.
    # /width conserves the mean: narrower shells get denser, not fainter.
    # In voxels, so the result is independent of the graph's world scale.
    info.fragment_source(glsl("volume_slice_frag"))
    return gpu.shader.create_from_info(info)


# Depth of field: golden-angle spiral scatter-as-gather, after "Bokeh depth of
# field in a single pass" (Dennis Gustafsson, 2018). Samples behind the center
# clamp to twice the center's CoC, so background does not bleed over in-focus
# foreground, and accumulation is premultiplied so transparent film blurs right.
# The viewport variant samples DEPTH_2D, the render one a FLOAT_2D copy.

# Eye distance from depth, thin-lens CoC.
_DOF_HELPERS = glsl("dof_helpers")

# Bokeh shape: u_blades-sided polygon (circle under 3), rotated and stretched.
# u_boost (luminance^2) keeps out-of-focus highlights as crisp bokeh discs.
# Interleaved gradient noise: undersampling shows as grain, not banding.
# Contributes once its own shaped CoC reaches this spiral radius.
_DOF_FRAG = _DOF_HELPERS + glsl("dof_common_frag")


def _build_resolve():
    """Average each s x s block of a supersampled buffer; the saving is the readback,
    not the averaging (18 ms in numpy). ``gl_FragCoord`` avoids a half-texel shift.
    ``u_premul`` also folds in the clip and the alpha associate: see resolve_frag."""
    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "src")
    info.push_constant('INT', "u_s")
    info.push_constant('INT', "u_premul")
    info.vertex_in(0, 'VEC2', "pos")
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("resolve_vert"))
    info.fragment_source(glsl("resolve_frag"))
    return gpu.shader.create_from_info(info)


def _build_depth_resolve():
    """Reduce and linearize the depth buffer on the GPU, so the beauty pass never
    reads a supersampled depth back. Needs the depth as a sampler, which is why
    _render_beauty draws into a GPUFrameBuffer instead of a GPUOffScreen."""
    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'DEPTH_2D', "src")
    info.push_constant('INT', "u_s")
    info.push_constant('FLOAT', "u_a")
    info.push_constant('FLOAT', "u_b")
    info.vertex_in(0, 'VEC2', "pos")
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("resolve_vert"))
    info.fragment_source(glsl("depth_resolve_frag"))
    return gpu.shader.create_from_info(info)


def _dof_push_constants(info):
    info.push_constant('VEC2', "u_texel")
    info.push_constant('FLOAT', "u_a")
    info.push_constant('FLOAT', "u_b")
    info.push_constant('FLOAT', "u_focus")
    info.push_constant('FLOAT', "u_coc_scale")
    info.push_constant('FLOAT', "u_max_radius")


def _build_dof_common(iface_name, depth_sampler):
    vout = gpu.types.GPUStageInterfaceInfo(iface_name)
    vout.smooth('VEC2', "uv")

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "color")
    info.sampler(1, depth_sampler, "depth")
    _dof_push_constants(info)
    info.push_constant('FLOAT', "u_rad_scale")
    info.push_constant('FLOAT', "u_seed")
    info.push_constant('FLOAT', "u_blades")
    info.push_constant('FLOAT', "u_blade_rot")
    info.push_constant('FLOAT', "u_ratio")
    info.push_constant('FLOAT', "u_boost")
    info.vertex_in(0, 'VEC2', "pos")
    info.vertex_in(1, 'VEC2', "texCoord")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("dof_common_vert"))
    info.fragment_source(_DOF_FRAG)
    return gpu.shader.create_from_info(info)


def _build_dof():
    return _build_dof_common("scig_dof_iface", 'DEPTH_2D')


def _build_dof_tex():
    return _build_dof_common("scig_dof_tex_iface", 'FLOAT_2D')


def _build_dof_comp():
    """Composite for the half-resolution viewport gather, blended in by CoC. Ring
    taps dilate nearer neighbors so foreground bokeh bleeds over sharp background."""
    vout = gpu.types.GPUStageInterfaceInfo("scig_dof_comp_iface")
    vout.smooth('VEC2', "uv")

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "color")
    info.sampler(1, 'FLOAT_2D', "blur")
    info.sampler(2, 'DEPTH_2D', "depth")
    _dof_push_constants(info)
    info.vertex_in(0, 'VEC2', "pos")
    info.vertex_in(1, 'VEC2', "texCoord")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("dof_comp_vert"))
    info.fragment_source(_DOF_HELPERS + glsl("dof_comp_frag"))
    return gpu.shader.create_from_info(info)


_BUILDERS = {
    "round_point": _build_round_point,
    "round_point_f": lambda: _build_round_point(filtered=True),
    "round_point_d": lambda: _build_round_point(animated=True),
    "round_point_fd": lambda: _build_round_point(filtered=True, animated=True),
    "sphere": _build_sphere,
    "sphere_f": lambda: _build_sphere(filtered=True),
    "sphere_d": lambda: _build_sphere(animated=True),
    "sphere_fd": lambda: _build_sphere(filtered=True, animated=True),
    "sphere_id": _build_sphere_id,
    "arrow_d": _build_arrow_dynamic,
    "arrow_flat": _build_arrow_flat,
    "arrow_flat_d": lambda: _build_arrow_flat(animated=True),
    "ribbon": _build_ribbon,
    "ribbon_f": lambda: _build_ribbon(filtered=True),
    "ribbon_d": lambda: _build_ribbon(animated=True),
    "ribbon_fd": lambda: _build_ribbon(filtered=True, animated=True),
    "line_f": _build_line_filter,
    "line_d": _build_line_dynamic,
    "line_wide_d": _build_line_dynamic_wide,
    "line_fd": lambda: _build_line_dynamic(filtered=True),
    "ribbon_id": _build_ribbon_id,
    "style_line": _build_style_line,
    "style_line_f": lambda: _build_style_line(filtered=True),
    "style_wide": _build_style_line_wide,
    "style_wide_f": lambda: _build_style_line_wide(filtered=True),
    "style_ribbon": _build_style_ribbon,
    "style_ribbon_f": lambda: _build_style_ribbon(filtered=True),
    "arrow": _build_arrow,
    "line_count": _build_line_count,
    "heb_line": _build_heb_line,
    "heb_line_wide": _build_heb_line_wide,
    "heb_count": _build_heb_count,
    "volume": _build_volume_slice,
    "resolve": _build_resolve,
    "depth_resolve": _build_depth_resolve,
    "dof": _build_dof,
    "dof_tex": _build_dof_tex,
    "dof_comp": _build_dof_comp,
}
_COMPILED = {}
_TRIED = set()


def _gpu_ready():
    try:
        return gpu.platform.backend_type_get() != 'NONE'
    except Exception:  # noqa: BLE001 - raises outright before gpu.init()
        return False


def _compile(name):
    if name in _TRIED:
        return _COMPILED.get(name)
    if not _gpu_ready():
        # No context is not a rejected shader: caching this failure would
        # permanently disable any feature queried before the first draw.
        return None
    _TRIED.add(name)
    try:
        _COMPILED[name] = _BUILDERS[name]()
    except Exception:  # noqa: BLE001 - unsupported backend / compile error
        _COMPILED[name] = None
    return _COMPILED[name]


def _variant(base, filtered, animated):
    suffix = {(False, False): "", (True, False): "_f",
              (False, True): "_d", (True, True): "_fd"}[(bool(filtered),
                                                         bool(animated))]
    return base + suffix


def get_round_point_shader(filtered=False, animated=False):
    return _compile(_variant("round_point", filtered, animated))


def get_line_filter_shader():
    return _compile("line_f")


def get_heb_line_shader():
    return _compile("heb_line")


def get_heb_count_shader():
    return _compile("heb_count")


def get_sphere_shader(filtered=False, animated=False):
    return _compile(_variant("sphere", filtered, animated))


def get_arrow_dynamic_shader():
    return _compile("arrow_d")


def get_arrow_flat_shader(animated=False):
    return _compile("arrow_flat_d" if animated else "arrow_flat")


def get_sphere_id_shader():
    return _compile("sphere_id")


def get_ribbon_shader(filtered=False, animated=False):
    return _compile(_variant("ribbon", filtered, animated))


def get_line_dynamic_shader(filtered=False):
    return _compile("line_fd" if filtered else "line_d")


def get_ribbon_id_shader():
    return _compile("ribbon_id")


def get_arrow_shader():
    return _compile("arrow")


def get_line_count_shader():
    return _compile("line_count")


def get_volume_shader():
    return _compile("volume")


_VOLUME_BATCH = None


def get_volume_batch(shader):
    global _VOLUME_BATCH
    if _VOLUME_BATCH is None:
        from gpu_extras.batch import batch_for_shader
        try:
            _VOLUME_BATCH = batch_for_shader(
                shader, 'TRI_FAN',
                {"corner": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))},
            )
        except Exception:  # noqa: BLE001
            return None
    return _VOLUME_BATCH


# The _failed / _available pairs tell "compiled and rejected" from "never asked".

def volume_failed():
    return "volume" in _TRIED and _COMPILED.get("volume") is None


def get_dof_shader():
    return _compile("dof")


def get_resolve_shader():
    return _compile("resolve")


def get_depth_resolve_shader():
    return _compile("depth_resolve")


def get_dof_tex_shader():
    return _compile("dof_tex")


def get_dof_comp_shader():
    return _compile("dof_comp")


def round_point_available():
    return "round_point" in _TRIED and _COMPILED.get("round_point") is not None


def round_point_failed():
    return "round_point" in _TRIED and _COMPILED.get("round_point") is None


def sphere_failed():
    return "sphere" in _TRIED and _COMPILED.get("sphere") is None
