# Twin of ../shaders.py. Vertex inputs stay byte-identical to the lit twin in
# every variant, so the deciding pieces are imported from there. At mode == LIT
# with glyph == ROUND the two draw the same thing.

import gpu

from . import TOON_UBO_SLOT
from .glyphs import GLYPH_GLSL
from .shading import GLSL as _SHADING_GLSL, TOON_STRUCT

from ..glsl_source import glsl
from ..shaders import (
    _LIGHT_RIG_STRUCT,
    _add_filter,
    _add_pos,
    _gpu_ready,
    _guard,
    _prelude,
    _pos_prelude,
    _variant,
)

_TYPEDEFS = _LIGHT_RIG_STRUCT + TOON_STRUCT


# Both stages call this, and INNER has to return exactly 1.0, so keep one copy.
_GROW_GLSL = glsl("toon_grow")


# OUTER grows the quad by (1 + width), so the beauty pass covers more fragments
# than the ungrown ID pass and clicking the ink misses the node. INNER is default.

def _build_toon_sphere(filtered=False, animated=False):
    vout = gpu.types.GPUStageInterfaceInfo("scig_toon_sphere_iface")
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
                    typedefs=_TYPEDEFS)
    else:
        info.typedef_source(_TYPEDEFS)
    info.uniform_buf(TOON_UBO_SLOT, "ToonParams", "u_toon")
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
        _prelude(filtered) + _pos_prelude(animated) + _GROW_GLSL +
        "void main()"
        "{"
        "  " + _guard(filtered) +
        f"  vec4 view_center = u_view * vec4({src}, 1.0);"
        "  vec3 vc = view_center.xyz;"
        "  vec3 view_pos = vc + vec3(corner * (radius * scig_toon_grow()), 0.0);"
        "  gl_Position = u_proj * vec4(view_pos, 1.0);"
        "  v_uv = corner;"
        "  v_view_center = vc;"
        "  v_radius = radius;"
        "  v_color = color;"
        "}"
    )
    info.fragment_source(
        GLYPH_GLSL + _SHADING_GLSL + _GROW_GLSL + glsl("toon_sphere_frag")
    )
    return gpu.shader.create_from_info(info)


# Edge ribbon: no glyphs, and the transverse coordinate serves as both rim and
# outline. The quad never grows, since a wider tube would misreport edge weight.

def _build_toon_ribbon(filtered=False, animated=False):
    vout = gpu.types.GPUStageInterfaceInfo("scig_toon_ribbon_iface")
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
                    typedefs=_TYPEDEFS)
    else:
        info.typedef_source(_TYPEDEFS)
    info.uniform_buf(TOON_UBO_SLOT, "ToonParams", "u_toon")
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
    info.fragment_source(_SHADING_GLSL + glsl("toon_ribbon_frag"))
    return gpu.shader.create_from_info(info)


# Round point: FLAT whatever the mode says. A sprite carries only gl_PointCoord
# and a color, and sprite size is fixed-function state, so the quad cannot grow.
# scig_toon_shade reads a u_light this program has no rig for.
_LIGHT_RIG_STUB = (
    "LightRig u_light = LightRig(vec4(0.0), vec4(0.0), vec4(0.0),"
    " vec4(0.0), vec4(0.0));"
)


def _build_toon_point(filtered=False, animated=False):
    vout = gpu.types.GPUStageInterfaceInfo("scig_toon_round_iface")
    vout.smooth('VEC4', "f_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_mvp")
    if animated:
        info.vertex_in(0, 'FLOAT', "n_idx")
        _add_pos(info, sampler_slot=1 if filtered else 0)
    else:
        info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC4', "color")
    if filtered:
        _add_filter(info, sampler_slot=0, attr_slot=2, ubo_slot=0,
                    typedefs=_TYPEDEFS)
    else:
        info.typedef_source(_TYPEDEFS)
    info.uniform_buf(TOON_UBO_SLOT, "ToonParams", "u_toon")
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
    info.fragment_source(
        GLYPH_GLSL + _LIGHT_RIG_STUB + _SHADING_GLSL + glsl("toon_point_frag")
    )
    return gpu.shader.create_from_info(info)


# Static mesh only: _build_arrow_dynamic must match batches.py vertex for vertex.

def _build_toon_arrow():
    vout = gpu.types.GPUStageInterfaceInfo("scig_toon_arrow_iface")
    vout.smooth('VEC3', "v_normal")
    vout.smooth('VEC4', "v_color")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "u_view")
    info.push_constant('MAT4', "u_proj")
    info.typedef_source(_TYPEDEFS)
    info.uniform_buf(0, "LightRig", "u_light")
    info.uniform_buf(TOON_UBO_SLOT, "ToonParams", "u_toon")
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC3', "nrm")
    info.vertex_in(2, 'VEC4', "color")
    info.vertex_out(vout)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(glsl("toon_arrow_vert"))
    info.fragment_source(_SHADING_GLSL + glsl("toon_arrow_frag"))
    return gpu.shader.create_from_info(info)



_BUILDERS = {
    "toon_sphere": _build_toon_sphere,
    "toon_sphere_f": lambda: _build_toon_sphere(filtered=True),
    "toon_sphere_d": lambda: _build_toon_sphere(animated=True),
    "toon_sphere_fd": lambda: _build_toon_sphere(filtered=True, animated=True),
    "toon_ribbon": _build_toon_ribbon,
    "toon_ribbon_f": lambda: _build_toon_ribbon(filtered=True),
    "toon_ribbon_d": lambda: _build_toon_ribbon(animated=True),
    "toon_ribbon_fd": lambda: _build_toon_ribbon(filtered=True, animated=True),
    "toon_point": _build_toon_point,
    "toon_point_f": lambda: _build_toon_point(filtered=True),
    "toon_point_d": lambda: _build_toon_point(animated=True),
    "toon_point_fd": lambda: _build_toon_point(filtered=True, animated=True),
    "toon_arrow": _build_toon_arrow,
}
_COMPILED = {}
_TRIED = set()


def _compile(name):
    if name in _TRIED:
        return _COMPILED.get(name)
    if not _gpu_ready():
        # No context yet is not a failure; recording one kills the style for good.
        return None
    _TRIED.add(name)
    try:
        _COMPILED[name] = _BUILDERS[name]()
    except Exception:  # noqa: BLE001 - unsupported backend / compile error
        _COMPILED[name] = None
    return _COMPILED[name]


def get_toon_sphere_shader(filtered=False, animated=False):
    """Same vertex layout as _build_sphere; None if the backend cannot build it."""
    return _compile(_variant("toon_sphere", filtered, animated))


def get_toon_ribbon_shader(filtered=False, animated=False):
    """Same vertex layout as _build_ribbon. Bind its light rig with
    ``with_rim=False`` or LIT mode gains a rim the lit path does not have."""
    return _compile(_variant("toon_ribbon", filtered, animated))


def get_toon_point_shader(filtered=False, animated=False):
    """Same vertex layout as _build_round_point; takes no light rig."""
    return _compile(_variant("toon_point", filtered, animated))


def get_toon_arrow_shader(animated=False):
    """Always None when ``animated``; callers fall back to the lit cone."""
    if animated:
        return None
    return _compile("toon_arrow")


def toon_sphere_failed():
    return "toon_sphere" in _TRIED and _COMPILED.get("toon_sphere") is None


def toon_ribbon_failed():
    return "toon_ribbon" in _TRIED and _COMPILED.get("toon_ribbon") is None


def available():
    """Compiles the sphere on the first call; a failed attempt is not retried."""
    return get_toon_sphere_shader() is not None
