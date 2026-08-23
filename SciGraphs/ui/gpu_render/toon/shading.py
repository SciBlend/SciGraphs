# typedef_source is injected before the UBO declarations, so nothing that reads
# u_light or u_toon can live there. mode == LIT must match _build_sphere.

from ..glsl_source import glsl

# All vec4 for a trivial std140 layout, 7 vec4 = 112 bytes per draw. bands: steps,
# softness, spec strength, spec size. outline: rgb, w = width over radius. shadow:
# rgb tint, w = rim. warm/cool: Gooch ends, w = mode id / glyph id. screen: dot
# scale, angle, outline mode, glyph rotation. misc: gloss exponent, glyph bulge.
TOON_STRUCT = glsl("toon_struct")


# Constants, not UBO fields, since each field costs 16 bytes in a block four
# programs write per draw. 0.2 and 0.6 are from Gooch et al. (1998).
GLSL = glsl("toon_shading")
