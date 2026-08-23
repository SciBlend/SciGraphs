# Packs a Settings snapshot into the 112-byte std140 struct in toon/shading.py.
# No defaults here; one would drift from properties.py. The two ids ride in a
# color's alpha, which std140 pads a lone float out to anyway.

import gpu
import numpy as np

from . import GLYPH_IDS, MODE_IDS, OUTLINE_IDS, is_stylized


def mode_id(st):
    # Indexed, not .get(x, 0): a miss means MODE_IDS and the enum drifted.
    return MODE_IDS[st.toon_mode]


def glyph_id(st):
    return GLYPH_IDS[st.toon_glyph]


def stylized_edges(st):
    return bool(is_stylized(st) and st.toon_edges)


def _pack(st):
    """Split out of ``toon_ubo`` so tests can assert on the numbers."""
    tint = st.toon_shadow_tint
    warm = st.toon_warm
    cool = st.toon_cool
    line = st.toon_outline_color
    return np.array([
        # bands
        float(st.toon_steps), float(st.toon_softness),
        float(st.toon_spec), float(st.toon_spec_size),
        # outline
        line[0], line[1], line[2], float(st.toon_outline_width),
        # shadow
        tint[0], tint[1], tint[2], float(st.toon_rim),
        # warm
        warm[0], warm[1], warm[2], float(mode_id(st)),
        # cool
        cool[0], cool[1], cool[2], float(glyph_id(st)),
        # screen
        float(st.toon_halftone_scale), float(st.toon_halftone_angle),
        float(OUTLINE_IDS[st.toon_outline_mode]), float(st.toon_glyph_rotation),
        # misc
        float(st.toon_gloss), float(st.toon_glyph_bulge), 0.0, 0.0,
    ], dtype=np.float32)


def toon_ubo(st):
    """Hold the returned GPUUniformBuf until after the draw call. Drop it
    earlier and the backing buffer can be freed mid-draw."""
    data = _pack(st)
    buf = gpu.types.Buffer('FLOAT', data.size, data)
    return gpu.types.GPUUniformBuf(buf)
