# Stylized (toon / NPR) shading: a second lighting model over the same quad and
# the same GPUBatch. A UBO rather than push constants, since two mat4 already
# fill the guaranteed 128-byte block. Toon is block 2, after light 0, filter 1.

TOON_UBO_SLOT = 2

# One list, read by the property enum, the packer and the GLSL.
MODE_IDS = {
    'LIT': 0,
    'TOON': 1,
    'GOOCH': 2,
    'FLAT': 3,
    'HALFTONE': 4,
}

# Travels as a float in ToonParams.screen.z; GLSL tests ``> 0.5``, not equality.
OUTLINE_IDS = {
    'INNER': 0,
    'OUTER': 1,
}

GLYPH_IDS = {
    'ROUND': 0,
    'SQUARE': 1,
    'DIAMOND': 2,
    'TRIANGLE': 3,
    'HEXAGON': 4,
    'STAR': 5,
    'RING': 6,
    'CROSS': 7,
}


def is_stylized(st):
    """False means nothing here runs. Outline width counts and defaults to 0."""
    return (getattr(st, "toon_mode", 'LIT') != 'LIT'
            or getattr(st, "toon_glyph", 'ROUND') != 'ROUND'
            or float(getattr(st, "toon_outline_width", 0.0)) > 0.0)
