# The ``scigraphs_preview_toon_*`` prefix is load-bearing: host._rna_properties
# introspects it, so a declaration below is all that makes a property reach the
# snapshot as ``st.toon_*``, and its declared default is the only default.
# _on_setting_update is a local copy of ../properties.py's, whose import would
# be circular. Never _on_rebuild_update: it drops batches on a drag.

import bpy

from ..state import tag_redraw
from . import MODE_IDS, OUTLINE_IDS
from .glyphs import GLYPH_ITEMS


def _on_setting_update(self, context):
    tag_redraw()


# Built from MODE_IDS, so the enum cannot offer a mode the packer has no id for.
_MODE_TEXT = {
    'LIT': ("Lit", "The standard impostor lighting. Leave it here and only the "
                   "glyph shape changes"),
    'TOON': ("Cel", "Quantize the light into bands with a hard stepped "
                    "specular: the comic-book look"),
    'GOOCH': ("Gooch", "Cool-to-warm technical illustration ramp. Shape reads "
                       "through the hue rather than through the brightness, so "
                       "nothing goes black"),
    'FLAT': ("Flat", "Unlit constant color. The vector-figure look, and what "
                     "makes an outline read"),
    'HALFTONE': ("Halftone", "The cel ramp printed as a rotated screen of dots: "
                             "darker means bigger dots"),
}

MODE_ITEMS = tuple(
    (mode, _MODE_TEXT[mode][0], _MODE_TEXT[mode][1])
    for mode in sorted(MODE_IDS, key=MODE_IDS.get)
)

_OUTLINE_TEXT = {
    'INNER': ("Inner", "Draw the outline inside the silhouette. The glyph keeps "
                       "its size and the beauty pass keeps the ID pass's "
                       "silhouette, which is why this is the default"),
    'OUTER': ("Outer", "Expand the quad and draw the outline outside the "
                       "silhouette. Thicker and more even, but the drawn "
                       "silhouette is wider than the one picking sees"),
}

OUTLINE_ITEMS = tuple(
    (mode, _OUTLINE_TEXT[mode][0], _OUTLINE_TEXT[mode][1])
    for mode in sorted(OUTLINE_IDS, key=OUTLINE_IDS.get)
)


def register_properties():
    S = bpy.types.Scene

    S.scigraphs_preview_toon_mode = bpy.props.EnumProperty(
        name="Shading",
        description=(
            "How nodes and edges are shaded. Everything but Lit is a "
            "non-photorealistic model evaluated in the same impostor, so "
            "switching costs a redraw and never a rebuild"
        ),
        items=MODE_ITEMS,
        default='LIT', update=_on_setting_update,
    )
    S.scigraphs_preview_toon_steps = bpy.props.IntProperty(
        name="Bands",
        description="How many brightness bands the light is quantized into",
        default=3, min=2, max=8, update=_on_setting_update,
    )
    S.scigraphs_preview_toon_softness = bpy.props.FloatProperty(
        name="Band Softness",
        description=(
            "How far each band edge is feathered. A little is not a stylistic "
            "choice: a hard step aliases badly on a curved impostor"
        ),
        default=0.06, min=0.0, max=0.5, update=_on_setting_update,
    )
    S.scigraphs_preview_toon_spec = bpy.props.FloatProperty(
        name="Specular",
        description="Strength of the hard stepped highlight (0 removes it)",
        default=0.35, min=0.0, max=1.0, subtype='FACTOR',
        update=_on_setting_update,
    )
    S.scigraphs_preview_toon_spec_size = bpy.props.FloatProperty(
        name="Specular Size",
        description="How much of the impostor the stepped highlight covers",
        default=0.25, min=0.0, max=1.0, subtype='FACTOR',
        update=_on_setting_update,
    )
    S.scigraphs_preview_toon_gloss = bpy.props.FloatProperty(
        name="Gloss",
        description=(
            "Exponent of the highlight lobe before it is stepped. Higher is a "
            "tighter, harder dot"
        ),
        default=32.0, min=4.0, max=128.0, update=_on_setting_update,
    )
    S.scigraphs_preview_toon_rim = bpy.props.FloatProperty(
        name="Rim",
        description=(
            "Strength of a stepped rim light at the silhouette. Separate from "
            "the lighting rig's smooth rim, which the stylized modes do not use"
        ),
        default=0.0, min=0.0, max=2.0, update=_on_setting_update,
    )
    S.scigraphs_preview_toon_shadow_tint = bpy.props.FloatVectorProperty(
        name="Shadow Tint",
        description=(
            "The darkest band multiplies the node color by this. A cool tint "
            "instead of plain black is what keeps a cel render from reading as "
            "a hole in the image"
        ),
        subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.45, 0.50, 0.62), update=_on_setting_update,
    )
    S.scigraphs_preview_toon_warm = bpy.props.FloatVectorProperty(
        name="Warm",
        description="Lit end of the Gooch ramp",
        subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.85, 0.55, 0.15), update=_on_setting_update,
    )
    S.scigraphs_preview_toon_cool = bpy.props.FloatVectorProperty(
        name="Cool",
        description="Unlit end of the Gooch ramp",
        subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.10, 0.20, 0.55), update=_on_setting_update,
    )
    S.scigraphs_preview_toon_outline_width = bpy.props.FloatProperty(
        name="Outline Width",
        description=(
            "Outline thickness as a fraction of the node radius, so it stays "
            "proportional under any zoom. 0 switches the outline off, and is "
            "the default because a non-zero one would put every untouched "
            "scene on the stylized path (see toon.is_stylized)"
        ),
        default=0.0, min=0.0, max=0.5, update=_on_setting_update,
    )
    S.scigraphs_preview_toon_outline_color = bpy.props.FloatVectorProperty(
        name="Outline Color",
        description="Color mixed in near the silhouette",
        subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0), update=_on_setting_update,
    )
    S.scigraphs_preview_toon_outline_mode = bpy.props.EnumProperty(
        name="Outline Side",
        description="Whether the outline eats into the glyph or grows past it",
        items=OUTLINE_ITEMS,
        default='INNER', update=_on_setting_update,
    )
    S.scigraphs_preview_toon_halftone_scale = bpy.props.FloatProperty(
        name="Dot Scale",
        description=(
            "Halftone dots per 100 device pixels. The screen is screen-space, "
            "so the dot size does not change when the graph is zoomed"
        ),
        default=14.0, min=1.0, max=64.0, update=_on_setting_update,
    )
    S.scigraphs_preview_toon_halftone_angle = bpy.props.FloatProperty(
        name="Dot Angle",
        description="Rotation of the halftone screen, in radians",
        default=0.4, min=0.0, max=3.15, subtype='ANGLE',
        update=_on_setting_update,
    )
    S.scigraphs_preview_toon_glyph = bpy.props.EnumProperty(
        name="Glyph",
        description=(
            "Shape of a node. Independent of the shading mode: the impostor "
            "quad is unchanged and only the mask and the relief normal differ, "
            "so a square node can still be lit"
        ),
        items=GLYPH_ITEMS,
        default='ROUND', update=_on_setting_update,
    )
    S.scigraphs_preview_toon_glyph_rotation = bpy.props.FloatProperty(
        name="Glyph Rotation",
        description="Rotation of the glyph in its quad, in radians",
        default=0.0, min=0.0, max=6.30, subtype='ANGLE',
        update=_on_setting_update,
    )
    S.scigraphs_preview_toon_glyph_bulge = bpy.props.FloatProperty(
        name="Relief",
        description=(
            "How much the glyph is inflated out of its plane. 1 is full "
            "spherical relief (a round glyph is then exactly the sphere "
            "impostor), 0 is a flat plate facing the camera"
        ),
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        update=_on_setting_update,
    )
    S.scigraphs_preview_toon_edges = bpy.props.BoolProperty(
        name="Stylize Edges",
        description=(
            "Shade the edge tubes and the arrow cones with the same model, so "
            "a stylized render is coherent instead of toon nodes among lit "
            "tubes"
        ),
        default=True, update=_on_setting_update,
    )


_PROP_NAMES = (
    "scigraphs_preview_toon_mode",
    "scigraphs_preview_toon_steps",
    "scigraphs_preview_toon_softness",
    "scigraphs_preview_toon_spec",
    "scigraphs_preview_toon_spec_size",
    "scigraphs_preview_toon_gloss",
    "scigraphs_preview_toon_rim",
    "scigraphs_preview_toon_shadow_tint",
    "scigraphs_preview_toon_warm",
    "scigraphs_preview_toon_cool",
    "scigraphs_preview_toon_outline_width",
    "scigraphs_preview_toon_outline_color",
    "scigraphs_preview_toon_outline_mode",
    "scigraphs_preview_toon_halftone_scale",
    "scigraphs_preview_toon_halftone_angle",
    "scigraphs_preview_toon_glyph",
    "scigraphs_preview_toon_glyph_rotation",
    "scigraphs_preview_toon_glyph_bulge",
    "scigraphs_preview_toon_edges",
)


def unregister_properties():
    S = bpy.types.Scene
    for name in _PROP_NAMES:
        if hasattr(S, name):
            delattr(S, name)
