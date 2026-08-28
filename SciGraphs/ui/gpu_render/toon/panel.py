# Sits under SCIGRAPHS_RENDER_PT_lighting: the rig is what these modes quantize.

import bpy

_BANDS = {'TOON', 'HALFTONE'}
_SPECULAR = {'TOON', 'GOOCH', 'HALFTONE'}
_RIM = {'TOON', 'HALFTONE'}
_GOOCH = {'GOOCH'}
_HALFTONE = {'HALFTONE'}


class SCIGRAPHS_RENDER_PT_toon(bpy.types.Panel):
    """Non-photorealistic shading and node glyphs."""
    bl_label = "Stylize"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_order = 7
    COMPAT_ENGINES = {'SCIGRAPHS'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        mode = scene.scigraphs_preview_toon_mode
        glyph = scene.scigraphs_preview_toon_glyph

        layout.prop(scene, "scigraphs_preview_toon_mode", text="Shading")

        if mode == 'FLAT':
            # Flat is unlit and the rest inert; an empty panel looks broken.
            layout.label(text="Unlit: node color straight through",
                         icon='INFO')

        if mode in _BANDS:
            col = layout.column(align=True)
            col.prop(scene, "scigraphs_preview_toon_steps", text="Bands")
            col.prop(scene, "scigraphs_preview_toon_softness", text="Softness")
            col.prop(scene, "scigraphs_preview_toon_shadow_tint",
                     text="Shadow Tint")

        if mode in _GOOCH:
            col = layout.column(align=True)
            col.prop(scene, "scigraphs_preview_toon_warm", text="Warm")
            col.prop(scene, "scigraphs_preview_toon_cool", text="Cool")

        if mode in _SPECULAR:
            col = layout.column(align=True)
            col.prop(scene, "scigraphs_preview_toon_spec", text="Specular")
            sub = col.column(align=True)
            sub.active = scene.scigraphs_preview_toon_spec > 0.0
            sub.prop(scene, "scigraphs_preview_toon_spec_size", text="Size")
            sub.prop(scene, "scigraphs_preview_toon_gloss", text="Gloss")

        if mode in _RIM:
            layout.prop(scene, "scigraphs_preview_toon_rim", text="Rim")

        if mode in _HALFTONE:
            col = layout.column(align=True)
            col.prop(scene, "scigraphs_preview_toon_halftone_scale",
                     text="Dot Scale")
            col.prop(scene, "scigraphs_preview_toon_halftone_angle",
                     text="Dot Angle")

        layout.separator()
        header = layout.row()
        header.label(text="Glyph Shape", icon='MESH_CIRCLE')
        col = layout.column(align=True)
        col.prop(scene, "scigraphs_preview_toon_glyph", text="Shape")
        rot = col.row(align=True)
        rot.active = glyph != 'ROUND'
        rot.prop(scene, "scigraphs_preview_toon_glyph_rotation",
                 text="Rotation")
        col.prop(scene, "scigraphs_preview_toon_glyph_bulge", text="Relief")
        note = layout.row()
        note.label(text="Glyphs work in any shading mode, Lit included",
                   icon='INFO')

        layout.separator()
        row = layout.row()
        # Mirrors toon.is_stylized and has to stay in step with it.
        row.active = (mode != 'LIT' or glyph != 'ROUND'
                      or scene.scigraphs_preview_toon_outline_width > 0.0)
        row.prop(scene, "scigraphs_preview_toon_edges", text="Stylize Edges")


class SCIGRAPHS_RENDER_PT_toon_outline(bpy.types.Panel):
    """Silhouette outline. Its own panel because it works in every mode."""
    bl_label = "Outline"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_parent_id = "SCIGRAPHS_RENDER_PT_toon"
    COMPAT_ENGINES = {'SCIGRAPHS'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene

        layout.prop(scene, "scigraphs_preview_toon_outline_width",
                    text="Width")
        col = layout.column(align=True)
        col.active = scene.scigraphs_preview_toon_outline_width > 0.0
        col.prop(scene, "scigraphs_preview_toon_outline_color", text="Color")
        col.prop(scene, "scigraphs_preview_toon_outline_mode", text="Side")


CLASSES = (SCIGRAPHS_RENDER_PT_toon, SCIGRAPHS_RENDER_PT_toon_outline)
