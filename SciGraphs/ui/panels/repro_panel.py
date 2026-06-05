# Reproducibility Panel for SciGraphs
#
# Lightweight sidebar panel for managing reproducible pipelines.

import bpy
from bpy.types import Panel
from bpy.props import StringProperty, PointerProperty


class SCIGRAPHS_PT_reproducibility(Panel):
    """Reproducible pipeline management panel."""
    bl_label = "Reproducibility"
    bl_idname = "SCIGRAPHS_PT_reproducibility"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SciGraphs"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs_repro

        # Pipeline file selector
        box = layout.box()
        box.label(text="Pipeline", icon='FILE_CACHE')

        col = box.column(align=True)
        col.prop(props, "pipeline_path", text="")

        row = col.row(align=True)
        row.operator("scigraphs.validate_pipeline", text="Validate", icon='CHECKMARK')
        row.operator("scigraphs.run_pipeline", text="Run", icon='PLAY')

        # Quick actions
        layout.separator()
        box = layout.box()
        box.label(text="Templates", icon='FILE_NEW')

        col = box.column(align=True)
        col.operator("scigraphs.export_pipeline_template", text="Export Template", icon='EXPORT')
        col.operator("scigraphs.export_current_repro_spec", text="Export Current Scene", icon='SCENE_DATA')
        col.operator("scigraphs.export_repro_reference", text="Export Options Reference", icon='HELP')

        # Output folder
        layout.separator()
        box = layout.box()
        box.label(text="Artifacts", icon='FOLDER_REDIRECT')

        col = box.column(align=True)
        col.prop(props, "artifacts_path", text="")
        col.operator("scigraphs.open_artifacts_folder", text="Open Folder", icon='FILEBROWSER')

        # Info box
        obj = context.active_object
        if obj and "num_nodes" in obj:
            layout.separator()
            box = layout.box()
            box.label(text="Current Graph", icon='OUTLINER_OB_MESH')

            col = box.column(align=True)
            col.label(text=f"Nodes: {obj['num_nodes']}")
            col.label(text=f"Edges: {obj.get('num_edges', 'N/A')}")

            if obj.get("is_osmnx"):
                col.label(text="Source: OSMnx", icon='WORLD')
            elif obj.get("is_geospatial"):
                col.label(text="Source: Geospatial", icon='WORLD_DATA')
            else:
                col.label(text="Source: Custom", icon='MESH_DATA')


class ScigraphsReproProperties(bpy.types.PropertyGroup):
    """Properties for reproducibility panel."""

    pipeline_path: StringProperty(
        name="Pipeline File",
        description="Path to pipeline YAML/JSON file",
        subtype='FILE_PATH',
        default="",
    )

    artifacts_path: StringProperty(
        name="Artifacts Folder",
        description="Output folder for pipeline artifacts",
        subtype='DIR_PATH',
        default="//repro/",
    )


classes = (
    ScigraphsReproProperties,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.scigraphs_repro = PointerProperty(type=ScigraphsReproProperties)


def unregister():
    del bpy.types.Scene.scigraphs_repro

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
