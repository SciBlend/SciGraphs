# Sidebar panel for managing reproducible pipelines.

import bpy
from bpy.types import Panel
from bpy.props import StringProperty, PointerProperty


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
