# Reproducible pipeline operators for SciGraphs
#
# Provides UI operators for validating, executing, and exporting
# reproducible pipelines.

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper, ExportHelper
import os


class SCIGRAPHS_OT_RunPipeline(Operator, ImportHelper):
    """Execute a reproducible pipeline from YAML/JSON file"""
    bl_idname = "scigraphs.run_pipeline"
    bl_label = "Run Pipeline"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".yaml"
    filter_glob: StringProperty(
        default="*.yaml;*.yml;*.json",
        options={'HIDDEN'},
    )

    stop_on_error: BoolProperty(
        name="Stop on Error",
        description="Stop execution if any step fails",
        default=True,
    )

    verbose: BoolProperty(
        name="Verbose",
        description="Print detailed progress to console",
        default=True,
    )

    def invoke(self, context, event):
        props = getattr(context.scene, "scigraphs_repro", None)
        if props and props.pipeline_path:
            self.filepath = bpy.path.abspath(props.pipeline_path)
            return self.execute(context)
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        from ....core.repro import parse_pipeline, PipelineExecutor

        filepath = bpy.path.abspath(self.filepath)
        if not os.path.isfile(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        try:
            # Parse pipeline
            schema, raw_dict, pipeline_hash = parse_pipeline(filepath)

            # Execute
            executor = PipelineExecutor(
                stop_on_error=self.stop_on_error,
                verbose=self.verbose,
            )
            result = executor.execute(schema, raw_dict, pipeline_hash)

            if result.success:
                self.report({'INFO'}, f"Pipeline executed successfully. Output: {result.output_dir}")
            else:
                errors = "; ".join(result.errors[:3])
                self.report({'WARNING'}, f"Pipeline completed with errors: {errors}")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Pipeline execution failed: {e}")
            return {'CANCELLED'}


class SCIGRAPHS_OT_ValidatePipeline(Operator, ImportHelper):
    """Validate a pipeline file without executing it"""
    bl_idname = "scigraphs.validate_pipeline"
    bl_label = "Validate Pipeline"
    bl_options = {'REGISTER'}

    filename_ext = ".yaml"
    filter_glob: StringProperty(
        default="*.yaml;*.yml;*.json",
        options={'HIDDEN'},
    )

    def invoke(self, context, event):
        props = getattr(context.scene, "scigraphs_repro", None)
        if props and props.pipeline_path:
            self.filepath = bpy.path.abspath(props.pipeline_path)
            return self.execute(context)
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        from ....core.repro.parser import parse_pipeline
        from ....core.repro.schema import ValidationError

        filepath = bpy.path.abspath(self.filepath)
        if not os.path.isfile(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        try:
            schema, raw_dict, pipeline_hash = parse_pipeline(filepath)
            self.report({'INFO'}, f"Pipeline is valid! Hash: {pipeline_hash[:16]}...")
            return {'FINISHED'}

        except ValidationError as e:
            self.report({'ERROR'}, f"Validation failed: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error reading pipeline: {e}")
            return {'CANCELLED'}


class SCIGRAPHS_OT_ExportPipelineTemplate(Operator, ExportHelper):
    """Export a pipeline template for reproducible workflows"""
    bl_idname = "scigraphs.export_pipeline_template"
    bl_label = "Export Pipeline Template"
    bl_options = {'REGISTER'}

    filename_ext = ".yaml"
    filter_glob: StringProperty(
        default="*.yaml;*.yml;*.json",
        options={'HIDDEN'},
    )

    template_type: EnumProperty(
        name="Template Type",
        items=[
            ('MINIMAL', "Minimal", "Basic pipeline structure"),
            ('OSMNX', "OSMnx Network", "Template for OSMnx street networks"),
            ('GEXF', "GEXF/GraphML", "Template for file-based graphs"),
            ('FULL', "Full Example", "Complete pipeline with all sections"),
        ],
        default='OSMNX',
    )

    format: EnumProperty(
        name="Format",
        items=[
            ('YAML', "YAML", "Human-readable YAML format"),
            ('JSON', "JSON", "JSON format"),
        ],
        default='YAML',
    )

    def execute(self, context):
        import json

        templates = {
            'MINIMAL': {
                "meta": {
                    "title": "minimal_pipeline",
                    "seed": 42,
                    "output_dir": "//repro/output",
                },
            },
            'OSMNX': {
                "meta": {
                    "title": "osmnx_network_analysis",
                    "seed": 42,
                    "output_dir": "//repro/osmnx_output",
                    "description": "OSMnx street network analysis pipeline",
                },
                "dataset": {
                    "source": "osmnx",
                    "method": "PLACE",
                    "query": "Granada, Spain",
                    "network_type": "walk",
                    "simplify": True,
                    "cache": True,
                },
                "analysis": {
                    "metrics": ["degree", "betweenness", "closeness"],
                    "clustering": {
                        "algorithm": "louvain",
                        "resolution": 1.0,
                    },
                },
                "layout": {
                    "algorithm": "YIFAN_HU",
                    "scale": 5.0,
                    "iterations": 50,
                },
                "visual": {
                    "setup_geometry_nodes": True,
                    "node_color": "betweenness",
                    "node_size": "degree",
                },
                "exports": {
                    "graph": "network.gexf",
                    "positions": "positions.csv",
                    "statistics": "stats.txt",
                },
            },
            'GEXF': {
                "meta": {
                    "title": "gexf_visualization",
                    "seed": 42,
                    "output_dir": "//repro/gexf_output",
                },
                "dataset": {
                    "source": "gexf",
                    "filepath": "//data/network.gexf",
                    "auto_layout": True,
                },
                "layout": {
                    "algorithm": "SPRING_3D",
                    "scale": 5.0,
                    "iterations": 100,
                },
                "visual": {
                    "setup_geometry_nodes": True,
                    "edge_style": "CYTOSCAPE_BEZIER",
                },
                "render": {
                    "engine": "CYCLES",
                    "resolution": [1920, 1080],
                    "samples": 128,
                    "output": "figure.png",
                },
            },
            'FULL': {
                "meta": {
                    "title": "full_analysis_pipeline",
                    "seed": 42,
                    "output_dir": "//repro/full_output",
                    "description": "Complete example with all pipeline sections",
                    "version": "1.0",
                },
                "dataset": {
                    "source": "osmnx",
                    "method": "PLACE",
                    "query": "Barcelona, Spain",
                    "network_type": "drive",
                    "simplify": True,
                    "cache": True,
                },
                "analysis": {
                    "metrics": ["degree", "betweenness", "closeness", "pagerank"],
                    "clustering": {
                        "algorithm": "louvain",
                        "resolution": 1.0,
                    },
                    "normalize": True,
                },
                "layout": {
                    "algorithm": "YIFAN_HU",
                    "scale": 10.0,
                    "iterations": 100,
                },
                "visual": {
                    "setup_geometry_nodes": True,
                    "node_color": "betweenness",
                    "node_size": "degree",
                    "edge_style": "GEPHI_DEFAULT",
                    "colormap": "viridis",
                },
                "render": {
                    "engine": "CYCLES",
                    "resolution": [3840, 2160],
                    "samples": 256,
                    "output": "figure_4k.png",
                    "transparent": False,
                    "denoise": True,
                },
                "exports": {
                    "graph": "network.gexf",
                    "positions": "positions.csv",
                    "statistics": "statistics.txt",
                    "blend": "scene.blend",
                },
                "ops": [
                    {
                        "id": "scigraphs.setup_lighting",
                        "props": {},
                    },
                ],
            },
        }

        template = templates.get(self.template_type, templates['MINIMAL'])
        filepath = self.filepath

        # Ensure correct extension
        if self.format == 'YAML':
            if not filepath.endswith(('.yaml', '.yml')):
                filepath = os.path.splitext(filepath)[0] + '.yaml'
        else:
            if not filepath.endswith('.json'):
                filepath = os.path.splitext(filepath)[0] + '.json'

        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                if self.format == 'YAML':
                    try:
                        import yaml
                        yaml.safe_dump(template, f, default_flow_style=False, sort_keys=False)
                    except ImportError:
                        json.dump(template, f, indent=2)
                        self.report({'WARNING'}, "YAML not available, saved as JSON-formatted YAML")
                else:
                    json.dump(template, f, indent=2)

            self.report({'INFO'}, f"Template exported to: {filepath}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}


class SCIGRAPHS_OT_ExportCurrentReproSpec(Operator, ExportHelper):
    """Export an approximate pipeline spec from the current scene"""
    bl_idname = "scigraphs.export_current_repro_spec"
    bl_label = "Export Current as Pipeline"
    bl_options = {'REGISTER'}

    filename_ext = ".yaml"
    filter_glob: StringProperty(
        default="*.yaml;*.yml;*.json",
        options={'HIDDEN'},
    )

    format: EnumProperty(
        name="Format",
        items=[
            ('YAML', "YAML", "Human-readable YAML format"),
            ('JSON', "JSON", "JSON format"),
        ],
        default='YAML',
    )

    def execute(self, context):
        import json

        # Find active graph object
        obj = context.active_object
        if obj is None or "num_nodes" not in obj:
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}

        props = context.scene.scigraphs

        # Build spec from current state
        spec = {
            "meta": {
                "title": f"scene_{obj.name}",
                "seed": 42,
                "output_dir": "//repro/exported",
                "description": f"Exported from scene with {obj['num_nodes']} nodes",
            },
        }

        # Detect source type
        if obj.get("is_osmnx"):
            spec["dataset"] = {
                "source": "osmnx",
                "method": props.osmnx_method if hasattr(props, 'osmnx_method') else "PLACE",
                "query": props.osmnx_place if hasattr(props, 'osmnx_place') else "",
                "network_type": props.osmnx_network_type if hasattr(props, 'osmnx_network_type') else "drive",
            }
        elif obj.get("is_geospatial"):
            spec["dataset"] = {
                "source": "gexf",
                "filepath": "//data/graph.gexf",
            }

        # Layout info
        if "layout_algorithm" in obj:
            spec["layout"] = {
                "algorithm": obj["layout_algorithm"],
                "scale": props.layout_scale if hasattr(props, 'layout_scale') else 5.0,
            }

        # Visual settings
        spec["visual"] = {
            "setup_geometry_nodes": True,
        }
        if hasattr(props, 'node_color_attribute') and props.node_color_attribute:
            spec["visual"]["node_color"] = props.node_color_attribute
        if hasattr(props, 'edge_style_preset') and props.edge_style_preset:
            spec["visual"]["edge_style"] = props.edge_style_preset

        # Render settings
        render = context.scene.render
        spec["render"] = {
            "engine": render.engine,
            "resolution": [render.resolution_x, render.resolution_y],
            "output": "render.png",
        }

        filepath = self.filepath

        # Ensure correct extension
        if self.format == 'YAML':
            if not filepath.endswith(('.yaml', '.yml')):
                filepath = os.path.splitext(filepath)[0] + '.yaml'
        else:
            if not filepath.endswith('.json'):
                filepath = os.path.splitext(filepath)[0] + '.json'

        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                if self.format == 'YAML':
                    try:
                        import yaml
                        yaml.safe_dump(spec, f, default_flow_style=False, sort_keys=False)
                    except ImportError:
                        json.dump(spec, f, indent=2)
                else:
                    json.dump(spec, f, indent=2)

            self.report({'INFO'}, f"Pipeline spec exported to: {filepath}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}


class SCIGRAPHS_OT_ExportReproReference(Operator, ExportHelper):
    """Generate a Markdown reference of every pipeline option"""
    bl_idname = "scigraphs.export_repro_reference"
    bl_label = "Export Options Reference"
    bl_options = {'REGISTER'}

    filename_ext = ".md"
    filter_glob: StringProperty(
        default="*.md",
        options={'HIDDEN'},
    )

    def execute(self, context):
        from ....core.repro.reference import write_reference_markdown

        filepath = self.filepath
        if not filepath.endswith(".md"):
            filepath = os.path.splitext(filepath)[0] + ".md"

        try:
            write_reference_markdown(filepath)
            self.report({'INFO'}, f"Pipeline options reference written to: {filepath}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Reference export failed: {e}")
            return {'CANCELLED'}


class SCIGRAPHS_OT_OpenArtifactsFolder(Operator):
    """Open the pipeline artifacts folder in file browser"""
    bl_idname = "scigraphs.open_artifacts_folder"
    bl_label = "Open Artifacts Folder"
    bl_options = {'REGISTER'}

    folder_path: StringProperty(
        name="Folder Path",
        default="",
    )

    def execute(self, context):
        import subprocess
        import platform

        path = self.folder_path
        if not path:
            # Use default repro folder relative to blend file
            blend_path = bpy.data.filepath
            if blend_path:
                path = os.path.join(os.path.dirname(blend_path), "repro")
            else:
                self.report({'WARNING'}, "No folder path specified and no blend file saved")
                return {'CANCELLED'}

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        # Open folder in system file browser
        system = platform.system()
        try:
            if system == 'Windows':
                subprocess.run(['explorer', path], check=False)
            elif system == 'Darwin':
                subprocess.run(['open', path], check=False)
            else:
                subprocess.run(['xdg-open', path], check=False)
            self.report({'INFO'}, f"Opened: {path}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Could not open folder: {e}")
            return {'CANCELLED'}


# Registration
classes = (
    SCIGRAPHS_OT_RunPipeline,
    SCIGRAPHS_OT_ValidatePipeline,
    SCIGRAPHS_OT_ExportPipelineTemplate,
    SCIGRAPHS_OT_ExportCurrentReproSpec,
    SCIGRAPHS_OT_ExportReproReference,
    SCIGRAPHS_OT_OpenArtifactsFolder,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
