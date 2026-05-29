"""GIS-interchange export operators for the OSMnx panel.

Supports:
    - GeoPackage (.gpkg)      — ox.io.save_graph_geopackage
    - OSM XML (.osm)          — ox.io.save_graph_xml  (sets all_oneway first)
    - Gephi-compatible GraphML — ox.io.save_graphml(gephi=True)
    - SVG                      — matplotlib vector render of the graph
"""

import os
import bpy
from bpy.props import StringProperty

from .utils import _get_osmnx_graph


def _save_graph_geopackage(G, filepath):
    """Try modern API then legacy fallbacks."""
    try:
        import osmnx as ox
    except ImportError:
        return False, "OSMnx not installed"

    try:
        if hasattr(ox.io, "save_graph_geopackage"):
            ox.io.save_graph_geopackage(G, filepath=filepath)
        else:
            ox.save_graph_geopackage(G, filepath=filepath)
        return True, f"Saved GeoPackage: {os.path.basename(filepath)}"
    except Exception as e:
        return False, f"GeoPackage export failed: {e}"


def _save_graph_osm_xml(G, filepath):
    try:
        import osmnx as ox
    except ImportError:
        return False, "OSMnx not installed"

    try:
        prev = getattr(ox.settings, "all_oneway", None)
        ox.settings.all_oneway = True
        try:
            if hasattr(ox.io, "save_graph_xml"):
                ox.io.save_graph_xml(G, filepath=filepath)
            else:
                ox.save_graph_xml(G, filepath=filepath)
        finally:
            if prev is not None:
                ox.settings.all_oneway = prev
        return True, f"Saved OSM XML: {os.path.basename(filepath)}"
    except Exception as e:
        return False, f"OSM XML export failed: {e}"


def _save_graph_graphml_gephi(G, filepath):
    try:
        import osmnx as ox
    except ImportError:
        return False, "OSMnx not installed"

    try:
        if hasattr(ox.io, "save_graphml"):
            ox.io.save_graphml(G, filepath=filepath, gephi=True)
        else:
            ox.save_graphml(G, filepath=filepath, gephi=True)
        return True, f"Saved Gephi GraphML: {os.path.basename(filepath)}"
    except Exception as e:
        return False, f"Gephi GraphML export failed: {e}"


def _save_graph_svg(G, filepath):
    try:
        import osmnx as ox
    except ImportError:
        return False, "OSMnx not installed"

    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass

    try:
        # Project to metric CRS for correct SVG aspect when possible.
        try:
            Gp = ox.project_graph(G)
        except Exception:
            Gp = G
        fig, ax = ox.plot_graph(
            Gp, show=False, close=True, save=True, filepath=filepath,
            node_size=0, edge_color="black", edge_linewidth=0.5, bgcolor="white",
        )
        return True, f"Saved SVG: {os.path.basename(filepath)}"
    except Exception as e:
        return False, f"SVG export failed: {e}"


class SCIGRAPHS_OT_OSMnxExport(bpy.types.Operator):
    """Export the OSMnx graph to a GIS-interchange format."""
    bl_idname = "scigraphs.osmnx_export"
    bl_label = "Export Graph"
    bl_description = "Export the current OSMnx graph to GeoPackage / OSM XML / Gephi GraphML / SVG"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)

    def invoke(self, context, event):
        props = context.scene.scigraphs
        fmt = props.osmnx_export_format
        ext = {'GEOPACKAGE': '.gpkg', 'OSM_XML': '.osm',
               'GRAPHML_GEPHI': '.graphml', 'SVG': '.svg'}[fmt]
        name = context.active_object.name.replace(" ", "_")
        default = props.osmnx_export_filepath or f"//{name}{ext}"
        if not default.endswith(ext):
            default = os.path.splitext(default)[0] + ext
        self.filepath = default
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = context.scene.scigraphs
        fmt = props.osmnx_export_format
        obj = context.active_object

        G = _get_osmnx_graph(obj)
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}

        filepath = bpy.path.abspath(self.filepath)

        if fmt == 'GEOPACKAGE':
            ok, msg = _save_graph_geopackage(G, filepath)
        elif fmt == 'OSM_XML':
            ok, msg = _save_graph_osm_xml(G, filepath)
        elif fmt == 'GRAPHML_GEPHI':
            ok, msg = _save_graph_graphml_gephi(G, filepath)
        elif fmt == 'SVG':
            ok, msg = _save_graph_svg(G, filepath)
        else:
            self.report({'ERROR'}, f"Unknown format {fmt}")
            return {'CANCELLED'}

        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        props.osmnx_export_filepath = filepath
        self.report({'INFO'}, msg)
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_OSMnxExport,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
