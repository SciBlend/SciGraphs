"""Operators to load OD matrices, turn them into graphs, and draw the flows.

``core.city2graph.mobility`` computes and returns GeoDataFrames and NetworkX
graphs; the two helpers here are the only step that touches bpy, which is what
keeps that module importable without Blender.
"""

import bpy
from bpy.props import StringProperty


def create_od_graph_blender(od_data, zones_gdf, zone_id_col, osmnx_obj=None,
                            matrix_type="edgelist", directed=True,
                            weight_cols=None, threshold=None):
    """Create a Blender mesh graph from OD matrix data, or None on failure.

    ``matrix_type`` is 'edgelist' or 'adjacency'; ``osmnx_obj`` aligns the
    result with an existing street network.
    """
    from scigraphs_core.city2graph.mobility import od_matrix_to_graph
    from ....core.city2graph.morphology import create_graph_from_networkx
    from scigraphs_core.logger import log

    graph = od_matrix_to_graph(
        od_data=od_data,
        zones_gdf=zones_gdf,
        zone_id_col=zone_id_col,
        matrix_type=matrix_type,
        weight_cols=weight_cols,
        threshold=threshold,
        directed=directed,
        as_nx=True,
    )

    if graph is None:
        return None

    try:
        graph_obj = create_graph_from_networkx(
            graph,
            name="OD_Graph",
            use_positions=True,
            osmnx_obj=osmnx_obj,
        )

        if graph_obj:
            graph_obj["is_od_graph"] = True
            graph_obj["od_directed"] = directed
            if threshold:
                graph_obj["od_threshold"] = threshold

        return graph_obj

    except Exception as e:
        log(f"Error creating Blender OD graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_od_flows(od_data, zones_gdf, zone_id_col, feature_obj=None,
                       weight_col=None, threshold=None, curve_thickness=0.0002,
                       limit=1000):
    """Draw OD flows as Blender curves and return the created objects.

    ``mobility.od_flow_edges`` produces the geometry; this only turns the
    GeoDataFrame into curves. ``feature_obj`` supplies the coordinate transform.
    """
    from scigraphs_core.city2graph.mobility import od_flow_edges
    from ....core.mesh.geo_mesh import create_curves_from_gdf
    from scigraphs_core.logger import log

    edges_gdf = od_flow_edges(
        od_data=od_data,
        zones_gdf=zones_gdf,
        zone_id_col=zone_id_col,
        weight_col=weight_col,
        threshold=threshold,
        limit=limit,
    )

    if edges_gdf is None or len(edges_gdf) == 0:
        return []

    try:
        curves_obj = create_curves_from_gdf(
            edges_gdf,
            name="OD_Flows",
            feature_obj=feature_obj,
            thickness=curve_thickness,
            limit=limit,
        )

        if curves_obj:
            curves_obj["is_od_flow"] = True
            log(f"Created OD flow curves with {len(edges_gdf)} edges")
            return [curves_obj]

        return []

    except Exception as e:
        log(f"Error visualizing OD flows: {e}")
        import traceback
        traceback.print_exc()
        return []


class SCIGRAPHS_OT_C2G_LoadODMatrix(bpy.types.Operator):
    """Load an OD matrix CSV file into the scene."""
    bl_idname = "scigraphs.c2g_load_od_matrix"
    bl_label = "Load OD Matrix"
    bl_description = "Load Origin-Destination matrix from CSV (edgelist or adjacency)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.csv", options={'HIDDEN'})

    def execute(self, context):
        props = context.scene.city2graph

        path = self.filepath or bpy.path.abspath(props.od_matrix_path)
        if not path:
            self.report({'ERROR'}, "No OD matrix file specified")
            return {'CANCELLED'}

        try:
            import pandas as pd
            od_data = pd.read_csv(path)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read CSV: {e}")
            return {'CANCELLED'}

        import pickle
        import base64
        context.scene["c2g_od_data"] = base64.b64encode(pickle.dumps(od_data)).decode("ascii")
        context.scene["c2g_od_path"] = path

        self.report({'INFO'}, f"Loaded OD matrix: {len(od_data)} rows from {path}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SCIGRAPHS_OT_C2G_ODToGraph(bpy.types.Operator):
    """Convert loaded OD matrix into a spatial graph."""
    bl_idname = "scigraphs.c2g_od_to_graph"
    bl_label = "OD Matrix → Graph"
    bl_description = "Convert the loaded OD data and zones into a Blender graph"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return "c2g_od_data" in context.scene

    def execute(self, context):
        from ....core.city2graph import utils as c2g_utils

        props = context.scene.city2graph

        zones_obj = props.od_zones_object
        if not zones_obj or zones_obj.type != 'MESH':
            self.report({'ERROR'}, "Select a zones object (mesh with zone polygons)")
            return {'CANCELLED'}

        zones_gdf = c2g_utils.blender_to_geopandas(zones_obj)
        if zones_gdf is None or len(zones_gdf) == 0:
            self.report({'ERROR'}, "Could not extract zone geometries")
            return {'CANCELLED'}

        import pickle
        import base64
        od_data = pickle.loads(base64.b64decode(context.scene["c2g_od_data"]))

        zone_id_col = props.od_zone_id_col.strip() or None
        matrix_type = props.od_matrix_type.lower()
        threshold = props.od_threshold if props.od_threshold > 0 else None
        directed = props.od_directed

        weight_cols = None
        wc = props.od_weight_col.strip()
        if wc:
            weight_cols = [wc]

        osmnx_obj = context.active_object if (
            context.active_object and context.active_object.get("is_osmnx")
        ) else None

        self.report({'INFO'}, "Converting OD matrix to graph...")

        graph_obj = create_od_graph_blender(
            od_data=od_data,
            zones_gdf=zones_gdf,
            zone_id_col=zone_id_col,
            osmnx_obj=osmnx_obj,
            matrix_type=matrix_type,
            directed=directed,
            weight_cols=weight_cols,
            threshold=threshold,
        )

        if graph_obj is None:
            self.report({'ERROR'}, "Failed to create OD graph")
            return {'CANCELLED'}

        context.view_layer.objects.active = graph_obj
        self.report({'INFO'}, f"OD graph created: {graph_obj.get('num_nodes', 0)} nodes, {graph_obj.get('num_edges', 0)} edges")
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_C2G_LoadODMatrix,
    SCIGRAPHS_OT_C2G_ODToGraph,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
