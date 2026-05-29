# SuiteSparse Matrix Collection import operators

import bpy
from ...view_utils import focus_graph_in_top_view


class SCIGRAPHS_OT_DownloadSuiteSparse(bpy.types.Operator):
    """Download a matrix from SuiteSparse and create a graph."""
    bl_idname = "scigraphs.download_suitesparse"
    bl_label = "Download & Import"
    bl_description = "Download matrix from SuiteSparse Matrix Collection and create graph"
    
    def execute(self, context):
        from ....core import suitesparse_importer, geometry
        from .data_operators import SCIGRAPHS_AutoLayoutOnImport
        
        props = context.scene.scigraphs
        identifier = props.suitesparse_id.strip()
        
        if not identifier:
            self.report({'WARNING'}, "Please enter a matrix identifier (e.g. Grund/bayer09)")
            return {'CANCELLED'}
        
        # Validate identifier format
        group, name = suitesparse_importer.parse_matrix_id(identifier)
        if not group or not name:
            self.report({'ERROR'}, f"Invalid identifier: '{identifier}'. Use 'Group/Name' format.")
            return {'CANCELLED'}
        
        props.suitesparse_status = f"Downloading {group}/{name}..."
        
        # Download and build graph
        graph_data = suitesparse_importer.load_suitesparse_graph(
            identifier,
            mode=props.suitesparse_mode,
            giant_only=props.suitesparse_giant_only,
        )
        
        if graph_data is None:
            props.suitesparse_status = "Download failed. Check console."
            self.report({'ERROR'}, "Could not download or parse matrix. Check console for details.")
            return {'CANCELLED'}
        
        # Create graph object using existing pipeline
        obj = geometry.create_graph_object(graph_data, is_directed=False)
        auto_layout_applied = SCIGRAPHS_AutoLayoutOnImport.apply(context, obj, self)
        focus_graph_in_top_view(context, obj)
        
        mode_label = "bipartite" if props.suitesparse_mode == 'BIPARTITE' else "symmetric"
        coord_label = " + coords" if getattr(graph_data, 'has_coordinates', False) else ""
        layout_label = f" + {props.layout_algorithm} layout" if auto_layout_applied else ""
        status = f"{group}/{name} ({mode_label}{coord_label}): {len(graph_data.nodes)} nodes, {len(graph_data.edges)} edges"
        status = f"{status}{layout_label}"
        props.suitesparse_status = status
        
        self.report({'INFO'}, f"SuiteSparse graph created: {status}")
        return {'FINISHED'}


class SCIGRAPHS_OT_BrowseSuiteSparse(bpy.types.Operator):
    """Open SuiteSparse website in browser."""
    bl_idname = "scigraphs.browse_suitesparse"
    bl_label = "Browse Collection"
    bl_description = "Open SuiteSparse Matrix Collection website in your browser"
    
    def execute(self, context):
        import webbrowser
        webbrowser.open("https://sparse.tamu.edu")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_DownloadSuiteSparse)
    bpy.utils.register_class(SCIGRAPHS_OT_BrowseSuiteSparse)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_OT_BrowseSuiteSparse)
    bpy.utils.unregister_class(SCIGRAPHS_OT_DownloadSuiteSparse)
