import bpy
import os
from bpy.props import StringProperty
from ....core import osmnx_analysis
from .utils import _get_osmnx_graph, _store_osmnx_graph


class SCIGRAPHS_OT_SaveGraphML(bpy.types.Operator):
    """Save the OSMnx graph to GraphML format."""
    bl_idname = "scigraphs.osmnx_save_graphml"
    bl_label = "Save GraphML"
    bl_description = "Export the graph to GraphML format for use in other tools"
    bl_options = {'REGISTER'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to save the GraphML file",
        subtype='FILE_PATH',
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        if props.osmnx_graphml_path:
            self.filepath = props.osmnx_graphml_path
        else:
            self.filepath = "//network.graphml"
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph found")
            return {'CANCELLED'}
        
        filepath = bpy.path.abspath(self.filepath)
        
        if not filepath.lower().endswith('.graphml'):
            filepath += '.graphml'
        
        success = osmnx_analysis.save_graph_graphml(G, filepath)
        
        if not success:
            self.report({'ERROR'}, "Failed to save GraphML file")
            return {'CANCELLED'}
        
        props.osmnx_graphml_path = filepath
        self.report({'INFO'}, f"Graph saved to {filepath}")
        return {'FINISHED'}


class SCIGRAPHS_OT_LoadGraphML(bpy.types.Operator):
    """Load an OSMnx graph from GraphML format."""
    bl_idname = "scigraphs.osmnx_load_graphml"
    bl_label = "Load GraphML"
    bl_description = "Import a graph from GraphML format"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to the GraphML file",
        subtype='FILE_PATH',
    )
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        if props.osmnx_graphml_path:
            self.filepath = props.osmnx_graphml_path
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        filepath = bpy.path.abspath(self.filepath)
        
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}
        
        G = osmnx_analysis.load_graph_graphml(filepath)
        
        if G is None:
            self.report({'ERROR'}, "Failed to load GraphML file")
            return {'CANCELLED'}
        
        from ....core import geometry, importer
        
        graph_data, edge_geometries = importer.osmnx_to_graph_data(G, retain_geometry=True)
        
        if graph_data is None:
            self.report({'ERROR'}, "Failed to convert loaded graph")
            return {'CANCELLED'}
        
        props = context.scene.scigraphs
        scale = props.osmnx_scale
        
        obj = geometry.create_osmnx_graph_object(
            graph_data, edge_geometries, scale=scale, retain_geometry=True
        )
        
        if obj:
            _store_osmnx_graph(obj, G)
            obj["osmnx_scale"] = scale
            
            props.osmnx_graphml_path = filepath
            filename = os.path.basename(filepath)
            self.report({'INFO'}, f"Loaded graph from {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return {'FINISHED'}
        
        self.report({'ERROR'}, "Failed to create graph object")
        return {'CANCELLED'}


class SCIGRAPHS_OT_SaveToCache(bpy.types.Operator):
    """Save the OSMnx graph to cache directory for automatic reloading."""
    bl_idname = "scigraphs.osmnx_save_to_cache"
    bl_label = "Save to Cache"
    bl_description = "Save graph to cache directory for automatic reloading when Blender restarts"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.get("is_osmnx", False)
    
    def execute(self, context):
        obj = context.active_object
        G = _get_osmnx_graph(obj)
        
        if G is None:
            self.report({'ERROR'}, "No OSMnx graph in memory. Cannot save to cache.")
            return {'CANCELLED'}
        
        from ....core.osmnx import cache
        
        success, filepath, message = cache.save_graph_to_cache(obj, G)
        
        if success:
            filename = os.path.basename(filepath) if filepath else ""
            self.report({'INFO'}, f"Graph cached as: {filename}")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, f"Failed to save to cache: {message}")
            return {'CANCELLED'}


classes = [
    SCIGRAPHS_OT_SaveGraphML,
    SCIGRAPHS_OT_LoadGraphML,
    SCIGRAPHS_OT_SaveToCache,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

