"""
OSMnx Cache Management Operators

Operators for viewing and managing cached OSMnx graphs.
"""

import bpy
import os
from datetime import datetime
from bpy.props import StringProperty, IntProperty


class SCIGRAPHS_OT_ViewCachedGraphs(bpy.types.Operator):
    """View and manage cached OSMnx graphs."""
    bl_idname = "scigraphs.osmnx_view_cached_graphs"
    bl_label = "Manage Cached Graphs"
    bl_description = "View and delete cached OSMnx graphs"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=600)
    
    def draw(self, context):
        layout = self.layout
        
        from ....core.osmnx import cache
        
        cached_graphs = cache.list_cached_graphs()
        
        if not cached_graphs:
            layout.label(text="No cached graphs found", icon='INFO')
            
            box = layout.box()
            cache_dir = cache.get_cache_directory()
            box.label(text=f"Cache directory: {cache_dir}")
            
            return
        
        layout.label(text=f"Found {len(cached_graphs)} cached graph(s):", icon='FILE_CACHE')
        
        box = layout.box()
        cache_dir = cache.get_cache_directory()
        box.label(text=f"Cache directory: {cache_dir}")
        
        layout.separator()
        
        # List all cached graphs
        for i, (filename, filepath, size_mb, modified_time) in enumerate(cached_graphs):
            box = layout.box()
            
            col = box.column(align=True)
            
            # Filename
            row = col.row()
            row.label(text=filename, icon='FILE')
            
            # File info
            row = col.row()
            row.label(text=f"Size: {size_mb:.2f} MB")
            
            # Modified time
            try:
                mod_date = datetime.fromtimestamp(modified_time)
                date_str = mod_date.strftime("%Y-%m-%d %H:%M:%S")
                row = col.row()
                row.label(text=f"Modified: {date_str}")
            except:
                pass
            
            # Action buttons
            row = col.row(align=True)
            op = row.operator("scigraphs.osmnx_load_from_cache", text="Load", icon='IMPORT')
            op.filepath = filepath
            op.filename = filename
            
            op = row.operator("scigraphs.osmnx_delete_cached_graph", text="Delete", icon='TRASH')
            op.filepath = filepath
            op.filename = filename
        
        layout.separator()
        
        # Clear all button
        row = layout.row()
        row.operator("scigraphs.osmnx_clear_cache", text="Clear All Cache", icon='TRASH')


class SCIGRAPHS_OT_DeleteCachedGraph(bpy.types.Operator):
    """Delete a cached OSMnx graph."""
    bl_idname = "scigraphs.osmnx_delete_cached_graph"
    bl_label = "Delete Cached Graph"
    bl_description = "Delete this cached graph file"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to the cached graph file",
    )
    
    filename: StringProperty(
        name="Filename",
        description="Name of the cached graph file",
    )
    
    def execute(self, context):
        from ....core.osmnx import cache
        
        success = cache.delete_cached_graph(self.filepath)
        
        if success:
            self.report({'INFO'}, f"Deleted: {self.filename}")
            # Refresh the view
            bpy.ops.scigraphs.osmnx_view_cached_graphs('INVOKE_DEFAULT')
        else:
            self.report({'ERROR'}, f"Failed to delete: {self.filename}")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    
    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Delete {self.filename}?")
        layout.label(text="This action cannot be undone.", icon='ERROR')


class SCIGRAPHS_OT_ClearCache(bpy.types.Operator):
    """Clear all cached OSMnx graphs."""
    bl_idname = "scigraphs.osmnx_clear_cache"
    bl_label = "Clear All Cache"
    bl_description = "Delete all cached OSMnx graphs"
    bl_options = {'REGISTER', 'INTERNAL'}
    
    def execute(self, context):
        from ....core.osmnx import cache
        
        success_count, error_count = cache.clear_all_cache()
        
        if success_count > 0:
            self.report({'INFO'}, f"Deleted {success_count} cached graph(s)")
        
        if error_count > 0:
            self.report({'WARNING'}, f"Failed to delete {error_count} file(s)")
        
        if success_count == 0 and error_count == 0:
            self.report({'INFO'}, "No cached graphs to delete")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    
    def draw(self, context):
        layout = self.layout
        layout.label(text="Delete all cached graphs?")
        layout.label(text="This action cannot be undone.", icon='ERROR')


class SCIGRAPHS_OT_OpenCacheDirectory(bpy.types.Operator):
    """Open the cache directory in file browser."""
    bl_idname = "scigraphs.osmnx_open_cache_directory"
    bl_label = "Open Cache Directory"
    bl_description = "Open the cache directory in your system's file browser"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        from ....core.osmnx import cache
        import subprocess
        import sys
        
        cache_dir = cache.get_cache_directory()
        
        if not os.path.exists(cache_dir):
            cache.ensure_cache_directory()
        
        try:
            if sys.platform == 'win32':
                os.startfile(cache_dir)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', cache_dir])
            else:  # linux
                subprocess.Popen(['xdg-open', cache_dir])
            
            self.report({'INFO'}, f"Opened: {cache_dir}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to open directory: {e}")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_LoadFromCache(bpy.types.Operator):
    """Load an OSMnx graph from cache."""
    bl_idname = "scigraphs.osmnx_load_from_cache"
    bl_label = "Load from Cache"
    bl_description = "Load this cached graph into the scene"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to the cached graph file",
    )
    
    filename: StringProperty(
        name="Filename",
        description="Name of the cached graph file",
    )
    
    def execute(self, context):
        from ....core.osmnx import cache
        from ....core import osmnx_analysis
        
        if not os.path.exists(self.filepath):
            self.report({'ERROR'}, f"File not found: {self.filepath}")
            return {'CANCELLED'}
        
        # Load graph from GraphML
        G = osmnx_analysis.load_graph_graphml(self.filepath)
        
        if G is None:
            self.report({'ERROR'}, "Failed to load GraphML file")
            return {'CANCELLED'}
        
        # Convert to graph_data
        from ....core import importer, geometry
        
        graph_data, edge_geometries = importer.osmnx_to_graph_data(G, retain_geometry=True)
        
        if graph_data is None:
            self.report({'ERROR'}, "Failed to convert loaded graph")
            return {'CANCELLED'}
        
        # Create Blender object
        props = context.scene.scigraphs
        scale = props.osmnx_scale
        
        obj = geometry.create_osmnx_graph_object(
            graph_data, edge_geometries, scale=scale, retain_geometry=True
        )
        
        if obj:
            # Store in memory cache
            from .utils import _store_osmnx_graph
            _store_osmnx_graph(obj, G)
            obj["osmnx_scale"] = scale
            
            # Try to extract metadata from filename
            # Format: Location_Name_networktype.graphml
            base_name = os.path.splitext(self.filename)[0]
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2:
                obj["osmnx_query_name"] = parts[0].replace('_', ' ')
                obj["osmnx_network_type"] = parts[1]
            
            self.report({'INFO'}, f"Loaded: {self.filename} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            return {'FINISHED'}
        
        self.report({'ERROR'}, "Failed to create graph object")
        return {'CANCELLED'}


classes = [
    SCIGRAPHS_OT_ViewCachedGraphs,
    SCIGRAPHS_OT_DeleteCachedGraph,
    SCIGRAPHS_OT_ClearCache,
    SCIGRAPHS_OT_OpenCacheDirectory,
    SCIGRAPHS_OT_LoadFromCache,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

